from argparse import ArgumentParser
from pathlib import Path
import re
import random

import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.func import functional_call
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lib.canonicalizer import Canonicalizer
from lib.serial_model import _build_overrides, load_serialized, serialize_model
from lib.serial_params import Symmeters


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "google/multiberts-seed_0"
DEFAULT_SERIALIZED_PATH = ROOT_DIR / "data" / "multiberts" / "serialized" / "seed_0.pt"
DEFAULT_TEXTS = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
    "The quick brown fox jumps over the dog.",
    "Machine learning is a subset of science.",
]


def parse_args():
    parser = ArgumentParser(description="Overfit canonicalization against a deterministic random permutation.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--serialized-path", type=Path, default=DEFAULT_SERIALIZED_PATH)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--perm-seed", type=int, default=0)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-decay", type=float, default=0.9995)
    parser.add_argument("--tau-min", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=1,
        help="How many initial global optimization steps to log very verbosely. Set to 0 to disable.",
    )
    parser.add_argument("--canonicalizer-device", default="cuda")
    parser.add_argument("--model-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--layers-per-stage",
        type=int,
        default=2,
        help="How many encoder layers to canonicalize together in each default layer stage.",
    )
    parser.add_argument(
        "--stage-order",
        default=None,
        help="Comma-separated stage names. Supports encoder, decoder, layerN, or layersAtoB. Overrides --layers-per-stage.",
    )
    parser.add_argument(
        "--stage-sampling",
        choices=("random", "sequential"),
        default="random",
        help="How to choose which stage to train next. Random shuffles stages each round; sequential keeps a fixed order.",
    )
    parser.add_argument(
        "--stage-sample-seed",
        type=int,
        default=None,
        help="Optional RNG seed for random stage sampling. Defaults to perm-seed.",
    )
    parser.add_argument("--project", default="canonicalization-overfit", help="Weights & Biases project name.")
    parser.add_argument("--run-name", default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--disable-wandb", action="store_true", help="Skip Weights & Biases logging.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    return torch.device(device_name)


def clone_symmeters_to_device(symmeters: Symmeters, device: torch.device) -> Symmeters:
    moved = Symmeters(symmeters.symmetry_names)
    for symmetry_name, component_name, component in symmeters.iter_components():
        tensor = component.tensor.detach().to(device)
        tensor.requires_grad_(False)
        moved.add_component(
            symmetry_name,
            component_name,
            tensor,
            axes=component.axes,
            kind=component.kind,
            layout=component.layout,
            parameter_keys=component.parameter_keys,
        )
    return moved


def rand_perm_matrix(size: int, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(size, generator=generator)
    return torch.eye(size)[perm]


def deterministic_permutations(symmeters: Symmeters, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perms: dict[str, torch.Tensor] = {}
    for symmetry_name in symmeters.ordered_transform_names():
        if symmetry_name.endswith(".head"):
            continue
        component_specs = symmeters.components_with_axis(symmetry_name)
        if not component_specs:
            continue
        bank_axis = symmeters.transform_bank_axis(symmetry_name)
        size = symmeters.symmetry_size(symmetry_name)
        if bank_axis is None:
            perms[symmetry_name] = rand_perm_matrix(size, generator)
            continue
        bank_size = symmeters.symmetry_size(bank_axis)
        perms[symmetry_name] = torch.stack([rand_perm_matrix(size, generator) for _ in range(bank_size)])

    head_names = [name for name in symmeters.ordered_transform_names() if name.endswith(".head")]
    if head_names:
        perms["head"] = rand_perm_matrix(symmeters.symmetry_size(head_names[0]), generator)
    return perms


def lerp_symmeters(a: Symmeters, b: Symmeters, alpha: float) -> Symmeters:
    out = Symmeters([])
    for symmetry_name, component_name, component_a in a.iter_components():
        component_b = b.component(symmetry_name, component_name)
        out.add_component(
            symmetry_name,
            component_name,
            torch.lerp(component_a.tensor, component_b.tensor, alpha),
            axes=component_a.axes,
            kind=component_a.kind,
            layout=component_a.layout,
            parameter_keys=component_a.parameter_keys,
        )
    return out


def move_overrides(overrides: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in overrides.items()}


def module_key(symmetry_name: str) -> str:
    return symmetry_name.replace(".", "__dot__")


def load_source_symmeters(args, canonicalizer_device: torch.device) -> Symmeters:
    if args.serialized_path.exists():
        try:
            return clone_symmeters_to_device(Symmeters.load(str(args.serialized_path)), canonicalizer_device)
        except ValueError as exc:
            print(f"ignoring serialized checkpoint at {args.serialized_path}: {exc}")

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).eval()
    return clone_symmeters_to_device(serialize_model(model), canonicalizer_device)


def cuda_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.memory_allocated(device) / 1024 ** 3


def trace_enabled(args, global_step: int) -> bool:
    return global_step < args.trace_steps


def tensor_summary(tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    shape = tuple(detached.shape)
    summary = f"shape={shape} dtype={detached.dtype} device={detached.device}"
    if detached.numel() == 0:
        return summary + " numel=0"

    flattened = detached.reshape(-1)
    is_floating = torch.is_floating_point(detached) or torch.is_complex(detached)
    if not is_floating:
        min_value = flattened.min().item()
        max_value = flattened.max().item()
        return summary + f" min={min_value} max={max_value}"

    finite_mask = torch.isfinite(flattened)
    finite_count = int(finite_mask.sum().item())
    summary += f" finite={finite_count}/{flattened.numel()}"
    if finite_count == 0:
        return summary

    finite_values = flattened[finite_mask]
    mean_value = finite_values.mean().item()
    std_value = finite_values.std(unbiased=False).item() if finite_values.numel() > 1 else 0.0
    min_value = finite_values.min().item()
    max_value = finite_values.max().item()
    abs_max = finite_values.abs().max().item()
    return (
        summary
        + f" mean={mean_value:.6g} std={std_value:.6g} min={min_value:.6g}"
        + f" max={max_value:.6g} abs_max={abs_max:.6g}"
    )


def trace_header(title: str):
    print(f"TRACE {title}")


def trace_batch(batch: dict[str, torch.Tensor], tokenizer: AutoTokenizer):
    trace_header("batch")
    for key, value in batch.items():
        print(f"  batch[{key}] {tensor_summary(value)}")
    if "input_ids" in batch:
        decoded = tokenizer.batch_decode(batch["input_ids"].detach().cpu(), skip_special_tokens=False)
        for index, text in enumerate(decoded):
            print(f"  decoded[{index}]={text}")


def trace_symmeters(symmeters: Symmeters, title: str, symmetry_names: tuple[str, ...] | None = None):
    trace_header(title)
    names = tuple(symmetry_names) if symmetry_names is not None else tuple(symmeters.symmetry_names)
    print(f"  symmetry_names={names}")
    for symmetry_name in names:
        owned = symmeters.owned_components(symmetry_name)
        print(f"  symmetry={symmetry_name} owned_component_count={len(owned)}")
        for component_name, component in owned.items():
            print(
                f"    component={component_name} axes={component.axes} kind={component.kind} "
                f"layout={component.layout} parameter_keys={component.parameter_keys} {tensor_summary(component.tensor)}"
            )


def trace_permutations(permutations: dict[str, torch.Tensor]):
    trace_header("permutations")
    for symmetry_name, matrix in permutations.items():
        print(f"  permutation[{symmetry_name}] {tensor_summary(matrix)}")


def trace_transport_debug(
    canonicalizer: Canonicalizer,
    symmeters: Symmeters,
    active_symmetry_names: tuple[str, ...],
    tau: float,
    permutations: dict[str, torch.Tensor],
):
    trace_header("transport_inputs")
    for symmetry_name in active_transport_names(active_symmetry_names):
        key = module_key(symmetry_name)
        print(f"  symmetry={symmetry_name} tau={tau:.6f}")
        if key in canonicalizer.dimension_aligners:
            evidence = canonicalizer._evidence_tensor(symmeters, symmetry_name)
            aligner = canonicalizer.dimension_aligners[key]
            print(f"    evidence {tensor_summary(evidence)}")
            print(f"    W_q {tensor_summary(aligner.W_q)}")
            print(f"    W_k {tensor_summary(aligner.W_k)}")
        elif key in canonicalizer.head_aligners:
            aligner = canonicalizer.head_aligners[key]
            print(f"    head_aligner_scale={aligner.scale:.6f} sinkhorn_iters={aligner.sinkhorn_iters}")
        else:
            print("    no aligner module found")
            continue

        with torch.no_grad():
            transport = compute_transport_matrices(
                canonicalizer,
                symmeters,
                (symmetry_name,) if symmetry_name != "model" else (),
                tau=tau,
            )[symmetry_name]
        print(f"    transport {tensor_summary(transport)}")
        target = target_transport_for_symmetry(symmetry_name, permutations)
        if target is not None:
            print(f"    target_transport {tensor_summary(target)}")
            for metric_name, metric_value in transport_metrics(transport, target=target).items():
                print(f"    metric[{metric_name}]={metric_value:.6g}")


def trace_overrides(overrides: dict[str, torch.Tensor], alpha: float):
    trace_header(f"overrides alpha={alpha}")
    for name, value in overrides.items():
        print(f"  override[{name}] {tensor_summary(value)}")


def trace_iteration_context(
    *,
    args,
    stage_name: str,
    stage_step: int,
    global_step: int,
    tau: float,
    batch: dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    source_symmeters: Symmeters,
    current_symmeters: Symmeters,
    active_symmetry_names: tuple[str, ...],
    permutations: dict[str, torch.Tensor],
    canonicalizer: Canonicalizer,
    target_logits: torch.Tensor,
):
    if not trace_enabled(args, global_step):
        return
    trace_header("iteration_context")
    print(
        f"  stage={stage_name} stage_step={stage_step} global_step={global_step} tau={tau:.6f} "
        f"active_symmetry_names={active_symmetry_names}"
    )
    trace_batch(batch, tokenizer)
    trace_symmeters(source_symmeters, "source_symmeters")
    trace_symmeters(current_symmeters, "current_symmeters", symmetry_names=active_transport_names(active_symmetry_names))
    trace_permutations(permutations)
    trace_transport_debug(canonicalizer, current_symmeters, active_symmetry_names, tau, permutations)
    trace_header("target_logits")
    print(f"  target_logits {tensor_summary(target_logits)}")


def layer_stage_names(symmeters: Symmeters) -> list[str]:
    ordered = symmeters.ordered_transform_names()
    layer_prefixes = sorted(
        {name.split(".", 1)[0] for name in ordered if name.startswith("L")},
        key=lambda prefix: int(prefix.removeprefix("L")),
    )
    return [f"layer{prefix[1:]}" for prefix in layer_prefixes]


def layer_indices(symmeters: Symmeters) -> list[int]:
    return [int(stage_name.removeprefix("layer")) for stage_name in layer_stage_names(symmeters)]


def _layer_stage_label(layer_group: list[int]) -> str:
    if len(layer_group) == 1:
        return f"layer{layer_group[0]}"
    return f"layers{layer_group[0]}to{layer_group[-1]}"


def _active_symmetry_names_for_layers(symmeters: Symmeters, layer_group: list[int]) -> tuple[str, ...]:
    layer_prefixes = tuple(f"L{layer_index}." for layer_index in layer_group)
    return tuple(
        symmetry_name
        for symmetry_name in symmeters.ordered_transform_names()
        if symmetry_name.startswith(layer_prefixes)
    )


def default_stage_specs(symmeters: Symmeters, layers_per_stage: int) -> list[tuple[str, tuple[str, ...]]]:
    if layers_per_stage <= 0:
        raise ValueError("layers_per_stage must be positive.")

    specs: list[tuple[str, tuple[str, ...]]] = [("encoder", ())]
    indices = layer_indices(symmeters)
    for start in range(0, len(indices), layers_per_stage):
        layer_group = indices[start:start + layers_per_stage]
        specs.append((_layer_stage_label(layer_group), _active_symmetry_names_for_layers(symmeters, layer_group)))

    if "decoder" in symmeters:
        specs.append(("decoder", ("decoder",)))
    return specs


def _manual_stage_spec(symmeters: Symmeters, stage_name: str) -> tuple[str, tuple[str, ...]]:
    if stage_name == "encoder":
        return (stage_name, ())
    if stage_name == "decoder":
        if "decoder" not in symmeters:
            return (stage_name, ())
        return (stage_name, ("decoder",))
    if stage_name.startswith("layer"):
        layer_index = stage_name.removeprefix("layer")
        return (stage_name, _active_symmetry_names_for_layers(symmeters, [int(layer_index)]))

    match = re.fullmatch(r"layers(\d+)to(\d+)", stage_name)
    if match is not None:
        start, end = (int(match.group(1)), int(match.group(2)))
        if end < start:
            raise ValueError(f"Invalid stage {stage_name}: end layer must be >= start layer.")
        layer_group = list(range(start, end + 1))
        return (stage_name, _active_symmetry_names_for_layers(symmeters, layer_group))

    raise ValueError(f"Unknown stage {stage_name}.")


def stage_specs_from_args(args, symmeters: Symmeters) -> list[tuple[str, tuple[str, ...]]]:
    if args.stage_order is None:
        return default_stage_specs(symmeters, layers_per_stage=args.layers_per_stage)
    return [
        _manual_stage_spec(symmeters, stage_name.strip())
        for stage_name in args.stage_order.split(",")
        if stage_name.strip()
    ]


def filtered_stage_specs(stage_specs: list[tuple[str, tuple[str, ...]]]) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (stage_name, active_symmetry_names)
        for stage_name, active_symmetry_names in stage_specs
        if stage_name == "encoder" or active_symmetry_names
    ]


def stage_execution_order(
    args,
    active_stage_specs: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...], int]]:
    sample_seed = args.perm_seed if args.stage_sample_seed is None else args.stage_sample_seed
    sampler = random.Random(sample_seed)
    execution_plan: list[tuple[str, tuple[str, ...], int]] = []
    for round_index in range(args.steps):
        round_specs = list(active_stage_specs)
        if args.stage_sampling == "random":
            sampler.shuffle(round_specs)
        execution_plan.extend(
            (stage_name, active_symmetry_names, round_index)
            for stage_name, active_symmetry_names in round_specs
        )
    return execution_plan


def target_transport_for_symmetry(symmetry_name: str, permutations: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if symmetry_name == "model":
        matrix = permutations.get("model")
    elif symmetry_name.endswith(".head"):
        matrix = permutations.get("head")
    else:
        matrix = permutations.get(symmetry_name)
    if matrix is None:
        return None
    return matrix.transpose(-1, -2)


def active_transport_names(active_symmetry_names: tuple[str, ...]) -> tuple[str, ...]:
    return ("model", *active_symmetry_names)


def compute_transport_matrices(
    canonicalizer: Canonicalizer,
    symmeters: Symmeters,
    active_symmetry_names: tuple[str, ...],
    tau: float,
) -> dict[str, torch.Tensor]:
    transports: dict[str, torch.Tensor] = {}
    for symmetry_name in active_transport_names(active_symmetry_names):
        key = module_key(symmetry_name)
        if key in canonicalizer.dimension_aligners:
            evidence = canonicalizer._evidence_tensor(symmeters, symmetry_name)
            if evidence.ndim == 2:
                matrix = canonicalizer.dimension_aligners[key](evidence.unsqueeze(0), tau=tau).squeeze(0)
            else:
                matrix = canonicalizer.dimension_aligners[key](evidence, tau=tau)
            transports[symmetry_name] = matrix
            continue
        if key in canonicalizer.head_aligners:
            transports[symmetry_name] = canonicalizer.head_aligners[key](symmeters, tau=tau)
    return transports


def rowwise_top2_gap(transport: torch.Tensor) -> torch.Tensor:
    flat = transport.reshape(-1, transport.shape[-1])
    top2 = flat.topk(k=min(2, flat.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def transport_metrics(transport: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, float]:
    flat = transport.reshape(-1, transport.shape[-1])
    max_prob = flat.max(dim=-1).values
    entropy = -(flat.clamp_min(1e-9) * flat.clamp_min(1e-9).log()).sum(dim=-1)
    normalized_entropy = entropy / torch.log(torch.tensor(flat.shape[-1], device=flat.device, dtype=flat.dtype))
    metrics = {
        "max_prob_mean": max_prob.mean().item(),
        "max_prob_min": max_prob.min().item(),
        "top2_gap_mean": rowwise_top2_gap(transport).mean().item(),
        "normalized_entropy_mean": normalized_entropy.mean().item(),
    }
    if target is not None:
        target_flat = target.reshape(-1, target.shape[-1]).to(device=flat.device, dtype=flat.dtype)
        metrics["target_mass_mean"] = (flat * target_flat).sum(dim=-1).mean().item()
    return metrics


def build_wandb_payload(
    *,
    stage_name: str,
    stage_step: int,
    global_step: int,
    tau: float,
    total_loss: float,
    grad_norm: float | None,
    gpu_mem: float | None,
    skipped_step: bool,
    active_symmetry_names: tuple[str, ...],
    transport_matrices: dict[str, torch.Tensor],
    permutations: dict[str, torch.Tensor],
) -> dict[str, float]:
    payload: dict[str, float] = {
        "train/loss": total_loss,
        "train/tau": tau,
        "train/global_step": float(global_step),
        "train/stage_step": float(stage_step),
        "train/active_symmetry_count": float(len(active_symmetry_names) + 1),
        "train/skipped_step": float(skipped_step),
    }
    if grad_norm is not None:
        payload["train/grad_norm"] = grad_norm
    if gpu_mem is not None:
        payload["system/gpu_mem_gb"] = gpu_mem

    stage_prefix = f"stage/{stage_name}"
    payload[f"{stage_prefix}/loss"] = total_loss
    payload[f"{stage_prefix}/tau"] = tau
    payload[f"{stage_prefix}/skipped_step"] = float(skipped_step)
    if grad_norm is not None:
        payload[f"{stage_prefix}/grad_norm"] = grad_norm

    for symmetry_name, transport in transport_matrices.items():
        target = target_transport_for_symmetry(symmetry_name, permutations)
        metrics = transport_metrics(transport.detach(), target=target)
        symmetry_prefix = f"transport/{symmetry_name}"
        for metric_name, metric_value in metrics.items():
            payload[f"{symmetry_prefix}/{metric_name}"] = metric_value
            payload[f"{stage_prefix}/{symmetry_name}/{metric_name}"] = metric_value
    return payload


def main():
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    canonicalizer_device = resolve_device(args.canonicalizer_device)
    model_device = resolve_device(args.model_device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    batch = tokenizer(DEFAULT_TEXTS, return_tensors="pt", padding=True, truncation=True)
    batch = {key: value.to(model_device) for key, value in batch.items()}

    source_symmeters = load_source_symmeters(args, canonicalizer_device)
    permuted_symmeters = source_symmeters.clone()
    permutations = deterministic_permutations(permuted_symmeters, seed=args.perm_seed)
    permuted_symmeters.apply_transforms(permutations)

    canonicalizer = Canonicalizer(permuted_symmeters, sinkhorn_iters=args.sinkhorn_iters).to(canonicalizer_device)
    optimizer = torch.optim.AdamW(canonicalizer.parameters(), lr=args.learning_rate)

    shell_model, target_overrides = load_serialized(
        source_symmeters,
        args.model_name,
        local_files_only=args.local_files_only,
    )
    shell_model = shell_model.to(model_device).eval()
    for param in shell_model.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        target_logits = functional_call(
            shell_model,
            move_overrides(target_overrides, model_device),
            (),
            batch,
        ).logits.detach()

    if args.trace_steps > 0:
        trace_header("initial_objects")
        print(f"  source_symmetry_names={tuple(source_symmeters.symmetry_names)}")
        print(f"  permuted_symmetry_names={tuple(permuted_symmeters.symmetry_names)}")
        print(f"  target_overrides_count={len(target_overrides)}")
        print(f"  shell_model_device={model_device} canonicalizer_device={canonicalizer_device}")

    alphas = [0.25, 0.5, 0.75, 1.0]
    stage_specs = stage_specs_from_args(args, source_symmeters)
    active_stage_specs = filtered_stage_specs(stage_specs)
    execution_plan = stage_execution_order(args, active_stage_specs)
    current_symmeters = permuted_symmeters
    wandb_run = None
    if not args.disable_wandb:
        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            dir=str(ROOT_DIR),
            config={
                "model_name": args.model_name,
                "serialized_path": str(args.serialized_path),
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "grad_clip_norm": args.grad_clip_norm,
                "sinkhorn_iters": args.sinkhorn_iters,
                "perm_seed": args.perm_seed,
                "stage_sampling": args.stage_sampling,
                "stage_sample_seed": args.stage_sample_seed,
                "tau_start": args.tau_start,
                "tau_decay": args.tau_decay,
                "tau_min": args.tau_min,
                "layers_per_stage": args.layers_per_stage,
                "stage_order": args.stage_order,
                "active_stages": [name for name, _ in active_stage_specs],
                "canonicalizer_device": str(canonicalizer_device),
                "model_device": str(model_device),
                "alphas": alphas,
                "default_stages": [name for name, _ in stage_specs],
            },
        )
    print(
        f"canonicalizer_device={canonicalizer_device} model_device={model_device} "
        f"steps_per_stage={args.steps} serialized_path={args.serialized_path} "
        f"stages={[name for name, _ in stage_specs]} stage_sampling={args.stage_sampling}"
    )

    global_step = 0
    stage_step_counts = {stage_name: 0 for stage_name, _ in active_stage_specs}
    progress = tqdm(execution_plan, desc="Canonicalization training", total=len(execution_plan))
    try:
        for stage_name, active_symmetry_names, round_index in progress:
            stage_step = stage_step_counts[stage_name]
            tau = max(args.tau_min, args.tau_start * (args.tau_decay ** stage_step))
            progress.set_postfix_str(
                f"stage={stage_name} round={round_index} stage_step={stage_step} tau={tau:.4f}"
            )
            if stage_step == 0:
                print(f"starting stage={stage_name} active_symmetry_names={active_symmetry_names}")

            optimizer.zero_grad(set_to_none=True)
            trace_iteration_context(
                args=args,
                stage_name=stage_name,
                stage_step=stage_step,
                global_step=global_step,
                tau=tau,
                batch=batch,
                tokenizer=tokenizer,
                source_symmeters=source_symmeters,
                current_symmeters=current_symmeters,
                active_symmetry_names=active_symmetry_names,
                permutations=permutations,
                canonicalizer=canonicalizer,
                target_logits=target_logits,
            )
            canonicalized = canonicalizer(
                current_symmeters,
                tau=tau,
                active_symmetry_names=active_symmetry_names,
            )

            if trace_enabled(args, global_step):
                trace_symmeters(
                    canonicalized,
                    "canonicalized_symmeters",
                    symmetry_names=active_transport_names(active_symmetry_names),
                )

            total_loss = 0.0
            encountered_nonfinite = False
            for alpha_index, alpha in enumerate(alphas):
                mixed = lerp_symmeters(source_symmeters, canonicalized, alpha)
                if trace_enabled(args, global_step):
                    trace_symmeters(
                        mixed,
                        f"mixed_symmeters alpha={alpha}",
                        symmetry_names=active_transport_names(active_symmetry_names),
                    )
                overrides = move_overrides(_build_overrides(mixed), model_device)
                if trace_enabled(args, global_step):
                    trace_overrides(overrides, alpha)
                logits = functional_call(shell_model, overrides, (), batch).logits
                if trace_enabled(args, global_step):
                    trace_header(f"branch_logits alpha={alpha}")
                    print(f"  logits {tensor_summary(logits)}")
                branch_loss = F.mse_loss(logits, target_logits) / len(alphas)
                if trace_enabled(args, global_step):
                    trace_header(f"branch_loss alpha={alpha}")
                    print(f"  branch_loss={branch_loss.detach().item():.6g}")
                if not torch.isfinite(branch_loss):
                    encountered_nonfinite = True
                    del mixed, overrides, logits, branch_loss
                    break
                branch_loss.backward(retain_graph=alpha_index < len(alphas) - 1)
                total_loss += branch_loss.detach().item()
                del mixed, overrides, logits, branch_loss

            grad_norm = None
            skipped_step = encountered_nonfinite
            if not encountered_nonfinite:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    canonicalizer.parameters(),
                    max_norm=args.grad_clip_norm,
                )
                grad_norm = float(grad_norm_tensor.item())
                if trace_enabled(args, global_step):
                    trace_header("gradient_clipping")
                    print(f"  grad_norm_before_clip={grad_norm:.6g} clip_threshold={args.grad_clip_norm:.6g}")
                if torch.isfinite(grad_norm_tensor):
                    optimizer.step()
                    if trace_enabled(args, global_step):
                        trace_header("optimizer_step")
                        print("  optimizer.step() applied")
                else:
                    skipped_step = True
                    optimizer.zero_grad(set_to_none=True)
                    if trace_enabled(args, global_step):
                        trace_header("optimizer_step")
                        print("  optimizer.step() skipped due to non-finite gradient norm")
            else:
                optimizer.zero_grad(set_to_none=True)
                if trace_enabled(args, global_step):
                    trace_header("optimizer_step")
                    print("  optimizer.step() skipped due to non-finite branch loss")

            del canonicalized

            with torch.no_grad():
                current_symmeters = clone_symmeters_to_device(
                    canonicalizer(
                        current_symmeters,
                        tau=args.tau_min,
                        active_symmetry_names=active_symmetry_names,
                    ),
                    canonicalizer_device,
                )

            global_step += 1
            stage_step_counts[stage_name] += 1

            if stage_step % args.log_every == 0:
                gpu_mem = cuda_memory_gb(model_device)
                with torch.no_grad():
                    transport_matrices = compute_transport_matrices(
                        canonicalizer,
                        current_symmeters,
                        active_symmetry_names,
                        tau=tau,
                    )
                if gpu_mem is None:
                    print(
                        f"stage={stage_name} step={stage_step} global_step={global_step} "
                        f"loss={total_loss:.6f} tau={tau:.4f} "
                        f"grad_norm={grad_norm if grad_norm is not None else float('nan'):.4f} skipped={skipped_step}"
                    )
                else:
                    print(
                        f"stage={stage_name} step={stage_step} global_step={global_step} "
                        f"loss={total_loss:.6f} tau={tau:.4f} gpu_mem_gb={gpu_mem:.3f} "
                        f"grad_norm={grad_norm if grad_norm is not None else float('nan'):.4f} skipped={skipped_step}"
                    )

                if wandb_run is not None:
                    wandb.log(
                        build_wandb_payload(
                            stage_name=stage_name,
                            stage_step=stage_step,
                            global_step=global_step,
                            tau=tau,
                            total_loss=total_loss,
                            grad_norm=grad_norm,
                            gpu_mem=gpu_mem,
                            skipped_step=skipped_step,
                            active_symmetry_names=active_symmetry_names,
                            transport_matrices=transport_matrices,
                            permutations=permutations,
                        ),
                        step=global_step,
                    )

            if canonicalizer_device.type == "cuda" or model_device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        progress.close()
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()