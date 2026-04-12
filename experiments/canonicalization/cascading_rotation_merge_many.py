# %%
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForMaskedLM
from tqdm import tqdm

from lib.canonicalizer import Canonicalizer, CascadingTemplateCanonicalizer, _flatten_component_for_axis
from lib.serial_model import serialize_model

torch.manual_seed(0)
device = torch.device("cuda")

NUM_TRAIN = 20
NUM_TEST = 5
NUM_TOTAL = NUM_TRAIN + NUM_TEST
OUTPUT_PATH = Path("data/alignment_models/cascading_rotation_merge_many.pt")
ROTATION_MSE_RTOL = 1e-4
ROTATION_MSE_ATOL = 1e-8
ROTATION_MSE_PATIENCE = 3


def attention_dual_roles(symmetry_name):
    return CascadingTemplateCanonicalizer.attention_dual_roles(symmetry_name)


def attention_dual_evidence(symmeters, symmetry_name):
    return {
        role: tensor.detach().float().to(device)
        for role, tensor in CascadingTemplateCanonicalizer.attention_dual_evidence(symmeters, symmetry_name).items()
    }


def combine_attention_dual_evidence(symmetry_name, evidence):
    roles = attention_dual_roles(symmetry_name)
    cat_dim = 1 if evidence[roles[0]].ndim == 3 else 0
    return torch.cat([evidence[role] for role in roles], dim=cat_dim)


def symmetry_evidence(symmeters, symmetry_name):
    if symmetry_name.endswith(".head"):
        return head_descriptor(symmeters, symmetry_name).detach().float().to(device)
    if symmetry_name.endswith((".qk", ".ov")):
        return combine_attention_dual_evidence(symmetry_name, attention_dual_evidence(symmeters, symmetry_name))
    return Canonicalizer._evidence_tensor(symmeters, symmetry_name).detach().float().to(device)


def pairwise_mse_values(evidence_list):
    return np.array([
        (evidence_list[i] - evidence_list[j]).pow(2).mean().item()
        for i, j in itertools.combinations(range(len(evidence_list)), 2)
    ])


def summarize_alignment(symmetry_name, raw_symmeters, aligned_symmeters):
    raw_evidence = [symmetry_evidence(sym, symmetry_name) for sym in raw_symmeters]
    aligned_evidence = [symmetry_evidence(sym, symmetry_name) for sym in aligned_symmeters]

    mse_identity = pairwise_mse_values(raw_evidence)
    mse_canonical = pairwise_mse_values(aligned_evidence)
    identity_mean = mse_identity.mean()
    frac = 0.0 if identity_mean == 0.0 else (identity_mean - mse_canonical.mean()) / identity_mean * 100

    tqdm.write(f"\n--- Held-out {symmetry_name} ---")
    tqdm.write(f"MSE (identity):            {identity_mean:.6f} ± {mse_identity.std():.6f}")
    tqdm.write(f"MSE (canonical cascade):   {mse_canonical.mean():.6f} ± {mse_canonical.std():.6f}")
    tqdm.write(f"Canonical closes {frac:.1f}% of identity MSE")


def infer_transform(symmeters, symmetry_name, template):
    if symmetry_name.endswith(".head"):
        template = {
            kind: {role: tensor.to(device=device) for role, tensor in template[kind].items()}
            for kind in ("qk", "ov")
        }
        return head_permutation_align(template, head_evidence(symmeters, symmetry_name))[0]

    if symmetry_name.endswith((".qk", ".ov")):
        roles = attention_dual_roles(symmetry_name)
        template = {role: tensor.to(device=device) for role, tensor in template.items()}
        evidence = attention_dual_evidence(symmeters, symmetry_name)
        return CascadingTemplateCanonicalizer.polar_align(template[roles[0]], evidence[roles[0]])

    template = template.to(device=device)
    evidence = Canonicalizer._evidence_tensor(symmeters, symmetry_name).detach().float().to(device)
    return procrustes_align(template, evidence)


def apply_symmetry_transform(symmeters, symmetry_name, matrix):
    if symmetry_name.endswith(".head"):
        symmeters.apply_head_transport(symmetry_name, matrix)
    elif symmetry_name.endswith((".qk", ".ov")):
        symmeters.apply_attention_dual_transform(symmetry_name, matrix)
    else:
        symmeters.apply_transform(symmetry_name, matrix)


def fit_symmetry(
    symmetry_name,
    evidence_list,
    align_fn,
    apply_fn,
    n_iters=50,
    tol=1e-6,
    stop_mode="delta",
    mse_rtol=ROTATION_MSE_RTOL,
    mse_atol=ROTATION_MSE_ATOL,
    mse_patience=ROTATION_MSE_PATIENCE,
):
    aligned = torch.stack(evidence_list).clone()
    sample = evidence_list[0]
    previous_mse = None
    stagnant_steps = 0

    if sample.ndim == 2:
        dim = sample.shape[-1]
        transforms = [torch.eye(dim, device=aligned.device) for _ in evidence_list]
    else:
        num_heads, dim = sample.shape[0], sample.shape[-1]
        transforms = [
            torch.eye(dim, device=aligned.device).expand(num_heads, dim, dim).clone()
            for _ in evidence_list
        ]

    progress = tqdm(range(n_iters), desc=f"fit {symmetry_name}", leave=False)
    for _ in progress:
        mean = aligned.mean(dim=0)
        max_delta = 0.0
        for i, evidence in enumerate(evidence_list):
            new_transform = align_fn(mean, evidence)
            aligned[i] = apply_fn(evidence, new_transform)
            max_delta = max(max_delta, (new_transform - transforms[i]).norm().item())
            transforms[i] = new_transform

        mse = pairwise_mse_values([aligned[i] for i in range(aligned.shape[0])]).mean()
        mse_change = float("nan") if previous_mse is None else abs(mse - previous_mse)
        progress.set_postfix(train_mse=f"{mse:.6f}", mse_change=f"{mse_change:.3e}", delta=f"{max_delta:.3e}")

        if stop_mode == "mse":
            if previous_mse is not None and np.isclose(mse, previous_mse, rtol=mse_rtol, atol=mse_atol):
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            previous_mse = mse
            if stagnant_steps >= mse_patience:
                break
            continue

        if max_delta < tol:
            break

    progress.close()
    return transforms, aligned.mean(dim=0)


def fit_attention_dual_symmetry(
    symmetry_name,
    evidence_list,
    n_iters=50,
    tol=1e-6,
    stop_mode="delta",
    mse_rtol=ROTATION_MSE_RTOL,
    mse_atol=ROTATION_MSE_ATOL,
    mse_patience=ROTATION_MSE_PATIENCE,
):
    roles = attention_dual_roles(symmetry_name)
    aligned = {
        role: torch.stack([evidence[role] for evidence in evidence_list]).clone()
        for role in roles
    }
    sample = evidence_list[0][roles[0]]
    previous_mse = None
    stagnant_steps = 0

    if sample.ndim == 2:
        dim = sample.shape[-1]
        transforms = [torch.eye(dim, device=sample.device) for _ in evidence_list]
    else:
        num_heads, dim = sample.shape[0], sample.shape[-1]
        transforms = [
            torch.eye(dim, device=sample.device).expand(num_heads, dim, dim).clone()
            for _ in evidence_list
        ]

    progress = tqdm(range(n_iters), desc=f"fit {symmetry_name}", leave=False)
    for _ in progress:
        mean = {role: aligned[role].mean(dim=0) for role in roles}
        max_delta = 0.0
        for i, evidence in enumerate(evidence_list):
            new_transform = CascadingTemplateCanonicalizer.polar_align(mean[roles[0]], evidence[roles[0]])
            aligned_evidence = CascadingTemplateCanonicalizer.apply_attention_dual_evidence_transform(
                evidence,
                new_transform,
                symmetry_name,
            )
            for role in roles:
                aligned[role][i] = aligned_evidence[role]
            max_delta = max(max_delta, (new_transform - transforms[i]).norm().item())
            transforms[i] = new_transform

        mse = pairwise_mse_values([
            combine_attention_dual_evidence(
                symmetry_name,
                {role: aligned[role][i] for role in roles},
            )
            for i in range(aligned[roles[0]].shape[0])
        ]).mean()
        mse_change = float("nan") if previous_mse is None else abs(mse - previous_mse)
        progress.set_postfix(train_mse=f"{mse:.6f}", mse_change=f"{mse_change:.3e}", delta=f"{max_delta:.3e}")

        if stop_mode == "mse":
            if previous_mse is not None and np.isclose(mse, previous_mse, rtol=mse_rtol, atol=mse_atol):
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            previous_mse = mse
            if stagnant_steps >= mse_patience:
                break
            continue

        if max_delta < tol:
            break

    progress.close()
    return transforms, {role: aligned[role].mean(dim=0) for role in roles}


def head_evidence(symmeters, symmetry_name):
    prefix = symmetry_name.rsplit(".", 1)[0]
    return {
        kind: attention_dual_evidence(symmeters, f"{prefix}.{kind}")
        for kind in ("qk", "ov")
    }


def apply_head_permutation(evidence, matrix):
    return torch.einsum("h...,hj->j...", evidence, matrix)


def head_permutation_align(target, source):
    num_heads = source["qk"]["query"].shape[0]
    costs = torch.empty((num_heads, num_heads), device=source["qk"]["query"].device)
    for target_head in range(num_heads):
        for source_head in range(num_heads):
            cost = 0.0
            for kind in ("qk", "ov"):
                symmetry = f"L0.{kind}"
                roles = attention_dual_roles(symmetry)
                matrix = procrustes_align(
                    target[kind][roles[0]][target_head],
                    source[kind][roles[0]][source_head],
                )
                aligned = CascadingTemplateCanonicalizer.apply_attention_dual_evidence_transform(
                    {
                        role: source[kind][role][source_head]
                        for role in roles
                    },
                    matrix,
                    symmetry,
                )
                for role in roles:
                    cost += (aligned[role] - target[kind][role][target_head]).pow(2).mean()
            costs[target_head, source_head] = cost
    _, col_ind = linear_sum_assignment(costs.detach().cpu().numpy())
    permutation = torch.eye(num_heads, device=source["qk"]["query"].device)[col_ind]
    mean_cost = costs[torch.arange(num_heads, device=costs.device), torch.as_tensor(col_ind, device=costs.device)].mean().item()
    return permutation, mean_cost


def fit_head_symmetry(symmetry_name, evidence_list, n_iters=30, tol=1e-6):
    num_heads = evidence_list[0]["qk"]["query"].shape[0]
    permutations = [torch.eye(num_heads, device=device) for _ in evidence_list]
    template = {
        kind: {
            role: evidence_list[0][kind][role].clone()
            for role in attention_dual_roles(f"L0.{kind}")
        }
        for kind in ("qk", "ov")
    }

    progress = tqdm(range(n_iters), desc=f"fit {symmetry_name}", leave=False)
    for _ in progress:
        aligned = {kind: [] for kind in ("qk", "ov")}
        max_delta = 0.0
        mean_cost = 0.0

        for i, evidence in enumerate(evidence_list):
            permutation, cost = head_permutation_align(template, evidence)
            mean_cost += cost
            max_delta = max(max_delta, (permutation - permutations[i]).norm().item())
            permutations[i] = permutation

            for kind in ("qk", "ov"):
                symmetry = f"L0.{kind}"
                roles = attention_dual_roles(symmetry)
                permuted = {
                    role: apply_head_permutation(evidence[kind][role], permutation)
                    for role in roles
                }
                aligned_heads = {role: [] for role in roles}
                for head in range(num_heads):
                    matrix = procrustes_align(
                        template[kind][roles[0]][head],
                        permuted[roles[0]][head],
                    )
                    aligned_head = CascadingTemplateCanonicalizer.apply_attention_dual_evidence_transform(
                        {
                            role_name: permuted[role_name][head]
                            for role_name in roles
                        },
                        matrix,
                        symmetry,
                    )
                    for role in roles:
                        aligned_heads[role].append(aligned_head[role])
                aligned[kind].append({
                    role: torch.stack(aligned_heads[role])
                    for role in roles
                })

        template = {
            kind: {
                role: torch.stack([item[role] for item in aligned[kind]]).mean(dim=0)
                for role in attention_dual_roles(f"L0.{kind}")
            }
            for kind in ("qk", "ov")
        }
        progress.set_postfix(cost=f"{mean_cost / len(evidence_list):.6f}", delta=f"{max_delta:.3e}")
        if max_delta < tol:
            break

    progress.close()
    return permutations, template


def procrustes_align(target, source):
    # target/source: (K, D) or (H, K, D)
    M = source.transpose(-1, -2) @ target
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


def apply_evidence_transform(evidence, matrix):
    if evidence.ndim == 2:
        return evidence @ matrix
    if evidence.ndim == 3:
        return torch.einsum("hkd,hde->hke", evidence, matrix)
    raise ValueError(f"Unexpected evidence rank: {evidence.ndim}")


def permutation_align(target, source):
    # target/source: (F, H), columns are heads
    scores = (source.T @ target).detach().cpu().numpy()
    _, col_ind = linear_sum_assignment(scores, maximize=True)
    return torch.eye(source.shape[1], device=source.device)[col_ind]


def head_descriptor(symmeters, head_symmetry_name):
    pieces = []
    for _, _, component in symmeters.components_with_axis(head_symmetry_name):
        flat = _flatten_component_for_axis(component, head_symmetry_name)
        if flat is not None:
            pieces.append(flat.float())
    if not pieces:
        raise ValueError(f"No head evidence for {head_symmetry_name}")
    return torch.cat(pieces, dim=0)  # (F, H)


def default_cascade_order(symmeters):
    ordered = list(symmeters.ordered_transform_names())

    # Resolve head correspondence before fitting per-head qk/ov rotations.
    custom = []
    layer_prefixes = sorted(
        {name.split(".", 1)[0] for name in ordered if name.startswith("L")},
        key=lambda x: int(x[1:]),
    )

    if "model" in ordered:
        custom.append("model")
    if "decoder" in ordered:
        custom.append("decoder")

    for prefix in layer_prefixes:
        for suffix in ("head", "qk", "ov", "mlp"):
            name = f"{prefix}.{suffix}"
            if name in ordered:
                custom.append(name)

    leftovers = [name for name in ordered if name not in custom]
    return custom + leftovers


all_symmeters = []
for seed in tqdm(range(NUM_TOTAL), desc="load models"):
    model = AutoModelForMaskedLM.from_pretrained(
        f"google/multiberts-seed_{seed}",
        local_files_only=True,
    ).eval()
    all_symmeters.append(serialize_model(model))

train_original = all_symmeters[:NUM_TRAIN]
test_original = all_symmeters[NUM_TRAIN:]

train_working = [sym.clone() for sym in train_original]
base = train_working[0]
order = default_cascade_order(base)

templates = {}
test_working = [sym.clone() for sym in test_original]

cascade_progress = tqdm(order, desc="cascade", total=len(order))
for symmetry_name in cascade_progress:
    cascade_progress.set_postfix_str(f"current={symmetry_name}")

    if symmetry_name == "L0.head":
        tqdm.write(f"\n[baseline] L0.qk before fitting:")
        summarize_alignment("L0.qk", test_original, test_working)

    if symmetry_name.endswith(".head"):
        evidences = [head_evidence(sym, symmetry_name) for sym in train_working]
        Ps, template = fit_head_symmetry(symmetry_name, evidences, n_iters=30)

        for i, P in enumerate(Ps):
            apply_symmetry_transform(train_working[i], symmetry_name, P)

        templates[symmetry_name] = {
            kind: {role: tensor.detach().cpu() for role, tensor in template[kind].items()}
            for kind in ("qk", "ov")
        }
    elif symmetry_name.endswith((".qk", ".ov")):
        evidences = [attention_dual_evidence(sym, symmetry_name) for sym in train_working]
        Ts, template = fit_attention_dual_symmetry(
            symmetry_name,
            evidences,
            stop_mode="mse",
        )

        for i, T in enumerate(Ts):
            apply_symmetry_transform(train_working[i], symmetry_name, T)

        templates[symmetry_name] = {role: tensor.detach().cpu() for role, tensor in template.items()}
    else:
        evidences = [
            Canonicalizer._evidence_tensor(sym, symmetry_name).detach().float().to(device)
            for sym in train_working
        ]
        Ts, template = fit_symmetry(
            symmetry_name,
            evidences,
            procrustes_align,
            apply_evidence_transform,
            stop_mode="mse",
        )

        for i, T in enumerate(Ts):
            apply_symmetry_transform(train_working[i], symmetry_name, T)

        templates[symmetry_name] = template.detach().cpu()

    for test_symmeters in tqdm(test_working, desc=f"apply {symmetry_name} to test", leave=False):
        matrix = infer_transform(test_symmeters, symmetry_name, templates[symmetry_name])
        apply_symmetry_transform(test_symmeters, symmetry_name, matrix)

    summarize_alignment(symmetry_name, test_original, test_working)
    if symmetry_name.endswith(".head"):
        prefix = symmetry_name.rsplit(".", 1)[0]
        summarize_alignment(f"{prefix}.qk", test_original, test_working)
        summarize_alignment(f"{prefix}.ov", test_original, test_working)

    if symmetry_name.endswith(".head"):
        del evidences, Ps
    else:
        del evidences, Ts
    del template, matrix
    if device.type == "cuda":
        torch.cuda.empty_cache()

cascade_progress.close()


fitted_canonicalizer = CascadingTemplateCanonicalizer(order, templates)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fitted_canonicalizer.save(str(OUTPUT_PATH))
tqdm.write(f"Saved fitted canonicalizer to {OUTPUT_PATH}")

final_test_working = []
for sym in test_original:
    final_test_working.append(fitted_canonicalizer(sym))

summarize_alignment("model", test_original, final_test_working)

# %%