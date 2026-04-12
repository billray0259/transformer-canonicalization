import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from lib.canonicalizer import Canonicalizer, DimensionAligner
from lib.serial_model import _build_overrides, load_serialized, serialize_model
from lib.serial_params import Symmeters
from tqdm import tqdm


def print_matrix(title: str, matrix: torch.Tensor):
    print(title)
    print(np.round(matrix.detach().cpu().numpy() * 100) / 100)


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


torch.manual_seed(0)
required_free_gb = 2.0
if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024 ** 3
    total_gb = total_bytes / 1024 ** 3
    if free_gb >= required_free_gb:
        aligner_device = torch.device("cuda")
    else:
        aligner_device = torch.device("cpu")
        print(
            f"Falling back to CPU for aligners: free CUDA memory {free_gb:.2f} GiB / {total_gb:.2f} GiB "
            f"is below required threshold {required_free_gb:.2f} GiB"
        )
else:
    aligner_device = torch.device("cpu")
model_device = torch.device("cpu")
print(f"aligner_device={aligner_device} model_device={model_device}")
texts = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
]

model_a = AutoModelForMaskedLM.from_pretrained(
    "google/multiberts-seed_0",
    local_files_only=True,
).eval()
model_b = AutoModelForMaskedLM.from_pretrained(
    "google/multiberts-seed_1",
    local_files_only=True,
).eval()

symmeters_a = serialize_model(model_a)
symmeters_b = serialize_model(model_b)

x_a_cpu = Canonicalizer._evidence_tensor(symmeters_a, "model").detach()
x_b_cpu = Canonicalizer._evidence_tensor(symmeters_b, "model").detach()
x_a = x_a_cpu.to(aligner_device)
x_b = x_b_cpu.to(aligner_device)

tokenizer = AutoTokenizer.from_pretrained(
    "google/multiberts-seed_0",
    local_files_only=True,
)
batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

shell_model, _ = load_serialized(
    symmeters_a,
    "google/multiberts-seed_0",
    local_files_only=True,
)
shell_model = shell_model.to(model_device).eval()
for param in shell_model.parameters():
    param.requires_grad_(False)

with torch.no_grad():
    logits_a = functional_call(shell_model, _build_overrides(symmeters_a), (), batch).logits.detach()
    logits_b = functional_call(shell_model, _build_overrides(symmeters_b), (), batch).logits.detach()
    raw_midpoint = lerp_symmeters(symmeters_a, symmeters_b, 0.5)
    raw_midpoint_logits = functional_call(shell_model, _build_overrides(raw_midpoint), (), batch).logits.detach()

raw_logit_mse = F.mse_loss(logits_a, logits_b)
raw_midpoint_mse = F.mse_loss(raw_midpoint_logits, 0.5 * (logits_a + logits_b))
print(f"Raw endpoint logit MSE: {raw_logit_mse.item():.6f}")
print(f"Raw midpoint logit MSE: {raw_midpoint_mse.item():.6f}")

aligner_a = DimensionAligner(x_a.shape[0], x_a.shape[1]).to(aligner_device)
aligner_b = DimensionAligner(x_b.shape[0], x_b.shape[1]).to(aligner_device)

with torch.no_grad():
    pinv_a = torch.linalg.pinv(x_a_cpu.T)
    pinv_b = torch.linalg.pinv(x_b_cpu.T)
    identity = torch.eye(x_a.shape[-1])
    aligner_a.W_q.copy_(pinv_a.to(aligner_device))
    aligner_a.W_k.copy_((pinv_a @ (8.0 * identity)).to(aligner_device))
    aligner_b.W_q.copy_(pinv_b.to(aligner_device))
    aligner_b.W_k.copy_((pinv_b @ (8.0 * identity)).to(aligner_device))

optimizer = torch.optim.Adam(
    list(aligner_a.parameters()) + list(aligner_b.parameters()),
    lr=1e-5,
)

n_steps = 20
for step in tqdm(range(n_steps)):
    tau = 0.2
    P_a = aligner_a(x_a.unsqueeze(0), tau=tau).squeeze(0)
    P_b = aligner_b(x_b.unsqueeze(0), tau=tau).squeeze(0)
    canonicalized_a = symmeters_a.clone()
    canonicalized_a.apply_transform("model", P_a)
    canonicalized_b = symmeters_b.clone()
    canonicalized_b.apply_transform("model", P_b)

    logits_canonicalized_a = functional_call(shell_model, _build_overrides(canonicalized_a), (), batch).logits
    logits_canonicalized_b = functional_call(shell_model, _build_overrides(canonicalized_b), (), batch).logits
    midpoint = lerp_symmeters(canonicalized_a, canonicalized_b, 0.5)
    logits_midpoint = functional_call(shell_model, _build_overrides(midpoint), (), batch).logits

    preserve_loss = 0.5 * (
        F.mse_loss(logits_canonicalized_a, logits_a)
        + F.mse_loss(logits_canonicalized_b, logits_b)
    )
    midpoint_loss = F.mse_loss(logits_midpoint, 0.5 * (logits_a + logits_b))
    agreement_mse = F.mse_loss(logits_canonicalized_a, logits_canonicalized_b)
    loss = preserve_loss + midpoint_loss

    if step % 25 == 0 or step == n_steps - 1:
        print(
            f"Step {step}: "
            f"loss={loss.item():.6f} "
            f"preserve_loss={preserve_loss.item():.6f} "
            f"midpoint_loss={midpoint_loss.item():.6f} "
            f"endpoint_logit_mse={agreement_mse.item():.6f} "
            f"tau={tau:.4f}"
        )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print_matrix("seed 0 evidence (first 10 rows)", x_a[:10])
print_matrix("seed 1 evidence (first 10 rows)", x_b[:10])
print_matrix("learned transport seed 0", P_a)
print_matrix("learned transport seed 1", P_b)