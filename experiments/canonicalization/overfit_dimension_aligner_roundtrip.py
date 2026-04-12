# %%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from transformers import AutoModelForMaskedLM

from lib.canonicalizer import Canonicalizer, DimensionAligner
from lib.serial_model import serialize_model
from tqdm import tqdm
import matplotlib.pyplot as plt

def col_diffs(matrix):
    col_min = matrix.min(dim=-1).values
    col_max = matrix.max(dim=-1).values
    col_diff = col_max - col_min
    return col_diff

def row_diffs(matrix):
    row_min = matrix.min(dim=-2).values
    row_max = matrix.max(dim=-2).values
    row_diff = row_max - row_min
    return row_diff

def all_diffs(matrix):
    return torch.cat([col_diffs(matrix), row_diffs(matrix)])


torch.manual_seed(0)
device = torch.device("cuda")

model = AutoModelForMaskedLM.from_pretrained(
    "google/multiberts-seed_0",
    local_files_only=True,
).eval()
symmeters = serialize_model(model)

x = Canonicalizer._evidence_tensor(symmeters, "model").detach().to(device)

perm_generator = torch.Generator(device=device).manual_seed(1)

aligner = DimensionAligner(x.shape[0], x.shape[1]).to(device)

optimizer = torch.optim.Adam(
    aligner.parameters(),
    lr=1e-3,
)

n_steps = 200
batch_size = 14
for step in tqdm(range(n_steps)):
    tau = max(0.05, 1.0 * (0.99 ** step))  # warm -> cold
    with torch.no_grad():
        perms = torch.stack([
            torch.eye(x.shape[-1], device=device)[torch.randperm(x.shape[-1], generator=perm_generator, device=device)]
            for _ in range(batch_size)
        ])
        x_batch = x.unsqueeze(0) @ perms  # (batch, known, unknown)
    targets = perms.transpose(-1, -2).argmax(dim=-1)  # (batch, unknown)
    P, logits = aligner(x_batch, tau=tau)
    
    reconstruction = torch.bmm(x_batch, P)  # (batch, known, 768)
    mean_recon = reconstruction.mean(dim=0, keepdim=True).expand_as(reconstruction)  # (batch, known, 768)
    roundtrip = torch.bmm(mean_recon, P.transpose(-1, -2))  # (batch, known, 768)
    loss = F.mse_loss(roundtrip, x_batch)
    accuracy = (P.argmax(dim=-1) == targets).float().mean().item()

    if step % 25 == 0 or step == n_steps - 1:
        diffs = all_diffs(P)
        plt.figure(figsize=(6, 6))
        plt.hist(diffs.detach().cpu().numpy().flatten(), bins=20, color="blue", alpha=0.7, range=(0, 1))
        plt.title(f"Step {step}: Average Diff={torch.mean(diffs).item():.6f}, Accuracy={accuracy:.6f}")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# %%

with torch.no_grad():
    P1 = torch.eye(x.shape[-1], device=device)[torch.randperm(x.shape[-1], device=device)]
    P2 = torch.eye(x.shape[-1], device=device)[torch.randperm(x.shape[-1], device=device)]
    x_p1 = (x @ P1).unsqueeze(0)
    x_p2 = (x @ P2).unsqueeze(0)
    # for tau in [0.1, 0.05, 0.01, 0.005, 0.001]:
    #     soft1, _ = aligner(x_p1, tau=tau)
    #     soft2, _ = aligner(x_p2, tau=tau)
    #     try:
    #         inv1 = torch.linalg.inv(soft1)
    #         inv2 = torch.linalg.inv(soft2)
    #         assert torch.isfinite(inv1).all() and torch.isfinite(inv2).all()
    #         print(f"tau={tau}: inversion succeeded")
    #         break
    #     except Exception as e:
    #         print(f"tau={tau}: inversion failed ({e})")
    soft1, _ = aligner(x_p1, tau=0.05)
    soft2, _ = aligner(x_p2, tau=0.05)
    inv1 = soft1.transpose(-1, -2)
    inv2 = soft2.transpose(-1, -2)
    
    c1 = torch.bmm(x_p1, soft1)
    c2 = torch.bmm(x_p2, soft2)

    avg = (c1 + c2) / 2

    r1 = torch.bmm(avg, inv1)
    r2 = torch.bmm(avg, inv2)

    print(f"Canonical agreement: {F.mse_loss(c1, c2).item():.6f}")
    print(f"Recon vs x@P1: {F.mse_loss(r1, x_p1).item():.6f}")
    print(f"Recon vs x@P2: {F.mse_loss(r2, x_p2).item():.6f}")

# %%

with torch.no_grad():
    reconstruction = torch.bmm(x_batch, P)
    # Should be ~0 if working
    print("Pairwise disagreement:", (reconstruction[0] - reconstruction[1]).abs().mean().item())
    # Should be ~0 if it found a true permutation (not a blend)
    print("Reconstruction vs some permutation of x:", 
          F.mse_loss(reconstruction[0], x).item())
    
# %%

with torch.no_grad():
    hard_P = torch.zeros_like(P)
    hard_P.scatter_(-1, P.argmax(dim=-1, keepdim=True), 1.0)
    hard_recon = torch.bmm(x_batch, hard_P)
    print("Hard recon vs x:", F.mse_loss(hard_recon[0], x).item())
    
# %%

with torch.no_grad():
    # For each column in reconstruction, find nearest column in x
    dists = torch.cdist(reconstruction[0].T, x.T)  # (768, 768)
    nearest = dists.min(dim=-1).values
    print(f"Perfect matches: {(nearest < 1e-3).sum().item()} / 768")
    print(f"Worst column error: {nearest.max().item():.6f}")
    
    # histogram of dists
    plt.figure(figsize=(6, 6))
    plt.hist(dists.detach().cpu().numpy().flatten(), bins=20, color="green", alpha=0.7)
    plt.title("Histogram of Column Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
# %%
with torch.no_grad():
    assignments = P[0].argmax(dim=-1)
    print(f"Unique assignments: {assignments.unique().shape[0]} / 768")
    print(f"hard_recon column norms: {hard_recon[0].norm(dim=0)[:5]}")
    print(f"x column norms: {x.norm(dim=0)[:5]}")
    print(f"Nearest distances: min={nearest.min():.6f} median={nearest.median():.6f} max={nearest.max():.6f}")

with torch.no_grad():
    assignments = P[0].argmax(dim=-1)
    print(f"Unique assignments: {assignments.unique().shape[0]} / 768")
    
print(f"< 1e-1: {(nearest < 0.1).sum().item()}")
print(f"< 1e-2: {(nearest < 0.01).sum().item()}")
print(f"< 1e-3: {(nearest < 0.001).sum().item()}")
# %%
from scipy.optimize import linear_sum_assignment

with torch.no_grad():
    cost = -P[0].detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    exact_P = torch.zeros_like(P[0])
    exact_P[row_ind, col_ind] = 1.0
    exact_recon = x_batch[0:1] @ exact_P.unsqueeze(0)
    dists = torch.cdist(exact_recon[0].T, x.T)
    nearest = dists.min(dim=-1).values
    print(f"Hungarian: <1e-3: {(nearest < 1e-3).sum().item()} / 768")
    print(f"min={nearest.min():.6f} median={nearest.median():.6f} max={nearest.max():.6f}")
# %%
with torch.no_grad():
    # Direct indexing instead of matmul — no accumulation error
    col_perm = col_ind  # from Hungarian
    indexed_recon = x_batch[0, :, col_perm]  # exact column selection
    dists = torch.cdist(indexed_recon.T, x.T)
    nearest = dists.min(dim=-1).values
    print(f"Indexed: <1e-6: {(nearest < 1e-6).sum().item()} / 768")
# %%