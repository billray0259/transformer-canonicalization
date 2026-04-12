# %%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from transformers import AutoModelForMaskedLM

from lib.canonicalizer import Canonicalizer, PermutationAligner, RotationAligner
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

all_evidence = []
for seed in tqdm(range(10)):
    model = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed}", local_files_only=True).eval()
    all_evidence.append(Canonicalizer._evidence_tensor(serialize_model(model), "model").detach().half().to(device))
all_evidence = torch.stack(all_evidence)  # fp16, shape (25, 31113, 768)
evidence_shape = all_evidence[0].shape

aligner = PermutationAligner(evidence_shape[0], evidence_shape[1]).to(device)

n_steps = 2500
tau_start = 1.0
tau_end = 0.05
tau_decay = (tau_end / tau_start) ** (1.0 / n_steps)

optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)


batch_size = 10
rand_gen = torch.Generator(device=device).manual_seed(1)
losses = []

for step in tqdm(range(n_steps)):
    log_override = False
    tau = max(tau_end, tau_start * (tau_decay ** step))
    # tau = 1.0
    with torch.no_grad():
        perms = torch.stack([
            torch.eye(evidence_shape[-1], device=device)[torch.randperm(evidence_shape[-1], generator=rand_gen, device=device)]
            for _ in range(batch_size)
        ])
        
        # sample with balanced representation across seeds
        n_models = all_evidence.shape[0]
        base_count = batch_size // n_models
        remainder = batch_size % n_models
        counts = torch.full((n_models,), base_count, dtype=torch.long, device=device)
        extra_seeds = torch.randperm(n_models, generator=rand_gen, device=device)[:remainder]
        counts[extra_seeds] += 1
        rand_idx = torch.cat([
            torch.full((counts[i].item(),), i, dtype=torch.long, device=device)
            for i in range(n_models)
        ])
        rand_idx = rand_idx[torch.randperm(batch_size, generator=rand_gen, device=device)]
        rand_batch = all_evidence[rand_idx].float()  # cast fp16 -> fp32 for computation
        rand_batch @= perms

    P, logits = aligner(rand_batch, tau=tau)
    # R = aligner(rand_batch)

    reconstruction = torch.bmm(rand_batch, P)
    mean_recon = reconstruction.mean(dim=0, keepdim=True).expand_as(reconstruction)
    # inv_logits = torch.inverse(logits)
    roundtrip = torch.bmm(mean_recon, P.transpose(1, 2))
    loss = F.mse_loss(roundtrip, rand_batch)
    losses.append(loss.item())
    if log_override or step % 50 == 0 or step == n_steps - 1:
        with torch.no_grad():
            diffs = all_diffs(P)
            # Per-sample MSE, matching the baseline metric: (A - B_aligned).pow(2).mean()
            mse_per_sample = (roundtrip - rand_batch).pow(2).mean(dim=0).flatten()

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].plot(losses)
        axes[0].set_title(f"Loss over time")
        axes[0].set_yscale("log")

        axes[1].hist(diffs.cpu().flatten().numpy(), bins=50, color="blue", alpha=0.7, range=(0, 1))
        axes[1].set_title(f"R sharpness (avg={diffs.mean():.4f})")
        axes[1].set_yscale("log")
        
        mse_np = mse_per_sample.cpu().numpy()
        axes[2].hist(mse_np, bins=50, color="orange", alpha=0.7)
        axes[2].set_title(f"Roundtrip MSE per position (mean={mse_per_sample.mean():.6f})")
        axes[2].axvline(mse_per_sample.mean().item(), color="black", linewidth=0.8, linestyle="--")
        axes[2].set_yscale("log")

        fig.suptitle(f"Step {step} | tau={tau:.4f} | loss={loss.item():.6f}")
        plt.tight_layout()
        plt.show()
    
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
# %%

