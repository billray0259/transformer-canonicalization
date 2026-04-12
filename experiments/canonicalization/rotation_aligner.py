# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from lib.canonicalizer import Canonicalizer, RotationAligner
from lib.serial_model import serialize_model
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda")

all_evidence = []
for seed in tqdm(range(10)):
    model = AutoModelForMaskedLM.from_pretrained(f"google/multiberts-seed_{seed}", local_files_only=True).eval()
    all_evidence.append(Canonicalizer._evidence_tensor(serialize_model(model), "model").detach().half().to(device))
all_evidence = torch.stack(all_evidence)  # fp16, shape (25, 31113, 768)
evidence_shape = all_evidence[0].shape

aligner = RotationAligner(evidence_shape[0], evidence_shape[1]).to(device)

n_steps = 2500
optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)


batch_size = 10
rand_gen = torch.Generator(device=device).manual_seed(1)
losses = []

for step in tqdm(range(n_steps)):
    log_override = False
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

    # P, logits = aligner(rand_batch, tau=tau)
    R = aligner(rand_batch)

    reconstruction = torch.bmm(rand_batch, R)
    mean_recon = reconstruction.mean(dim=0, keepdim=True).expand_as(reconstruction)
    # inv_logits = torch.inverse(logits)
    roundtrip = torch.bmm(mean_recon, R.transpose(1, 2))
    loss = F.mse_loss(roundtrip, rand_batch)
    losses.append(loss.item())
    if log_override or step % 25 == 0 or step == n_steps - 1:
        with torch.no_grad():
            raw_mean = rand_batch.mean(dim=0, keepdim=True)
            raw_mse_per_coord = (rand_batch - raw_mean).pow(2).mean(dim=0).flatten()
            aligned_mse_per_coord = (reconstruction - mean_recon).pow(2).mean(dim=0).flatten()
            roundtrip_mse_per_coord = (roundtrip - rand_batch).pow(2).mean(dim=0).flatten()
            disagreement_reduction = 1.0 - aligned_mse_per_coord.mean() / raw_mse_per_coord.mean().clamp_min(1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].plot(losses)
        axes[0].set_title(f"Loss over time (latest={loss.item():.6f})")
        axes[0].set_yscale("log")

        raw_mse_np = raw_mse_per_coord.cpu().numpy()
        aligned_mse_np = aligned_mse_per_coord.cpu().numpy()
        axes[1].hist(raw_mse_np, bins=50, color="gray", alpha=0.55, label="raw")
        axes[1].hist(aligned_mse_np, bins=50, color="green", alpha=0.55, label="aligned")
        axes[1].axvline(raw_mse_per_coord.mean().item(), color="black", linewidth=0.8, linestyle="--")
        axes[1].axvline(aligned_mse_per_coord.mean().item(), color="darkgreen", linewidth=0.8, linestyle="--")
        axes[1].set_title(
            f"Batch disagreement per coordinate ({disagreement_reduction.item():.1%} reduction)"
        )
        axes[1].set_yscale("log")
        axes[1].legend()
        
        roundtrip_mse_np = roundtrip_mse_per_coord.cpu().numpy()
        axes[2].hist(roundtrip_mse_np, bins=50, color="orange", alpha=0.7)
        axes[2].set_title(
            f"Roundtrip reconstruction MSE per coordinate (mean={roundtrip_mse_per_coord.mean():.6f})"
        )
        axes[2].axvline(roundtrip_mse_per_coord.mean().item(), color="black", linewidth=0.8, linestyle="--")
        axes[2].set_yscale("log")

        fig.suptitle(f"Step {step} | loss={loss.item():.6f}")
        plt.tight_layout()
        plt.show()
    
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
# %%

