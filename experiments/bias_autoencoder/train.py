from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from lib.bias_autoencoder import BiasAutoencoder
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "bias_autoencoder" / "dataset" / "train_rows.pt"
DEFAULT_VALIDATION_DATASET_PATH = ROOT_DIR / "data" / "bias_autoencoder" / "dataset" / "validation_rows.pt"
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "data" / "bias_autoencoder" / "checkpoints" / "best.pt"

def random_permutation_matrix(d_model, device):
    return torch.eye(d_model, device=device)[torch.randperm(d_model, device=device)]

def reconstruction_loss(model, x, gamma=1.0):
    reconstruction = model(x)
    v, b = x[:, :-1], x[:, -1:]
    reconstruction_v, reconstruction_b = reconstruction[:, :-1], reconstruction[:, -1:]
    return torch.mean((v - reconstruction_v) ** 2) + gamma * torch.mean((b - reconstruction_b) ** 2)

def encoder_equivariance_loss(model, perm, x):
    full_perm = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    full_perm[:perm.shape[0], :perm.shape[1]] = perm.to(device=x.device, dtype=x.dtype)
    permuted_x = x @ full_perm
    encoded_permuted_x = model.encode(permuted_x)
    permuted_encoded_x = model.encode(x) @ perm.to(device=x.device, dtype=x.dtype)
    return torch.mean((encoded_permuted_x - permuted_encoded_x) ** 2)

def decoder_equivariance_loss(model, perm, z):
    full_perm = torch.eye(model.decoder.shape[1], device=z.device, dtype=z.dtype)
    full_perm[:perm.shape[0], :perm.shape[1]] = perm.to(device=z.device, dtype=z.dtype)
    decoded_permuted_z = model.decode(z @ perm.to(device=z.device, dtype=z.dtype))
    permuted_decoded_z = model.decode(z) @ full_perm
    return torch.mean((decoded_permuted_z - permuted_decoded_z) ** 2)

def anchor_loss(model, x):
    v = x[:, :-1]
    zero_bias_x = torch.cat([v, torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)], dim=1)
    encoded_zero_bias_x = model.encode(zero_bias_x)
    return torch.mean((encoded_zero_bias_x - v) ** 2)

def parse_args():
    parser = ArgumentParser(description="Train the bias autoencoder and log metrics to Weights & Biases.")
    parser.add_argument("--project", default="bias-autoencoder", help="Weights & Biases project name.")
    parser.add_argument("--run-name", default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the training rows tensor.")
    parser.add_argument("--validation-data-path", type=Path, default=DEFAULT_VALIDATION_DATASET_PATH, help="Path to the validation rows tensor.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to save the best-loss checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.25e-4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--rec-loss-weight", type=float, default=1.0)
    parser.add_argument("--equiv-loss-weight", type=float, default=1.0)
    parser.add_argument("--dec-equiv-loss-weight", type=float, default=1.0)
    parser.add_argument("--anc-loss-weight", type=float, default=0.25)
    parser.add_argument("--disable-wandb", action="store_true", help="Skip Weights & Biases logging.")
    return parser.parse_args()


def main():
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiasAutoencoder(d_model=args.d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    validation_rows = torch.load(args.validation_data_path)

    wandb_run = None
    if not args.disable_wandb:
        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            dir=str(ROOT_DIR),
            config={
                "data_path": str(args.data_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "d_model": args.d_model,
                "rec_loss_weight": args.rec_loss_weight,
                "equiv_loss_weight": args.equiv_loss_weight,
                "dec_equiv_loss_weight": args.dec_equiv_loss_weight,
                "anc_loss_weight": args.anc_loss_weight,
            },
        )

    global_step = 0

    try:
        for epoch in tqdm(range(args.epochs), desc="Training epochs"):
            train_rows = torch.load(args.data_path)

            data_perm = torch.randperm(train_rows.shape[0])
            train_rows = train_rows[data_perm]

            epoch_totals = {
                "train/reconstruction_loss": 0.0,
                "train/encoder_equivariance_loss": 0.0,
                "train/decoder_equivariance_loss": 0.0,
                "train/anchor_loss": 0.0,
                "train/total_loss": 0.0,
            }
            epoch_batches = 0

            for i in tqdm(range(0, train_rows.shape[0], args.batch_size), desc="Training batches", leave=False):
                batch = train_rows[i:i + args.batch_size].to(device)

                random_perm = random_permutation_matrix(args.d_model, device)

                rec_loss = reconstruction_loss(model, batch)
                enc_equiv_loss = encoder_equivariance_loss(model, random_perm, batch)
                z = model.encode(batch).detach()  # Detach to avoid backprop through encoder twice
                dec_equiv_loss = decoder_equivariance_loss(model, random_perm, z)
                anc_loss = anchor_loss(model, batch)

                total_loss = (
                    args.rec_loss_weight * rec_loss
                    + args.equiv_loss_weight * enc_equiv_loss
                    + args.dec_equiv_loss_weight * dec_equiv_loss
                    + args.anc_loss_weight * anc_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                metrics = {
                    "train/reconstruction_loss": rec_loss.item(),
                    "train/encoder_equivariance_loss": enc_equiv_loss.item(),
                    "train/decoder_equivariance_loss": dec_equiv_loss.item(),
                    "train/anchor_loss": anc_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                for name, value in metrics.items():
                    if name in epoch_totals:
                        epoch_totals[name] += value

                if wandb_run is not None and global_step % 25 == 0:
                    wandb.log(metrics, step=global_step)

                global_step += 1
                epoch_batches += 1

                if global_step % 250 == 0:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = 0.0
                        validation_batches = 0
                        for j in range(0, validation_rows.shape[0], args.batch_size):
                            validation_batch = validation_rows[j:j + args.batch_size].to(device)
                            validation_loss += reconstruction_loss(model, validation_batch).item()
                            validation_batches += 1
                    model.train()
                    validation_loss /= validation_batches

                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), args.checkpoint_path)

                    if wandb_run is not None:
                        wandb.log({"validation/loss": validation_loss}, step=global_step)

            if epoch_batches > 0:
                epoch_loss = epoch_totals["train/total_loss"] / epoch_batches

                if wandb_run is not None:
                    wandb.log(
                        {
                            "epoch/reconstruction_loss": epoch_totals["train/reconstruction_loss"] / epoch_batches,
                            "epoch/encoder_equivariance_loss": epoch_totals["train/encoder_equivariance_loss"] / epoch_batches,
                            "epoch/decoder_equivariance_loss": epoch_totals["train/decoder_equivariance_loss"] / epoch_batches,
                            "epoch/anchor_loss": epoch_totals["train/anchor_loss"] / epoch_batches,
                            "epoch/total_loss": epoch_loss,
                            "epoch/index": epoch,
                        },
                        step=global_step,
                    )
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()