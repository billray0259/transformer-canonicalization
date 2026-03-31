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
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "data" / "bias_autoencoder" / "checkpoints" / "best_v2.pt"

def reconstruction_loss(model, x, gamma=1.0):
    reconstruction = model(x)
    v, b = x[:, :-1], x[:, -1:]
    reconstruction_v, reconstruction_b = reconstruction[:, :-1], reconstruction[:, -1:]
    return torch.mean((v - reconstruction_v) ** 2) + gamma * torch.mean((b - reconstruction_b) ** 2)

def latent_preservation_loss(model, x):
    encoded_x = model.encode(x)
    return torch.mean((encoded_x - x[:, :-1]) ** 2)

def bias_isolation_loss(model, x, gamma=1.0):
    bias_only_x = torch.zeros_like(x)
    bias_only_x[:, -1] = x[:, -1]
    reconstructed = model(bias_only_x)
    return torch.mean(reconstructed[:, :-1] ** 2) + gamma * torch.mean((reconstructed[:, -1:] - x[:, -1:]) ** 2)

def anchor_loss(model, x):
    v = x[:, :-1]
    zero_bias_x = torch.cat([v, torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)], dim=1)
    encoded_zero_bias_x = model.encode(zero_bias_x)
    return torch.mean((encoded_zero_bias_x - v) ** 2)

def initialization_regularization_loss(model):
    d_model = model.encoder.shape[1]
    encoder_target = torch.cat(
        [
            torch.eye(d_model, device=model.encoder.device, dtype=model.encoder.dtype),
            torch.zeros((1, d_model), device=model.encoder.device, dtype=model.encoder.dtype),
        ],
        dim=0,
    )
    decoder_target = torch.cat(
        [
            torch.eye(d_model, device=model.decoder.device, dtype=model.decoder.dtype),
            torch.zeros((d_model, 1), device=model.decoder.device, dtype=model.decoder.dtype),
        ],
        dim=1,
    )
    return torch.mean((model.encoder - encoder_target) ** 2) + torch.mean((model.decoder - decoder_target) ** 2)

def permutation_invariant_bias_loss(model):
    encoder_bias_row = model.encoder[-1]
    decoder_weight_block = model.decoder[:, :-1]
    decoder_bias_column = model.decoder[:, -1]
    uniform_direction = torch.full(
        (decoder_weight_block.shape[0],),
        1.0 / (decoder_weight_block.shape[0] ** 0.5),
        device=decoder_weight_block.device,
        dtype=decoder_weight_block.dtype,
    )

    encoder_uniform_loss = torch.mean((encoder_bias_row - encoder_bias_row.mean()) ** 2)
    decoder_uniform_loss = torch.mean((decoder_bias_column - decoder_bias_column.mean()) ** 2)
    weight_leakage_loss = torch.mean((uniform_direction @ decoder_weight_block) ** 2)
    return encoder_uniform_loss + decoder_uniform_loss + weight_leakage_loss

def compute_loss_terms(model, batch):
    rec_loss = reconstruction_loss(model, batch)
    latent_loss = latent_preservation_loss(model, batch)
    bias_loss = bias_isolation_loss(model, batch)
    anc_loss = anchor_loss(model, batch)
    init_loss = initialization_regularization_loss(model)
    invariant_bias_loss = permutation_invariant_bias_loss(model)
    return {
        "reconstruction_loss": rec_loss,
        "latent_preservation_loss": latent_loss,
        "bias_isolation_loss": bias_loss,
        "anchor_loss": anc_loss,
        "initialization_regularization_loss": init_loss,
        "permutation_invariant_bias_loss": invariant_bias_loss,
    }

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
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--bias-loss-weight", type=float, default=1.0)
    parser.add_argument("--anc-loss-weight", type=float, default=0.25)
    parser.add_argument("--init-loss-weight", type=float, default=0.1)
    parser.add_argument("--invariant-bias-loss-weight", type=float, default=0.1)
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
                "latent_loss_weight": args.latent_loss_weight,
                "bias_loss_weight": args.bias_loss_weight,
                "anc_loss_weight": args.anc_loss_weight,
                "init_loss_weight": args.init_loss_weight,
                "invariant_bias_loss_weight": args.invariant_bias_loss_weight,
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
                "train/latent_preservation_loss": 0.0,
                "train/bias_isolation_loss": 0.0,
                "train/anchor_loss": 0.0,
                "train/initialization_regularization_loss": 0.0,
                "train/permutation_invariant_bias_loss": 0.0,
                "train/total_loss": 0.0,
            }
            epoch_batches = 0

            for i in tqdm(range(0, train_rows.shape[0], args.batch_size), desc="Training batches", leave=False):
                batch = train_rows[i:i + args.batch_size].to(device)

                loss_terms = compute_loss_terms(model, batch)

                total_loss = (
                    args.rec_loss_weight * loss_terms["reconstruction_loss"]
                    + args.latent_loss_weight * loss_terms["latent_preservation_loss"]
                    + args.bias_loss_weight * loss_terms["bias_isolation_loss"]
                    + args.anc_loss_weight * loss_terms["anchor_loss"]
                    + args.init_loss_weight * loss_terms["initialization_regularization_loss"]
                    + args.invariant_bias_loss_weight * loss_terms["permutation_invariant_bias_loss"]
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                metrics = {
                    "train/reconstruction_loss": loss_terms["reconstruction_loss"].item(),
                    "train/latent_preservation_loss": loss_terms["latent_preservation_loss"].item(),
                    "train/bias_isolation_loss": loss_terms["bias_isolation_loss"].item(),
                    "train/anchor_loss": loss_terms["anchor_loss"].item(),
                    "train/initialization_regularization_loss": loss_terms["initialization_regularization_loss"].item(),
                    "train/permutation_invariant_bias_loss": loss_terms["permutation_invariant_bias_loss"].item(),
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
                        validation_totals = {
                            "validation/reconstruction_loss": 0.0,
                            "validation/latent_preservation_loss": 0.0,
                            "validation/bias_isolation_loss": 0.0,
                            "validation/anchor_loss": 0.0,
                            "validation/initialization_regularization_loss": 0.0,
                            "validation/permutation_invariant_bias_loss": 0.0,
                            "validation/total_loss": 0.0,
                        }
                        validation_batches = 0
                        for j in range(0, validation_rows.shape[0], args.batch_size):
                            validation_batch = validation_rows[j:j + args.batch_size].to(device)
                            validation_terms = compute_loss_terms(model, validation_batch)
                            validation_total_loss = (
                                args.rec_loss_weight * validation_terms["reconstruction_loss"]
                                + args.latent_loss_weight * validation_terms["latent_preservation_loss"]
                                + args.bias_loss_weight * validation_terms["bias_isolation_loss"]
                                + args.anc_loss_weight * validation_terms["anchor_loss"]
                                + args.init_loss_weight * validation_terms["initialization_regularization_loss"]
                                + args.invariant_bias_loss_weight * validation_terms["permutation_invariant_bias_loss"]
                            )
                            validation_totals["validation/reconstruction_loss"] += validation_terms["reconstruction_loss"].item()
                            validation_totals["validation/latent_preservation_loss"] += validation_terms["latent_preservation_loss"].item()
                            validation_totals["validation/bias_isolation_loss"] += validation_terms["bias_isolation_loss"].item()
                            validation_totals["validation/anchor_loss"] += validation_terms["anchor_loss"].item()
                            validation_totals["validation/initialization_regularization_loss"] += validation_terms["initialization_regularization_loss"].item()
                            validation_totals["validation/permutation_invariant_bias_loss"] += validation_terms["permutation_invariant_bias_loss"].item()
                            validation_totals["validation/total_loss"] += validation_total_loss.item()
                            validation_batches += 1
                    model.train()
                    validation_metrics = {
                        name: value / validation_batches
                        for name, value in validation_totals.items()
                    }

                    if validation_metrics["validation/total_loss"] < best_loss:
                        best_loss = validation_metrics["validation/total_loss"]
                        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), args.checkpoint_path)

                    if wandb_run is not None:
                        wandb.log(validation_metrics, step=global_step)

            if epoch_batches > 0:
                epoch_loss = epoch_totals["train/total_loss"] / epoch_batches

                if wandb_run is not None:
                    wandb.log(
                        {
                            "epoch/reconstruction_loss": epoch_totals["train/reconstruction_loss"] / epoch_batches,
                            "epoch/latent_preservation_loss": epoch_totals["train/latent_preservation_loss"] / epoch_batches,
                            "epoch/bias_isolation_loss": epoch_totals["train/bias_isolation_loss"] / epoch_batches,
                            "epoch/anchor_loss": epoch_totals["train/anchor_loss"] / epoch_batches,
                            "epoch/initialization_regularization_loss": epoch_totals["train/initialization_regularization_loss"] / epoch_batches,
                            "epoch/permutation_invariant_bias_loss": epoch_totals["train/permutation_invariant_bias_loss"] / epoch_batches,
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