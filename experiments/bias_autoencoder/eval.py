from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lib.bias_autoencoder import BiasAutoencoder
from lib.serial_model import SerialAutoModelForMaskedLM
from lib.serial_params import NamedSerialParameters
from lib.utils import masked_token_pseudo_perplexity


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "data" / "bias_autoencoder" / "checkpoints" / "best_v2.pt"
DEFAULT_VALIDATION_SEEDS = [20, 21, 22]
DEFAULT_TEXTS = [
    "The capital of France is Paris.",
    "Water freezes at zero degrees Celsius.",
    "The Earth orbits the Sun once every year.",
    "A triangle has three sides and three angles.",
    "Photosynthesis converts sunlight into chemical energy.",
]


def parse_args():
    parser = ArgumentParser(description="Evaluate the bias autoencoder on validation MultiBERT seeds.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to the best bias autoencoder checkpoint.")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=DEFAULT_VALIDATION_SEEDS, help="Validation MultiBERT seed ids.")
    parser.add_argument("--text-file", type=Path, default=None, help="Optional newline-delimited evaluation text file.")
    return parser.parse_args()


def load_texts(text_file):
    if text_file is None:
        return DEFAULT_TEXTS
    return [line.strip() for line in text_file.read_text().splitlines() if line.strip()]


def load_bias_autoencoder(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = BiasAutoencoder(d_model=state_dict["encoder"].shape[1]).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def autoencode_params(serialized_params, bias_autoencoder, device):
    original_device = serialized_params.vectors.device
    vectors = serialized_params.vectors.clone().to(device)
    bias_mask = ~torch.isnan(vectors[:, -1])

    with torch.no_grad():
        if bias_mask.any():
            vectors[bias_mask] = bias_autoencoder(vectors[bias_mask])

    return NamedSerialParameters.from_vector_list(serialized_params.names, [vectors.to(original_device)])

def encode_bias(serialized_params, bias_autoencoder, device):
    original_device = serialized_params.vectors.device
    vectors = serialized_params.vectors.clone().to(device)
    bias_mask = ~torch.isnan(vectors[:, -1])
    encoded_vectors = torch.zeros(vectors.shape[0], bias_autoencoder.encoder.shape[1], device=vectors.device, dtype=vectors.dtype)

    with torch.no_grad():
        if bias_mask.any():
            encoded_vectors[bias_mask] = bias_autoencoder.encode(vectors[bias_mask])
    if (~bias_mask).any():
        encoded_vectors[~bias_mask] = vectors[~bias_mask][:, :-1]

    return NamedSerialParameters.from_vector_list(serialized_params.names, [encoded_vectors.to(original_device)]), bias_mask.to(original_device)


def build_output_permutation(permutation, output_dim, device, dtype):
    full_permutation = torch.eye(output_dim, device=device, dtype=dtype)
    full_permutation[:permutation.shape[0], :permutation.shape[1]] = permutation.to(device=device, dtype=dtype)
    return full_permutation

def decode_bias(serialized_params, bias_autoencoder, bias_mask, device, permutation=None):
    original_device = serialized_params.vectors.device
    vectors = serialized_params.vectors.clone().to(device)
    bias_mask = bias_mask.to(device)
    decoder = bias_autoencoder.decoder
    if permutation is not None:
        permutation = permutation.to(device=device, dtype=vectors.dtype)
        full_permutation = build_output_permutation(permutation, decoder.shape[1], device, vectors.dtype)
        decoder = permutation.T @ decoder @ full_permutation
    decoded_vectors = torch.zeros(vectors.shape[0], bias_autoencoder.decoder.shape[1], device=vectors.device, dtype=vectors.dtype)

    with torch.no_grad():
        if bias_mask.any():
            decoded_vectors[bias_mask] = vectors[bias_mask] @ decoder
    if (~bias_mask).any():
        decoded_vectors[~bias_mask, :-1] = vectors[~bias_mask]
        decoded_vectors[~bias_mask, -1] = float("nan")

    return NamedSerialParameters.from_vector_list(serialized_params.names, [decoded_vectors.to(original_device)])

def unpermute_decoded(serialized_params, bias_mask, permutation, device):
    original_device = serialized_params.vectors.device
    vectors = serialized_params.vectors.clone().to(device)
    bias_mask = bias_mask.to(device)
    permutation = permutation.to(device=device, dtype=vectors.dtype)
    full_permutation = build_output_permutation(permutation, vectors.shape[1], device, vectors.dtype)

    if bias_mask.any():
        vectors[bias_mask] = vectors[bias_mask] @ full_permutation.T
    if (~bias_mask).any():
        vectors[~bias_mask, :-1] = vectors[~bias_mask, :-1] @ permutation.T
        vectors[~bias_mask, -1] = float("nan")

    return NamedSerialParameters.from_vector_list(serialized_params.names, [vectors.to(original_device)])

def build_permuted_model_params(serialized_params, bias_autoencoder, permutation, device):
    permutation = permutation.to(device=device, dtype=serialized_params.vectors.dtype)
    encoded_params, bias_mask = encode_bias(serialized_params, bias_autoencoder, device)
    permuted_vectors = encoded_params.vectors.to(device) @ permutation
    permuted_encoded_params = NamedSerialParameters.from_vector_list(
        encoded_params.names,
        [permuted_vectors.to(encoded_params.vectors.device)],
    )
    return decode_bias(permuted_encoded_params, bias_autoencoder, bias_mask, device, permutation=permutation)


def build_permuted_roundtrip_params(serialized_params, bias_autoencoder, permutation, device):
    permutation = permutation.to(device=device, dtype=serialized_params.vectors.dtype)
    encoded_params, bias_mask = encode_bias(serialized_params, bias_autoencoder, device)
    permuted_vectors = encoded_params.vectors.to(device) @ permutation
    permuted_encoded_params = NamedSerialParameters.from_vector_list(
        encoded_params.names,
        [permuted_vectors.to(encoded_params.vectors.device)],
    )
    permuted_decoded_params = decode_bias(permuted_encoded_params, bias_autoencoder, bias_mask, device, permutation=permutation)
    return unpermute_decoded(permuted_decoded_params, bias_mask, permutation, device)


def evaluate_serialized_params(model_name, serialized_params, tokenizer, texts, device):
    model, overrides = SerialAutoModelForMaskedLM.load_serialized(serialized_params, model_name)
    model = model.to(device)
    model.eval()
    return masked_token_pseudo_perplexity(
        model,
        tokenizer,
        texts,
        overrides=overrides,
    )

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = load_texts(args.text_file)
    bias_autoencoder = load_bias_autoencoder(args.checkpoint_path, device)

    tokenizer = AutoTokenizer.from_pretrained(f"google/multiberts-seed_{args.validation_seeds[0]}")

    results = []

    for seed in args.validation_seeds:
        model_name = f"google/multiberts-seed_{seed}"
        source_model = SerialAutoModelForMaskedLM.from_pretrained(model_name).to(device)
        source_model.eval()

        original_perplexity = masked_token_pseudo_perplexity(source_model, tokenizer, texts)

        serialized_params = source_model.serialize()
        random_perm = torch.eye(serialized_params.vectors.shape[1]-1, device=device)[torch.randperm(serialized_params.vectors.shape[1]-1, device=device)]
        autoencoded_params = autoencode_params(serialized_params, bias_autoencoder, device)
        permuted_params = build_permuted_model_params(serialized_params, bias_autoencoder, random_perm, device)
        sanity_check_params = build_permuted_roundtrip_params(serialized_params, bias_autoencoder, random_perm, device)

        autoencoded_perplexity = evaluate_serialized_params(model_name, autoencoded_params, tokenizer, texts, device)
        permuted_perplexity = evaluate_serialized_params(model_name, permuted_params, tokenizer, texts, device)
        sanity_check_perplexity = evaluate_serialized_params(model_name, sanity_check_params, tokenizer, texts, device)
        sanity_diff = (autoencoded_params.vectors - sanity_check_params.vectors).abs()
        finite_sanity_diff = sanity_diff[torch.isfinite(sanity_diff)]

        results.append(
            {
                "seed": seed,
                "original_perplexity": original_perplexity,
                "autoencoded_perplexity": autoencoded_perplexity,
                "autoencoded_ratio": autoencoded_perplexity / original_perplexity,
                "permuted_perplexity": permuted_perplexity,
                "permuted_ratio": permuted_perplexity / original_perplexity,
                "sanity_check_perplexity": sanity_check_perplexity,
                "sanity_check_ratio": sanity_check_perplexity / original_perplexity,
                "sanity_check_max_abs_diff": finite_sanity_diff.max().item(),
                "sanity_check_mean_abs_diff": finite_sanity_diff.mean().item(),
            }
        )

    for result in results:
        print(
            f"Seed {result['seed']}: original_ppl={result['original_perplexity']:.6f}, "
            f"autoencoded_ppl={result['autoencoded_perplexity']:.6f}, "
            f"permuted_ppl={result['permuted_perplexity']:.6f}, "
            f"sanity_check_ppl={result['sanity_check_perplexity']:.6f}"
        )
        print(
            f"Seed {result['seed']}: autoencoded_ratio={result['autoencoded_ratio']:.6f}, "
            f"permuted_ratio={result['permuted_ratio']:.6f}, "
            f"sanity_check_ratio={result['sanity_check_ratio']:.6f}, "
            f"sanity_check_max_abs_diff={result['sanity_check_max_abs_diff']:.6e}, "
            f"sanity_check_mean_abs_diff={result['sanity_check_mean_abs_diff']:.6e}"
        )

    mean_original = sum(result["original_perplexity"] for result in results) / len(results)
    mean_autoencoded = sum(result["autoencoded_perplexity"] for result in results) / len(results)
    mean_permuted = sum(result["permuted_perplexity"] for result in results) / len(results)
    mean_sanity_check = sum(result["sanity_check_perplexity"] for result in results) / len(results)
    mean_autoencoded_ratio = sum(result["autoencoded_ratio"] for result in results) / len(results)
    mean_permuted_ratio = sum(result["permuted_ratio"] for result in results) / len(results)
    mean_sanity_check_ratio = sum(result["sanity_check_ratio"] for result in results) / len(results)

    print(f"mean_original_ppl={mean_original:.6f}")
    print(f"mean_autoencoded_ppl={mean_autoencoded:.6f}")
    print(f"mean_permuted_ppl={mean_permuted:.6f}")
    print(f"mean_sanity_check_ppl={mean_sanity_check:.6f}")
    print(f"mean_autoencoded_ratio={mean_autoencoded_ratio:.6f}")
    print(f"mean_permuted_ratio={mean_permuted_ratio:.6f}")
    print(f"mean_sanity_check_ratio={mean_sanity_check_ratio:.6f}")


if __name__ == "__main__":
    main()
