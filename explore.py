# %%
import torch
from transformers import AutoTokenizer

from lib.serial_model import SerialAutoModelForMaskedLM
from lib.utils import masked_token_pseudo_perplexity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_0 = "google/multiberts-seed_0"
model_name_1 = "google/multiberts-seed_1"

tokenizer = AutoTokenizer.from_pretrained(model_name_0)
texts = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
    "The quick brown fox jumps over the dog.",
    "Machine learning is a subset of science.",
    "She enjoys reading books in her free time.",
    "Jupiter is the largest planet in our solar system.",
    "Water freezes at zero degrees Celsius.",
    "Shakespeare wrote many famous plays.",
    "The human brain contains billions of neurons.",
    "Photosynthesis converts sunlight into energy.",
]

# Load both models and serialize parameters.
serial_model_0 = SerialAutoModelForMaskedLM.from_pretrained(model_name_0).to(device).eval()
serial_model_1 = SerialAutoModelForMaskedLM.from_pretrained(model_name_1).to(device).eval()
symmeters_0 = serial_model_0.serialize()
symmeters_1 = serial_model_1.serialize()

# %%

print("symmetry; parameters shape; equivalency prefixes")
for symmetry, parameters in symmeters_0.items():
    print(symmetry, tuple(parameters.shape), symmeters_0.get_equivalence_class(symmetry))

# %%
ppl_0_before = masked_token_pseudo_perplexity(serial_model_0, tokenizer, texts)
ppl_1_before = masked_token_pseudo_perplexity(serial_model_1, tokenizer, texts)
print(f"Before swap: Seed 0 perplexity = {ppl_0_before:.4f}, Seed 1 perplexity = {ppl_1_before:.4f}")

# %%
# Swap only vocabulary bias symmetry between seeds.
symmeters_0["vocab"], symmeters_1["vocab"] = symmeters_1["vocab"], symmeters_0["vocab"]

swapped_model_0, swapped_overrides_0 = SerialAutoModelForMaskedLM.load_serialized(symmeters_0, model_name_0)
swapped_model_1, swapped_overrides_1 = SerialAutoModelForMaskedLM.load_serialized(symmeters_1, model_name_1)
swapped_model_0 = swapped_model_0.to(device).eval()
swapped_model_1 = swapped_model_1.to(device).eval()

ppl_0_after = masked_token_pseudo_perplexity(swapped_model_0, tokenizer, texts, overrides=swapped_overrides_0)
ppl_1_after = masked_token_pseudo_perplexity(swapped_model_1, tokenizer, texts, overrides=swapped_overrides_1)
print(f"After swap: Seed 0 perplexity = {ppl_0_after:.4f}, Seed 1 perplexity = {ppl_1_after:.4f}")

# %%
