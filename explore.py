# %%
from lib.serial_model import SerialAutoModelForMaskedLM
import torch
from tqdm import tqdm
# %%
# Load MultiBERT seed 0 from Hugging Face
model_name = "google/multiberts-seed_0"
serial_model = SerialAutoModelForMaskedLM.from_pretrained(model_name)
multi_params = serial_model.serialize()

    
# %%
for key, item in multi_params.items():
    equiv_class = multi_params.get_equivalence_class(key)
    print(f"{key} {tuple(item.vectors.shape)} {equiv_class}")

    
# %%
def random_permutation(size: int, device: torch.device = "cpu") -> torch.Tensor:
    return torch.eye(size, device=device)[torch.randperm(size, device=device)]

stream_names = list(multi_params)
for stream_name in tqdm(stream_names):
    named_params = multi_params[stream_name]
    perm = random_permutation(named_params.vectors.shape[1], named_params.vectors.device)
    multi_params.apply_square_matrix(perm, stream_name)

# %% 
permuted_model = SerialAutoModelForMaskedLM.from_serial_params(multi_params)

# check that permuted model has the same activations as original model

random_tokens = torch.randint(0, serial_model.config.vocab_size, (1, 10))
with torch.no_grad():
    original_output = serial_model(random_tokens).logits
    permuted_output = permuted_model(random_tokens).logits
    
torch.testing.assert_close(original_output, permuted_output)