# %%
from lib.serial_model import SerialAutoModelForMaskedLM
from lib.serial_params import NamedSerialParameters
import torch
import random
from tqdm import tqdm
# %%
# Load MultiBERT seed 0 from Hugging Face
model_name = "google/multiberts-seed_0"
serial_model = SerialAutoModelForMaskedLM.from_pretrained(model_name)
serialize_params = serial_model.serialize() 
# %%

names = serialize_params.names
vectors = serialize_params.vectors

# %%
print(sum(["head.0" in name for name in names]))

# %%

nan_count = sum(torch.isnan(v[-1]) if isinstance(v, torch.Tensor) and v.numel() > 0 else False for v in vectors)
print(f"Rows with NaN in last column: {nan_count}")

total_nans = sum(torch.isnan(v).sum().item() if isinstance(v, torch.Tensor) else 0 for v in vectors)
print(f"Total NaN values: {total_nans}")

print(f"Total Rows: {len(vectors)}")
# %%

has_nan_mask = torch.tensor([torch.isnan(v[-1]) if isinstance(v, torch.Tensor) and v.numel() > 0 else False for v in vectors])
nan_names = [name for name, has_nan in zip(names, has_nan_mask) if has_nan]
print(f"10 names of rows with NaN in last column: {random.sample(nan_names, min(10, len(nan_names)))}")
# %%


    
# %%




    
# %%
