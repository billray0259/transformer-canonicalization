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

class BiasAutoencoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # (d_model+1, d_model) [I, 0]
        self.encoder = torch.nn.Parameter(torch.cat([torch.eye(d_model), torch.zeros((1, d_model))], dim=0))
        # (d_model, d_model+1) [I, 0]^T
        self.decoder = torch.nn.Parameter(torch.cat([torch.eye(d_model), torch.zeros((d_model, 1))], dim=1))
    
    def encode(self, x):
        return x @ self.encoder
    
    def decode(self, z):
        return z @ self.decoder
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
# %%

def random_permutation_matrix(d_model):
    return torch.eye(d_model)[torch.randperm(d_model)]

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

# %%
train_model_seeds = range(20)
test_model_seeds = range(20, 25)

train_rows = []
for i in tqdm(train_model_seeds):
    path = f"data/multiberts/serialized/seed_{i}.pt"
    params = NamedSerialParameters.load(path)
    train_rows.extend([v for v in params.vectors if not torch.isnan(v[-1])])

train_rows = torch.stack(train_rows)
print(f"Total train rows collected: {train_rows.shape[0]}")
    
# %%
