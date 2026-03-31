from lib.serial_model import SerialAutoModelForMaskedLM
from lib.serial_params import NamedSerialParameters
import torch
from tqdm import tqdm

train_model_seeds = range(20)
validation_model_seeds = range(20, 23)
test_model_seeds = range(23, 25)

train_rows = []
for i in tqdm(train_model_seeds, desc="Processing training seeds"):
    path = f"data/multiberts/serialized/seed_{i}.pt"
    params = NamedSerialParameters.load(path)
    train_rows.extend([v for v in params.vectors if not torch.isnan(v[-1])])

train_rows = torch.stack(train_rows)
print(f"Total train rows collected: {train_rows.shape[0]}")

validation_rows = []
for i in tqdm(validation_model_seeds, desc="Processing validation seeds"):
    path = f"data/multiberts/serialized/seed_{i}.pt"
    params = NamedSerialParameters.load(path)
    validation_rows.extend([v for v in params.vectors if not torch.isnan(v[-1])])
    
validation_rows = torch.stack(validation_rows)
print(f"Total validation rows collected: {validation_rows.shape[0]}")

test_rows = []
for i in tqdm(test_model_seeds, desc="Processing test seeds"):
    path = f"data/multiberts/serialized/seed_{i}.pt"
    params = NamedSerialParameters.load(path)
    test_rows.extend([v for v in params.vectors if not torch.isnan(v[-1])])
    
test_rows = torch.stack(test_rows)
print(f"Total test rows collected: {test_rows.shape[0]}")

# save the collected rows for later use
torch.save(train_rows.detach(), "data/bias_autoencoder/dataset/train_rows.pt")
torch.save(validation_rows.detach(), "data/bias_autoencoder/dataset/validation_rows.pt")
torch.save(test_rows.detach(), "data/bias_autoencoder/dataset/test_rows.pt")