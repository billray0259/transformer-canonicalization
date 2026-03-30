# %%

from transformers import AutoModelForMaskedLM
from lib.serial_model import SerialAutoModelForMaskedLM

# %%
# Load MultiBERT seed 0 from Hugging Face
model_name = "google/multiberts-seed_0"
mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

print("Encoder")
for name, param in mlm_model.base_model.named_parameters():
    print(f"{name} {param.shape}")

print("MLM Head")
for name, param in mlm_model.cls.named_parameters():
    print(f"{name} {param.shape}")
    
# %%

# for name, param in mlm_model.base_model.encoder.layer:
#     print(f"{name} {param.shape}")

print(type(mlm_model.base_model))

# %%


    
    



serial_model = SerialAutoModelForMaskedLM.from_pretrained(model_name)    
# %%


serialized_params = serial_model.serialize()
print(f"Serialized parameters: {[serialized_params.names[i] for i in range(0, len(serialized_params.names), 10000)]}")
print(f"Serialized parameters shape: {serialized_params.vectors.shape}")
# %% 

print("Total original parameters:\t", sum(p.numel() for p in mlm_model.parameters()))
print("Total serialized parameters:\t", serialized_params.vectors.numel())
print("Difference:\t\t\t", serialized_params.vectors.numel() - sum(p.numel() for p in mlm_model.parameters()))
# %%