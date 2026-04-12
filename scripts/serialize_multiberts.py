from transformers import AutoModelForMaskedLM
from tqdm import tqdm

from lib.serial_model import serialize_model

for seed in tqdm(range(25)):
    model_name = f"google/multiberts-seed_{seed}"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    serialize_params = serialize_model(model)
    serialize_params.save(f"data/multiberts/serialized/seed_{seed}.pt")