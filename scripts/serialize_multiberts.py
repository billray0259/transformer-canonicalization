from lib.serial_model import SerialAutoModelForMaskedLM
from tqdm import tqdm

for seed in tqdm(range(25)):
    model_name = f"google/multiberts-seed_{seed}"
    serial_model = SerialAutoModelForMaskedLM.from_pretrained(model_name)
    serialize_params = serial_model.serialize()
    serialize_params.save(f"../data/multiberts/serialized/seed_{seed}.pt")