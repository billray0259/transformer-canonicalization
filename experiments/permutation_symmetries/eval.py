from lib.serial_model import SerialAutoModelForMaskedLM
from lib.utils import masked_token_pseudo_perplexity
import torch


def random_permutation(size: int, device: torch.device) -> torch.Tensor:
    return torch.eye(size, device=device)[torch.randperm(size, device=device)]

model = SerialAutoModelForMaskedLM.from_pretrained("multiBERT/seed-0")

multi_params = model.serialize()
