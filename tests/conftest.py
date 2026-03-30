from types import MethodType

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from lib.serial_model import SerialAutoModelForMaskedLM


SERIAL_METHOD_NAMES = (
    "serialize_matrix",
    "serialize_layernorm",
    "serialize_embeddings",
    "serialize_attention",
    "serialize_encoder_layer",
    "serialize_encoder",
    "serialize_mlm_head",
    "serialize",
)


def attach_serial_methods(model):
    for method_name in SERIAL_METHOD_NAMES:
        method = getattr(SerialAutoModelForMaskedLM, method_name)
        setattr(model, method_name, MethodType(method, model))
    return model


@pytest.fixture
def tiny_config():
    return BertConfig(
        vocab_size=11,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        max_position_embeddings=7,
        type_vocab_size=3,
    )


@pytest.fixture
def tiny_serial_model(tiny_config):
    torch.manual_seed(0)
    model = BertForMaskedLM(tiny_config)
    model.eval()
    return attach_serial_methods(model)
