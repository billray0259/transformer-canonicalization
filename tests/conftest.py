import pytest
import torch
from transformers import BertConfig, BertForMaskedLM


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="run tests marked expensive",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        return

    skip_expensive = pytest.mark.skip(reason="need --run-expensive option to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)


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
def tiny_model(tiny_config):
    torch.manual_seed(0)
    model = BertForMaskedLM(tiny_config)
    model.eval()
    return model
