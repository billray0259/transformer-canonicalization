from types import MethodType

import pytest
import torch
from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM

from lib.serial_model import SerialAutoModelForMaskedLM


SERIAL_METHOD_NAMES = (
    "serialize_matrix",
    "serialize_bias",
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


def patch_working_serialize_bias(model):
    def fixed_serialize_bias(self, bias, name=None):
        assert bias.shape[0] == self.config.hidden_size, "Bias must have the same size as the model's hidden size."
        names = [f"{name}.bias" if name is not None else "bias"]
        padded_bias = torch.cat(
            [bias.unsqueeze(0), torch.zeros(1, 1, device=bias.device, dtype=bias.dtype)],
            dim=1,
        )
        from lib.serial_params import NamedSerialParameters

        return NamedSerialParameters.from_vector_list(names, [padded_bias])

    model.serialize_bias = MethodType(fixed_serialize_bias, model)
    return model


def test_from_pretrained_attaches_bound_serialization_methods(monkeypatch, tiny_config):
    dummy_model = BertForMaskedLM(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        assert model_args == ()
        assert kwargs == {}
        return dummy_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model = SerialAutoModelForMaskedLM.from_pretrained("dummy-model")

    assert loaded_model is dummy_model
    for method_name in SERIAL_METHOD_NAMES:
        method = getattr(loaded_model, method_name)
        assert callable(method)
        assert method.__self__ is dummy_model


def test_serialize_matrix_without_bias_pads_zero_column(tiny_serial_model):
    matrix = torch.arange(12, dtype=torch.float32).view(3, 4)

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj")

    assert serialized.names == ["proj.0", "proj.1", "proj.2"]
    assert serialized.vectors.shape == (3, 5)
    torch.testing.assert_close(serialized.vectors[:, :4], matrix)
    torch.testing.assert_close(serialized.vectors[:, 4], torch.zeros(3))


def test_serialize_matrix_with_bias_appends_bias_as_last_column(tiny_serial_model):
    matrix = torch.arange(8, dtype=torch.float32).view(2, 4)
    bias = torch.tensor([0.5, -1.5])

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj", bias=bias)

    assert serialized.names == ["proj.0", "proj.1"]
    assert serialized.vectors.shape == (2, 5)
    torch.testing.assert_close(serialized.vectors[:, :4], matrix)
    torch.testing.assert_close(serialized.vectors[:, 4], bias)


def test_serialize_matrix_rejects_hidden_size_mismatch(tiny_serial_model):
    matrix = torch.randn(2, 5)

    with pytest.raises(AssertionError, match="hidden size"):
        tiny_serial_model.serialize_matrix(matrix)


def test_serialize_bias_creates_single_padded_row(tiny_serial_model):
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])

    serialized = tiny_serial_model.serialize_bias(bias, name="proj")

    assert serialized.names == ["proj.bias"]
    assert serialized.vectors.shape == (1, 5)
    torch.testing.assert_close(serialized.vectors[0, :4], bias)
    torch.testing.assert_close(serialized.vectors[0, 4], torch.tensor(0.0))


def test_serialize_bias_rejects_hidden_size_mismatch(tiny_serial_model):
    with pytest.raises(AssertionError, match="same size as the model's hidden size"):
        tiny_serial_model.serialize_bias(torch.randn(3), name="proj")


def test_serialize_layernorm_uses_weight_and_bias_rows(tiny_serial_model):
    layernorm = tiny_serial_model.base_model.embeddings.LayerNorm

    serialized = tiny_serial_model.serialize_layernorm(layernorm, name="norm")

    assert serialized.names == ["norm.weight", "norm.bias"]
    assert serialized.vectors.shape == (2, 5)
    torch.testing.assert_close(serialized.vectors[0, :4], layernorm.weight)
    torch.testing.assert_close(serialized.vectors[1, :4], layernorm.bias)
    torch.testing.assert_close(serialized.vectors[:, 4], torch.zeros(2))


def test_serialize_embeddings_contains_embedding_tables_and_layernorm(tiny_serial_model, tiny_config):
    serialized = tiny_serial_model.serialize_embeddings()

    assert len(serialized.names) == tiny_config.vocab_size + tiny_config.max_position_embeddings + tiny_config.type_vocab_size + 2
    assert serialized.vectors.shape == (23, tiny_config.hidden_size + 1)
    assert serialized.names[:3] == [
        "embeddings.word_embeddings.weight.0",
        "embeddings.word_embeddings.weight.1",
        "embeddings.word_embeddings.weight.2",
    ]
    assert serialized.names[-2:] == ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias"]
    torch.testing.assert_close(
        serialized.vectors[0, :4],
        tiny_serial_model.base_model.embeddings.word_embeddings.weight[0],
    )
    torch.testing.assert_close(serialized.vectors[:, 4], torch.zeros(23))


def test_serialize_embeddings_skips_missing_layernorm(tiny_serial_model, tiny_config):
    tiny_serial_model.base_model.embeddings.LayerNorm = None

    serialized = tiny_serial_model.serialize_embeddings()

    assert len(serialized.names) == tiny_config.vocab_size + tiny_config.max_position_embeddings + tiny_config.type_vocab_size
    assert serialized.vectors.shape == (21, tiny_config.hidden_size + 1)
    assert all("LayerNorm" not in name for name in serialized.names)


def test_serialize_attention_serializes_read_biases_inline_and_write_bias_separately(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    attention = tiny_serial_model.base_model.encoder.layer[0].attention

    serialized = tiny_serial_model.serialize_attention(attention, name="attn")

    assert len(serialized.names) == 19
    assert serialized.vectors.shape == (19, tiny_config.hidden_size + 1)
    assert serialized.names[0] == "attn.self.query.0"
    assert serialized.names[12] == "attn.output.dense.weight.0"
    assert serialized.names[16] == "attn.output.dense.bias"
    assert serialized.names[-2:] == ["attn.output.LayerNorm.weight", "attn.output.LayerNorm.bias"]
    torch.testing.assert_close(serialized.vectors[0, :4], attention.self.query.weight[0])
    torch.testing.assert_close(serialized.vectors[:4, 4], attention.self.query.bias)
    torch.testing.assert_close(serialized.vectors[12, :4], attention.output.dense.weight.T[0])
    torch.testing.assert_close(serialized.vectors[12:16, 4], torch.zeros(4))
    torch.testing.assert_close(serialized.vectors[16, :4], attention.output.dense.bias)
    torch.testing.assert_close(serialized.vectors[16, 4], torch.tensor(0.0))


def test_serialize_attention_skips_missing_output_layernorm(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    attention = tiny_serial_model.base_model.encoder.layer[0].attention
    attention.output.LayerNorm = None

    serialized = tiny_serial_model.serialize_attention(attention, name="attn")

    assert len(serialized.names) == 17
    assert serialized.vectors.shape == (17, tiny_config.hidden_size + 1)
    assert all("output.LayerNorm" not in name for name in serialized.names)


def test_serialize_encoder_layer_inlines_intermediate_bias_and_separates_output_bias(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    layer = tiny_serial_model.base_model.encoder.layer[0]

    serialized = tiny_serial_model.serialize_encoder_layer(layer, name="encoder.layer.0")

    assert len(serialized.names) == 38
    assert serialized.vectors.shape == (38, tiny_config.hidden_size + 1)
    assert "encoder.layer.0.intermediate.dense.0" in serialized.names
    assert "encoder.layer.0.output.dense.bias" in serialized.names
    assert all(not name.startswith("encoder.layer.0.intermediate.dense.bias") for name in serialized.names)
    intermediate_start = serialized.names.index("encoder.layer.0.intermediate.dense.0")
    torch.testing.assert_close(
        serialized.vectors[intermediate_start:intermediate_start + 8, 4],
        layer.intermediate.dense.bias,
    )


def test_serialize_encoder_layer_skips_missing_output_layernorm(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    layer = tiny_serial_model.base_model.encoder.layer[0]
    layer.output.LayerNorm = None

    serialized = tiny_serial_model.serialize_encoder_layer(layer, name="encoder.layer.0")

    assert len(serialized.names) == 36
    assert serialized.vectors.shape == (36, tiny_config.hidden_size + 1)
    assert all(
        not name.startswith("encoder.layer.0.output.LayerNorm")
        for name in serialized.names
    )


def test_serialize_mlm_head_inlines_transform_and_decoder_biases(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    serialized = tiny_serial_model.serialize_mlm_head()

    assert len(serialized.names) == 17
    assert serialized.vectors.shape == (17, tiny_config.hidden_size + 1)
    transform_start = serialized.names.index("predictions.transform.dense.weight.0")
    decoder_start = serialized.names.index("predictions.decoder.weight.0")
    torch.testing.assert_close(
        serialized.vectors[transform_start:transform_start + 4, 4],
        tiny_serial_model.cls.predictions.transform.dense.bias,
    )
    torch.testing.assert_close(
        serialized.vectors[decoder_start:decoder_start + tiny_config.vocab_size, 4],
        tiny_serial_model.cls.predictions.bias,
    )


def test_serialize_mlm_head_skips_missing_layernorm(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    tiny_serial_model.cls.predictions.transform.LayerNorm = None

    serialized = tiny_serial_model.serialize_mlm_head()

    assert len(serialized.names) == tiny_config.hidden_size + tiny_config.vocab_size
    assert serialized.vectors.shape == (15, tiny_config.hidden_size + 1)
    assert all("transform.LayerNorm" not in name for name in serialized.names)
    torch.testing.assert_close(serialized.vectors[-1, -1], tiny_serial_model.cls.predictions.bias[-1])


def test_serialize_aggregates_embeddings_encoder_and_mlm_head(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    serialized = tiny_serial_model.serialize()

    assert len(serialized.names) == 116
    assert serialized.vectors.shape == (116, tiny_config.hidden_size + 1)

