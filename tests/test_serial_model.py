from types import MethodType

import pytest
import torch
from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM

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


def test_serialize_matrix_separate_without_bias_preserves_rows(tiny_serial_model):
    matrix = torch.arange(12, dtype=torch.float32).view(3, 4)

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj", bias_method="separate")

    assert serialized.names == ["proj.0", "proj.1", "proj.2"]
    torch.testing.assert_close(serialized.vectors, matrix)


def test_serialize_matrix_concat_without_bias_pads_zero_column(tiny_serial_model):
    matrix = torch.arange(8, dtype=torch.float32).view(2, 4)

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj", bias_method="concat")

    assert serialized.names == ["proj.0", "proj.1"]
    assert serialized.vectors.shape == (2, 5)
    torch.testing.assert_close(serialized.vectors[:, :4], matrix)
    torch.testing.assert_close(serialized.vectors[:, 4], torch.zeros(2))


def test_serialize_matrix_rejects_hidden_size_mismatch(tiny_serial_model):
    matrix = torch.randn(2, 5)

    with pytest.raises(AssertionError, match="hidden size"):
        tiny_serial_model.serialize_matrix(matrix)


def test_serialize_matrix_rejects_incompatible_bias_shape(tiny_serial_model):
    matrix = torch.randn(2, 4)
    bias = torch.randn(3)

    with pytest.raises(AssertionError, match="Incompatible matrix and bias"):
        tiny_serial_model.serialize_matrix(matrix, bias=bias, bias_method="concat")


def test_serialize_layernorm_uses_weight_and_bias_rows(tiny_serial_model):
    layernorm = tiny_serial_model.base_model.embeddings.LayerNorm

    separate = tiny_serial_model.serialize_layernorm(layernorm, name="norm", bias_method="separate")
    concat = tiny_serial_model.serialize_layernorm(layernorm, name="norm", bias_method="concat")

    assert separate.names == ["norm.weight", "norm.bias"]
    torch.testing.assert_close(separate.vectors[0], layernorm.weight)
    torch.testing.assert_close(separate.vectors[1], layernorm.bias)
    assert concat.vectors.shape == (2, 5)
    torch.testing.assert_close(concat.vectors[:, :4], separate.vectors)
    torch.testing.assert_close(concat.vectors[:, 4], torch.zeros(2))


def test_serialize_embeddings_separate_contains_embedding_tables_and_layernorm(tiny_serial_model, tiny_config):
    serialized = tiny_serial_model.serialize_embeddings(bias_method="separate")

    assert len(serialized.names) == tiny_config.vocab_size + tiny_config.max_position_embeddings + tiny_config.type_vocab_size + 2
    assert serialized.vectors.shape == (23, tiny_config.hidden_size)
    assert serialized.names[:3] == [
        "embeddings.word_embeddings.weight.0",
        "embeddings.word_embeddings.weight.1",
        "embeddings.word_embeddings.weight.2",
    ]
    assert serialized.names[-2:] == ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias"]
    torch.testing.assert_close(
        serialized.vectors[0],
        tiny_serial_model.base_model.embeddings.word_embeddings.weight[0],
    )


def test_serialize_embeddings_skips_missing_layernorm(tiny_serial_model, tiny_config):
    tiny_serial_model.base_model.embeddings.LayerNorm = None

    serialized = tiny_serial_model.serialize_embeddings(bias_method="separate")

    assert len(serialized.names) == tiny_config.vocab_size + tiny_config.max_position_embeddings + tiny_config.type_vocab_size
    assert serialized.vectors.shape == (21, tiny_config.hidden_size)
    assert all("LayerNorm" not in name for name in serialized.names)


def test_serialize_attention_separate_serializes_qkv_output_and_layernorm(tiny_serial_model, tiny_config):
    attention = tiny_serial_model.base_model.encoder.layer[0].attention

    serialized = tiny_serial_model.serialize_attention(attention, name="attn", bias_method="separate")

    assert len(serialized.names) == 22
    assert serialized.vectors.shape == (22, tiny_config.hidden_size)
    assert serialized.names[0] == "attn.self.query.0"
    assert serialized.names[4] == "attn.self.query.bias"
    assert serialized.names[-2:] == ["attn.output.LayerNorm.weight", "attn.output.LayerNorm.bias"]
    torch.testing.assert_close(serialized.vectors[0], attention.self.query.weight[0])
    torch.testing.assert_close(serialized.vectors[4], attention.self.query.bias)
    torch.testing.assert_close(serialized.vectors[15], attention.output.dense.weight.T[0])


def test_serialize_attention_skips_missing_output_layernorm(tiny_serial_model, tiny_config):
    attention = tiny_serial_model.base_model.encoder.layer[0].attention
    attention.output.LayerNorm = None

    serialized = tiny_serial_model.serialize_attention(attention, name="attn", bias_method="separate")

    assert len(serialized.names) == 20
    assert serialized.vectors.shape == (20, tiny_config.hidden_size)
    assert all("output.LayerNorm" not in name for name in serialized.names)


def test_serialize_encoder_layer_separate_serializes_intermediate_bias_rows(tiny_serial_model, tiny_config):
    layer = tiny_serial_model.base_model.encoder.layer[0]

    serialized = tiny_serial_model.serialize_encoder_layer(layer, name="encoder.layer.0", bias_method="separate")

    assert len(serialized.names) == 43
    assert serialized.vectors.shape == (43, tiny_config.hidden_size)
    assert "encoder.layer.0.intermediate.dense.bias.0" in serialized.names
    assert "encoder.layer.0.intermediate.dense.bias.1" in serialized.names


def test_serialize_encoder_layer_separate_skips_missing_output_layernorm(tiny_serial_model, tiny_config):
    layer = tiny_serial_model.base_model.encoder.layer[0]
    layer.output.LayerNorm = None

    serialized = tiny_serial_model.serialize_encoder_layer(layer, name="encoder.layer.0", bias_method="separate")

    assert len(serialized.names) == 41
    assert serialized.vectors.shape == (41, tiny_config.hidden_size)
    assert all(
        not name.startswith("encoder.layer.0.output.LayerNorm")
        for name in serialized.names
    )


@pytest.mark.xfail(reason="concat serialization for encoder layer output dense still rejects rectangular matrices")
def test_serialize_encoder_layer_concat_inlines_biases_and_skips_output_layernorm(tiny_serial_model):
    layer = tiny_serial_model.base_model.encoder.layer[0]
    layer.output.LayerNorm = None

    serialized = tiny_serial_model.serialize_encoder_layer(layer, name="encoder.layer.0", bias_method="concat")

    assert len(serialized.names) == 30
    assert serialized.vectors.shape == (30, tiny_serial_model.config.hidden_size + 1)
    assert all(not name.startswith("encoder.layer.0.intermediate.dense.bias.") for name in serialized.names)
    assert all("output.LayerNorm" not in name for name in serialized.names)


def test_serialize_encoder_layer_separate_requires_divisible_intermediate_bias():
    config = BertConfig(
        vocab_size=11,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=6,
        max_position_embeddings=7,
        type_vocab_size=3,
    )
    model = attach_serial_methods(BertForMaskedLM(config).eval())

    with pytest.raises(AssertionError, match="divisible by hidden_size"):
        model.serialize_encoder_layer(model.base_model.encoder.layer[0], bias_method="separate")


def test_serialize_mlm_head_separate_returns_decoder_bias_out_of_band(tiny_serial_model, tiny_config):
    serialized, cls_bias = tiny_serial_model.serialize_mlm_head(bias_method="separate")

    assert len(serialized.names) == 18
    assert serialized.vectors.shape == (18, tiny_config.hidden_size)
    assert cls_bias.shape == (tiny_config.vocab_size,)
    torch.testing.assert_close(cls_bias, tiny_serial_model.cls.predictions.bias)
    assert all(not name.endswith("decoder.bias") for name in serialized.names)


def test_serialize_mlm_head_concat_skips_missing_layernorm_and_returns_no_bias(tiny_serial_model, tiny_config):
    tiny_serial_model.cls.predictions.transform.LayerNorm = None

    serialized, cls_bias = tiny_serial_model.serialize_mlm_head(bias_method="concat")

    assert len(serialized.names) == tiny_config.hidden_size + tiny_config.vocab_size
    assert serialized.vectors.shape == (15, tiny_config.hidden_size + 1)
    assert cls_bias is None
    assert all("transform.LayerNorm" not in name for name in serialized.names)
    torch.testing.assert_close(serialized.vectors[-1, -1], tiny_serial_model.cls.predictions.bias[-1])


def test_serialize_separate_aggregates_embeddings_encoder_and_mlm_head(tiny_serial_model, tiny_config):
    serialized, cls_bias = tiny_serial_model.serialize(bias_method="separate")

    assert len(serialized.names) == 127
    assert serialized.vectors.shape == (127, tiny_config.hidden_size)
    assert cls_bias.shape == (tiny_config.vocab_size,)


def test_serialize_matrix_concat_with_bias_appends_bias_as_last_column(tiny_serial_model):
    matrix = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    bias = torch.tensor([0.5, -1.5])

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj", bias=bias, bias_method="concat")

    assert serialized.names == ["proj.0", "proj.1"]
    assert serialized.vectors.shape == (2, 5)
    torch.testing.assert_close(serialized.vectors[:, :4], matrix)
    torch.testing.assert_close(serialized.vectors[:, 4], bias)