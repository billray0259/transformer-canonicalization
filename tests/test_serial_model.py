from types import MethodType

import pytest
import torch
from torch.func import functional_call
from transformers import AutoModelForMaskedLM, BertForMaskedLM

from lib.serial_model import SerialAutoModelForMaskedLM
from lib.serial_params import MultiStreamSerialParameters, NamedSerialParameters
from lib.serial_reader import SerializedParameterOverrides

SERIAL_METHOD_NAMES = (
    "serialize_matrix",
    "serialize_bias",
    "serialize_head_biases",
    "serialize_layernorm",
    "serialize_embeddings",
    "serialize_attention",
    "serialize_encoder_layer",
    "serialize_encoder",
    "serialize_mlm_head",
    "serialize",
)


def has_tied_input_output_embeddings(model):
    return (
        model.base_model.embeddings.word_embeddings.weight.data_ptr()
        == model.cls.predictions.decoder.weight.data_ptr()
    )


def untie_input_output_embeddings(model):
    model.cls.predictions.decoder.weight = torch.nn.Parameter(
        model.cls.predictions.decoder.weight.detach().clone()
    )
    return model


def build_shell_model(tiny_config, seed=1234):
    torch.manual_seed(seed)
    model = BertForMaskedLM(tiny_config)
    model.eval()
    return model


def clone_stream(stream):
    if stream.vectors is None:
        return NamedSerialParameters()
    return NamedSerialParameters.from_vector_list(
        stream.names,
        [stream.vectors.detach().clone().requires_grad_(True)],
    )


def clone_multistream(serialized):
    return MultiStreamSerialParameters.from_stream_dict(
        {
            stream_name: (
                NamedSerialParameters()
                if stream.vectors is None
                else NamedSerialParameters.from_vector_list(
                    stream.names,
                    [stream.vectors.detach().clone()],
                )
            )
            for stream_name, stream in serialized.items()
        },
        equivalence_classes={
            stream_name: set(prefixes)
            for stream_name, prefixes in serialized.equivalence_classes.items()
        },
    )


def permutation_matrix(size, device):
    return torch.eye(size, device=device)[torch.randperm(size, device=device)]


def apply_multibert_permutation_family(permuted, source_model, mode, device):
    if mode == "model":
        permuted.apply_square_matrix(permutation_matrix(permuted["model"].vectors.shape[1], device), "model")
        return
    for layer_idx in range(source_model.config.num_hidden_layers):
        if mode == "qk":
            for head_idx in range(source_model.config.num_attention_heads):
                permuted.apply_square_matrix(
                    permutation_matrix(permuted[f"L{layer_idx}.H{head_idx}.qk"].vectors.shape[1], device),
                    f"L{layer_idx}.H{head_idx}.qk",
                )
        elif mode == "ov":
            for head_idx in range(source_model.config.num_attention_heads):
                permuted.apply_square_matrix(
                    permutation_matrix(permuted[f"L{layer_idx}.H{head_idx}.ov"].vectors.shape[1], device),
                    f"L{layer_idx}.H{head_idx}.ov",
                )
        elif mode == "head":
            permuted.apply_attention_head_matrix(
                permutation_matrix(source_model.config.num_attention_heads, device),
                layer_idx,
            )
        elif mode == "mlp":
            permuted.apply_square_matrix(
                permutation_matrix(permuted[f"L{layer_idx}.mlp"].vectors.shape[1], device),
                f"L{layer_idx}.mlp",
            )
        else:
            raise ValueError(f"Unknown permutation mode: {mode}")


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
        assert isinstance(method, MethodType)
        assert method.__self__ is dummy_model


def test_serialize_attention_routes_weights_and_biases_into_expected_streams(tiny_serial_model):
    attention = tiny_serial_model.base_model.encoder.layer[0].attention

    serialized = tiny_serial_model.serialize_attention(attention, layer_idx=0, name="attn")

    assert serialized.stream_names == [
        "model",
        "L0.H0.qk",
        "L0.H1.qk",
        "L0.H0.ov",
        "L0.H1.ov",
    ]
    assert serialized["L0.H0.qk"].names == ["attn.self.query.head.0.bias", "attn.self.key.head.0.bias"]
    assert serialized["L0.H1.qk"].names == ["attn.self.query.head.1.bias", "attn.self.key.head.1.bias"]
    assert serialized["L0.H0.ov"].names == ["attn.self.value.head.0.bias"]
    assert serialized["L0.H1.ov"].names == ["attn.self.value.head.1.bias"]
    assert serialized.get_equivalence_class("L0.H0.qk") == {
        "attn.self.query.weight.head.0",
        "attn.self.key.weight.head.0",
    }
    assert serialized.get_equivalence_class("L0.H0.ov") == {
        "attn.self.value.weight.head.0",
        "attn.output.dense.weight.head.0",
    }
    assert "attn.output.dense.bias" in serialized["model"].names
    torch.testing.assert_close(serialized["model"].vectors[0], attention.self.query.weight[0])
    torch.testing.assert_close(serialized["L0.H0.qk"].vectors[0], attention.self.query.bias[:2])
    torch.testing.assert_close(serialized["L0.H1.qk"].vectors[0], attention.self.query.bias[2:])
    torch.testing.assert_close(serialized["L0.H0.ov"].vectors[0], attention.self.value.bias[:2])
    torch.testing.assert_close(serialized["L0.H1.ov"].vectors[0], attention.self.value.bias[2:])


def test_serialize_encoder_layer_routes_mlp_bias_to_layer_stream(tiny_serial_model):
    layer = tiny_serial_model.base_model.encoder.layer[0]

    serialized = tiny_serial_model.serialize_encoder_layer(layer, layer_idx=0, name="encoder.layer.0")

    assert serialized["L0.mlp"].names == ["encoder.layer.0.intermediate.dense.bias"]
    assert serialized.get_equivalence_class("L0.mlp") == {
        "encoder.layer.0.intermediate.dense",
        "encoder.layer.0.output.dense.weight",
    }
    torch.testing.assert_close(serialized["L0.mlp"].vectors[0], layer.intermediate.dense.bias)
    assert "encoder.layer.0.output.dense.bias" in serialized["model"].names


def test_equivalent_model_rows_return_same_class_weight_rows(tiny_serial_model):
    serialized = tiny_serial_model.serialize_attention(
        tiny_serial_model.base_model.encoder.layer[0].attention,
        layer_idx=0,
        name="attn",
    )

    equivalent_rows = serialized.equivalent_model_rows("L0.H1.qk")

    assert equivalent_rows.names == [
        "attn.self.query.weight.head.1.0",
        "attn.self.query.weight.head.1.1",
        "attn.self.key.weight.head.1.0",
        "attn.self.key.weight.head.1.1",
    ]


def test_serialize_merges_tied_decoder_aux_rows_into_model_stream(tiny_serial_model):
    serialized = tiny_serial_model.serialize()

    assert has_tied_input_output_embeddings(tiny_serial_model)
    assert "decoder" not in serialized
    assert f"cls.predictions.transform.dense.weight.{tiny_serial_model.config.hidden_size - 1}" in serialized["model"].names
    assert "cls.predictions.transform.dense.bias" in serialized["model"].names
    assert "cls.predictions.transform.LayerNorm.weight" in serialized["model"].names
    assert serialized.get_equivalence_class("model") == {"cls.predictions.transform.dense.weight"}
    assert all(not name.startswith("cls.predictions.decoder.weight") for name in serialized["model"].names)
    assert serialized["vocab"].names == ["cls.predictions.decoder.bias"]


def test_serialize_keeps_decoder_stream_when_embeddings_are_untied(tiny_serial_model):
    untie_input_output_embeddings(tiny_serial_model)

    serialized = tiny_serial_model.serialize()

    assert not has_tied_input_output_embeddings(tiny_serial_model)
    assert "decoder" in serialized
    assert f"cls.predictions.transform.dense.weight.{tiny_serial_model.config.hidden_size - 1}" in serialized["decoder"].names
    assert "cls.predictions.transform.dense.bias" in serialized["decoder"].names
    assert all(not name.startswith("cls.predictions.transform.dense.weight") for name in serialized["model"].names)
    assert f"cls.predictions.decoder.weight.{tiny_serial_model.config.vocab_size - 1}" in serialized["decoder"].names


def test_serialize_helpers_cover_default_and_custom_naming_paths(tiny_serial_model):
    tiny_serial_model.base_model.embeddings.LayerNorm = None
    tiny_serial_model.base_model.encoder.layer[0].output.LayerNorm = None

    embeddings = tiny_serial_model.serialize_embeddings(name="custom.embeddings")
    attention = tiny_serial_model.serialize_attention(tiny_serial_model.base_model.encoder.layer[0].attention, layer_idx=0)
    encoder_layer = tiny_serial_model.serialize_encoder_layer(tiny_serial_model.base_model.encoder.layer[0], layer_idx=0)
    encoder = tiny_serial_model.serialize_encoder(name="custom.encoder")

    assert all(not name.startswith("custom.embeddings.LayerNorm") for name in embeddings.names)
    assert attention["model"].names[0].startswith("bert.encoder.layer.0.attention.self.query.weight")
    assert "bert.encoder.layer.0.output.dense.bias" in encoder_layer["model"].names
    assert "custom.encoder.layer.0.intermediate.dense.0" in encoder["model"].names


def test_deserialize_helpers_accept_custom_prefixes(tiny_serial_model, tiny_config):
    shell_model = build_shell_model(tiny_config, seed=999)

    embedding_overrides = SerializedParameterOverrides(
        MultiStreamSerialParameters.from_stream_dict(
            {"model": tiny_serial_model.serialize_embeddings(name="custom.embeddings")}
        )
    )
    encoder_overrides = SerializedParameterOverrides(tiny_serial_model.serialize_encoder(name="custom.encoder"))

    SerialAutoModelForMaskedLM._deserialize_embeddings(shell_model, embedding_overrides, name="custom.embeddings")
    SerialAutoModelForMaskedLM._deserialize_encoder(shell_model, encoder_overrides, name="custom.encoder")

    embedding_overrides.assert_done()
    encoder_overrides.assert_done()


def test_serialize_handles_tied_models_even_when_decoder_stream_is_absent(monkeypatch, tiny_serial_model):
    def fake_serialize_mlm_head(self, name="cls.predictions"):
        return MultiStreamSerialParameters.from_stream_dict(
            {
                "model": NamedSerialParameters(),
                "vocab": NamedSerialParameters(),
            }
        )

    monkeypatch.setattr(tiny_serial_model, "serialize_mlm_head", MethodType(fake_serialize_mlm_head, tiny_serial_model))

    serialized = tiny_serial_model.serialize()

    assert "decoder" not in serialized


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_load_serialized_matches_source_model_outputs(monkeypatch, tiny_serial_model, tiny_config, tied_embeddings):
    source_model = tiny_serial_model
    if not tied_embeddings:
        untie_input_output_embeddings(source_model)

    serialized = source_model.serialize()
    shell_model = build_shell_model(tiny_config, seed=4321)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        assert model_args == ()
        assert kwargs == {}
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model, overrides = SerialAutoModelForMaskedLM.load_serialized(serialized, "dummy-model")
    loaded_outputs = functional_call(
        loaded_model,
        overrides,
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    source_outputs = source_model(input_ids=input_ids, attention_mask=attention_mask)

    assert loaded_model is shell_model
    assert has_tied_input_output_embeddings(loaded_model) is tied_embeddings
    torch.testing.assert_close(loaded_outputs.logits, source_outputs.logits)


def test_serialize_and_load_handle_missing_optional_layernorms(monkeypatch, tiny_serial_model, tiny_config):
    tiny_serial_model.base_model.embeddings.LayerNorm = None
    tiny_serial_model.base_model.encoder.layer[0].attention.output.LayerNorm = None
    tiny_serial_model.base_model.encoder.layer[1].output.LayerNorm = None
    tiny_serial_model.cls.predictions.transform.LayerNorm = None

    serialized = tiny_serial_model.serialize()
    shell_model = build_shell_model(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model, overrides = SerialAutoModelForMaskedLM.load_serialized(serialized, "dummy-model")

    assert loaded_model.base_model.embeddings.LayerNorm is None
    assert loaded_model.base_model.encoder.layer[0].attention.output.LayerNorm is None
    assert loaded_model.base_model.encoder.layer[1].output.LayerNorm is None
    assert loaded_model.cls.predictions.transform.LayerNorm is None
    assert "bert.embeddings.LayerNorm.weight" not in overrides
    assert "bert.encoder.layer.0.attention.output.LayerNorm.weight" not in overrides
    assert "bert.encoder.layer.1.output.LayerNorm.weight" not in overrides
    assert "cls.predictions.transform.LayerNorm.weight" not in overrides


def test_load_serialized_rejects_unexpected_row_names(monkeypatch, tiny_serial_model, tiny_config):
    serialized = tiny_serial_model.serialize()
    broken_serialized = MultiStreamSerialParameters.from_stream_dict(
        {
            **serialized,
            "model": NamedSerialParameters.from_vector_list(
                ["wrong.prefix", *serialized["model"].names[1:]],
                [serialized["model"].vectors],
            ),
        }
    )
    shell_model = build_shell_model(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    with pytest.raises(AssertionError, match="Expected rows"):
        SerialAutoModelForMaskedLM.load_serialized(broken_serialized, "dummy-model")


def test_load_serialized_rejects_flat_named_serialized_params(monkeypatch, tiny_serial_model, tiny_config):
    flat_serialized = NamedSerialParameters()
    for stream_name in tiny_serial_model.serialize().stream_names:
        flat_serialized += tiny_serial_model.serialize()[stream_name]

    shell_model = build_shell_model(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    with pytest.raises(TypeError, match="MultiStreamSerialParameters"):
        SerialAutoModelForMaskedLM.load_serialized(flat_serialized, "dummy-model")


def test_load_serialized_backprops_to_multistream_vectors(monkeypatch, tiny_serial_model, tiny_config):
    untie_input_output_embeddings(tiny_serial_model)
    serialized = tiny_serial_model.serialize()
    differentiable_serialized = MultiStreamSerialParameters.from_stream_dict(
        {stream_name: clone_stream(stream) for stream_name, stream in serialized.items()}
    )
    shell_model = build_shell_model(tiny_config, seed=5678)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    functional_model, overrides = SerialAutoModelForMaskedLM.load_serialized(
        differentiable_serialized,
        "dummy-model",
    )
    loss = functional_call(functional_model, overrides, (), {"input_ids": input_ids}).logits.square().mean()
    loss.backward()

    grads = [
        stream.vectors.grad
        for stream in differentiable_serialized.values()
        if stream.vectors is not None and stream.vectors.grad is not None
    ]
    assert grads
    assert sum(torch.count_nonzero(grad).item() for grad in grads) > 0


@pytest.mark.parametrize(
    ("mode", "atol", "rtol"),
    [
        ("model", 5e-5, 1e-4),
        ("qk", 5e-5, 1e-4),
        ("ov", 5e-5, 1e-4),
        ("head", 5e-5, 1e-4),
        ("mlp", 2e-4, 2e-4),
    ],
)
def test_multibert_permutation_equivalence_classes_preserve_outputs(mode, atol, rtol):
    model_name = "google/multiberts-seed_0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        source_model = SerialAutoModelForMaskedLM.from_pretrained(model_name).to(device)
    except Exception as exc:
        pytest.skip(f"Hugging Face model assets unavailable: {exc}")

    source_model.eval()
    serialized = source_model.serialize()
    permuted = clone_multistream(serialized)

    torch.manual_seed(0)
    apply_multibert_permutation_family(permuted, source_model, mode, device)

    permuted_model, overrides = SerialAutoModelForMaskedLM.load_serialized(permuted, model_name)
    permuted_model = permuted_model.to(device)
    input_ids = torch.randint(0, source_model.config.vocab_size, (2, 8), device=device)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]], device=device)

    with torch.no_grad():
        source_logits = source_model(input_ids=input_ids, attention_mask=attention_mask).logits
        permuted_logits = functional_call(
            permuted_model,
            overrides,
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        ).logits

    torch.testing.assert_close(permuted_logits, source_logits, atol=atol, rtol=rtol)

