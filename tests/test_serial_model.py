from types import MethodType

import pytest
import torch
from torch.func import functional_call
from transformers import AutoModelForMaskedLM, BertForMaskedLM

from lib.serial_model import SerialAutoModelForMaskedLM
from lib.serial_params import NamedSerialParameters


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
            [bias.unsqueeze(0), torch.full((1, 1), float("nan"), device=bias.device, dtype=bias.dtype)],
            dim=1,
        )
        from lib.serial_params import NamedSerialParameters

        return NamedSerialParameters.from_vector_list(names, [padded_bias])

    model.serialize_bias = MethodType(fixed_serialize_bias, model)
    return model


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


def load_pretrained_serial_model_or_skip(model_name="google/multiberts-seed_0"):
    try:
        model = SerialAutoModelForMaskedLM.from_pretrained(model_name)
    except Exception as exc:
        pytest.skip(f"Could not load pretrained model {model_name}: {exc}")

    model.eval()
    return model


def run_and_capture_self_attention_context(model, layer_index, run_forward):
    captured = {}

    def hook(_module, _inputs, outputs):
        captured["context"] = outputs[0].detach().clone()

    handle = model.base_model.encoder.layer[layer_index].attention.self.register_forward_hook(hook)
    try:
        run_forward()
    finally:
        handle.remove()

    return captured["context"]


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


def test_serialize_matrix_without_bias_pads_nan_column(tiny_serial_model):
    matrix = torch.arange(12, dtype=torch.float32).view(3, 4)

    serialized = tiny_serial_model.serialize_matrix(matrix, name="proj")

    assert serialized.names == ["proj.0", "proj.1", "proj.2"]
    assert serialized.vectors.shape == (3, 5)
    torch.testing.assert_close(serialized.vectors[:, :4], matrix)
    assert torch.isnan(serialized.vectors[:, 4]).all()


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
    assert torch.isnan(serialized.vectors[0, 4])


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
    assert torch.isnan(serialized.vectors[:, 4]).all()


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_serialize_embeddings_contains_expected_tables_and_layernorm(tiny_serial_model, tiny_config, tied_embeddings):
    if not tied_embeddings:
        untie_input_output_embeddings(tiny_serial_model)

    serialized = tiny_serial_model.serialize_embeddings()
    skips_word_embeddings = has_tied_input_output_embeddings(tiny_serial_model)
    expected_rows = tiny_config.max_position_embeddings + tiny_config.type_vocab_size + 2
    if not skips_word_embeddings:
        expected_rows += tiny_config.vocab_size

    assert len(serialized.names) == expected_rows
    assert serialized.vectors.shape == (expected_rows, tiny_config.hidden_size + 1)
    if skips_word_embeddings:
        assert serialized.names[:3] == [
            "embeddings.position_embeddings.weight.0",
            "embeddings.position_embeddings.weight.1",
            "embeddings.position_embeddings.weight.2",
        ]
    else:
        assert serialized.names[:3] == [
            "embeddings.word_embeddings.weight.0",
            "embeddings.word_embeddings.weight.1",
            "embeddings.word_embeddings.weight.2",
        ]
    assert serialized.names[-2:] == ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias"]
    first_embedding_table = tiny_serial_model.base_model.embeddings.position_embeddings
    if not skips_word_embeddings:
        first_embedding_table = tiny_serial_model.base_model.embeddings.word_embeddings
    torch.testing.assert_close(serialized.vectors[0, :4], first_embedding_table.weight[0])
    assert torch.isnan(serialized.vectors[:, 4]).all()


def test_serialize_embeddings_skips_missing_layernorm(tiny_serial_model, tiny_config):
    tiny_serial_model.base_model.embeddings.LayerNorm = None

    serialized = tiny_serial_model.serialize_embeddings()
    expected_rows = tiny_config.max_position_embeddings + tiny_config.type_vocab_size
    if not has_tied_input_output_embeddings(tiny_serial_model):
        expected_rows += tiny_config.vocab_size

    assert len(serialized.names) == expected_rows
    assert serialized.vectors.shape == (expected_rows, tiny_config.hidden_size + 1)
    assert all("LayerNorm" not in name for name in serialized.names)


def test_serialize_attention_serializes_read_biases_inline_and_write_bias_separately(tiny_serial_model, tiny_config):
    patch_working_serialize_bias(tiny_serial_model)
    attention = tiny_serial_model.base_model.encoder.layer[0].attention

    serialized = tiny_serial_model.serialize_attention(attention, name="attn")

    assert len(serialized.names) == 19
    assert serialized.vectors.shape == (19, tiny_config.hidden_size + 1)
    assert serialized.names[0] == "attn.self.query.head.0.0"
    assert serialized.names[1] == "attn.self.query.head.0.1"
    assert serialized.names[2] == "attn.self.query.head.1.0"
    assert serialized.names[12] == "attn.output.dense.weight.head.0.0"
    assert serialized.names[16] == "attn.output.dense.bias"
    assert serialized.names[-2:] == ["attn.output.LayerNorm.weight", "attn.output.LayerNorm.bias"]
    torch.testing.assert_close(serialized.vectors[0, :4], attention.self.query.weight[0])
    torch.testing.assert_close(serialized.vectors[:4, 4], attention.self.query.bias)
    torch.testing.assert_close(serialized.vectors[12, :4], attention.output.dense.weight.T[0])
    assert torch.isnan(serialized.vectors[12:16, 4]).all()
    torch.testing.assert_close(serialized.vectors[16, :4], attention.output.dense.bias)
    assert torch.isnan(serialized.vectors[16, 4])


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


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_serialize_aggregates_embeddings_encoder_and_mlm_head(tiny_serial_model, tiny_config, tied_embeddings):
    if not tied_embeddings:
        untie_input_output_embeddings(tiny_serial_model)

    patch_working_serialize_bias(tiny_serial_model)
    serialized = tiny_serial_model.serialize()

    embedding_rows = tiny_config.max_position_embeddings + tiny_config.type_vocab_size + 2
    if not has_tied_input_output_embeddings(tiny_serial_model):
        embedding_rows += tiny_config.vocab_size
    expected_rows = embedding_rows + tiny_config.num_hidden_layers * 38 + 17

    assert len(serialized.names) == expected_rows
    assert serialized.vectors.shape == (expected_rows, tiny_config.hidden_size + 1)


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_serialize_parameter_count_matches_expected_padding_overhead(tiny_serial_model, tiny_config, tied_embeddings):
    if not tied_embeddings:
        untie_input_output_embeddings(tiny_serial_model)

    patch_working_serialize_bias(tiny_serial_model)

    original_parameter_count = sum(parameter.numel() for parameter in tiny_serial_model.parameters())
    serialized = tiny_serial_model.serialize()
    serialized_parameter_count = serialized.vectors.numel()

    expected_extra_parameters = (
        + tiny_config.max_position_embeddings
        + tiny_config.type_vocab_size
        + 2
        + tiny_config.num_hidden_layers * (tiny_config.hidden_size + tiny_config.intermediate_size + 6)
        + 2
    )

    if not has_tied_input_output_embeddings(tiny_serial_model):
        expected_extra_parameters += tiny_config.vocab_size

    assert serialized_parameter_count - original_parameter_count == expected_extra_parameters
    assert serialized_parameter_count == original_parameter_count + expected_extra_parameters


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_serialize_preserves_non_nan_parameter_count(tiny_serial_model, tied_embeddings):
    if not tied_embeddings:
        untie_input_output_embeddings(tiny_serial_model)

    patch_working_serialize_bias(tiny_serial_model)

    original_parameter_count = sum(parameter.numel() for parameter in tiny_serial_model.parameters())
    serialized = tiny_serial_model.serialize()
    non_nan_serialized_parameter_count = torch.count_nonzero(~torch.isnan(serialized.vectors)).item()

    assert non_nan_serialized_parameter_count == original_parameter_count


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_load_serialized_matches_source_model_outputs(monkeypatch, tiny_serial_model, tiny_config, tied_embeddings):
    source_model = tiny_serial_model
    if not tied_embeddings:
        untie_input_output_embeddings(source_model)

    patch_working_serialize_bias(source_model)
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


def test_load_serialized_restores_missing_optional_layernorms(monkeypatch, tiny_serial_model, tiny_config):
    source_model = tiny_serial_model
    patch_working_serialize_bias(source_model)
    source_model.base_model.embeddings.LayerNorm = None
    source_model.base_model.encoder.layer[0].attention.output.LayerNorm = None
    source_model.base_model.encoder.layer[1].output.LayerNorm = None
    source_model.cls.predictions.transform.LayerNorm = None

    serialized = source_model.serialize()
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
    patch_working_serialize_bias(tiny_serial_model)
    serialized = tiny_serial_model.serialize()
    broken_serialized = NamedSerialParameters.from_vector_list(
        ["wrong.prefix"] + serialized.names[1:],
        [serialized.vectors],
    )
    shell_model = build_shell_model(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    with pytest.raises(AssertionError, match="Expected rows"):
        SerialAutoModelForMaskedLM.load_serialized(broken_serialized, "dummy-model")


def test_load_serialized_backprops_to_serialized_vectors(monkeypatch, tiny_serial_model, tiny_config):
    source_model = tiny_serial_model
    patch_working_serialize_bias(source_model)
    serialized = source_model.serialize()
    differentiable_vectors = serialized.vectors.detach().clone().requires_grad_(True)
    differentiable_serialized = NamedSerialParameters.from_vector_list(serialized.names, [differentiable_vectors])
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

    assert differentiable_vectors.grad is not None
    assert torch.count_nonzero(differentiable_vectors.grad).item() > 0


def test_load_serialized_zeroed_attention_head_only_affects_that_head(monkeypatch):
    model_name = "google/multiberts-seed_0"
    source_model = load_pretrained_serial_model_or_skip(model_name)
    patch_working_serialize_bias(source_model)
    serialized = source_model.serialize()
    modified_vectors = serialized.vectors.clone()
    layer_index = 0
    head_index = 0
    head_dim = source_model.config.hidden_size // source_model.config.num_attention_heads
    target_prefixes = [
        f"encoder.layer.{layer_index}.attention.self.query.head.{head_index}.",
        f"encoder.layer.{layer_index}.attention.self.key.head.{head_index}.",
        f"encoder.layer.{layer_index}.attention.self.value.head.{head_index}.",
        f"encoder.layer.{layer_index}.attention.output.dense.weight.head.{head_index}.",
    ]
    no_bias_prefix = f"encoder.layer.{layer_index}.attention.output.dense.weight.head.{head_index}."
    target_rows = [
        index
        for index, name in enumerate(serialized.names)
        if any(name.startswith(prefix) for prefix in target_prefixes)
    ]
    assert len(target_rows) == 4 * head_dim
    for row_index in target_rows:
        modified_vectors[row_index, :-1] = 0.0
        if not serialized.names[row_index].startswith(no_bias_prefix):
            modified_vectors[row_index, -1] = 0.0
    modified_serialized = NamedSerialParameters.from_vector_list(serialized.names, [modified_vectors])
    shell_model = load_pretrained_serial_model_or_skip(model_name)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == model_name
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    baseline_context = run_and_capture_self_attention_context(
        source_model,
        layer_index,
        lambda: source_model(input_ids=input_ids, attention_mask=attention_mask),
    )
    loaded_model, overrides = SerialAutoModelForMaskedLM.load_serialized(modified_serialized, model_name)
    zeroed_context = run_and_capture_self_attention_context(
        loaded_model,
        layer_index,
        lambda: functional_call(
            loaded_model,
            overrides,
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        ),
    )

    baseline_context = baseline_context.view(
        baseline_context.shape[0],
        baseline_context.shape[1],
        source_model.config.num_attention_heads,
        head_dim,
    )
    zeroed_context = zeroed_context.view(
        zeroed_context.shape[0],
        zeroed_context.shape[1],
        source_model.config.num_attention_heads,
        head_dim,
    )

    assert torch.count_nonzero(baseline_context[:, :, head_index]).item() > 0
    torch.testing.assert_close(
        zeroed_context[:, :, head_index],
        torch.zeros_like(zeroed_context[:, :, head_index]),
    )
    torch.testing.assert_close(
        zeroed_context[:, :, 1 - head_index],
        baseline_context[:, :, 1 - head_index],
    )

