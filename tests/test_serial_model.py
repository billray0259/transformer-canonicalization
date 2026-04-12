import pytest
import torch
from torch.func import functional_call
from transformers import AutoModelForMaskedLM, BertForMaskedLM

from lib.serial_model import (
    has_tied_input_output_embeddings,
    load_serialized,
    serialize_attention,
    serialize_encoder_layer,
    serialize_model,
    untie_input_output_embeddings,
)


def build_shell_model(tiny_config, seed=1234):
    torch.manual_seed(seed)
    model = BertForMaskedLM(tiny_config)
    model.eval()
    return model


def rand_perm_matrix(size, device):
    return torch.eye(size, device=device)[torch.randperm(size, device=device)]


def bank_axis_name(symmetry_name):
    if not symmetry_name.startswith("L"):
        return None
    if symmetry_name.endswith(".qk") or symmetry_name.endswith(".ov"):
        layer_name = symmetry_name.rsplit(".", 1)[0]
        return f"{layer_name}.head"
    return None


def count_distinct_permutations(permutations):
    return sum(matrix.shape[0] if matrix.ndim == 3 else 1 for matrix in permutations.values())


def full_rand_perms(symmeters):
    permutations = {}
    for symmetry_name in symmeters.ordered_transform_names():
        if symmetry_name.endswith(".head"):
            continue

        component_specs = symmeters.components_with_axis(symmetry_name)
        if not component_specs:
            continue
        device = component_specs[0][2].tensor.device

        bank_axis = bank_axis_name(symmetry_name)
        if bank_axis is not None:
            bank_size = symmeters.symmetry_size(bank_axis)
            permutations[symmetry_name] = torch.stack(
                [rand_perm_matrix(symmeters.symmetry_size(symmetry_name), device) for _ in range(bank_size)]
            )
            continue

        permutations[symmetry_name] = rand_perm_matrix(symmeters.symmetry_size(symmetry_name), device)

    head_symmetry_names = [symmetry_name for symmetry_name in symmeters.ordered_transform_names() if symmetry_name.endswith(".head")]
    if head_symmetry_names:
        head_size = symmeters.symmetry_size(head_symmetry_names[0])
        device = symmeters.components_with_axis(head_symmetry_names[0])[0][2].tensor.device
        permutations["head"] = rand_perm_matrix(head_size, device)
    return permutations


def assert_pretrained_model_permutations_preserve_outputs(
    model_name: str,
    *,
    untie_embeddings_first: bool,
    atol: float,
    rtol: float,
):
    device = torch.device("cuda")
    try:
        source_model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception as exc:
        pytest.skip(f"Hugging Face model assets unavailable locally: {exc}")

    source_model.eval()
    if untie_embeddings_first:
        untie_input_output_embeddings(source_model)

    source_symmeters = serialize_model(source_model)
    permuted_symmeters = source_symmeters.clone()

    torch.manual_seed(0)
    permutations = full_rand_perms(permuted_symmeters)

    explicit_symmetry_names = {
        *[
            symmetry_name
            for symmetry_name in permuted_symmeters.ordered_transform_names()
            if not symmetry_name.endswith(".head")
        ],
        *( ["head"] if any(symmetry_name.endswith(".head") for symmetry_name in permuted_symmeters.ordered_transform_names()) else [] ),
    }
    assert set(permutations) == explicit_symmetry_names

    expected_distinct_permutations = (
        1
        + int("decoder" in permuted_symmeters)
        + source_model.config.num_hidden_layers * source_model.config.num_attention_heads
        + source_model.config.num_hidden_layers * source_model.config.num_attention_heads
        + source_model.config.num_hidden_layers
        + int(any(symmetry_name.endswith(".head") for symmetry_name in permuted_symmeters.ordered_transform_names()))
    )
    assert count_distinct_permutations(permutations) == expected_distinct_permutations

    permuted_symmeters.apply_transforms(permutations)

    loaded_model, overrides = load_serialized(
        permuted_symmeters,
        model_name,
        local_files_only=True,
    )
    loaded_model = loaded_model.to(device).eval()

    input_ids = torch.tensor(
        [
            [101, 1996, 4937, 2938, 2006, 1996, 13523, 102],
            [101, 2023, 3231, 8667, 23651, 2594, 8043, 102],
        ],
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        source_logits = source_model(input_ids=input_ids, attention_mask=attention_mask).logits
        permuted_logits = functional_call(
            loaded_model,
            overrides,
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        ).logits

    torch.testing.assert_close(permuted_logits, source_logits, atol=atol, rtol=rtol)


def test_serialize_attention_uses_layer_level_head_stacked_components(tiny_model):
    attention = tiny_model.base_model.encoder.layer[0].attention
    symmeters = serialize_attention(tiny_model, attention, layer_idx=0, name="attn")

    assert symmeters.symmetry_names == ["model", "L0.qk", "L0.ov", "L0.head"]
    assert "L0.H0.qk" not in symmeters
    assert symmeters.component("L0.qk", "attn.self.query.weight").axes == ("L0.head", "L0.qk", "model")
    assert tuple(symmeters.tensor("L0.qk", "attn.self.query.weight").shape) == (2, 2, 4)
    assert tuple(symmeters.tensor("L0.qk", "attn.self.query.bias").shape) == (2, 2)
    assert tuple(symmeters.tensor("L0.ov", "attn.output.dense.weight").shape) == (2, 2, 4)


def test_serialize_encoder_layer_keeps_mlp_bias_as_owned_block(tiny_model):
    layer = tiny_model.base_model.encoder.layer[0]
    symmeters = serialize_encoder_layer(tiny_model, layer, layer_idx=0, name="encoder.layer.0")

    assert "encoder.layer.0.intermediate.dense.bias" in symmeters["L0.mlp"]
    assert symmeters.component("L0.mlp", "encoder.layer.0.intermediate.dense.bias").axes == ("L0.mlp",)
    assert "encoder.layer.0.output.dense.bias" in symmeters["model"]


def test_serialize_model_tracks_tied_and_untied_decoder_blocks(tiny_model):
    tied_symmeters = serialize_model(tiny_model)

    assert has_tied_input_output_embeddings(tiny_model)
    assert "decoder" not in tied_symmeters
    assert tied_symmeters.component("model", "cls.predictions.transform.dense.weight").axes == ("model", "model")
    assert not tied_symmeters.has_component("cls.predictions.decoder.weight")

    untie_input_output_embeddings(tiny_model)
    untied_symmeters = serialize_model(tiny_model)

    assert not has_tied_input_output_embeddings(tiny_model)
    assert "decoder" in untied_symmeters
    assert untied_symmeters.component("decoder", "cls.predictions.transform.dense.weight").axes == ("decoder", "model")
    assert untied_symmeters.component("decoder", "cls.predictions.decoder.weight").axes == ("vocab_items", "decoder")


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_load_serialized_matches_source_model_outputs(monkeypatch, tiny_model, tiny_config, tied_embeddings):
    source_model = tiny_model
    if not tied_embeddings:
        untie_input_output_embeddings(source_model)

    serialized_symmeters = serialize_model(source_model)
    shell_model = build_shell_model(tiny_config, seed=4321)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        assert model_args == ()
        assert kwargs == {}
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model, overrides = load_serialized(serialized_symmeters, "dummy-model")
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


def test_serialize_and_load_handle_missing_optional_layernorms(monkeypatch, tiny_model, tiny_config):
    tiny_model.base_model.embeddings.LayerNorm = None
    tiny_model.base_model.encoder.layer[0].attention.output.LayerNorm = None
    tiny_model.base_model.encoder.layer[1].output.LayerNorm = None
    tiny_model.cls.predictions.transform.LayerNorm = None

    serialized_symmeters = serialize_model(tiny_model)
    shell_model = build_shell_model(tiny_config)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model, overrides = load_serialized(serialized_symmeters, "dummy-model")

    assert loaded_model.base_model.embeddings.LayerNorm is None
    assert loaded_model.base_model.encoder.layer[0].attention.output.LayerNorm is None
    assert loaded_model.base_model.encoder.layer[1].output.LayerNorm is None
    assert loaded_model.cls.predictions.transform.LayerNorm is None
    assert "bert.embeddings.LayerNorm.weight" not in overrides
    assert "bert.encoder.layer.0.attention.output.LayerNorm.weight" not in overrides
    assert "bert.encoder.layer.1.output.LayerNorm.weight" not in overrides
    assert "cls.predictions.transform.LayerNorm.weight" not in overrides


def test_load_serialized_rejects_non_symmeters():
    with pytest.raises(TypeError, match="Symmeters"):
        load_serialized({}, "dummy-model")


def test_load_serialized_backprops_to_component_tensors(monkeypatch, tiny_model, tiny_config):
    untie_input_output_embeddings(tiny_model)
    differentiable_symmeters = serialize_model(tiny_model).clone()
    shell_model = build_shell_model(tiny_config, seed=5678)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    functional_model, overrides = load_serialized(
        differentiable_symmeters,
        "dummy-model",
    )
    loss = functional_call(functional_model, overrides, (), {"input_ids": input_ids}).logits.square().mean()
    loss.backward()

    grads = [
        component.tensor.grad
        for _, _, component in differentiable_symmeters.iter_components()
        if component.tensor.grad is not None
    ]
    assert grads
    assert sum(torch.count_nonzero(grad).item() for grad in grads) > 0


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_apply_transforms_preserve_outputs(monkeypatch, tiny_model, tiny_config, tied_embeddings):
    source_model = tiny_model
    if not tied_embeddings:
        untie_input_output_embeddings(source_model)

    source_model.eval()
    source_symmeters = serialize_model(source_model)
    permuted_symmeters = source_symmeters.clone()

    torch.manual_seed(0)
    permuted_symmeters.apply_transforms(full_rand_perms(permuted_symmeters))

    shell_model = build_shell_model(tiny_config, seed=999)

    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path == "dummy-model"
        return shell_model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", classmethod(fake_from_pretrained))

    loaded_model, overrides = load_serialized(permuted_symmeters, "dummy-model")
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    with torch.no_grad():
        source_logits = source_model(input_ids=input_ids, attention_mask=attention_mask).logits
        permuted_logits = functional_call(
            loaded_model,
            overrides,
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        ).logits

    torch.testing.assert_close(permuted_logits, source_logits, atol=1e-5, rtol=1e-5)


@pytest.mark.expensive
def test_full_bert_randomly_permuting_every_symmetry_preserves_outputs():
    assert_pretrained_model_permutations_preserve_outputs(
        "bert-base-uncased",
        untie_embeddings_first=True,
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.expensive
def test_full_bert_random_permutation_count_is_303():
    try:
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Hugging Face model assets unavailable locally: {exc}")

    untie_input_output_embeddings(model)
    permutations = full_rand_perms(serialize_model(model))

    assert count_distinct_permutations(permutations) == 303


