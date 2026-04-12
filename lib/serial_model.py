from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForMaskedLM

from lib.serial_params import Symmeters


def has_tied_input_output_embeddings(model: Any) -> bool:
    return (
        model.base_model.embeddings.word_embeddings.weight.data_ptr()
        == model.cls.predictions.decoder.weight.data_ptr()
    )


def untie_input_output_embeddings(model: Any):
    model.cls.predictions.decoder.weight = torch.nn.Parameter(
        model.cls.predictions.decoder.weight.detach().clone()
    )
    return model


def _stack_head_rows(matrix: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if matrix.shape[0] != num_heads * head_dim:
        raise ValueError("Matrix row count must split evenly across attention heads.")
    return matrix.reshape(num_heads, head_dim, matrix.shape[1])


def _stack_head_bias(bias: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if bias.shape[0] != num_heads * head_dim:
        raise ValueError("Bias length must split evenly across attention heads.")
    return bias.reshape(num_heads, head_dim)


def _serialize_layernorm(layernorm: torch.nn.LayerNorm) -> torch.Tensor:
    return torch.stack([layernorm.weight, layernorm.bias])


def serialize_embeddings(model: Any, name: str | None = None) -> Symmeters:
    prefix = name or f"{model.base_model_prefix}.embeddings"
    params = model.base_model.embeddings
    symmeters = Symmeters(["model"])
    symmeters.add_component(
        "model",
        f"{prefix}.word_embeddings.weight",
        params.word_embeddings.weight,
        axes=("vocab_items", "model"),
        kind="weight",
        layout="identity",
        parameter_keys=f"{prefix}.word_embeddings.weight",
    )
    symmeters.add_component(
        "model",
        f"{prefix}.position_embeddings.weight",
        params.position_embeddings.weight,
        axes=("position_items", "model"),
        kind="weight",
        layout="identity",
        parameter_keys=f"{prefix}.position_embeddings.weight",
    )
    symmeters.add_component(
        "model",
        f"{prefix}.token_type_embeddings.weight",
        params.token_type_embeddings.weight,
        axes=("token_type_items", "model"),
        kind="weight",
        layout="identity",
        parameter_keys=f"{prefix}.token_type_embeddings.weight",
    )
    if params.LayerNorm is not None:
        symmeters.add_component(
            "model",
            f"{prefix}.LayerNorm",
            _serialize_layernorm(params.LayerNorm),
            axes=("layernorm_param", "model"),
            kind="layernorm",
            layout="layernorm",
            parameter_keys=(f"{prefix}.LayerNorm.weight", f"{prefix}.LayerNorm.bias"),
        )
    return symmeters


def serialize_attention(
    model: Any,
    attention: Any,
    layer_idx: int,
    name: str | None = None,
) -> Symmeters:
    prefix = name or f"{model.base_model_prefix}.encoder.layer.{layer_idx}.attention"
    num_heads = attention.self.num_attention_heads
    head_dim = attention.self.attention_head_size
    head_symmetry = f"L{layer_idx}.head"
    qk_symmetry = f"L{layer_idx}.qk"
    ov_symmetry = f"L{layer_idx}.ov"

    symmeters = Symmeters(["model", qk_symmetry, ov_symmetry, head_symmetry])

    for projection_name in ("query", "key"):
        projection = getattr(attention.self, projection_name)
        component_name = f"{prefix}.self.{projection_name}.weight"
        symmeters.add_component(
            qk_symmetry,
            component_name,
            _stack_head_rows(projection.weight, num_heads, head_dim),
            axes=(head_symmetry, qk_symmetry, "model"),
            kind="weight",
            layout="head_rows",
            parameter_keys=component_name,
        )
        component_name = f"{prefix}.self.{projection_name}.bias"
        symmeters.add_component(
            qk_symmetry,
            component_name,
            _stack_head_bias(projection.bias, num_heads, head_dim),
            axes=(head_symmetry, qk_symmetry),
            kind="bias",
            layout="head_bias",
            parameter_keys=component_name,
        )

    value = attention.self.value
    component_name = f"{prefix}.self.value.weight"
    symmeters.add_component(
        ov_symmetry,
        component_name,
        _stack_head_rows(value.weight, num_heads, head_dim),
        axes=(head_symmetry, ov_symmetry, "model"),
        kind="weight",
        layout="head_rows",
        parameter_keys=component_name,
    )
    component_name = f"{prefix}.self.value.bias"
    symmeters.add_component(
        ov_symmetry,
        component_name,
        _stack_head_bias(value.bias, num_heads, head_dim),
        axes=(head_symmetry, ov_symmetry),
        kind="bias",
        layout="head_bias",
        parameter_keys=component_name,
    )

    component_name = f"{prefix}.output.dense.weight"
    symmeters.add_component(
        ov_symmetry,
        component_name,
        _stack_head_rows(attention.output.dense.weight.T, num_heads, head_dim),
        axes=(head_symmetry, ov_symmetry, "model"),
        kind="weight",
        layout="head_rows_transposed",
        parameter_keys=component_name,
    )
    component_name = f"{prefix}.output.dense.bias"
    symmeters.add_component(
        "model",
        component_name,
        attention.output.dense.bias,
        axes=("model",),
        kind="bias",
        layout="identity",
        parameter_keys=component_name,
    )
    if attention.output.LayerNorm is not None:
        symmeters.add_component(
            "model",
            f"{prefix}.output.LayerNorm",
            _serialize_layernorm(attention.output.LayerNorm),
            axes=("layernorm_param", "model"),
            kind="layernorm",
            layout="layernorm",
            parameter_keys=(f"{prefix}.output.LayerNorm.weight", f"{prefix}.output.LayerNorm.bias"),
        )

    return symmeters


def serialize_encoder_layer(
    model: Any,
    layer: Any,
    layer_idx: int,
    name: str | None = None,
) -> Symmeters:
    prefix = name or f"{model.base_model_prefix}.encoder.layer.{layer_idx}"
    symmeters = serialize_attention(model, layer.attention, layer_idx=layer_idx, name=f"{prefix}.attention")
    mlp_symmetry = f"L{layer_idx}.mlp"
    symmeters.add_symmetry(mlp_symmetry)

    component_name = f"{prefix}.intermediate.dense.weight"
    symmeters.add_component(
        mlp_symmetry,
        component_name,
        layer.intermediate.dense.weight,
        axes=(mlp_symmetry, "model"),
        kind="weight",
        layout="identity",
        parameter_keys=component_name,
    )
    component_name = f"{prefix}.intermediate.dense.bias"
    symmeters.add_component(
        mlp_symmetry,
        component_name,
        layer.intermediate.dense.bias,
        axes=(mlp_symmetry,),
        kind="bias",
        layout="identity",
        parameter_keys=component_name,
    )
    component_name = f"{prefix}.output.dense.weight"
    symmeters.add_component(
        mlp_symmetry,
        component_name,
        layer.output.dense.weight.T,
        axes=(mlp_symmetry, "model"),
        kind="weight",
        layout="transpose",
        parameter_keys=component_name,
    )
    component_name = f"{prefix}.output.dense.bias"
    symmeters.add_component(
        "model",
        component_name,
        layer.output.dense.bias,
        axes=("model",),
        kind="bias",
        layout="identity",
        parameter_keys=component_name,
    )
    if layer.output.LayerNorm is not None:
        symmeters.add_component(
            "model",
            f"{prefix}.output.LayerNorm",
            _serialize_layernorm(layer.output.LayerNorm),
            axes=("layernorm_param", "model"),
            kind="layernorm",
            layout="layernorm",
            parameter_keys=(f"{prefix}.output.LayerNorm.weight", f"{prefix}.output.LayerNorm.bias"),
        )
    return symmeters


def serialize_encoder(model: Any, name: str | None = None) -> Symmeters:
    prefix = name or f"{model.base_model_prefix}.encoder"
    symmeters = Symmeters([])
    for layer_idx, layer in enumerate(model.base_model.encoder.layer):
        symmeters += serialize_encoder_layer(model, layer, layer_idx=layer_idx, name=f"{prefix}.layer.{layer_idx}")
    return symmeters


def serialize_mlm_head(model: Any, name: str = "cls.predictions") -> Symmeters:
    params = model.cls.predictions
    tied = has_tied_input_output_embeddings(model)
    owner = "model" if tied else "decoder"
    transform_axes = ("model", "model") if tied else ("decoder", "model")

    symmeters = Symmeters(["model", "vocab"])
    if not tied:
        symmeters.add_symmetry("decoder")

    component_name = f"{name}.transform.dense.weight"
    symmeters.add_component(
        owner,
        component_name,
        params.transform.dense.weight,
        axes=transform_axes,
        kind="weight",
        layout="identity",
        parameter_keys=component_name,
    )
    component_name = f"{name}.transform.dense.bias"
    symmeters.add_component(
        owner,
        component_name,
        params.transform.dense.bias,
        axes=(owner,),
        kind="bias",
        layout="identity",
        parameter_keys=component_name,
    )
    if params.transform.LayerNorm is not None:
        symmeters.add_component(
            owner,
            f"{name}.transform.LayerNorm",
            _serialize_layernorm(params.transform.LayerNorm),
            axes=("layernorm_param", owner),
            kind="layernorm",
            layout="layernorm",
            parameter_keys=(f"{name}.transform.LayerNorm.weight", f"{name}.transform.LayerNorm.bias"),
        )
    if not tied:
        component_name = f"{name}.decoder.weight"
        symmeters.add_component(
            "decoder",
            component_name,
            params.decoder.weight,
            axes=("vocab_items", "decoder"),
            kind="weight",
            layout="identity",
            parameter_keys=component_name,
        )
    component_name = f"{name}.decoder.bias"
    symmeters.add_component(
        "vocab",
        component_name,
        params.decoder.bias,
        axes=("vocab",),
        kind="bias",
        layout="identity",
        parameter_keys=component_name,
    )
    return symmeters


def serialize_model(model: Any) -> Symmeters:
    symmeters = Symmeters([])
    symmeters += serialize_embeddings(model)
    symmeters += serialize_encoder(model)
    symmeters += serialize_mlm_head(model)
    return symmeters


def _materialize_component_override(component) -> dict[str, torch.Tensor]:
    tensor = component.tensor
    parameter_keys = component.parameter_keys
    layout = component.layout

    if layout == "identity":
        return {parameter_keys[0]: tensor}
    if layout == "transpose":
        return {parameter_keys[0]: tensor.T}
    if layout == "head_rows":
        return {parameter_keys[0]: tensor.reshape(-1, tensor.shape[-1])}
    if layout == "head_rows_transposed":
        return {parameter_keys[0]: tensor.reshape(-1, tensor.shape[-1]).T}
    if layout == "head_bias":
        return {parameter_keys[0]: tensor.reshape(-1)}
    if layout == "layernorm":
        return {
            parameter_keys[0]: tensor[0],
            parameter_keys[1]: tensor[1],
        }
    raise ValueError(f"Unsupported parameter component layout {layout}.")


def _build_overrides(serialized_symmeters: Symmeters) -> dict[str, torch.Tensor]:
    overrides: dict[str, torch.Tensor] = {}
    for _, _, component in serialized_symmeters.iter_components():
        for parameter_key, tensor in _materialize_component_override(component).items():
            if parameter_key in overrides:
                raise ValueError(f"Duplicate parameter override for {parameter_key}.")
            overrides[parameter_key] = tensor
    return overrides


def _has_parameter_key(serialized_symmeters: Symmeters, parameter_key: str) -> bool:
    return any(parameter_key in component.parameter_keys for _, _, component in serialized_symmeters.iter_components())


def _configure_optional_modules(model: Any, serialized_symmeters: Symmeters):
    embeddings_prefix = f"{model.base_model_prefix}.embeddings.LayerNorm"
    if not serialized_symmeters.has_component(embeddings_prefix, symmetry_name="model"):
        model.base_model.embeddings.LayerNorm = None

    for layer_idx, layer in enumerate(model.base_model.encoder.layer):
        attention_ln = f"{model.base_model_prefix}.encoder.layer.{layer_idx}.attention.output.LayerNorm"
        if not serialized_symmeters.has_component(attention_ln, symmetry_name="model"):
            layer.attention.output.LayerNorm = None

        output_ln = f"{model.base_model_prefix}.encoder.layer.{layer_idx}.output.LayerNorm"
        if not serialized_symmeters.has_component(output_ln, symmetry_name="model"):
            layer.output.LayerNorm = None

    tied = not _has_parameter_key(serialized_symmeters, "cls.predictions.decoder.weight")
    mlm_owner = "model" if tied else "decoder"
    if not serialized_symmeters.has_component("cls.predictions.transform.LayerNorm", symmetry_name=mlm_owner):
        model.cls.predictions.transform.LayerNorm = None


def load_serialized(
    serialized_symmeters: Symmeters,
    pretrained_model_name_or_path: str,
    *model_args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    if not isinstance(serialized_symmeters, Symmeters):
        raise TypeError("load_serialized expects Symmeters.")

    model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    should_tie_embeddings = not _has_parameter_key(serialized_symmeters, "cls.predictions.decoder.weight")
    if should_tie_embeddings:
        model.cls.predictions.decoder.weight = model.base_model.embeddings.word_embeddings.weight
    else:
        untie_input_output_embeddings(model)

    _configure_optional_modules(model, serialized_symmeters)
    overrides = _build_overrides(serialized_symmeters)
    return model, overrides
