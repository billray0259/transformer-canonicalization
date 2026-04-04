from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForMaskedLM

from lib.serial_params import NamedSerialParameters, Symmeters
from lib.serial_reader import SerializedParameterOverrides


class SerialAutoModelForMaskedLM(AutoModelForMaskedLM):
    """Adds serialization and differentiable deserialization helpers to MLM models."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "SerialAutoModelForMaskedLM":
        """Load a pretrained MLM and bind the serialization helpers to the concrete instance."""
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        for method_name in [
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
            "load_serialized",
            "has_tied_input_output_embeddings",
            "untie_input_output_embeddings",
        ]:
            setattr(model, method_name, MethodType(getattr(cls, method_name), model))
        return model

    def serialize_matrix(
        self,
        matrix: torch.Tensor,
        name: str = "matrix",
        names: list[str] | None = None,
    ) -> NamedSerialParameters:
        """Serialize a matrix as row vectors."""
        assert matrix.shape[1] == self.config.hidden_size, "Matrix must have the same number of columns as the model's hidden size."
        if names is None:
            names = [f"{name}.{i}" for i in range(matrix.shape[0])]
        return NamedSerialParameters.from_vector_list(names, [matrix])

    def serialize_bias(self, bias: torch.Tensor, name: str | None = None) -> NamedSerialParameters:
        """Serialize a bias vector as a single row."""
        return NamedSerialParameters.from_vector_list(
            [f"{name}.bias" if name is not None else "bias"],
            [bias.unsqueeze(0)],
        )

    def serialize_head_biases(
        self,
        bias: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
        name: str,
    ) -> NamedSerialParameters:
        """Serialize a bias vector as one row per attention head."""
        assert bias.shape[0] == num_heads * head_dim, "Bias must split evenly across attention heads."
        return NamedSerialParameters.from_vector_list(
            [f"{name}.head.{head_idx}.bias" for head_idx in range(num_heads)],
            [bias.reshape(num_heads, head_dim)],
        )

    def serialize_layernorm(self, layernorm: torch.nn.LayerNorm, name: str = "LayerNorm") -> NamedSerialParameters:
        """Serialize a LayerNorm module as separate weight and bias rows."""
        return NamedSerialParameters.from_vector_list(
            [f"{name}.weight", f"{name}.bias"],
            [layernorm.weight.unsqueeze(0), layernorm.bias.unsqueeze(0)],
        )

    def serialize_embeddings(self, name: str | None = None) -> NamedSerialParameters:
        """Serialize the embedding tables and optional embedding LayerNorm."""
        if name is None:
            name = f"{self.base_model_prefix}.embeddings"
        params = self.base_model.embeddings
        serialized = NamedSerialParameters()
        serialized += self.serialize_matrix(params.word_embeddings.weight, name=f"{name}.word_embeddings.weight")
        serialized += self.serialize_matrix(params.position_embeddings.weight, name=f"{name}.position_embeddings.weight")
        serialized += self.serialize_matrix(params.token_type_embeddings.weight, name=f"{name}.token_type_embeddings.weight")
        if params.LayerNorm is not None:
            serialized += self.serialize_layernorm(params.LayerNorm, name=f"{name}.LayerNorm")
        return serialized

    def serialize_attention(
        self,
        attention: Any,
        layer_idx: int,
        name: str | None = None,
    ) -> Symmeters:
        """Serialize one attention block using the head-indexed row naming scheme."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder.layer.{layer_idx}.attention"
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size
        qk_symmetry_names = [f"L{layer_idx}.H{head_idx}.qk" for head_idx in range(num_heads)]
        ov_symmetry_names = [f"L{layer_idx}.H{head_idx}.ov" for head_idx in range(num_heads)]
        symmeters = Symmeters(
            [
                "model",
                *qk_symmetry_names,
                *ov_symmetry_names,
            ]
        )

        def head_names(prefix: str) -> list[str]:
            return [
                f"{prefix}.head.{head_idx}.{row_idx}"
                for head_idx in range(num_heads)
                for row_idx in range(head_dim)
            ]

        for head_idx in range(num_heads):
            symmeters.set_equivalence_class(
                qk_symmetry_names[head_idx],
                [
                    f"{name}.self.query.weight.head.{head_idx}",
                    f"{name}.self.key.weight.head.{head_idx}",
                ],
            )
            symmeters.set_equivalence_class(
                ov_symmetry_names[head_idx],
                [
                    f"{name}.self.value.weight.head.{head_idx}",
                    f"{name}.output.dense.weight.head.{head_idx}",
                ],
            )

        for matrix_name in ["query", "key", "value"]:
            qkv = getattr(attention.self, matrix_name)
            symmetry_names = qk_symmetry_names if matrix_name in {"query", "key"} else ov_symmetry_names
            symmeters["model"] += self.serialize_matrix(
                qkv.weight,
                names=head_names(f"{name}.self.{matrix_name}.weight"),
            )
            head_biases = self.serialize_head_biases(
                qkv.bias,
                num_heads=num_heads,
                head_dim=head_dim,
                name=f"{name}.self.{matrix_name}",
            )
            for head_idx, bias_name in enumerate(head_biases.names):
                symmeters[symmetry_names[head_idx]] += NamedSerialParameters.from_vector_list(
                    [bias_name],
                    [head_biases.vectors[head_idx : head_idx + 1]],
                )

        symmeters["model"] += self.serialize_matrix(
            attention.output.dense.weight.T,
            names=head_names(f"{name}.output.dense.weight"),
        )
        symmeters["model"] += self.serialize_bias(attention.output.dense.bias, name=f"{name}.output.dense")
        if attention.output.LayerNorm is not None:
            symmeters["model"] += self.serialize_layernorm(
                attention.output.LayerNorm,
                name=f"{name}.output.LayerNorm",
            )
        return symmeters

    def serialize_encoder_layer(
        self,
        layer: Any,
        layer_idx: int,
        name: str | None = None,
    ) -> Symmeters:
        """Serialize a single encoder layer including attention, MLP, and LayerNorm blocks."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder.layer.{layer_idx}"
        symmeters = self.serialize_attention(layer.attention, layer_idx=layer_idx, name=f"{name}.attention")
        symmeters[f"L{layer_idx}.mlp"] = NamedSerialParameters()
        symmeters.set_equivalence_class(
            f"L{layer_idx}.mlp",
            [f"{name}.intermediate.dense", f"{name}.output.dense.weight"],
        )
        symmeters["model"] += self.serialize_matrix(
            layer.intermediate.dense.weight,
            name=f"{name}.intermediate.dense",
        )
        symmeters[f"L{layer_idx}.mlp"] += self.serialize_bias(
            layer.intermediate.dense.bias,
            name=f"{name}.intermediate.dense",
        )
        symmeters["model"] += self.serialize_matrix(
            layer.output.dense.weight.T,
            name=f"{name}.output.dense.weight",
        )
        symmeters["model"] += self.serialize_bias(layer.output.dense.bias, name=f"{name}.output.dense")
        if layer.output.LayerNorm is not None:
            symmeters["model"] += self.serialize_layernorm(layer.output.LayerNorm, name=f"{name}.output.LayerNorm")
        return symmeters

    def serialize_encoder(self, name: str | None = None) -> Symmeters:
        """Serialize every encoder layer in order."""
        if name is None:
            name = f"{self.base_model_prefix}.encoder"
        symmeters = Symmeters()
        for layer_idx, layer in enumerate(self.base_model.encoder.layer):
            symmeters += self.serialize_encoder_layer(layer, layer_idx=layer_idx, name=f"{name}.layer.{layer_idx}")
        return symmeters

    def serialize_mlm_head(self, name: str = "cls.predictions") -> Symmeters:
        """Serialize the masked-language-model head."""
        params = self.cls.predictions
        symmeters = Symmeters(["model", "decoder", "vocab"])
        symmeters.set_equivalence_class("decoder", [f"{name}.transform.dense.weight"])
        symmeters["decoder"] += self.serialize_matrix(
            params.transform.dense.weight,
            name=f"{name}.transform.dense.weight",
        )
        symmeters["decoder"] += self.serialize_bias(params.transform.dense.bias, name=f"{name}.transform.dense")
        if params.transform.LayerNorm is not None:
            symmeters["decoder"] += self.serialize_layernorm(params.transform.LayerNorm, name=f"{name}.transform.LayerNorm")
        symmeters["decoder"] += self.serialize_matrix(params.decoder.weight, name=f"{name}.decoder.weight")
        symmeters["vocab"] += self.serialize_bias(params.decoder.bias, name=f"{name}.decoder")
        return symmeters

    def serialize(self) -> Symmeters:
        """Serialize embeddings, encoder layers, and MLM head into one flat row collection."""
        symmeters = Symmeters([])
        symmeters += self.serialize_embeddings()
        symmeters += self.serialize_encoder()
        symmeters += self.serialize_mlm_head()
        if self.cls.predictions.decoder.weight.data_ptr() == self.base_model.embeddings.word_embeddings.weight.data_ptr():
            if "decoder" in symmeters.symmetry_names:
                symmeters["model"] += symmeters["decoder"].filter(lambda name: "decoder.weight" not in name)
                symmeters.set_equivalence_class(
                    "model",
                    [*symmeters.get_equivalence_class("model"), "cls.predictions.transform.dense.weight"],
                )
                del symmeters["decoder"]
        return symmeters

    @classmethod
    def _deserialize_embeddings(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = f"{model.base_model_prefix}.embeddings"
        params = model.base_model.embeddings
        overrides.matrix(f"{name}.word_embeddings.weight", params.word_embeddings.num_embeddings)
        overrides.matrix(f"{name}.position_embeddings.weight", params.position_embeddings.num_embeddings)
        overrides.matrix(f"{name}.token_type_embeddings.weight", params.token_type_embeddings.num_embeddings)
        if overrides.optional_layernorm(f"{name}.LayerNorm") is None:
            params.LayerNorm = None

    @classmethod
    def _deserialize_attention(
        cls,
        attention: Any,
        overrides: SerializedParameterOverrides,
        layer_idx: int,
        name: str,
    ) -> None:
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size
        qk_symmetry_names = [f"L{layer_idx}.H{head_idx}.qk" for head_idx in range(num_heads)]
        ov_symmetry_names = [f"L{layer_idx}.H{head_idx}.ov" for head_idx in range(num_heads)]
        for matrix_name, suffix in [("query", "qk"), ("key", "qk"), ("value", "ov")]:
            overrides.head_matrix(
                f"{name}.self.{matrix_name}.weight",
                num_heads=num_heads,
                head_dim=head_dim,
            )
            overrides.head_bias(
                f"{name}.self.{matrix_name}.bias",
                symmetry_names=qk_symmetry_names if suffix == "qk" else ov_symmetry_names,
            )
        overrides.head_matrix(
            f"{name}.output.dense.weight",
            num_heads=num_heads,
            head_dim=head_dim,
            transpose=True,
        )
        overrides.bias(f"{name}.output.dense.bias")
        if overrides.optional_layernorm(f"{name}.output.LayerNorm") is None:
            attention.output.LayerNorm = None

    @classmethod
    def _deserialize_encoder_layer(
        cls,
        layer: Any,
        overrides: SerializedParameterOverrides,
        layer_idx: int,
        name: str,
    ) -> None:
        cls._deserialize_attention(layer.attention, overrides, layer_idx, name=f"{name}.attention")
        overrides.matrix(
            f"{name}.intermediate.dense.weight",
            layer.intermediate.dense.out_features,
            src=f"{name}.intermediate.dense",
        )
        overrides.bias(
            f"{name}.intermediate.dense.bias",
            symmetry=f"L{layer_idx}.mlp",
            src=f"{name}.intermediate.dense",
        )
        overrides.matrix(
            f"{name}.output.dense.weight",
            layer.output.dense.in_features,
            transpose=True,
        )
        overrides.bias(f"{name}.output.dense.bias")
        if overrides.optional_layernorm(f"{name}.output.LayerNorm") is None:
            layer.output.LayerNorm = None

    @classmethod
    def _deserialize_encoder(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = f"{model.base_model_prefix}.encoder"
        for layer_idx, layer in enumerate(model.base_model.encoder.layer):
            cls._deserialize_encoder_layer(layer, overrides, layer_idx, name=f"{name}.layer.{layer_idx}")

    @classmethod
    def _deserialize_mlm_head(
        cls,
        model: Any,
        overrides: SerializedParameterOverrides,
        tie_embeddings: bool,
        name: str = "cls.predictions",
    ) -> None:
        params = model.cls.predictions
        aux_symmetry = "model" if tie_embeddings else "decoder"
        overrides.matrix(f"{name}.transform.dense.weight", params.transform.dense.out_features, symmetry=aux_symmetry)
        overrides.bias(f"{name}.transform.dense.bias", symmetry=aux_symmetry)
        if overrides.optional_layernorm(f"{name}.transform.LayerNorm", symmetry=aux_symmetry) is None:
            params.transform.LayerNorm = None
        if not tie_embeddings:
            overrides.matrix(f"{name}.decoder.weight", params.decoder.out_features, symmetry="decoder")
        overrides.bias(f"{name}.decoder.bias", symmetry="vocab")

    @classmethod
    def load_serialized(
        cls,
        serialized_symmeters: Symmeters,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Build a shell model and differentiable overrides from serialized parameters."""
        if isinstance(serialized_symmeters, NamedSerialParameters):
            raise TypeError("load_serialized expects Symmeters.")
        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        overrides = SerializedParameterOverrides(serialized_symmeters)
        should_tie_embeddings = not overrides.has_prefix("cls.predictions.decoder.weight")
        if not should_tie_embeddings:
            model.cls.predictions.decoder.weight = torch.nn.Parameter(model.cls.predictions.decoder.weight.detach().clone())
        cls._deserialize_embeddings(model, overrides)
        cls._deserialize_encoder(model, overrides)
        cls._deserialize_mlm_head(model, overrides, should_tie_embeddings)
        if should_tie_embeddings:
            model.cls.predictions.decoder.weight = model.base_model.embeddings.word_embeddings.weight
        overrides.assert_done()
        return model, overrides
    
    def has_tied_input_output_embeddings(self):
        return (
            self.base_model.embeddings.word_embeddings.weight.data_ptr()
            == self.cls.predictions.decoder.weight.data_ptr()
        )


    def untie_input_output_embeddings(self):
        self.cls.predictions.decoder.weight = torch.nn.Parameter(
            self.cls.predictions.decoder.weight.detach().clone()
        )
        return self
