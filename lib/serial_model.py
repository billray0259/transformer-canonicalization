from transformers import AutoModelForMaskedLM
import torch
from types import MethodType

from lib.serial_params import NamedSerialParameters

class SerialAutoModelForMaskedLM(AutoModelForMaskedLM):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the model using the parent class's from_pretrained method
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # AutoModelForMaskedLM resolves to a concrete HF model class, so attach the
        # serialization helpers to the returned instance.
        model.serialize_matrix = MethodType(cls.serialize_matrix, model)
        model.serialize_bias = MethodType(cls.serialize_bias, model)
        model.serialize_layernorm = MethodType(cls.serialize_layernorm, model)
        model.serialize_embeddings = MethodType(cls.serialize_embeddings, model)
        model.serialize_attention = MethodType(cls.serialize_attention, model)
        model.serialize_encoder_layer = MethodType(cls.serialize_encoder_layer, model)
        model.serialize_encoder = MethodType(cls.serialize_encoder, model)
        model.serialize_mlm_head = MethodType(cls.serialize_mlm_head, model)
        model.serialize = MethodType(cls.serialize, model)

        return model
    
    def serialize_matrix(self, matrix, name="matrix", bias=None, names=None):        
        assert matrix.shape[1] == self.config.hidden_size, "Matrix must have the same number of columns as the model's hidden size."

        if names is None:
            names = [f"{name}.{i}" for i in range(matrix.shape[0])]
        else:
            assert len(names) == matrix.shape[0], "Names must match the number of matrix rows."
        vector_list = None
        if bias is None:
            # Pad with NaN sentinels to maintain consistent shape for concatenation, but don't include a bias name since there is no bias
            vector_list = [torch.cat([matrix, torch.full((matrix.shape[0], 1), float('nan'), device=matrix.device, dtype=matrix.dtype)], dim=1)]
        else:
            # Concatenate the bias to the matrix
            vector_list = [torch.cat([matrix, bias.unsqueeze(1)], dim=1)]            
        
        return NamedSerialParameters.from_vector_list(names, vector_list)
    
    
    def serialize_bias(self, bias, name=None):
        assert bias.shape[0] == self.config.hidden_size, "Bias must have the same size as the model's hidden size."
        names = [f"{name}.bias" if name is not None else "bias"]
        bias = bias.unsqueeze(0)
        vector_list = [torch.cat([bias, torch.full((1, 1), float('nan'), device=bias.device, dtype=bias.dtype)], dim=1)]
        return NamedSerialParameters.from_vector_list(names, vector_list)
        

    def serialize_layernorm(self, layernorm, name="LayerNorm"):
        weight_row = layernorm.weight.unsqueeze(0)
        bias_row = layernorm.bias.unsqueeze(0)
        weight_row = torch.cat([weight_row, torch.full((1, 1), float('nan'), device=weight_row.device, dtype=weight_row.dtype)], dim=1)
        bias_row = torch.cat([bias_row, torch.full((1, 1), float('nan'), device=bias_row.device, dtype=bias_row.dtype)], dim=1)
        return NamedSerialParameters.from_vector_list([f"{name}.weight", f"{name}.bias"], [weight_row, bias_row])
                
        
    def serialize_embeddings(self, name="embeddings"):
        params = self.base_model.embeddings
        
        serialized_params = NamedSerialParameters()
        
        if params.word_embeddings.weight.data_ptr() != self.cls.predictions.decoder.weight.data_ptr():
            serialized_params += self.serialize_matrix(params.word_embeddings.weight, name=f"{name}.word_embeddings.weight")
        
        serialized_params += self.serialize_matrix(params.position_embeddings.weight, name=f"{name}.position_embeddings.weight")
        serialized_params += self.serialize_matrix(params.token_type_embeddings.weight, name=f"{name}.token_type_embeddings.weight")
        
        if params.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(params.LayerNorm, name=f"{name}.LayerNorm")
        
        return serialized_params
    
    def serialize_attention(self, attention, name="attention"):
        serialized_params = NamedSerialParameters()
        num_heads = attention.self.num_attention_heads
        head_dim = attention.self.attention_head_size

        def head_names(prefix):
            return [
                f"{prefix}.head.{head_idx}.{row_idx}"
                for head_idx in range(num_heads)
                for row_idx in range(head_dim)
            ]
        
        # Query, Key, Value matrices and biases
        for matrix_name in ["query", "key", "value"]:
            qkv = getattr(attention.self, matrix_name)
            serialized_params += self.serialize_matrix(
                matrix=qkv.weight,
                bias=qkv.bias,
                names=head_names(f"{name}.self.{matrix_name}"),
            )
        
        # Output dense layer and bias
        serialized_params += self.serialize_matrix(matrix=attention.output.dense.weight.T, names=head_names(f"{name}.output.dense.weight"))
        serialized_params += self.serialize_bias(bias=attention.output.dense.bias, name=f"{name}.output.dense")

        # Output LayerNorm if it exists
        if attention.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(attention.output.LayerNorm, name=f"{name}.output.LayerNorm")
        
        return serialized_params
    
    def serialize_encoder_layer(self, layer, name="encoder.layer"):
        serialized_params = NamedSerialParameters()
        
        # Attention parameters
        serialized_params += self.serialize_attention(layer.attention, name=f"{name}.attention")
        
        # Intermediate dense layer and bias
        serialized_params += self.serialize_matrix(
            matrix=layer.intermediate.dense.weight,
            name=f"{name}.intermediate.dense",
            bias=layer.intermediate.dense.bias,
        )
        
        # Output dense layer and bias
        serialized_params += self.serialize_matrix(matrix=layer.output.dense.weight.T, name=f"{name}.output.dense.weight")
        serialized_params += self.serialize_bias(bias=layer.output.dense.bias, name=f"{name}.output.dense")

        # Output LayerNorm if it exists
        if layer.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(layer.output.LayerNorm, name=f"{name}.output.LayerNorm")
        
        return serialized_params
    
    def serialize_encoder(self, name="encoder"):
        serialized_params = NamedSerialParameters()
        
        for i, layer in enumerate(self.base_model.encoder.layer):
            serialized_params += self.serialize_encoder_layer(layer, name=f"{name}.layer.{i}")
        
        return serialized_params
    
    def serialize_mlm_head(self, name="predictions"):
        params = self.cls.predictions
        serialized_params = NamedSerialParameters()
        
        # Transform dense layer and bias
        serialized_params += self.serialize_matrix(
            matrix=params.transform.dense.weight,
            name=f"{name}.transform.dense.weight",
            bias=params.transform.dense.bias
        )
        
        # LayerNorm if it exists
        if params.transform.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(params.transform.LayerNorm, name=f"{name}.transform.LayerNorm")
        
        # Output decoder layer
        serialized_params += self.serialize_matrix(
            matrix=params.decoder.weight,
            name=f"{name}.decoder.weight",
            bias=params.bias
        )
        
        return serialized_params
    
    def serialize(self):
        serialized_params = NamedSerialParameters()
        
        # Serialize embeddings
        serialized_params += self.serialize_embeddings()
        
        # Serialize encoder layers
        serialized_params += self.serialize_encoder()
        
        # Serialize MLM head
        serialized_params += self.serialize_mlm_head()
        
        return serialized_params