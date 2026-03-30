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
        model.serialize_layernorm = MethodType(cls.serialize_layernorm, model)
        model.serialize_embeddings = MethodType(cls.serialize_embeddings, model)
        model.serialize_attention = MethodType(cls.serialize_attention, model)
        model.serialize_encoder_layer = MethodType(cls.serialize_encoder_layer, model)
        model.serialize_encoder = MethodType(cls.serialize_encoder, model)
        model.serialize_mlm_head = MethodType(cls.serialize_mlm_head, model)
        model.serialize = MethodType(cls.serialize, model)

        return model
    
    def serialize_matrix(self, matrix, name="matrix", bias=None, bias_method="separate"):
        # matrix (n, d), bias (n or d), bias_method in ["concat", "separate"]
        # concat -> (n, d + 1) padding with zeros if bias=None; names = [name.{i} for i in range(matrix.shape[0])]
        # separate -> (n + 1, d); names = [name.{i} for i in range(matrix.shape[0])] + [name.bias]
        
        assert matrix.shape[1] == self.config.hidden_size, "Matrix must have the same number of columns as the model's hidden size."
        
        if bias is not None:
            assert (bias.shape[0] == matrix.shape[0] and bias_method == "concat") or (bias.shape[0] == matrix.shape[1] and bias_method == "separate"), f"Incompatible matrix and bias for method '{bias_method}'. Matrix shape: {matrix.shape}, Bias shape: {bias.shape}, Bias method: {bias_method}"
        
        names = [f"{name}.{i}" for i in range(matrix.shape[0])]
        vector_list = []
        if bias is None:
            if bias_method == "concat":
                # Pad the matrix with zeros for the bias
                vector_list = [torch.cat([matrix, torch.zeros(matrix.shape[0], 1)], dim=1)]
            else:
                vector_list = [matrix]
        elif bias_method == "concat":
            # Concatenate the bias to the matrix
            vector_list = [torch.cat([matrix, bias.unsqueeze(1)], dim=1)]
        elif bias_method == "separate":
            # Add the bias as a separate row to the matrix
            vector_list = [matrix, bias.unsqueeze(0)]
            names += [f"{name}.bias"]
        
        return NamedSerialParameters.from_vector_list(names, vector_list)
    

    def serialize_layernorm(self, layernorm, name="LayerNorm", bias_method: str = "separate"):
        weight_row = layernorm.weight.unsqueeze(0)
        bias_row = layernorm.bias.unsqueeze(0)
        if bias_method == "concat":
            weight_row = torch.cat([weight_row, torch.zeros(1, 1, device=weight_row.device, dtype=weight_row.dtype)], dim=1)
            bias_row = torch.cat([bias_row, torch.zeros(1, 1, device=bias_row.device, dtype=bias_row.dtype)], dim=1)
        return NamedSerialParameters.from_vector_list([f"{name}.weight", f"{name}.bias"], [weight_row, bias_row])
                
        
    def serialize_embeddings(self, name="embeddings", bias_method: str = "separate"):
        params = self.base_model.embeddings
        
        serialized_params = NamedSerialParameters()
        
        serialized_params += self.serialize_matrix(
            params.word_embeddings.weight,
            name=f"{name}.word_embeddings.weight",
            bias_method=bias_method
        )
        serialized_params += self.serialize_matrix(
            params.position_embeddings.weight,
            name=f"{name}.position_embeddings.weight",
            bias_method=bias_method
        )
        serialized_params += self.serialize_matrix(
            params.token_type_embeddings.weight,
            name=f"{name}.token_type_embeddings.weight",
            bias_method=bias_method
        )
        
        if params.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(
                params.LayerNorm,
                name=f"{name}.LayerNorm",
                bias_method=bias_method
            )
        
        return serialized_params
    
    def serialize_attention(self, attention, name="attention", bias_method: str = "separate"):
        serialized_params = NamedSerialParameters()
        
        # Query, Key, Value matrices and biases
        for matrix_name in ["query", "key", "value"]:
            qkv = getattr(attention.self, matrix_name)
            serialized_params += self.serialize_matrix(
                matrix=qkv.weight,
                name=f"{name}.self.{matrix_name}",
                bias=qkv.bias,
                bias_method=bias_method
            )
        
        # Output dense layer and bias
        serialized_params += self.serialize_matrix(
            matrix=attention.output.dense.weight.T,
            name=f"{name}.output.dense",
            bias=attention.output.dense.bias,
            bias_method=bias_method
        )

        # Output LayerNorm if it exists
        if attention.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(
                attention.output.LayerNorm,
                name=f"{name}.output.LayerNorm",
                bias_method=bias_method
            )
        
        return serialized_params
    
    def serialize_encoder_layer(self, layer, name="encoder.layer", bias_method: str = "separate"):
        serialized_params = NamedSerialParameters()
        
        # Attention parameters
        serialized_params += self.serialize_attention(
            layer.attention,
            name=f"{name}.attention",
            bias_method=bias_method
        )
        
        # Intermediate dense layer and bias
        intermediate_dense_bias = layer.intermediate.dense.bias # (n * d)
        
        if bias_method == "separate":
            assert intermediate_dense_bias.shape[0] % self.config.hidden_size == 0, (
                "Intermediate dense bias size must be divisible by hidden_size for separate serialization. "
                f"Got bias shape {intermediate_dense_bias.shape} and hidden_size {self.config.hidden_size}."
            )
            reshaped_intermediate_dense_bias = intermediate_dense_bias.view(-1, self.config.hidden_size) # (n, d)
            intermediate_dense_bias = None
            
        serialized_params += self.serialize_matrix(
            matrix=layer.intermediate.dense.weight,
            name=f"{name}.intermediate.dense",
            bias=intermediate_dense_bias,
            bias_method=bias_method
        )
        
        if bias_method == "separate":
            serialized_params += self.serialize_matrix(
                matrix=reshaped_intermediate_dense_bias,
                name=f"{name}.intermediate.dense.bias",
                bias_method="separate"
            )
        
        # Output dense layer and bias
        serialized_params += self.serialize_matrix(
            matrix=layer.output.dense.weight.T,
            name=f"{name}.output.dense",
            bias=layer.output.dense.bias,
            bias_method=bias_method
        )

        # Output LayerNorm if it exists
        if layer.output.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(
                layer.output.LayerNorm,
                name=f"{name}.output.LayerNorm",
                bias_method=bias_method
            )
        
        return serialized_params
    
    def serialize_encoder(self, name="encoder", bias_method: str = "separate"):
        serialized_params = NamedSerialParameters()
        
        for i, layer in enumerate(self.base_model.encoder.layer):
            serialized_params += self.serialize_encoder_layer(
                layer,
                name=f"{name}.layer.{i}",
                bias_method=bias_method
            )
        
        return serialized_params
    
    def serialize_mlm_head(self, name="predictions", bias_method: str = "separate"):
        serialized_params = NamedSerialParameters()
        
        # Transform dense layer and bias
        serialized_params += self.serialize_matrix(
            matrix=self.cls.predictions.transform.dense.weight,
            name=f"{name}.transform.dense",
            bias=self.cls.predictions.transform.dense.bias,
            bias_method=bias_method
        )
        
        # LayerNorm if it exists
        if self.cls.predictions.transform.LayerNorm is not None:
            serialized_params += self.serialize_layernorm(
                self.cls.predictions.transform.LayerNorm,
                name=f"{name}.transform.LayerNorm",
                bias_method=bias_method
            )
        
        # Output decoder layer and if bias_method == "separate" return the bias as a separate tensor
        serialized_params += self.serialize_matrix(
            matrix=self.cls.predictions.decoder.weight,
            name=f"{name}.decoder",
            bias=self.cls.predictions.bias if bias_method == "concat" else None,
            bias_method=bias_method
        )
        
        if bias_method == "separate":
            return serialized_params, self.cls.predictions.bias
        
        return serialized_params, None
    
    def serialize(self, bias_method: str = "separate"):
        serialized_params = NamedSerialParameters()
        
        # Serialize embeddings
        serialized_params += self.serialize_embeddings(bias_method=bias_method)
        
        # Serialize encoder layers
        serialized_params += self.serialize_encoder(bias_method=bias_method)
        
        # Serialize MLM head
        mlm_head_params, cls_bias = self.serialize_mlm_head(bias_method=bias_method)
        serialized_params += mlm_head_params
        
        return serialized_params, cls_bias