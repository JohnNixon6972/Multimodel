from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values) 
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """
    Implements the SiglipAttention mechanism.

    Args:
        config: Configuration object containing model hyperparameters.

    Attributes:
        config: Stores the configuration object.
        embed_dim: Dimensionality of the embeddings.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor for the dot-product attention.
        dropout: Dropout probability for attention weights.
        k_proj: Linear layer to project input to key states.
        v_proj: Linear layer to project input to value states.
        q_proj: Linear layer to project input to query states.
        out_proj: Linear layer to project the output of the attention mechanism.

    Methods:
        forward(hidden_states):
            Computes the attention mechanism on the input hidden states.

            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor of shape (batch_size, seq_len, embed_dim) 
                and attention weights of shape (batch_size, num_heads, seq_len, seq_len).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model used in the Siglip project.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        config (object): Configuration object containing model parameters.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.

    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the MLP model.
            Applies a linear transformation followed by a GELU activation and another linear transformation.

    Example:
        >>> config = Config(hidden_size=128, intermediate_size=256)
        >>> model = SiglipMLP(config)
        >>> input_tensor = torch.randn(32, 128)
        >>> output_tensor = model(input_tensor)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    SiglipEncoderLayer is a custom neural network layer used in the Siglip model architecture. 
    It consists of a self-attention mechanism followed by a multi-layer perceptron (MLP) with 
    layer normalization applied before each component.
    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
    Attributes:
        embed_dim (int): Dimensionality of the embeddings.
        self_attn (SiglipAttention): Self-attention mechanism.
        layer_norm1 (nn.LayerNorm): Layer normalization applied before self-attention.
        mlp (SiglipMLP): Multi-layer perceptron.
        layer_norm2 (nn.LayerNorm): Layer normalization applied before MLP.
    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Passes the input tensor through the encoder layer, applying layer normalization, 
            self-attention, and MLP in sequence, with residual connections.
            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
            Returns:
                torch.Tensor: Output tensor of the same shape as input.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    SiglipEncoder is a neural network module that encodes input embeddings through a series of encoder layers.

    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        layers (nn.ModuleList): List of SiglipEncoderLayer modules.

    Methods:
        forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
            Passes the input embeddings through the encoder layers and returns the final hidden states.

    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    SiglipVisionTransformer is a vision transformer model for image processing tasks.

    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        embeddings (SiglipVisionEmbeddings): Embedding layer for input pixel values.
        encoder (SiglipEncoder): Encoder layer for processing embeddings.
        post_layernorm (nn.LayerNorm): Layer normalization applied after the encoder.

    Methods:
        forward(pixel_values: torch.Tensor) -> torch.Tensor:
            Forward pass of the model.

            Args:
                pixel_values (torch.Tensor): Input tensor containing pixel values of images.

            Returns:
                torch.Tensor: Output tensor containing the last hidden state after encoding and normalization.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    SiglipVisionModel is a neural network model for vision tasks using a Vision Transformer.

    Args:
        config (SiglipVisionConfig): Configuration object containing model parameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model parameters.
        vision_model (SiglipVisionTransformer): Vision Transformer model initialized with the given configuration.

    Methods:
        forward(pixel_values: torch.Tensor) -> Tuple:
            Performs a forward pass of the model on the input pixel values.

    Example:
        >>> config = SiglipVisionConfig(...)
        >>> model = SiglipVisionModel(config)
        >>> pixel_values = torch.randn(1, 3, 224, 224)
        >>> output = model.forward(pixel_values)
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values) 