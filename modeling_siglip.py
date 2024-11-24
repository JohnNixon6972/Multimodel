from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """
    Configuration class for the Siglip Vision model.

    Args:
        hidden_size (int, optional): The size of the hidden layers. Default is 768.
        intermediate_size (int, optional): The size of the intermediate layers. Default is 3072.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Default is 12.
        num_attention_heads (int, optional): The number of attention heads in each attention layer. Default is 12.
        num_channels (int, optional): The number of channels in the input images. Default is 3.
        image_size (int, optional): The size of the input images (height and width). Default is 224.
        patch_size (int, optional): The size of the patches to divide the input images into. Default is 16.
        layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-6.
        attention_dropout (float, optional): The dropout rate for the attention layers. Default is 0.0.
        num_image_tokens (int, optional): The number of image tokens. Default is None.
        **kwargs: Additional keyword arguments.

    Attributes:
        hidden_size (int): The size of the hidden layers.
        intermediate_size (int): The size of the intermediate layers.
        num_hidden_layers (int): The number of hidden layers in the model.
        num_attention_heads (int): The number of attention heads in each attention layer.
        num_channels (int): The number of channels in the input images.
        image_size (int): The size of the input images (height and width).
        patch_size (int): The size of the patches to divide the input images into.
        layer_norm_eps (float): The epsilon value for layer normalization.
        attention_dropout (float): The dropout rate for the attention layers.
        num_image_tokens (int): The number of image tokens.
    """

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
    """
    SiglipVisionEmbeddings is a neural network module that converts input images into a sequence of patch embeddings with positional encodings.

    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        embed_dim (int): Dimensionality of the patch embeddings.
        image_size (int): Size of the input image.
        patch_size (int): Size of each patch.
        patch_embedding (nn.Conv2d): Convolutional layer to generate patch embeddings.
        num_patches (int): Total number of patches in the image.
        num_positions (int): Total number of positions (same as num_patches).
        position_embedding (nn.Embedding): Embedding layer for positional encodings.
        position_ids (torch.Tensor): Tensor containing position indices for positional encodings.

    Methods:
        forward(pixel_values: torch.FloatTensor) -> torch.Tensor:
            Computes the patch embeddings with positional encodings for the input images.

            Args:
                pixel_values (torch.FloatTensor): Input tensor of shape [Batch_Size, Channels, Height, Width].

            Returns:
                torch.Tensor: Output tensor of shape [Batch_Size, Num_Patches, Embed_Dim] containing patch embeddings with positional encodings.
    """
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
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed attention mechanism from the 'Attention Is All You Need' paper.

    Args:
        config (object): Configuration object containing model hyperparameters.

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        embed_dim (int): Dimensionality of the embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for the dot product attention.
        dropout (float): Dropout probability for attention weights.
        k_proj (nn.Linear): Linear layer to project the input to key states.
        v_proj (nn.Linear): Linear layer to project the input to value states.
        q_proj (nn.Linear): Linear layer to project the input to query states.
        out_proj (nn.Linear): Linear layer to project the output of the attention mechanism.

    Methods:
        forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            Computes the multi-headed attention mechanism.

            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor of shape (batch_size, seq_len, embed_dim) 
                and attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
    """
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) class for processing hidden states in a neural network.

    Args:
        config (object): Configuration object containing the following attributes:
            - hidden_size (int): The size of the hidden layer.
            - intermediate_size (int): The size of the intermediate layer.

    Methods:
        forward(hidden_states: torch.Tensor) -> torch.Tensor:
            Forward pass through the MLP.
            
            Args:
                hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].
            
            Returns:
                torch.Tensor: Output tensor of shape [Batch_Size, Num_Patches, Embed_Dim].
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
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
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    SiglipEncoder is a neural network module that encodes input embeddings through a series of encoder layers.

    Args:
        config (SiglipVisionConfig): Configuration object containing parameters for the encoder, 
                                     including the number of hidden layers.

    Attributes:
        config (SiglipVisionConfig): Configuration object.
        layers (nn.ModuleList): List of encoder layers.

    Methods:
        forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
            Passes the input embeddings through the encoder layers and returns the final hidden states.

            Args:
                inputs_embeds (torch.Tensor): Input embeddings of shape [Batch_Size, Num_Patches, Embed_Dim].

            Returns:
                torch.Tensor: Output hidden states of shape [Batch_Size, Num_Patches, Embed_Dim].
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    SiglipVisionTransformer is a vision transformer model for image processing tasks.

    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        embeddings (SiglipVisionEmbeddings): Embedding layer that converts pixel values to embeddings.
        encoder (SiglipEncoder): Encoder layer that processes the embeddings.
        post_layernorm (nn.LayerNorm): Layer normalization applied after the encoder.

    Methods:
        forward(pixel_values: torch.Tensor) -> torch.Tensor:
            Forward pass of the model.

            Args:
                pixel_values (torch.Tensor): Input tensor of shape [Batch_Size, Channels, Height, Width].

            Returns:
                torch.Tensor: Output tensor of shape [Batch_Size, Num_Patches, Embed_Dim].
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
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
        vision_model (SiglipVisionTransformer): Vision Transformer model for processing images.

    Methods:
        forward(pixel_values: torch.Tensor) -> Tuple:
            Forward pass of the model. Takes in pixel values and returns the transformed output.

    Example:
        >>> config = SiglipVisionConfig(...)
        >>> model = SiglipVisionModel(config)
        >>> pixel_values = torch.randn(batch_size, channels, height, width)
        >>> output = model.forward(pixel_values)
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 