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


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
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
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = pixel_values.shape
        # Convolution for non-overlapping for embeddings using the patch_embedding
        # This operration will convert the 
        # [BatchSize, Channel, Height, Width] > [BatchSize, Embed_Dim,  Num_Patches_H, Num_Patches_W] 
        patch_embeddings = self.patch_embedding(pixel_values)

        # Transform the Num_Patches_H, Num_Patches_W into TotalPatches
        # [Batch_Size, Embed_Dim, Num_Patches]
        patch_embeddings = patch_embeddings.flatten(2)

        # Batch_Size, Embed_Dim, Num_Patches] > [Batch_Size, Num_Patches, Embed_Dim] 
        # Transpose the EmbedDim with Num_Patches
        patch_embeddings = patch_embeddings.permute(0,2,1)

        # Add Positions
        patch_embeddings = patch_embeddings + self.position_embedding(self.position_ids)

        return patch_embeddings


class SigLipAttention(nn.Modeule):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ =  self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbedding(embed_dim)
        self.encoder = SiglipEncoderLayer(self.config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=self.config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(pixel_values) # Embeddings of each of the image
        encodings = self.encoder(embeddings) # Encoding the embeddings
        normalized = self.post_layernorm(encodings)

        return normalized


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values)
    
