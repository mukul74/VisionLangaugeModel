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


class SiglipAttention(nn.Module):
    "Multi Head Attention mechanism from Attention is all you need"
    def __init__(self, config: SiglipVisionConfig):
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


    def forward(self, hidden_states):
        batch_size, seq_len, embed_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

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

        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.fc2 = nn.Linear(self.config.intermediate_size,self.config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = nn.functional.gelu(self.fc1(hidden_states), approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states
    
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.encoder_layer = SiglipEncoderLayer(config)

    def forward(self, hidden_states):
        hidden_state = self.encoder_layer(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        hidden_states, _ =  self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbedding(self.config)
        self.encoder = SiglipEncoder(self.config)
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
    




