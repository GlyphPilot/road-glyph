"""
InternViT-300M-448px encoder for RoadGlyph.

Loads the standalone InternViT-300M vision transformer and outputs
projected patch embeddings, optionally pixel-shuffle downsampled.
"""
import torch
from torch import nn
from transformers import AutoModel


def pixel_shuffle_downsample(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Spatial pixel-shuffle downsample: [B, H, W, C] → [B, H//s, W//s, C*s*s]."""
    B, H, W, C = x.shape
    assert H % scale == 0 and W % scale == 0
    x = x.reshape(B, H // scale, scale, W // scale, scale, C)
    x = x.permute(0, 1, 3, 4, 2, 5).reshape(B, H // scale, W // scale, C * scale * scale)
    return x


class InternViTEncoderModel(nn.Module):
    def __init__(
        self,
        variant: str = "OpenGVLab/InternViT-300M-448px",
        embed_dim: int = 512,
        freeze: bool = False,
        downsample_feature_grid_factor: int = 2,
        use_global_img: bool = False,
    ):
        super().__init__()
        self.num_cameras = 1
        self.num_frames = 1
        self.token_size = embed_dim
        self.downsample_feature_grid_factor = downsample_feature_grid_factor

        self.image_encoder = AutoModel.from_pretrained(variant, trust_remote_code=True)
        vit_hidden = self.image_encoder.config.hidden_size  # 1024

        # After pixel-shuffle downsample, channel dim multiplies by factor^2
        proj_in = vit_hidden * (downsample_feature_grid_factor ** 2)
        self.projection = nn.Linear(proj_in, embed_dim)

        self.temporal_encoding = nn.Parameter(
            0.02 * torch.randn(1, self.num_frames, 1, 1, embed_dim)
        )
        self.camera_encoding = nn.Parameter(
            0.02 * torch.randn(1, 1, self.num_cameras, 1, embed_dim)
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.projection.weight.requires_grad = True
            self.projection.bias.requires_grad = True
            self.temporal_encoding.requires_grad = True
            self.camera_encoding.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes=None,
        use_temporal_encoding: bool = True,
        use_camera_encoding: bool = True,
    ) -> torch.Tensor:
        # pixel_values: [BS, T, N, C, H, W]
        BS, num_frames, num_cams, C, H, W = pixel_values.shape

        flat = pixel_values.reshape(BS * num_frames * num_cams, C, H, W)

        # InternViT forward → last_hidden_state: [B', 1025, 1024] (1 CLS + 1024 patches)
        vit_out = self.image_encoder(pixel_values=flat).last_hidden_state
        vit_out = vit_out[:, 1:, :]  # remove CLS → [B', 1024, 1024]

        # Pixel-shuffle downsample
        if self.downsample_feature_grid_factor > 1:
            h = w = int(vit_out.shape[1] ** 0.5)  # 32
            vit_out = vit_out.reshape(vit_out.shape[0], h, w, -1)
            vit_out = pixel_shuffle_downsample(vit_out, self.downsample_feature_grid_factor)
            vit_out = vit_out.reshape(vit_out.shape[0], -1, vit_out.shape[-1])

        # Project to embed_dim
        patch_embeddings = self.projection(vit_out)
        patch_embeddings = patch_embeddings.view(
            BS, num_frames, num_cams, patch_embeddings.shape[-2], patch_embeddings.shape[-1]
        )

        input_sequence = patch_embeddings
        _, _, _, n_tokens, channels = input_sequence.shape

        if use_temporal_encoding:
            input_sequence = input_sequence + self.temporal_encoding
        if use_camera_encoding:
            input_sequence = input_sequence + self.camera_encoding

        embeds = input_sequence.view(BS, -1, channels)
        return embeds, (num_frames, n_tokens, channels)
