import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_emb: int) -> None:
        self.linear_1 = nn.Linear(n_emb, 4 * n_emb)
        self.linear_2 = nn.Linear(4 * n_emb, n_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # (1, 1280)
        return x


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, fetures, H, W) -> (batch_size, features, 2*H, 2*w)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET_ResidualBlock(nn.Module):
    """Here we are relating the latent with the time embedding"""

    def __init__(
        self, in_channels: int, out_channels: int, n_time=1280
    ) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_block = nn.Identity()
        else:
            self.residual_block = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(
        self, feature: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # feature: (batch_size, in_channels, H,W)
        # time: (1,n_times = 1280)
        residual = feature
        feature = self.groupnorm(feature)
        feature = F.silu(feature)
        # (batch_size, in_channels, H, W) -> (batch_size, out_channels, H, W)
        feature = self.conv(feature)
        time = F.silu(time)
        # (1, 1 , ntimes=1280) -> (1, 1 , out_channels)
        time = self.linear(time)
        # (batch_size, out_channels, H, W) + (1, out_channels, 1, 1) -> (batch_size, out_channels, H, W)
        merge = feature + time.unsqueeze(-1).unsqueeze(-1)
        merge = self.groupnorm_merged(merge)
        merge = F.silu(merge)
        # (batch_size, out_channels, H, W) -> (batch_size, out_channels, H, w)
        merge = self.conv_merged(merge)
        return merge + self.residual_block(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_emb: int, d_contex=768) -> None:
        super().__init__()
        self.__channels = n_heads * n_emb

        self.groupnorm = nn.GroupNorm(32, self.__channels, eps=1e-6)
        self.conv_input = nn.Conv2d(
            self.__channels, self.__channels, kernel_size=1, padding=0
        )

        self.layernorm_1 = nn.LayerNorm(self.__channels)
        self.self_attetion = SelfAttention(
            n_heads, self.__channels, in_proj_bias=False
        )
        self.layernorm_2 = nn.LayerNorm(self.__channels)
        self.cross_attetion = CrossAttention(
            n_heads, self.__channels, d_contex, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(self.__channels)
        self.linear_geglu_1 = nn.Linear(
            self.__channels, 4 * self.__channels * 2
        )
        self.linear_geglu_2 = nn.Linear(4 * self.__channels, self.__channels)
        self.conv_output = nn.Conv2d(
            self.__channels, self.__channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, H. W)
        # context: (batch_size, seq_len, dims)
        residual_long = x
        x = self.groupnorm(x)
        # (batch_size, channels, H, W) -> (batch_size, channels, H, W)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        # (batch_size, channels, H, W) -> (batch_size, channels, H*W)
        x = x.view((n, c, h * w))
        # (batch_size, channels, H*W) -> (batch_size, H*W, channels)
        x = x.transpose(-1, -2)
        # Normalize + SelfAttention with skip conection
        residual_short = x
        x = self.layernorm_1(x)
        x = self.self_attetion(x)
        x += residual_short
        # Normalize + CrossAttention with skip conection
        residual_short = x
        x = self.layernorm_2(x)
        x = self.cross_attetion(x, context)
        x += residual_short
        # Normalize + FF (GeGLU) with skip conection
        residual_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunks(2, dims=-1)
        x = x + F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residual_short
        # (batch_size, H*W, channels) -> (batch_size, channels, H*W)
        x = x.transpose(-1, -2)
        # (batch_size, channels, H*W) -> (batch_size, channels, H,W)
        x = x.view((n, c, h, w))
        output = self.conv_output(x) + residual_long
        # (batch_size, channels, H, W)
        return output


class UNET(nn.Module):
    def __int__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            # (batch_size, 4, H/8, W/8) -> (batch_size, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(
                UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
            ),
            SwitchSequential(
                UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
            ),
            # (batch_size, 320, H/8, W/8) -> (batch_size, 320, H/16, W/16)
            SwitchSequential(
                nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
            ),
            SwitchSequential(
                UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
            ),
            SwitchSequential(
                UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
            ),
            # (batch_size, 4, H/16, W/16) -> (batch_size, 320, H/32, W/32)
            SwitchSequential(
                nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
            ),
            SwitchSequential(
                UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
            ),
            # (batch_size, 4, H/32, W/32) -> (batch_size, 320, H/64, W/64)
            SwitchSequential(
                nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
            ),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # (batch_size, 1280, H/64, w/64) -> (batch_size, 1280, H/64, w/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList(
            # (batch_size, 2560, H/64, W/64) -> (batch_size, 1280, H/64, W/64)  The skip conection double the input 2*1280=2560
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            # (batch_size, 2560, H/64, W/64) -> (batch_size, 2560, H/32, W/32)  The skip conection double the input 2*1280=2560
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
            ),
            # (batch_size, 2560, H/32, W/32) -> (batch_size, 1280, H/16, W/16) The skip conection double the input 2*640 = 1280
            SwitchSequential(
                UNET_ResidualBlock(1920, 1280),
                UNET_AttentionBlock(8, 160),
                Upsample(1280),
            ),
            SwitchSequential(
                UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
            ),
            # (batch_size, 1280, H/16, W/16) -> (batch_size, 320, H/8, W/8) The skip conection double the input 2*640 = 1280
            SwitchSequential(
                UNET_ResidualBlock(960, 640),
                UNET_AttentionBlock(8, 80),
                Upsample(640),
            ),
            SwitchSequential(
                UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
            ),
            SwitchSequential(
                UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
            ),
            # (batch_size, 320, H/8, W/8)
            SwitchSequential(
                UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
            ),
        )


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, H, W) -> (batch_size, out_channels, H, W)
        x = self.groupnorm(x)
        x = F.silu(x)
        # (batch_size, 320, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):

    def __init__(self) -> None:
        self.time_emb = TimeEmbedding(320)
        self.unet = UNET()
        self.final_layer = UNET_OutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent (batch_size, 4, H/8, w/8) -> output of enconder
        # context (batch_size, seq_len, dim) -> output of the clip
        # time (1, 320)

        # (1,320) -> (1,1280)
        time = self.time_emb(time)
        # (batch_size, 4, H/8, W/8) -> (batch_size, 320, H/8, W/8)
        output = self.unet(latent, context, time)
        # (batch_size, 320, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        output = self.final_layer(output)
        # (batch_size, 4, H/8, W/8)
        return output
