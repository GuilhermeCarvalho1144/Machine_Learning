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


class UNET(nn.Module):
    def __int__(self):
        super.__init__()
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
        time - self.time_emb(time)
        # (batch_size, 4, H/8, W/8) -> (batch_size, 320, H/8, W/8)
        output = self.unet(latent, context, time)
        # (batch_size, 320, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        output = self.final_layer(output)
        # (batch_size, 4, H/8, W/8)
        return output
