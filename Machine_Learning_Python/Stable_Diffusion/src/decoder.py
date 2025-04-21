import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # residual block
        # to match the differents sizes
        # them we can add input + x
        if out_channels == in_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                out_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, In_Channels, H, W)
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual(residual)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attetion = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, channels, H, W)
        residual = x
        n, c, h, w = x.shape
        # (batch_size, channels, H, W) -> (batch_size, channels, H*W)
        x = x.view(n, c, h * w)
        # (batch_size, channels, H*W) -> (batch_size, H*W, channels)
        x = x.transpose(-1, -2)
        # (batch_size, H*W, channels) -> (batch_size, H*W, channels)
        x = self.attetion(x)
        # (batch_size, H*W, channels) -> (batch_size, channels,  H*W)
        x = x.transpose(-1, -2)
        # (batch_size, channels, H*W) -> (batch_size, channels, H, W)
        x = x.view(n, c, h, w)
        x += residual
        return x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, H/8, W/8) -> (batch_size, 4, H/8, W/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (batch_size, 4, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),
            # (batch_size, 512, H/2, W/2) -> (batch_size, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (batch_size, 512, H/2, W/2) -> (batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (batch_size, 256, H/2, W/2) -> (batch_size, 256, H, W)
            nn.Upsample(scale_factor=2),
            # (batch_size, 256, H, W) -> (batch_size, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (batch_size, 256, H, W) -> (batch_size, 128, H, W)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (batch_size, 128, H, W) -> (batch_size, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, H/8, W/8)
        # reverse the scaling
        x /= 0.18215
        for module in self:
            x = module(x)
        # (batch_size, 3, H, W)
        return x
