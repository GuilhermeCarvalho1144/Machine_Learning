import torch
from torch import nn
from torch.nn import functional as F
from attetion import SelfAttention


class VAR_ResidualBlock(nn.Module):
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
