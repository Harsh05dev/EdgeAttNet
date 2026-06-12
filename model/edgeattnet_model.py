import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EGMHSA(nn.Module):
    """Edge-Guided Multi-Head Self-Attention without positional encodings."""

    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_bias):
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        edge_tokens = edge_bias.flatten(2).transpose(1, 2)

        query = key = tokens + edge_tokens
        value = tokens

        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, channels)
        out = self.out_proj(out)
        out = self.norm(out + tokens)
        out = out.transpose(1, 2).reshape(batch_size, channels, height, width)
        return out


class UNetEdgeTransformer(nn.Module):
    """
    EdgeAttNet: U-Net backbone with edge-guided multi-head self-attention at the bottleneck.
    Returns (segmentation_logits, edge_map).
    """

    def __init__(self, in_channels=1, base_channels=64, num_heads=4):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.edge_proj = nn.Conv2d(1, channels[-1], kernel_size=1)

        self.enc1 = DoubleConv(in_channels, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(channels[3], channels[3])
        self.eg_mhsa1 = EGMHSA(channels[3], num_heads=num_heads)
        self.eg_mhsa2 = EGMHSA(channels[3], num_heads=num_heads)

        self.up4 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec4 = DoubleConv(channels[2] + channels[3], channels[2])
        self.up3 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec3 = DoubleConv(channels[1] + channels[2], channels[1])
        self.up2 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec2 = DoubleConv(channels[0] + channels[1], channels[0])
        self.up1 = nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[0] + channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def _edge_bias(self, edge_map, target_size):
        edge_resized = F.interpolate(
            edge_map,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.edge_proj(edge_resized)

    def forward(self, x):
        edge_map = self.edge_branch(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))
        edge_bias = self._edge_bias(edge_map, bottleneck.shape[-2:])
        bottleneck = self.eg_mhsa1(bottleneck, edge_bias)
        bottleneck = self.eg_mhsa2(bottleneck, edge_bias)

        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        return logits, edge_map
