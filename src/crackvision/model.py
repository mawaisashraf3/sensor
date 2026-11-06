"""Hierarchical Swin encoder/decoder with masked shifted-window attention."""

import torch
from torch import nn
from torch.nn import functional as F


def partition(x, w):
    b, h, width, c = x.shape
    return (
        x.reshape(b, h // w, w, width // w, w, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, w * w, c)
    )


def reverse(x, w, h, width, b):
    return (
        x.reshape(b, h // w, width // w, w, w, -1)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, h, width, -1)
    )


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding width must be divisible by attention heads")
        self.heads, self.window, self.scale = heads, window, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
        ).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer("index", rel[:, :, 0] * (2 * window - 1) + rel[:, :, 1])
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        a = (q * self.scale) @ k.transpose(-2, -1)
        a = a + self.bias[self.index.reshape(-1)].reshape(n, n, self.heads).permute(
            2, 0, 1
        )
        nw = mask.shape[0]
        a = a.reshape(b // nw, nw, self.heads, n, n) + mask[None, :, None, :, :]
        a = a.reshape(b, self.heads, n, n).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window=7, shift=False):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def attention_mask(self, h, w, hp, wp, s, device):
        win = self.window
        labels = torch.zeros((1, hp, wp, 1), device=device)
        if s:
            slices = (slice(0, -win), slice(-win, -s), slice(-s, None))
            for i, hs in enumerate(slices):
                for j, ws in enumerate(slices):
                    labels[:, hs, ws, :] = i * 3 + j
        labels = partition(labels, win).squeeze(-1)
        mask = labels[:, :, None] != labels[:, None, :]
        valid = torch.zeros((1, hp, wp, 1), device=device, dtype=torch.bool)
        valid[:, :h, :w] = True
        if s:
            valid = torch.roll(valid, (-s, -s), (1, 2))
        valid = partition(valid, win).squeeze(-1)
        # Mask padded keys; padded queries are discarded after reverse/crop.
        mask = mask | (~valid[:, None, :])
        return torch.zeros(mask.shape, device=device).masked_fill(mask, -10000.0)

    def forward(self, x):
        b, h, w, c = x.shape
        win = self.window
        s = win // 2 if self.shift and min(h, w) > win else 0
        z = F.pad(self.norm1(x), (0, 0, 0, (-w) % win, 0, (-h) % win))
        hp, wp = z.shape[1:3]
        if s:
            z = torch.roll(z, (-s, -s), (1, 2))
        z = reverse(
            self.attn(
                partition(z, win), self.attention_mask(h, w, hp, wp, s, x.device)
            ),
            win,
            hp,
            wp,
            b,
        )
        if s:
            z = torch.roll(z, (s, s), (1, 2))
        x = x + z[:, :h, :w]
        return x + self.mlp(self.norm2(x))


class Merge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        _, h, w, _ = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1
        )
        return self.reduce(self.norm(x))


class Expand(nn.Module):
    def __init__(self, dim, out_dim, scale=2):
        super().__init__()
        self.scale, self.out_dim = scale, out_dim
        self.proj = nn.Linear(dim, scale * scale * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        b, h, w, _ = x.shape
        s = self.scale
        x = (
            self.proj(x)
            .reshape(b, h, w, s, s, self.out_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, h * s, w * s, self.out_dim)
        )
        return self.norm(x)


class SpatialAttention(nn.Module):
    """Explicit design choice: mean/max channel pooling and 7x7 spatial gate."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        z = x.permute(0, 3, 1, 2)
        gate = self.conv(
            torch.cat([z.mean(1, keepdim=True), z.amax(1, keepdim=True)], 1)
        ).sigmoid()
        return (z * gate).permute(0, 2, 3, 1)


class CrackVisionNet(nn.Module):
    def __init__(
        self,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        window=7,
        spatial_attention=True,
    ):
        super().__init__()
        if len(depths) != 4 or len(heads) != 4 or any(d < 1 for d in depths):
            raise ValueError(
                "Four positive encoder depths and four head counts are required"
            )
        self.patch = nn.Conv2d(3, embed_dim, 4, stride=4)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.enc = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i, d in enumerate(depths):
            dim = embed_dim * 2**i
            self.enc.append(
                nn.Sequential(
                    *[SwinBlock(dim, heads[i], window, bool(j % 2)) for j in range(d)]
                )
            )
            if i < 3:
                self.merge.append(Merge(dim))
        for i in (2, 1, 0):
            dim = embed_dim * 2**i
            self.up.append(Expand(dim * 2, dim))
            self.fuse.append(nn.Linear(dim * 2, dim))
            self.gates.append(
                SpatialAttention() if spatial_attention else nn.Identity()
            )
            self.dec.append(
                nn.Sequential(
                    *[SwinBlock(dim, heads[i], window, bool(j % 2)) for j in range(2)]
                )
            )
        self.final = Expand(embed_dim, embed_dim, 4)
        self.out = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32))
        x = self.patch_norm(self.patch(x).permute(0, 2, 3, 1))
        skips = []
        for i, layer in enumerate(self.enc):
            x = layer(x)
            skips.append(x)
            if i < 3:
                x = self.merge[i](x)
        for up, fuse, gate, dec, skip in zip(
            self.up, self.fuse, self.gates, self.dec, reversed(skips[:-1])
        ):
            x = up(x)[:, : skip.shape[1], : skip.shape[2]]
            x = dec(fuse(torch.cat([x, gate(skip)], -1)))
        return self.out(self.final(x).permute(0, 3, 1, 2))[:, :, :h, :w]


class UNet(nn.Module):
    """Compact convolutional comparison model trained with the same protocol."""

    def __init__(self, base=32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(),
            )

        self.enc = nn.ModuleList(
            [block(3, base), block(base, base * 2), block(base * 2, base * 4)]
        )
        self.dec = nn.ModuleList([block(base * 6, base * 2), block(base * 3, base)])
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4))
        skips = []
        for i, m in enumerate(self.enc):
            x = m(x)
            skips.append(x)
            if i < 2:
                x = F.max_pool2d(x, 2)
        for m, s in zip(self.dec, reversed(skips[:-1])):
            x = m(
                torch.cat(
                    [
                        F.interpolate(
                            x, size=s.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        s,
                    ],
                    1,
                )
            )
        return self.out(x)[:, :, :h, :w]


def build_model(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("name", "swin")
    if kind == "swin":
        return CrackVisionNet(**cfg)
    if kind == "unet":
        return UNet(**cfg)
    raise ValueError(f"Unknown model: {kind}")
