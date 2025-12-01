import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)

class UNet(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, ch_mults=(1,2,4,8),
                 time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        # down
        self.downs = nn.ModuleList()
        ch = base_ch
        self.init_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        in_ch = ch
        self.downs_channels = []
        for mult in ch_mults:
            out_ch = base_ch * mult
            self.downs.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, time_emb_dim),
                ResidualBlock(out_ch, out_ch, time_emb_dim),
                nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # downsample
            ]))
            in_ch = out_ch
            self.downs_channels.append(out_ch)

        # bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, time_emb_dim),
            ResidualBlock(in_ch, in_ch, time_emb_dim)
        ])

        # up
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.ups.append(nn.ModuleList([
                ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim),
                ResidualBlock(out_ch, out_ch, time_emb_dim),
                nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  # upsample
            ]))
            in_ch = out_ch

        self.final_block = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, img_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        skips = []

        # down
        for res1, res2, downsample in self.downs:
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skips.append(x)
            x = downsample(x)

        # bottleneck
        res1, res2 = self.bottleneck
        x = res1(x, t_emb)
        x = res2(x, t_emb)

        # up
        for (res1, res2, upsample), skip in zip(self.ups, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)

        return self.final_block(x)
