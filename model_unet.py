# model_unet.py
import math
import torch
import torch.nn as nn


# ---------------------------
# Encodage temporel sinusoïdal
# ---------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : [B] (entiers ou floats)
        retourne : [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=device) / half_dim
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


# ---------------------------
# Bloc résiduel avec temps
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.residual = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_out = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_out
        h = self.block2(h)
        return h + self.residual(x)


# ---------------------------
# UNet fixe pour 64x64x3
# ---------------------------
class UNet(nn.Module):
    def __init__(self, img_channels: int = 3, time_emb_dim: int = 256):
        """
        Architecture fixe pour 64x64 :
        - base channels = 32 (GPU-friendly)
        Niveaux (spatial / canaux) :
        64x64  : 32
        32x32  : 32
        16x16  : 64
        8x8    : 128
        Bottleneck : 8x8, 128
        Puis remontée symétrique.
        """
        super().__init__()

        base_ch = 32

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --------
        # Encoder
        # --------
        self.init_conv = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        # 64x64 -> 32x32 (canaux 32)
        self.down1_res1 = ResidualBlock(base_ch, base_ch, time_emb_dim)
        self.down1_res2 = ResidualBlock(base_ch, base_ch, time_emb_dim)
        self.down1_down = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)

        # 32x32 -> 16x16 (canaux 64)
        self.down2_res1 = ResidualBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down2_res2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2_down = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # 16x16 -> 8x8 (canaux 128)
        self.down3_res1 = ResidualBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.down3_res2 = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3_down = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1)

        # --------
        # Bottleneck 8x8, canaux = 128
        # --------
        bottleneck_ch = base_ch * 4
        self.bot_res1 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)
        self.bot_res2 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # --------
        # Decoder (upsampling)
        # --------
        # 8x8 -> 16x16
        # x : 128 canaux, skip3 : 128 canaux → concat 256 → 128
        self.up3_up = nn.ConvTranspose2d(bottleneck_ch, bottleneck_ch, kernel_size=4, stride=2, padding=1)
        self.up3_res1 = ResidualBlock(bottleneck_ch * 2, bottleneck_ch, time_emb_dim)
        self.up3_res2 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # 16x16 -> 32x32
        # x : 128, skip2 : 64 → concat 192 → proj 64 via ResBlock
        self.up2_up = nn.ConvTranspose2d(bottleneck_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.up2_res1 = ResidualBlock(base_ch * 2 + base_ch * 2, base_ch * 2, time_emb_dim)
        self.up2_res2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        # 32x32 -> 64x64
        # x : 64, skip1 : 32 → concat 96 → proj 32
        self.up1_up = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.up1_res1 = ResidualBlock(base_ch + base_ch, base_ch, time_emb_dim)
        self.up1_res2 = ResidualBlock(base_ch, base_ch, time_emb_dim)

        # --------
        # Final
        # --------
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 3, 64, 64]
        t : [B]
        """
        t_emb = self.time_mlp(t)

        # Encoder
        x = self.init_conv(x)              # [B, 32, 64, 64]

        x = self.down1_res1(x, t_emb)
        x = self.down1_res2(x, t_emb)
        skip1 = x                          # [B, 32, 64, 64]
        x = self.down1_down(x)             # [B, 32, 32, 32]

        x = self.down2_res1(x, t_emb)      # [B, 64, 32, 32]
        x = self.down2_res2(x, t_emb)      # [B, 64, 32, 32]
        skip2 = x                          # [B, 64, 32, 32]
        x = self.down2_down(x)             # [B, 64, 16, 16]

        x = self.down3_res1(x, t_emb)      # [B, 128, 16, 16]
        x = self.down3_res2(x, t_emb)      # [B, 128, 16, 16]
        skip3 = x                          # [B, 128, 16, 16]
        x = self.down3_down(x)             # [B, 128, 8, 8]

        # Bottleneck
        x = self.bot_res1(x, t_emb)        # [B, 128, 8, 8]
        x = self.bot_res2(x, t_emb)        # [B, 128, 8, 8]

        # Decoder
        # 8 -> 16
        x = self.up3_up(x)                 # [B, 128, 16, 16]
        x = torch.cat([x, skip3], dim=1)   # [B, 256, 16, 16]
        x = self.up3_res1(x, t_emb)        # [B, 128, 16, 16]
        x = self.up3_res2(x, t_emb)        # [B, 128, 16, 16]

        # 16 -> 32
        x = self.up2_up(x)                 # [B, 64, 32, 32]
        x = torch.cat([x, skip2], dim=1)   # [B, 128, 32, 32]
        x = self.up2_res1(x, t_emb)        # [B, 64, 32, 32]
        x = self.up2_res2(x, t_emb)        # [B, 64, 32, 32]

        # 32 -> 64
        x = self.up1_up(x)                 # [B, 32, 64, 64]
        x = torch.cat([x, skip1], dim=1)   # [B, 64, 64, 64]
        x = self.up1_res1(x, t_emb)        # [B, 32, 64, 64]
        x = self.up1_res2(x, t_emb)        # [B, 32, 64, 64]

        return self.final_conv(x)          # [B, 3, 64, 64]
