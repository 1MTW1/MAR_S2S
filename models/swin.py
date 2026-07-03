"""
SST_swin/model.py — U-Transformer (Swin-Unet 기반) skin temperature 14일 예보 (instruction §3).

구조: Patch Embed(4×4) → Swin encoder(3 stage, 2 PatchMerging) → bottleneck
      → Swin decoder(2 stage, PatchExpand + skip concat) → FinalPatchExpand(×4) → (B,T,H,W).
입력의 시간축 T 를 채널처럼 다뤄 '과거 T일 → 미래 T일' 을 한 번에 예측(seq2seq direct).

Swin 블록(WindowAttention/relative position bias/cyclic shift/PatchMerging)은
microsoft/Swin-Transformer + Swin-Unet(Cao 2021) 공식 구현을 그대로 따른다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# window 유틸 (공식 구현)
# ──────────────────────────────────────────────────────────────────────────────
def window_partition(x, ws):
    """(B,H,W,C) → (num_windows·B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    """(num_windows·B, ws, ws, C) → (B,H,W,C)."""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ──────────────────────────────────────────────────────────────────────────────
# WindowAttention — window 내 MHSA + relative position bias B (공식)
#   Attention = SoftMax(QKᵀ/√d + B) V
# ──────────────────────────────────────────────────────────────────────────────
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size                              # (Wh, Ww)
        self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # relative position bias table: (2Wh-1)·(2Ww-1) × nH
        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        # 각 window 내 토큰쌍의 relative position index (고정 buffer)
        ch, cw = torch.meshgrid(torch.arange(window_size[0]), torch.arange(window_size[1]), indexing="ij")
        coords = torch.stack([ch.flatten(), cw.flatten()])         # (2, Wh·Ww)
        rel = coords[:, :, None] - coords[:, None, :]              # (2, N, N) -> i번째 토큰과 j번째 토큰의 상대좌표
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("rel_index", rel.sum(-1))            # (N, N)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """x: (num_windows·B, N, C). mask: (num_windows, N, N) 또는 None."""
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)             # (Bn, nH, N, N)
        bias = self.rpb_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)                          # + relative position bias
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bn // nw, nw, self.nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.nh, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(x))


# ──────────────────────────────────────────────────────────────────────────────
# SwinTransformerBlock — LN→(S)W-MSA→res→LN→MLP→res (공식, 식 1~4)
# ──────────────────────────────────────────────────────────────────────────────
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution                 # (H, W)
        self.ws = window_size
        self.shift = shift_size
        H, W = input_resolution
        if min(H, W) <= window_size:                             # 해상도 ≤ window → shift 없이 단일 window
            self.shift = 0
            self.ws = min(H, W)
        assert 0 <= self.shift < self.ws

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.ws, self.ws), num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

        if self.shift > 0:
            self.register_buffer("attn_mask", self._build_mask(H, W), persistent=False)
        else:
            self.attn_mask = None

    def _build_mask(self, H, W):
        """SW-MSA cyclic shift 시 윈도우 간 잘못된 attention 차단 마스크 (공식)."""
        img = torch.zeros((1, H, W, 1))
        hs = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        ws_ = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        cnt = 0
        for h in hs:
            for w in ws_:
                img[:, h, w, :] = cnt; cnt += 1
        mw = window_partition(img, self.ws).view(-1, self.ws * self.ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x):
        """x: (B, H·W, C)."""
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        xw = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        attn = self.attn(xw, self.attn_mask).view(-1, self.ws, self.ws, C)
        x = window_reverse(attn, self.ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


# ──────────────────────────────────────────────────────────────────────────────
# PatchMerging(2↓,2C) / PatchExpand(2↑,C/2) / FinalPatchExpand(×4) — 공식 Swin-Unet
# ──────────────────────────────────────────────────────────────────────────────
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0, x1, x2, x3 = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2] # 좌상단, 우상단, 좌하단, 우하단
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, 2, 2, C // 4).permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        return self.norm(x.view(B, -1, C // 4))


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale ** 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        s = self.dim_scale
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, s, s, C // s ** 2).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s, W * s, C // s ** 2)
        return self.norm(x.view(B, -1, C // s ** 2))


class BasicLayer(nn.Module):
    """Swin 블록 depth개 (W-MSA/SW-MSA 교대) + (옵션) downsample."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, input_resolution, num_heads, window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class BasicLayerUp(nn.Module):
    """디코더: Swin 블록 depth개 + (옵션) upsample(PatchExpand)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, input_resolution, num_heads, window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for i in range(depth)])
        self.upsample = upsample(input_resolution, dim) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    """입력 (B,T,H,W) → 4×4 패치 → (B, H/4·W/4, C). 시간 T 를 채널로 취급."""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.grid = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)              # (B, H/4·W/4, C)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# U-Transformer 본체
# ──────────────────────────────────────────────────────────────────────────────
class UTransformer(nn.Module):
    """과거 T일 → 미래 T일 (B,T,H,W)→(B,T,H,W). encoder 3 stage(2 merge) + decoder 2 stage.

    instruction §3.4: img_size 는 patch(4)·window·merging 호환이어야 함 → __init__ 에서 검증.
    """
    def __init__(self, img_size=(192, 384), patch_size=4, T=14, embed_dim=96,
                 depths=(2, 2, 2), depths_up=(2, 2), num_heads=(3, 6, 12),
                 window_size=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, ape=False,
                 in_chans=None):
        super().__init__()
        self.T = T
        H, W = img_size
        self._check_grid(H, W, patch_size, window_size, len(depths))
        # in_chans: 입력 채널 수 (기본 T — 기존 ckpt 호환). refine 등 다채널 입력 시 지정 (출력은 항상 T)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans or T, embed_dim)
        g0 = (H // patch_size, W // patch_size)                  # /4 해상도
        # ── 위치 인코딩: 기본은 ★relative position bias (WindowAttention 내, 항상 켜짐).
        #    APE(절대위치)는 옵션(기본 off) — 경도-roll 증강과 충돌하므로 relative bias 만 권장 ──
        self.ape = ape
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, g0[0] * g0[1], embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop)

        # ── encoder: stage0(/4,C) stage1(/8,2C) stage2(/16,4C=bottleneck) ──
        self.enc = nn.ModuleList()
        res = g0
        for i, d in enumerate(depths):
            dim = embed_dim * 2 ** i
            down = PatchMerging if i < len(depths) - 1 else None
            self.enc.append(BasicLayer(dim, res, d, num_heads[i], window_size,
                                       mlp_ratio, drop, attn_drop, downsample=down))
            if down:
                res = (res[0] // 2, res[1] // 2)
        self.bottleneck_dim = embed_dim * 2 ** (len(depths) - 1)
        self.bottleneck_res = res                               # /16

        # ── decoder: bottleneck → up → concat skip(/8) → up → concat skip(/4) ──
        self.dec = nn.ModuleList()
        self.concat_lin = nn.ModuleList()
        for i in range(len(depths_up)):                        # 0:/16→/8, 1:/8→/4
            dim_in = embed_dim * 2 ** (len(depths) - 1 - i)     # /16:4C, /8:2C
            res_in = (self.bottleneck_res[0] * 2 ** i, self.bottleneck_res[1] * 2 ** i)
            self.dec.append(BasicLayerUp(dim_in, res_in, depths_up[i],
                                         num_heads[len(depths) - 1 - i], window_size,
                                         mlp_ratio, drop, attn_drop, upsample=PatchExpand))
            # upsample 후 차원 dim_in//2 = skip 차원과 동일 → concat(2x) → linear 로 축소
            skip_dim = dim_in // 2
            self.concat_lin.append(nn.Linear(2 * skip_dim, skip_dim))

        self.norm_up = nn.LayerNorm(embed_dim)
        self.final_expand = FinalPatchExpand_X4(g0, embed_dim, patch_size)
        self.head = nn.Linear(embed_dim, T)                    # → T 채널(미래 T일)
        self.img_size = img_size
        self.apply(self._init)

    @staticmethod
    def _check_grid(H, W, patch, ws, n_stage):
        """§3.4 나눗셈 제약 검증 — 위반 시 명확히 실패."""
        assert H % patch == 0 and W % patch == 0, f"H,W 가 patch({patch}) 배수 아님: {(H, W)}"
        h, w = H // patch, W // patch
        for s in range(n_stage):
            assert h % ws == 0 and w % ws == 0, \
                f"stage{s} 해상도 {(h, w)} 가 window_size({ws}) 배수 아님 (img/pad 조정 필요)"
            if s < n_stage - 1:
                assert h % 2 == 0 and w % 2 == 0, f"stage{s} 해상도 {(h, w)} 가 PatchMerging(짝수) 불가"
                h, w = h // 2, w // 2

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B,T,H,W) → (B,T,H,W)."""
        B = x.shape[0]
        H, W = self.img_size
        x = self.patch_embed(x)
        if self.ape:                                            # 절대 위치(지리) 정보 주입
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        skips = []
        for i, layer in enumerate(self.enc):
            x, skip = layer(x)
            if i < len(self.enc) - 1:                           # bottleneck 제외 skip 보관
                skips.append(skip)
        # decoder: skip 은 역순(/8, /4)
        for i, layer in enumerate(self.dec):
            x = layer(x)                                        # up: 해상도 2배, 차원 절반
            skip = skips[-(i + 1)]
            x = self.concat_lin[i](torch.cat([x, skip], dim=-1))
        x = self.norm_up(x)
        x = self.final_expand(x)                                # (B, H·W, C)
        x = self.head(x)                                        # (B, H·W, T)
        return x.transpose(1, 2).reshape(B, self.T, H, W)
