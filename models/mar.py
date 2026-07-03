"""
S2S_inject — MM-DiT MAR (대기 latent + skt 패치). ★direct(OISST 1° 직접주입) 경로 전용 compact 버전.

아키텍처:
  · 대기 토큰(Cz) + skt 토큰(P²)을 **하나의 시퀀스**로 (프레임당 hw + hw).
  · trunk = **MM-DiT(dual-stream)**: 두 스트림 각자 weight + joint attention → single blocks.
  · 복원 head: 대기=diffusion, 육지 LST=diffusion, 보조 결정론 대기 head(det μ).

skt 결합필드(ocean=SST / land=LST) 를 skt_proj·ocean_proj 로 직접 임베딩:
  · ocean 토큰: 항상 visible(입력 전용, 예측 안 함) — ocean_proj.
  · land(LST) 토큰: 미래를 γ-MAR 로 마스크·복원 — skt_proj.
  · 대기 토큰: 미래를 γ-MAR 로 마스크·복원.

학습 forward_sst_direct / 추론 sample_sst_direct.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.diffloss import DiffLoss
from baseline.models.heads import DeterministicHead, deterministic_mse_loss
from baseline.models.mask_transformer import Block, build_3d_sincos_pos_embed


class DualStreamBlock(nn.Module):
    """두 스트림 A,B 각자 weight + joint attention(키/값 합집합) — MM-DiT 스타일."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.nh = num_heads
        self.hd = dim // num_heads
        self.n1a, self.n1b = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv_a, self.qkv_b = nn.Linear(dim, 3 * dim), nn.Linear(dim, 3 * dim)
        self.proj_a, self.proj_b = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.n2a, self.n2b = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        h = int(dim * mlp_ratio)
        self.mlp_a = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))
        self.mlp_b = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))

    def _split(self, qkv, B, N):
        q, k, v = qkv.reshape(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4) # (3, B, nh, N, hd)
        return q, k, v

    def forward(self, xa, xb):
        B, Na, _ = xa.shape
        Nb = xb.shape[1]
        qa, ka, va = self._split(self.qkv_a(self.n1a(xa)), B, Na)
        qb, kb, vb = self._split(self.qkv_b(self.n1b(xb)), B, Nb)
        k = torch.cat([ka, kb], dim=2)
        v = torch.cat([va, vb], dim=2)
        oa = F.scaled_dot_product_attention(qa, k, v).transpose(1, 2).reshape(B, Na, -1)
        ob = F.scaled_dot_product_attention(qb, k, v).transpose(1, 2).reshape(B, Nb, -1)
        xa = xa + self.drop(self.proj_a(oa))
        xb = xb + self.drop(self.proj_b(ob))
        xa = xa + self.mlp_a(self.n2a(xa))
        xb = xb + self.mlp_b(self.n2b(xb))
        return xa, xb


class S2SInjectTransformer(nn.Module):
    def __init__(
        self,
        latent_channels: int = 216,
        latent_h: int = 10,
        latent_w: int = 20,
        future_len: int = 14,
        cond_len: int = 1,
        skt_patch: int = 18,
        embed_dim: int = 768,
        num_heads: int = 12,
        n_dual: int = 6,
        n_single: int = 6,
        mlp_ratio: float = 4.0,
        diff_width: int = 2048,
        diff_depth: int = 6,
        diff_batch_mul: int = 2,
        num_sampling_timesteps: int = 32,
        sigma_data: float = 0.5,
        mask_ratio_min: float = 0.5,
        skt_recon_weight: float = 1.0,
        dropout: float = 0.0,
        det_weight: float = 0.5,        # 결정론 대기 head 손실 가중 (0=비활성)
        det_max_frames: int = 10,       # 지수감소 e^{-k} 적용 리드(=10일), 이후 0
        ocean_proj: bool = False,       # ocean 토큰 전용 Linear (SST↔LST 임베딩 분리; None 이면 skt_proj 공유)
    ):
        super().__init__()
        self.Cz, self.h, self.w = latent_channels, latent_h, latent_w
        self.hw = latent_h * latent_w
        self.T, self.cond_len = future_len, cond_len
        self.window = cond_len + future_len
        self.D = embed_dim
        self.P = skt_patch
        self.skt_dim = skt_patch * skt_patch
        self.mask_ratio_min = mask_ratio_min
        self.skt_recon_weight = skt_recon_weight
        self.det_weight = det_weight
        self.det_max_frames = det_max_frames

        self.atmo_proj = nn.Linear(latent_channels, embed_dim)
        self.skt_proj = nn.Linear(self.skt_dim, embed_dim)              # land(LST) 토큰 임베딩
        self.ocean_proj = nn.Linear(self.skt_dim, embed_dim) if ocean_proj else None  # ocean(SST) 전용
        self.atmo_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.skt_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(2, embed_dim)) # dual-stream A/B 구분용
        for p in (self.atmo_mask_token, self.skt_mask_token, self.type_embed):
            nn.init.normal_(p, std=0.02)
        self.season_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(),
                                        nn.Linear(embed_dim, embed_dim))
        pos = build_3d_sincos_pos_embed(embed_dim, self.window, latent_h, latent_w)
        self.register_buffer("pos", pos.reshape(1, self.window * self.hw, embed_dim), persistent=False)

        self.dual_blocks = nn.ModuleList(
            [DualStreamBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(n_dual)])
        self.single_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(n_single)])
        self.norm_out = nn.LayerNorm(embed_dim)

        # 대기·LST diffusion head (per-token 확률 복원)
        self.atmo_diff = DiffLoss(target_channels=latent_channels, z_channels=embed_dim,
                                  width=diff_width, depth=diff_depth,
                                  num_sampling_timesteps=num_sampling_timesteps,
                                  sigma_data=sigma_data, diffusion_batch_mul=diff_batch_mul)
        self.skt_diff = DiffLoss(target_channels=self.skt_dim, z_channels=embed_dim,
                                 width=diff_width, depth=diff_depth,
                                 num_sampling_timesteps=num_sampling_timesteps,
                                 sigma_data=sigma_data, diffusion_batch_mul=diff_batch_mul)
        # 결정론 대기 head (보조): za → μ. lead별 e^{-k}(앞 det_max_frames) 가중 MSE
        self.det_head = DeterministicHead(z_channels=embed_dim, target_channels=latent_channels,
                                          predict_sigma=False) if det_weight > 0 else None

    # ── 토큰화/임베딩 ──
    def _atmo_tokens(self, latents):
        B = latents.shape[0]
        return latents.permute(0, 1, 3, 4, 2).reshape(B, self.window * self.hw, self.Cz)

    def _frame_idx(self, device):
        return torch.arange(self.window, device=device).repeat_interleave(self.hw)

    def _ocean_pos(self, ocean_tok, device):
        """(h,w) bool → (N,) bool: ocean 토큰 위치(전 프레임 반복). frame→i→j 정합."""
        return ocean_tok.reshape(-1).to(device).repeat(self.window)

    def _season_embed(self, ts):
        ic = ts[:, min(self.cond_len - 1, ts.shape[1] - 1)].float()
        year = torch.floor(ic / 1e6); md = ic - year * 1e6
        month = torch.floor(md / 1e4); day = torch.floor((md - month * 1e4) / 100)
        frac = (((month - 1) * 30.4 + day) / 365.0).clamp(0, 1)
        half = self.D // 2
        freqs = torch.arange(1, half + 1, device=ts.device).float()
        phase = 2 * torch.pi * frac[:, None]
        emb = torch.cat([torch.sin(phase * freqs), torch.cos(phase * freqs)], dim=1)
        return self.season_mlp(emb[:, :self.D])

    def _embed_direct(self, atmo_tok, skt_f, atmo_mask, land_mask, ocean_pos, season):
        """direct 임베딩: ocean 토큰=ocean_proj(SST 전용, 없으면 skt_proj), land=skt_proj(LST).
        ocean 은 마스크 안 함(visible), land_mask 만 mask_token 치환."""
        xa = self.atmo_proj(atmo_tok) # atmo_tok은 뭐제? -> DCAE Latents
        xa = torch.where(atmo_mask[..., None], self.atmo_mask_token.to(xa.dtype), xa)
        e_land = self.skt_proj(skt_f) # LST랑 Ocean 둘다 같은 input인데 어떻게 구별하는거지? -> 그냥 unshuffle pixel을 다넣고 mask에 따라 선택
        e_ocean = self.ocean_proj(skt_f) if self.ocean_proj is not None else e_land
        xb = torch.where(ocean_pos[None, :, None], e_ocean, e_land)        # 위치별 라우팅
        xb = torch.where(land_mask[..., None], self.skt_mask_token.to(xb.dtype), xb)  # land 미래만 mask
        se = season[:, None]
        xa = xa + self.pos + self.type_embed[0] + se
        xb = xb + self.pos + self.type_embed[1] + se
        return xa, xb

    def _trunk(self, xa, xb):
        for blk in self.dual_blocks:
            xa, xb = blk(xa, xb)
        x = torch.cat([xa, xb], dim=1)
        for blk in self.single_blocks:
            x = blk(x)
        x = self.norm_out(x)
        Na = xa.shape[1]
        return x[:, :Na], x[:, Na:]

    def _maskable(self, ocean_tok, device):
        """(N,) bool: maskable_a=미래 대기, maskable_b=미래 ∩ 육지(LST). 해양은 마스크 대상 아님(항상 visible)."""
        fut = (self._frame_idx(device) >= self.cond_len)               # (N,) 미래 프레임
        land = (~ocean_tok.reshape(-1)).to(device).repeat(self.window)  # (N,) 육지(patch-level)
        return fut, fut & land

    def _mask_group(self, B, able, gamma, device):
        """able:(N,) bool 중 γ 비율을 배치별 랜덤 마스킹 → (B,N) bool."""
        K = int(able.sum().item())
        m = torch.zeros(B, able.shape[0], device=device)
        if K == 0 or gamma <= 0:
            return m.bool()
        n = int(np.ceil(K * gamma))
        score = torch.rand(B, able.shape[0], device=device).masked_fill(~able[None], float("inf"))
        order = score.argsort(dim=1)
        m.scatter_(1, order[:, :n], 1.0)
        return m.bool()

    def _diff_loss(self, diff, target, z, mask_flat):
        if mask_flat.any():
            return diff(target[mask_flat], z[mask_flat])
        return (z.sum() * 0.0)

    # ── 학습/추론 (direct 전용) ──
    def forward(self, latents, ts, skt_in, skt_tgt, ocean_tok, **_legacy):
        # ocean_tok: (B,h,w) bool, skt_in: (B,window,h,w,P²), skt_tgt: (B,window,h,w,P²)
        """direct 경로. skt_in = 결합필드(ocean=SST/land=LST). (skt_tgt·_legacy 인자는 호환용, 미사용.)"""
        return self.forward_sst_direct(latents, ts, skt_in, ocean_tok)

    def forward_sst_direct(self, latents, ts, skt, ocean_tok):
        """학습: 결합필드 skt(ocean=visible, land=LST). 대기 γ·land LST γ masking(ocean 불변).
        loss = atmo_diff + skt_recon·lst_diff(masked land) + det. ocean=ocean_proj, land=skt_proj."""
        B, device = latents.shape[0], latents.device
        atmo = self._atmo_tokens(latents)                                  # (B,N,Cz)
        N = atmo.shape[1]
        skt_f = skt.reshape(B, N, self.skt_dim)                            # 결합필드(ocean/land)
        season = self._season_embed(ts)
        ma_able, mb_able = self._maskable(ocean_tok[0], device)            # mb_able=미래∩land
        ocean_pos = self._ocean_pos(ocean_tok[0], device) # ocean_tok[0]로 batch차원 삭제
        gamma = float(np.random.uniform(self.mask_ratio_min, 1.0))
        ma = self._mask_group(B, ma_able, gamma, device)
        mb_land = self._mask_group(B, mb_able, gamma, device)              # ocean 미포함 → visible
        xa, xb = self._embed_direct(atmo, skt_f, ma, mb_land, ocean_pos, season)
        za, zb = self._trunk(xa, xb)
        atmo_loss = self._diff_loss(self.atmo_diff, atmo.reshape(-1, self.Cz),
                                    za.reshape(-1, self.D), ma.reshape(-1))
        lst_loss = self._diff_loss(self.skt_diff, skt_f.reshape(-1, self.skt_dim),
                                   zb.reshape(-1, self.D), mb_land.reshape(-1))
        out = {"atmo_loss": atmo_loss, "skt_loss": lst_loss}
        loss = atmo_loss + self.skt_recon_weight * lst_loss
        if self.det_head is not None:
            mu, _ = self.det_head(za.reshape(-1, self.D))
            frame = self._frame_idx(device)
            fut = (frame >= self.cond_len)[None].expand(B, -1).reshape(-1)
            lead = (frame - self.cond_len).clamp(min=0)[None].expand(B, -1).reshape(-1)
            det_loss = deterministic_mse_loss(mu[fut], atmo.reshape(-1, self.Cz)[fut], lead[fut],
                                              max_frames=self.det_max_frames, weighted=True)
            out["det_loss"] = det_loss
            loss = loss + self.det_weight * det_loss
        out["loss"] = loss
        return out

    @torch.no_grad()
    def sample_sst_direct(self, ic_latents, ts, skt, ocean_tok, num_iter=14, temperature=1.0,
                          det_atmo=False):
        """추론: 결합필드 ocean=visible 고정, 대기+land LST 를 cosine 스케줄 MAR 생성.
        skt:(B,window,h,w,P²) — ocean 전 프레임·land IC 는 visible, land 미래는 placeholder(생성)."""
        B, device = ic_latents.shape[0], ic_latents.device
        N = self.window * self.hw
        season = self._season_embed(ts)
        atmo = torch.zeros(B, N, self.Cz, device=device)
        atmo[:, :self.cond_len * self.hw] = ic_latents.permute(0, 1, 3, 4, 2).reshape(
            B, self.cond_len * self.hw, self.Cz) # IC 대기 토큰은 채워놓음
        skt_f = skt.reshape(B, N, self.skt_dim).clone()
        ma_able, mb_able = self._maskable(ocean_tok[0], device)
        ocean_pos = self._ocean_pos(ocean_tok[0], device)
        skt_f[:, mb_able] = 0.0                                            # 미래 land placeholder(마스킹됨)
        maskable = torch.cat([ma_able, mb_able])                          # ocean 제외(항상 visible)
        midx = maskable.nonzero(as_tuple=True)[0]; K = midx.numel()     # maskable 토큰 수가 K개
        orders = torch.argsort(torch.rand(B, K, device=device), dim=1) # Batch별로 랜덤 복원 순서
        kept = torch.ones(B, K, device=device)
        for step in range(num_iter):
            full = torch.zeros(B, 2 * N, device=device)
            full.scatter_(1, midx[None].expand(B, -1), kept)
            ma, mb = full[:, :N].bool(), full[:, N:].bool()
            xa, xb = self._embed_direct(atmo, skt_f, ma, mb, ocean_pos, season)
            za, zb = self._trunk(xa, xb)
            ratio = float(np.cos(np.pi / 2.0 * (step + 1) / num_iter))
            mask_len = int(np.floor(K * ratio))
            mask_len = min(max(mask_len, 1), K - 1) if step < num_iter - 1 else 0
            kept_next = torch.zeros(B, K, device=device)
            if mask_len > 0:
                kept_next.scatter_(1, orders[:, :mask_len], 1.0)
            to_pred = (kept.bool() & ~kept_next.bool())
            zaf, zbf = za.reshape(B * N, self.D), zb.reshape(B * N, self.D)
            af, sf = atmo.reshape(B * N, self.Cz), skt_f.reshape(B * N, self.skt_dim)
            bk = to_pred.nonzero(as_tuple=False) # (B*K, 2) -> (batch, maskable_idx), 이 때 maskable_idx는 2N-1의 범위
            if bk.numel() > 0: 
                gpos = midx[bk[:, 1]]; glob = bk[:, 0] * N + (gpos % N)
                is_atmo = gpos < N
                ia, isk = glob[is_atmo], glob[~is_atmo]
                if ia.numel() > 0:
                    af[ia] = (self.det_head(zaf[ia])[0] if det_atmo
                              else self.atmo_diff.sample(zaf[ia], temperature=temperature)).to(af.dtype)
                if isk.numel() > 0:                                        # land LST diffusion
                    sf[isk] = self.skt_diff.sample(zbf[isk], temperature=temperature).to(sf.dtype)
                atmo = af.reshape(B, N, self.Cz); skt_f = sf.reshape(B, N, self.skt_dim)
            kept = kept_next
        fut_atmo = atmo.reshape(B, self.window, self.hw, self.Cz)[:, self.cond_len:]
        fut_atmo = fut_atmo.permute(0, 1, 3, 2).reshape(B, self.T, self.Cz, self.h, self.w)
        return fut_atmo, skt_f.reshape(B, self.window, self.h, self.w, self.skt_dim)
