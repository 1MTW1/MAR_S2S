"""
S2S 마스크 트랜스포머 (s2s_instruction.md §A2, §A3, §B1, §B2).

MAR(`models/mar.py`)를 이식·수정:
  A3. MAE encoder-decoder (encoder=visible 만, decoder 에서 [MASK] 삽입, 양방향 full attention).
  A2. 마스킹 + 점진적 언마스킹 (random_masking, sample_orders, cosine 스케줄).
  B1. 토큰 2D→3D: (B, H·W, D) → (B, T·h·w, D). 미래 T프레임을 시간축까지 펼쳐 마스킹/attention.
  B2. class label → IC 조건 토큰: MAR class embedding 을 X0(초기조건) 인코딩 토큰 c 로 교체.
      c 는 항상 visible(마스킹 제외).

생성 방식: latent 공간에서 마스크된 미래 토큰을 per-token diffusion 으로 생성.
위치 임베딩은 Stage 1 에서 3D sincos 절대 임베딩(temporal, lat, lon). Stage 5(§B4)에서
GeoRoPE + 좌표 동반으로 교체한다(scaffold: `pos_embed_type`).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffloss import DiffLoss
from .heads import DeterministicHead


# ──────────────────────────────────────────────────────────────────────────
#  positional embedding (3D sincos: temporal, lat(h), lon(w))
# ──────────────────────────────────────────────────────────────────────────
def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def build_3d_sincos_pos_embed(dim: int, t: int, h: int, w: int) -> torch.Tensor:
    """(t*h*w, dim) 3D 위치 임베딩. dim 을 (temporal, lat, lon) 으로 분할."""
    d_t = dim // 4 * 1  # temporal 에 1/4
    d_t -= d_t % 2
    d_s = (dim - d_t) // 2  # lat, lon 각 절반
    d_s -= d_s % 2
    d_t = dim - 2 * d_s  # 나머지는 temporal 로 흡수 (짝수 보장)

    gt, gh, gw = np.meshgrid(np.arange(t), np.arange(h), np.arange(w), indexing="ij")
    emb_t = _sincos_1d(d_t, gt.reshape(-1))
    emb_h = _sincos_1d(d_s, gh.reshape(-1))
    emb_w = _sincos_1d(d_s, gw.reshape(-1))
    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (t*h*w, dim)
    return torch.from_numpy(emb).float()


# ──────────────────────────────────────────────────────────────────────────
#  transformer block (양방향 full attention)
# ──────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, rope=None, attn_mask=None):
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:                           # GeoRoPE (§B4): 격자 좌표에 회전 적용
            from .georope import apply_geo_rope
            q, k = apply_geo_rope(q, k, rope[0], rope[1])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # attn_mask=None=양방향
        out = out.transpose(1, 2).reshape(B, N, C)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


def modulate(x, shift, scale):
    """AdaLN 변조: x·(1+scale)+shift.  shift/scale: (B,N,C) per-token."""
    return x * (1 + scale) + shift


# ──────────────────────────────────────────────────────────────────────────
#  AdaLN-Zero causal block (§Option A: BC 강제 의존)
#    norm 의 affine 을 끄고(elementwise_affine=False), BC 가 만든 per-token
#    (shift,scale,gate)×2 로 attention/MLP 잔차를 변조한다. gate zero-init →
#    초기엔 항등(BC 무시), 학습이 진행되며 BC 의존을 켠다.  구조적으로 BC 를
#    못 무시하므로 "BC 최대 잠재력" 확인용.
# ──────────────────────────────────────────────────────────────────────────
class AdaLNCausalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mod, rope=None, attn_mask=None):
        """mod: 6-tuple (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), 각 (B,N,C)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        B, N, C = x.shape
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:
            from .georope import apply_geo_rope
            q, k = apply_geo_rope(q, k, rope[0], rope[1])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, C)
        x = x + gate_msa * self.drop(self.proj(out))
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.drop(self.mlp(h))
        return x


# ──────────────────────────────────────────────────────────────────────────
#  마스킹 유틸 (MAR 이식)
# ──────────────────────────────────────────────────────────────────────────
def sample_orders(bsz: int, seq_len: int, device) -> torch.Tensor:
    return torch.argsort(torch.rand(bsz, seq_len, device=device), dim=1)


def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len, device=device)
    src = torch.ones(bsz, seq_len, device=device)
    masking = torch.scatter(masking, dim=1, index=order[:, :mask_len.long()], src=src).bool()
    return masking


# ──────────────────────────────────────────────────────────────────────────
#  S2S 마스크 트랜스포머
# ──────────────────────────────────────────────────────────────────────────
class S2SMaskTransformer(nn.Module):
    def __init__(
        self,
        # ★ default = s2s.yaml(정식 풀스택) 값. DCAE latent (162,10,20), patch=1.
        latent_channels: int = 162,
        latent_h: int = 10,
        latent_w: int = 20,
        future_len: int = 44,
        cond_len: int = 1,              # 조건(history) 프레임 수: 1=IC(baseline), >1=seq2seq history
        patch_size: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,            # head_dim = 64
        mlp_ratio: float = 4.0,
        # diffusion head
        diff_width: int = 2048,
        diff_depth: int = 6,
        diff_batch_mul: int = 2,        # MAR diffusion_batch_mul (per-token noise/σ 샘플 수)
        num_sampling_timesteps: Optional[int] = 32,    # EDM Heun steps
        # 학습 마스킹 (§53): gamma ~ U[mask_ratio_min, 1.0]
        mask_ratio_min: float = 0.5,
        # 보조 head (§C1/§D2)
        deterministic: bool = True,
        predict_sigma: bool = True,
        causal_depth: int = 4,             # 보조 결정론 head 의 프레임-causal self-attn 블록 수
        det_weighted_loss: bool = True,    # det 손실 프레임 e^{-k} 가중. 짧은 horizon(3일)이면 False(균등)
        pos_embed_type: str = "georope",   # "sincos3d" | "georope"(§B4)
        # diffusion (§D1) — EDM 전용 (DDPM 경로 제거됨)
        diffusion_type: str = "edm",
        sigma_data: float = 0.5,
        # GeoRoPE (§B4) — sum(rope_axes_dim)=embed_dim//num_heads. None→자동([8,28,28]).
        rope_axes_dim: Optional[list] = None,
        rope_theta: float = 256.0,
        lat_start: float = 90.0, lat_end: float = -89.0,
        lon_start: float = 0.0, lon_end: float = 359.0,
        # seasonality (§C2) — IC timestamp 의 day-of-year 임베딩
        incl_time_elapsed: bool = True,
        # dropout (규제) — encoder/decoder Block 의 residual 경로에 적용. 0.0=비활성(기본).
        dropout: float = 0.0,
        # boundary condition (BC 주입) — skt+landmask 를 causal 트렁크 입력에 additive.
        use_boundary: bool = True,
        bc_in_channels: int = 2,        # skt, land_sea_mask
        bc_drop_prob: float = 0.15,     # 학습 시 e_bc→null 교체 확률 (CFG)
        bc_encoder_type: str = "conv",  # "conv"(SphereConv+AvgPool) | "sst_attn"(land=null+masked-conv patchify)
        bc_phys_h: int = 180,           # sst_attn: 물리격자 높이(셀=bc_phys_h//token_h)
        bc_phys_w: int = 360,           # sst_attn: 물리격자 너비
        bc_inject_type: Optional[str] = None,  # BC 주입 방식. None→자동(sst_attn=concat_attn, else additive).
        #   "additive"(트렁크 입력 가산) | "concat_attn"(BC 토큰 concat self-attn) | "adaln"(BC-driven AdaLN-Zero)
        # ocean-first 학습 마스킹 — 바다 토큰을 언마스킹 order 뒤(=먼저 복원될 위치)에 배치.
        #   추론의 ocean-first 스케줄(compare_ocean_first.py)과 조건분포 정합. landmask 는 bc 마지막 채널에서.
        mask_order: str = "random",        # "random"(기존) | "ocean_first"
        ocean_first_prob: float = 1.0,     # ocean_first 일 때 배치별 적용 확률(나머지는 random; <1=커버리지 혼합)
        ocean_thresh: float = 0.5,         # 셀 해양비율 ≥ thresh → 바다 토큰
        ocean_land_thresh: float = 0.5,    # landmask ≤ thresh → 해양 픽셀
    ):
        super().__init__()
        assert latent_h % patch_size == 0 and latent_w % patch_size == 0
        self.Cz = latent_channels
        self.p = patch_size
        self.h = latent_h // patch_size
        self.w = latent_w // patch_size
        self.T = future_len
        self.cond_len = cond_len
        self.hw = self.h * self.w
        self.cond_tok = cond_len * self.hw   # 조건 prefix 토큰 수 (cond_len 프레임)
        self.token_dim = latent_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        self.mask_ratio_min = mask_ratio_min
        self.deterministic = deterministic
        self.predict_sigma = predict_sigma
        self.det_weighted_loss = det_weighted_loss
        self.pos_embed_type = pos_embed_type
        self.use_rope = (pos_embed_type == "georope")
        self.incl_time_elapsed = incl_time_elapsed
        head_dim = embed_dim // num_heads
        self.rope_axes_dim = rope_axes_dim or [head_dim // 8 * 1 or 2,
                                               (head_dim - (head_dim // 8 or 2)) // 2,
                                               (head_dim - (head_dim // 8 or 2)) // 2]
        self.rope_theta = rope_theta

        # token <-> embed
        self.z_proj = nn.Linear(self.token_dim, embed_dim)      # 미래 토큰
        self.cond_proj = nn.Linear(self.token_dim, embed_dim)   # IC 조건 토큰 (B2)
        self.z_proj_ln = nn.LayerNorm(embed_dim)

        # mask token (decoder 삽입용)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 위치 임베딩 — (cond_len+T) 프레임: [:cond_len]=조건(history), [cond_len:]=미래.
        #   temporal index 0..cond_len-1 (history) → cond_len..cond_len+T-1 (future), 단조 증가.
        nframe = self.cond_len + self.T
        pos = build_3d_sincos_pos_embed(embed_dim, nframe, self.h, self.w)
        pos = pos.reshape(nframe, self.hw, embed_dim)
        self.register_buffer("cond_pos", pos[: self.cond_len].reshape(1, self.cond_tok, embed_dim))
        self.register_buffer("future_pos", pos[self.cond_len :].reshape(1, self.T * self.hw, embed_dim))

        # GeoRoPE 좌표 테이블 (§B4): (cond_len+T) 프레임의 (t,lat,lon).
        from .georope import make_coord_table
        coords = make_coord_table(nframe, self.h, self.w,
                                  lat_start, lat_end, lon_start, lon_end, deg2rad=True)
        coords = coords.reshape(nframe, self.hw, 3)
        self.register_buffer("cond_coords", coords[: self.cond_len].reshape(1, self.cond_tok, 3))
        self.register_buffer("future_coords", coords[self.cond_len :].reshape(1, self.T * self.hw, 3))

        # seasonality 임베딩 (§C2): day-of-year sincos -> embed_dim 에 가산(보조).
        if incl_time_elapsed:
            self.season_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        # encoder / decoder
        self.encoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # per-token diffusion head (공유). EDM 전용 — DDPM 제거됨.
        assert diffusion_type == "edm", \
            "DDPM 경로 제거됨 — diffusion_type 은 'edm' 만 지원."
        self.diffloss = DiffLoss(
            target_channels=self.token_dim,
            z_channels=embed_dim,
            width=diff_width,
            depth=diff_depth,
            num_sampling_timesteps=num_sampling_timesteps,
            sigma_data=sigma_data,
            diffusion_batch_mul=diff_batch_mul,
        )

        # ── 공유 causal self-attn 트렁크 (attention decoder 이후, MLP head 들 이전) ──
        #   dec_full[cond+future] → [프레임-causal self-attn × causal_depth] → LayerNorm
        #   → 미래부분(z) 이 Diffusion MLP 와 deterministic MLP 의 공통 입력.
        #   causal_depth=0 이면 트렁크 비활성(diffusion 이 decoder 출력을 직접 사용 — 구버전).
        # ── BC 주입 방식 결정 (구 ckpt 호환: 명시 안 하면 인코더로 추론) ──
        #   sst_attn 모델(concat_attn 으로 학습) / conv 모델(additive 로 학습) 자동 복원.
        if bc_inject_type is None:
            bc_inject_type = "concat_attn" if bc_encoder_type == "sst_attn" else "additive"
        assert bc_inject_type in ("additive", "concat_attn", "adaln"), bc_inject_type
        self.bc_inject_type = bc_inject_type

        self.use_causal = causal_depth > 0
        if self.use_causal:
            # adaln 모드는 AdaLN-Zero 블록, 그 외는 일반 Block.
            BlockCls = (AdaLNCausalBlock if (use_boundary and bc_inject_type == "adaln") else Block)
            self.causal_blocks = nn.ModuleList(
                [BlockCls(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(causal_depth)]
            )
            self.causal_norm = nn.LayerNorm(embed_dim)
            # 프레임 단위 causal 가산 마스크 (N,N): allow i→j ⇔ frame[j] ≤ frame[i]. (cond+future)
            self.register_buffer("causal_attn_mask", self._build_frame_causal_mask(),
                                 persistent=False)
        else:
            self.causal_blocks = None

        # 보조 결정론 MLP head (§C1, §D2) — causal 트렁크 출력 z 에서 μ(, logvar) 예측.
        # predict_sigma=False 면 μ-only (logvar 예측 끔).
        self.det_head = (
            DeterministicHead(embed_dim, self.token_dim, predict_sigma=predict_sigma)
            if deterministic else None
        )

        # ── boundary condition 인코더 (§BC) ──────────────────────────────
        #   skt+landmask (B, cond_len+T, 2, 180, 360) → e_bc (B, cond_tok+L, D).
        #   causal 트렁크 입력에 더해져 BC[0..t]→state[t] 인과를 frame-causal attn 이 전파.
        #   학습 시 prob 로 e_bc→bc_null (CFG uncond). 트렁크/head 만 2-pass 로 guidance.
        self.use_boundary = use_boundary
        self.bc_drop_prob = bc_drop_prob
        self.bc_encoder_type = bc_encoder_type
        # ocean-first 학습 마스킹
        assert mask_order in ("random", "ocean_first"), mask_order
        self.mask_order = mask_order
        self.ocean_first_prob = ocean_first_prob
        self.ocean_thresh = ocean_thresh
        self.ocean_land_thresh = ocean_land_thresh
        self._ocean_tok = None             # (L,) bool — bc landmask 에서 lazy 계산·캐시(정적)
        if use_boundary:
            assert self.use_causal, "use_boundary 는 causal 트렁크(causal_depth>0)가 필요"
            if bc_encoder_type == "conv":
                from .boundary import BoundaryEncoder
                self.boundary_encoder = BoundaryEncoder(
                    in_channels=bc_in_channels, embed_dim=embed_dim,
                    token_h=self.h, token_w=self.w,
                )
            elif bc_encoder_type == "avgpool":
                # 최소 인코더: avgpool(18×)→1×1 conv (SphereConv·다운샘플 없음). conv ablation.
                from .boundary import PoolEmbedBoundaryEncoder
                self.boundary_encoder = PoolEmbedBoundaryEncoder(
                    in_channels=bc_in_channels, embed_dim=embed_dim,
                    token_h=self.h, token_w=self.w,
                    phys_h=bc_phys_h, phys_w=bc_phys_w,
                )
            elif bc_encoder_type == "sst_attn":
                # land=null token + masked-conv patchify (정밀 SST, ocean-valid).
                from .sst_boundary import SSTBoundaryEncoder
                self.boundary_encoder = SSTBoundaryEncoder(
                    in_channels=bc_in_channels, embed_dim=embed_dim,
                    token_h=self.h, token_w=self.w,
                    phys_h=bc_phys_h, phys_w=bc_phys_w,
                )
            elif bc_encoder_type == "sst_soil":
                # ocean=SST + land=soil 결합 (bc=[skt, swvl…, landmask], in=2+n_soil).
                from .sst_boundary import CombinedBoundaryEncoder
                self.boundary_encoder = CombinedBoundaryEncoder(
                    in_channels=bc_in_channels, embed_dim=embed_dim,
                    token_h=self.h, token_w=self.w,
                    phys_h=bc_phys_h, phys_w=bc_phys_w,
                )
            else:
                raise ValueError(f"unknown bc_encoder_type: {bc_encoder_type}")
            self.bc_null = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CFG uncond 벡터
            nn.init.normal_(self.bc_null, std=0.02)
            if bc_inject_type == "concat_attn":
                # BC 토큰을 causal 트렁크 시퀀스에 concat → state 가 BC[0..t] attend. 확장 마스크 (2N,2N).
                self.register_buffer("causal_attn_mask_bc", self._build_frame_causal_mask_bc(),
                                     persistent=False)
            elif bc_inject_type == "adaln":
                # 블록별 AdaLN-Zero 변조기: e_bc → (shift,scale,gate)×2.  마지막 Linear zero-init=초기 항등.
                self.bc_adaln = nn.ModuleList(
                    [nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
                     for _ in range(causal_depth)]
                )
                for mod in self.bc_adaln:
                    nn.init.zeros_(mod[-1].weight); nn.init.zeros_(mod[-1].bias)

        nn.init.normal_(self.mask_token, std=0.02)

    # ── 프레임 causal 마스크 빌드 (S2S_det 와 동일) ──────────────────────
    def _build_frame_causal_mask(self) -> torch.Tensor:
        hw, T, cl = self.hw, self.T, self.cond_len
        cond_frames = torch.arange(cl).repeat_interleave(hw)             # 0..cl-1
        fut_frames = (cl + torch.arange(T)).repeat_interleave(hw)        # cl..cl+T-1
        fids = torch.cat([cond_frames, fut_frames])                     # (N,)
        allowed = fids[None, :] <= fids[:, None]                        # (N,N) i→j ⇔ frame[j]≤frame[i]
        mask = torch.zeros(allowed.shape, dtype=torch.float32)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask                                                     # (N,N) 가산 마스크

    # ── BC concat self-attn 용 확장 causal 마스크 (2N,2N) ────────────────
    def _build_frame_causal_mask_bc(self) -> torch.Tensor:
        """시퀀스 = [state(cond+future) , bc(cond+future)].  규칙:
          · 모든 query 는 frame ≤ 인 key 만 (frame-causal).
          · BC query 는 state key 를 보지 않음(BC=순수 forcing, state 역류 차단).
        state query 는 state+bc 모두 frame-causal 로 attend → BC[0..t]→state[t] 주입."""
        hw, T, cl = self.hw, self.T, self.cond_len
        cond_frames = torch.arange(cl).repeat_interleave(hw)
        fut_frames = (cl + torch.arange(T)).repeat_interleave(hw)
        fids_s = torch.cat([cond_frames, fut_frames])                  # (N_s,)
        Ns = fids_s.numel()
        fids = torch.cat([fids_s, fids_s])                             # (2N_s,) state+bc
        is_bc = torch.zeros(2 * Ns, dtype=torch.bool); is_bc[Ns:] = True
        allowed = fids[None, :] <= fids[:, None]                       # frame-causal
        forbid = is_bc[:, None] & (~is_bc[None, :])                    # bc query → state key 금지
        allowed = allowed & (~forbid)
        mask = torch.zeros(allowed.shape, dtype=torch.float32)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask                                                    # (2N,2N)

    # ── causal 트렁크: 프레임 causal self-attn + BC 주입 (3모드) ──────────
    def _causal_trunk(self, dec_full, e_bc=None):
        """dec_full:(B, cond_tok+L, D) → causal 처리된 (B, cond_tok+L, D). 반환은 항상 state(N_s)부분.
        BC 주입(self.bc_inject_type):
          · additive   : trunk_in = dec_full + e_bc (원본).
          · concat_attn: [state, bc] concat self-attn (state 가 BC[0..t] attend), state 부분만 반환.
          · adaln      : state-only self-attn 인데 각 블록 norm 을 e_bc 가 변조(AdaLN-Zero)."""
        if not self.use_causal:
            return dec_full
        B, Ns = dec_full.shape[0], dec_full.shape[1]
        coords_state = (torch.cat([self.cond_coords.expand(B, self.cond_tok, 3),
                                   self.future_coords.expand(B, self.T * self.hw, 3)], dim=1)
                        if self.use_rope else None)

        if self.bc_inject_type == "adaln":
            assert e_bc is not None, "adaln 트렁크는 e_bc 필요"
            rope = self._build_rope(coords_state) if self.use_rope else None
            attn_mask = self.causal_attn_mask.to(dec_full.dtype)
            x = dec_full
            for i, blk in enumerate(self.causal_blocks):
                mod = self.bc_adaln[i](e_bc).chunk(6, dim=-1)          # 각 (B,N_s,D)
                x = blk(x, mod, rope=rope, attn_mask=attn_mask)
            return self.causal_norm(x)

        if self.bc_inject_type == "concat_attn" and e_bc is not None:
            x = torch.cat([dec_full, e_bc], dim=1)                     # [state, bc]
            attn_mask = self.causal_attn_mask_bc.to(x.dtype)
            rope = (self._build_rope(torch.cat([coords_state, coords_state], dim=1))
                    if self.use_rope else None)                       # bc 좌표 = state 동일
            for blk in self.causal_blocks:
                x = blk(x, rope=rope, attn_mask=attn_mask)
            return self.causal_norm(x)[:, :Ns]                        # state 부분만

        # additive (또는 BC 없음)
        rope = self._build_rope(coords_state) if self.use_rope else None
        attn_mask = self.causal_attn_mask.to(dec_full.dtype)
        x = dec_full + e_bc if e_bc is not None else dec_full
        for blk in self.causal_blocks:
            x = blk(x, rope=rope, attn_mask=attn_mask)
        return self.causal_norm(x)

    # ── patchify ────────────────────────────────────────────────────────
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, Cz, T, Hz, Wz) -> tokens (B, T*h*w, token_dim)."""
        B, C, T, H, W = x.shape
        p = self.p
        x = x.reshape(B, C, T, H // p, p, W // p, p)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()  # B,T,h,w,C,p,p
        return x.reshape(B, T * (H // p) * (W // p), C * p * p)

    def unpatchify(self, tokens: torch.Tensor, T: int) -> torch.Tensor:
        """tokens (B, T*h*w, token_dim) -> (B, Cz, T, Hz, Wz)."""
        B = tokens.shape[0]
        p, C, h, w = self.p, self.Cz, self.h, self.w
        x = tokens.reshape(B, T, h, w, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()  # B,C,T,h,p,w,p
        return x.reshape(B, C, T, h * p, w * p)

    # ── ocean-first 토큰 마스크 (bc landmask → 토큰별 바다여부) ───────────
    def _ocean_tok_from_bc(self, bc: torch.Tensor) -> torch.Tensor:
        """bc:(B,F,C_in,H,W) 의 landmask(마지막 채널)에서 토큰별 바다여부 (L,) bool.
        셀(i,j)=18×18 블록의 해양비율 ≥ ocean_thresh 면 바다. 정적이라 1회 계산 후 캐시."""
        if self._ocean_tok is not None:
            return self._ocean_tok
        H, W = bc.shape[-2], bc.shape[-1]
        lm = bc[0, 0, -1]                                       # (H,W) 1=land (정적)
        ch, cw = H // self.h, W // self.w
        ocean = (lm <= self.ocean_land_thresh).float().reshape(
            self.h, ch, self.w, cw).mean(dim=(1, 3))           # (h,w) 해양비율
        cell = (ocean >= self.ocean_thresh).reshape(-1)        # (h·w,)
        self._ocean_tok = cell.repeat(self.T).to(bc.device)    # (T·h·w,) 프레임마다 동일
        return self._ocean_tok

    # ── 마스킹 (§53: gamma~U[min,1]) ────────────────────────────────────
    def random_masking(self, bsz: int, seq_len: int, device,
                       ocean_tok: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ocean_tok 주어지고 확률 통과 시: 바다=order 뒤(먼저 복원). 낮은 γ→land 만 마스크/예측,
        #   높은 γ→land+ocean. (β=0 추론 ocean-first 와 조건분포 정합.)
        if ocean_tok is not None and float(np.random.uniform()) < self.ocean_first_prob:
            rand = torch.rand(bsz, seq_len, device=device)
            score = rand + ocean_tok.view(1, seq_len).float()  # land [0,1), ocean [1,2)
            orders = score.argsort(dim=1)                      # land 앞, ocean 뒤
        else:
            orders = sample_orders(bsz, seq_len, device)
        # gamma ~ U[mask_ratio_min, 1.0] (배치 공통 길이로 단순화)
        gamma = float(np.random.uniform(self.mask_ratio_min, 1.0))
        mask_len = torch.tensor([int(np.ceil(seq_len * gamma))], device=device)
        return mask_by_order(mask_len, orders, bsz, seq_len, device).float()

    def fixed_masking(self, bsz: int, seq_len: int, ratio: float, device) -> torch.Tensor:
        """결정론 마스킹(validation 용). γ·순서 무작위성 제거 → epoch 간 비교 가능.
        ratio=1.0 이면 전부 마스킹(순서 무관). ratio<1 이면 마지막 mask_len 토큰 고정 마스킹."""
        mask_len = int(np.ceil(seq_len * ratio))
        mask = torch.zeros(bsz, seq_len, device=device)
        if mask_len > 0:
            mask[:, seq_len - mask_len :] = 1.0
        return mask

    def _build_rope(self, coords):
        """coords: (B, N, 3) -> (cos, sin) 또는 None(non-rope)."""
        if not self.use_rope:
            return None
        from .georope import build_geo_rope
        return build_geo_rope(coords, self.rope_axes_dim, self.rope_theta)

    # ── encoder: 조건 토큰(항상 visible) + visible 미래 토큰 ─────────────
    def forward_encoder(self, cond_emb, future_emb, mask):
        """cond_emb: (B,cond_tok,D) 항상 visible. future_emb: (B,L,D). mask: (B,L) 1=masked.
        반환: encoded (B, cond_tok + n_visible, D), keep, n_vis."""
        B, L, D = future_emb.shape
        # visible 미래 토큰만 모은다 (배치 내 mask 개수 동일 가정 → random_masking 이 보장)
        keep = (1 - mask).bool()  # (B,L) True=visible
        n_vis = int(keep[0].sum().item())
        vis_emb = future_emb[keep].reshape(B, n_vis, D)
        x = torch.cat([cond_emb, vis_emb], dim=1)  # 조건 prefix + visible
        x = self.z_proj_ln(x)
        # GeoRoPE 좌표 동반(§8-2): visible 좌표를 같은 keep 으로 운반
        rope = None
        if self.use_rope:
            fc = self.future_coords.expand(B, L, 3)
            vis_coords = fc[keep].reshape(B, n_vis, 3)
            coords = torch.cat([self.cond_coords.expand(B, self.cond_tok, 3), vis_coords], dim=1)
            rope = self._build_rope(coords)
        for blk in self.encoder_blocks:
            x = blk(x, rope)
        x = self.encoder_norm(x)
        return x, keep, n_vis

    # ── decoder: [MASK] 삽입 후 전체 미래 위치 복원 ──────────────────────
    def forward_decoder(self, encoded, keep, mask):
        """encoded: (B, hw+n_vis, D). 반환 dec_full: (B, cond_tok+L, D) 전체(조건+미래).
        diffusion head 는 future 부분만, causal head 는 cond+future 전체를 읽는다."""
        B = encoded.shape[0]
        L = mask.shape[1]
        D = self.embed_dim
        cond_part = encoded[:, : self.cond_tok]      # 조건 토큰
        vis_part = encoded[:, self.cond_tok :]       # visible 미래 토큰
        # 전체 미래 길이로 scatter: masked=mask_token, visible=encoded
        full = self.mask_token.expand(B, L, D).clone()
        full[keep] = vis_part.reshape(-1, D)
        if not self.use_rope:                        # GeoRoPE 면 가산 위치임베딩 미사용
            full = full + self.future_pos
            cond_part = cond_part + self.cond_pos
        x = torch.cat([cond_part, full], dim=1)
        rope = None
        if self.use_rope:
            coords = torch.cat([self.cond_coords.expand(B, self.cond_tok, 3),
                                self.future_coords.expand(B, L, 3)], dim=1)
            rope = self._build_rope(coords)
        for blk in self.decoder_blocks:
            x = blk(x, rope)
        x = self.decoder_norm(x)
        return x                                      # 전체 (B, cond_tok+L, D)

    # ── seasonality (§C2): IC timestamp -> day-of-year sincos ───────────
    def _season_embed(self, ts):
        """ts: (B, cond_len+T) 정수 YYYYMMDDHH. 예보 원점 t(마지막 history) 기준 연중진행도 (B, D)."""
        ic = ts[:, min(self.cond_len - 1, ts.shape[1] - 1)].float()
        year = torch.floor(ic / 1e6)
        md = ic - year * 1e6
        month = torch.floor(md / 1e4)
        day = torch.floor((md - month * 1e4) / 100)
        frac = (((month - 1) * 30.4 + day) / 365.0).clamp(0, 1)  # 근사 day-of-year
        half = self.embed_dim // 2
        freqs = torch.arange(1, half + 1, device=ts.device).float()
        phase = 2 * torch.pi * frac[:, None]
        emb = torch.cat([torch.sin(phase * freqs), torch.cos(phase * freqs)], dim=1)
        return self.season_mlp(emb[:, : self.embed_dim])

    # ── boundary condition 임베딩 (CFG drop 포함) ───────────────────────
    def _boundary_emb(self, bc, drop: bool = False):
        """bc: (B, cond_len+T, C_in, 180, 360) → e_bc (B, cond_tok+L, D).
        토큰 순서(frame→i→j)가 dec_full=[cond_tok | L] 과 정합.
        drop=True(학습): 표본별 prob 로 e_bc→bc_null (CFG uncond)."""
        e_bc = self.boundary_encoder(bc)                       # (B, cond_tok+L, D)
        if drop and self.bc_drop_prob > 0:
            B = e_bc.shape[0]
            m = (torch.rand(B, 1, 1, device=e_bc.device) < self.bc_drop_prob)
            e_bc = torch.where(m, self.bc_null.to(e_bc.dtype), e_bc)
        return e_bc

    # ── 학습 forward ────────────────────────────────────────────────────
    def forward(self, latents: torch.Tensor, ts: Optional[torch.Tensor] = None,
                mask_ratio: Optional[float] = None, bc: Optional[torch.Tensor] = None,
                return_pred: bool = False, bc_force_drop: bool = False):
        """latents: (B, cond_len+T, Cz, Hz, Wz)  [:cond_len]=조건(history), [cond_len:]=미래.
        ts: (B, cond_len+T) 옵션(계절성). mask_ratio: None=학습(랜덤 γ), 값=결정론 마스킹(val).
        bc: (B, cond_len+T, C_in, 180, 360) 옵션(boundary forcing).
        return_pred=True: det head μ 의 미래 latent 예측을 out["pred"] (B,Cz,T,Hz,Wz) 로 반환
          (val 의 '예보 task' 측정용. mask_ratio=1.0 과 함께 쓰면 IC+BC 만으로 전 미래 예측).
        bc_force_drop=True: BC 를 표본 전체 bc_null(uncond) 로 — BC 기여 격차 측정용.
        반환 dict(loss, diff_loss, [det_loss], [pred])."""
        B = latents.shape[0]
        cond = latents[:, : self.cond_len].permute(0, 2, 1, 3, 4)  # (B,Cz,cond_len,Hz,Wz)
        fut = latents[:, self.cond_len :].permute(0, 2, 1, 3, 4)   # (B,Cz,T,Hz,Wz)

        cond_tokens = self.patchify(cond)                  # (B, cond_tok, td)
        future_tokens = self.patchify(fut)                 # (B, L, td)
        L = future_tokens.shape[1]

        cond_emb = self.cond_proj(cond_tokens)
        if not self.use_rope:
            cond_emb = cond_emb + self.cond_pos
        future_emb = self.z_proj(future_tokens)
        if self.incl_time_elapsed and ts is not None:      # 계절성 가산(보조, §C2)
            se = self._season_embed(ts)[:, None]           # (B,1,D)
            cond_emb = cond_emb + se
            future_emb = future_emb + se

        # ocean-first 학습 마스킹: bc landmask 로 바다 토큰을 order 뒤(먼저 복원)에 배치.
        ot = (self._ocean_tok_from_bc(bc)
              if (self.mask_order == "ocean_first" and self.use_boundary and bc is not None)
              else None)
        mask = (self.random_masking(B, L, latents.device, ocean_tok=ot) if mask_ratio is None
                else self.fixed_masking(B, L, mask_ratio, latents.device))   # (B,L) 1=masked
        encoded, keep, _ = self.forward_encoder(cond_emb, future_emb, mask)
        dec_full = self.forward_decoder(encoded, keep, mask)  # (B, cond_tok+L, D)
        # ── BC 주입(§BC): causal 트렁크에 BC 토큰을 concat 해 self-attention 으로 주입.
        #   encoder/decoder(양방향)는 안 거쳐 미래 BC 역류 없음. frame-causal attn 이
        #   BC[0..t]→state[t] 전파. ──
        e_bc = (self._boundary_emb(bc, drop=self.training)
                if (self.use_boundary and bc is not None) else None)
        if bc_force_drop and e_bc is not None:             # BC off (uncond) — 기여 격차 측정
            e_bc = self.bc_null.to(e_bc.dtype).expand(e_bc.shape[0], e_bc.shape[1], -1)
        causal_full = self._causal_trunk(dec_full, e_bc)   # 공유 causal 트렁크 (B, cond_tok+L, D)
        z = causal_full[:, self.cond_tok :]                # 미래 토큰 부분 (B,L,D) — 두 MLP head 공통 입력
        # ★ head 입력 전 temporal+spatial 절대위치 주입 — MLP head(diffusion/det/σ)는 attention 이
        #   없어 RoPE 를 못 쓰므로, 위치(프레임 t + lat/lon)를 z 에 직접 더해줘야 한다.
        #   future_pos: (1, T*hw, D) 3D sincos. georope 모드에서도 항상 적용(그쪽은 위치가 RoPE 로만
        #   attention 에 들어가 z 에 절대좌표가 안 남으므로 특히 필수).
        z = z + self.future_pos.to(z.dtype)

        # per-token diffusion loss — 마스크된 토큰에만 (§53).
        # ★ 마스크 토큰만 gather 해서 head 에 통과 — visible 토큰은 loss 기여가 0 이라
        #   계산/메모리 낭비. diffusion_batch_mul 와 곱해지는 base 토큰 수를 줄여 OOM 완화.
        z_flat = z.reshape(B * L, self.embed_dim)
        tgt_flat = future_tokens.reshape(B * L, self.token_dim)
        m = mask.reshape(B * L).bool()
        diff_loss = self.diffloss(tgt_flat[m], z_flat[m])

        out = {"diff_loss": diff_loss, "loss": diff_loss}

        # 보조 결정론 MLP head (§C1, §D2) — causal 트렁크 출력 z 를 입력으로 μ(, logvar) 예측.
        if self.det_head is not None:
            from .heads import deterministic_mse_loss, heteroscedastic_loss
            frame_idx = torch.arange(self.T, device=latents.device).repeat_interleave(self.hw)
            frame_idx = frame_idx.repeat(B)
            mu, logvar = self.det_head(z_flat)
            if self.predict_sigma and logvar is not None:
                # ★ heteroscedastic_loss 가 μ·σ 를 모두 학습 — deterministic_mse_loss 를 따로 더하지 않음(중복)
                det_loss = heteroscedastic_loss(mu, logvar, tgt_flat, frame_idx)
            else:
                det_loss = deterministic_mse_loss(mu, tgt_flat, frame_idx,
                                                  weighted=self.det_weighted_loss)
            out["det_loss"] = det_loss
            out["loss"] = diff_loss + det_loss
            if return_pred:                                # det μ → 미래 latent 예보 (val task)
                out["pred"] = self.unpatchify(mu.reshape(B, L, self.token_dim), self.T)

        return out

    # ── 추론: 점진 언마스킹 (§54) ───────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        ic_latent: torch.Tensor,
        num_iter: int = 44,
        temperature: float = 1.3,
        beta: float = 0.0,
        ts: Optional[torch.Tensor] = None,
        bc: Optional[torch.Tensor] = None,
        bc_cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """ic_latent: (B, Cz, cond_len, Hz, Wz) 조건(history). 반환: (B, Cz, T, Hz, Wz) 미래 latent.
        cosine 스케줄, 무작위 순서(β=0=원본 OmniCast), temperature τ.
        β>0 이면 σ 가중 언마스킹(§D3, det_head+predict_sigma 필요).
        bc: (B, cond_len+T, C_in, 180, 360) boundary forcing(perfect-prog).
        bc_cfg_scale: BC classifier-free guidance 세기 w (1.0=guidance 없음)."""
        device = ic_latent.device
        B = ic_latent.shape[0]
        L = self.T * self.hw
        D = self.embed_dim
        # BC 임베딩은 윈도 전체에 대해 1회만 계산(고정 forcing). 트렁크/head 만 2-pass guidance.
        e_bc = self._boundary_emb(bc) if (self.use_boundary and bc is not None) else None

        cond_tokens = self.patchify(ic_latent)
        cond_emb = self.cond_proj(cond_tokens)
        if not self.use_rope:
            cond_emb = cond_emb + self.cond_pos
        se = self._season_embed(ts)[:, None] if (self.incl_time_elapsed and ts is not None) else None
        if se is not None:
            cond_emb = cond_emb + se

        tokens = torch.zeros(B, L, self.token_dim, device=device)
        mask = torch.ones(B, L, device=device)            # 전부 마스크(γ=1) 시작
        orders = sample_orders(B, L, device)

        for step in range(num_iter):
            cur_emb = self.z_proj(tokens)
            if se is not None:
                cur_emb = cur_emb + se
            encoded, keep, _ = self.forward_encoder(cond_emb, cur_emb, mask)
            dec_full = self.forward_decoder(encoded, keep, mask)  # (B, cond_tok+L, D)
            # ── BC 주입(self-attn concat) + CFG: encoder/decoder 는 1회, 트렁크만 2-pass ──
            fpos = self.future_pos.to(dec_full.dtype)
            z_u = None
            if e_bc is not None:
                z = self._causal_trunk(dec_full, e_bc)[:, self.cond_tok :] + fpos
                if bc_cfg_scale != 1.0:
                    e_bc_null = self.bc_null.to(dec_full.dtype).expand(
                        dec_full.shape[0], e_bc.shape[1], -1)
                    z_u = self._causal_trunk(dec_full, e_bc_null)[:, self.cond_tok :] + fpos
            else:
                z = self._causal_trunk(dec_full)[:, self.cond_tok :] + fpos

            # cosine 스케줄: 다음 스텝까지 남길 마스크 개수
            ratio = float(np.cos(np.pi / 2.0 * (step + 1) / num_iter))
            mask_len = torch.tensor([int(np.floor(L * ratio))], device=device)
            mask_len = torch.clamp(mask_len, min=1 if step < num_iter - 1 else 0,
                                   max=int(mask.sum(dim=1).min().item()) - 1 if step < num_iter - 1 else L)

            mask_next = mask_by_order(mask_len, orders, B, L, device).float()
            if step >= num_iter - 1:
                mask_next = torch.zeros_like(mask)
            # 이번에 새로 언마스킹할 토큰 = 현재 masked & 다음에 visible
            to_pred = torch.logical_and(mask.bool(), ~mask_next.bool())  # (B,L)

            # σ 가중 선택(§D3): β>0 이면 σ 낮은(확실한) 토큰일수록 높은 확률로 뽑되, 가끔
            # 불확실한 토큰도 열리는 '확률적 비복원 샘플링'. 확률 P ∝ exp(β·(−σ)) = softmax(β·(−σ)),
            # β 는 분포 날카로움(temperature) 하이퍼파라미터(β↑ 결정론에 가까움, β↓ 무작위에 가까움).
            # (β=0 이면 이 블록을 안 타고 orders 기반 순수 random — baseline 일치, §8-8.)
            if beta > 0 and self.det_head is not None and self.predict_sigma:
                _, logvar = self.det_head(z.reshape(B * L, D))
                logvar = logvar.reshape(B, L, self.token_dim).mean(-1)   # (B,L)
                sigma = torch.exp(0.5 * logvar)
                # visible 토큰은 −inf → softmax 후 확률 0 (후보에서 제외)
                logits = (beta * (-sigma)).masked_fill(~mask.bool(), float("-inf"))
                probs = torch.softmax(logits, dim=1)                    # (B,L), P ∝ exp(β·(−σ))
                # ★ random_masking 이 배치 공통 단일 γ 를 쓰므로 행마다 마스크(후보) 수가 동일 →
                #   언마스킹 개수 k 도 모든 행에서 같다. multinomial(replacement=False) 은 행마다
                #   nonzero 확률 ≥ k 를 요구하므로, 안전을 위해 후보(=masked) 최소 개수로 k 를 클램프.
                k = int(to_pred[0].sum().item())
                k = min(k, int(mask.sum(dim=1).min().item()))
                if k > 0:
                    # σ 가중 확률로 k 개를 비복원 추출 (top-k 결정론 ✗ → β 가 실제로 작동)
                    # .float(): autocast(bf16) 추론에서 multinomial 입력 dtype 안전 확보
                    sel = torch.multinomial(probs.float(), k, replacement=False)  # (B,k)
                    to_pred = torch.zeros_like(to_pred)
                    to_pred.scatter_(1, sel, True)
                    mask_next = torch.logical_and(mask.bool(), ~to_pred).float()

            # diffusion head 로 to_pred 토큰 샘플
            idx = to_pred.reshape(-1).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                z_sel = z.reshape(B * L, D)[idx]
                zu_sel = z_u.reshape(B * L, D)[idx] if z_u is not None else None
                sampled = self.diffloss.sample(z_sel, temperature=temperature,
                                               z_uncond=zu_sel, cfg_scale=bc_cfg_scale)
                tokens = tokens.reshape(B * L, self.token_dim)
                tokens[idx] = sampled.to(tokens.dtype)
                tokens = tokens.reshape(B, L, self.token_dim)

            mask = mask_next

        return self.unpatchify(tokens, self.T)
