"""
GeoRoPE — 3D 회전 위치 임베딩 (s2s_instruction.md §B4, Stage 5).

LaDCast2 `LaDCastRotaryPosEmbed_from_grid`(temporal, lat, lon) 의 정신을 잇되,
MAR 마스킹과 정합하도록 **격자 좌표(t,lat,lon)에 묶인** 형태로 재구성했다(§8-2).
토큰 shuffle 시 좌표를 같은 permutation 으로 함께 운반하므로(§B4), RoPE 가
"sequence position" 이 아니라 "격자 좌표"에 적용된다.

좌표 텐서 coords: (..., 3) = (t, lat_rad, lon_rad). 축별 차원 rope_axes_dim
(합 = head_dim, 각 짝수). GPT-NeoX 식 rotate_half 적용.
"""

from typing import List

import torch


def _axis_angles(coord: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """coord: (...,) -> (..., dim/2) 각도. dim 짝수."""
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=coord.device).float() / half))
    return coord[..., None].float() * inv_freq  # (..., dim/2)


def build_geo_rope(coords: torch.Tensor, rope_axes_dim: List[int], theta: float = 256.0):
    """coords: (B, N, 3) (t, lat_rad, lon_rad). 반환 cos,sin: (B, N, head_dim).
    head_dim = sum(rope_axes_dim)."""
    angles = []
    for a in range(3):
        angles.append(_axis_angles(coords[..., a], rope_axes_dim[a], theta))
    ang = torch.cat(angles, dim=-1)              # (B, N, head_dim/2)
    emb = torch.cat([ang, ang], dim=-1)          # (B, N, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_geo_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: (B, heads, N, head_dim). cos,sin: (B, N, head_dim) -> 회전된 q,k."""
    cos = cos[:, None].to(q.dtype)               # (B,1,N,hd) 헤드 broadcast
    sin = sin[:, None].to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def make_coord_table(T: int, h: int, w: int,
                     lat_start: float, lat_end: float,
                     lon_start: float, lon_end: float,
                     deg2rad: bool = True) -> torch.Tensor:
    """(T, h, w) 격자의 좌표 테이블 (T*h*w, 3). t=프레임 인덱스, lat/lon=격자(라디안).
    IC 프레임은 t=0, 미래는 1..T 로 호출측에서 슬라이싱(여기선 0..T-1 생성)."""
    import math
    ts = torch.arange(T).float()
    lat = torch.linspace(lat_start, lat_end, h)
    lon = torch.linspace(lon_start, lon_end, w)
    if deg2rad:
        lat = lat * math.pi / 180.0
        lon = lon * math.pi / 180.0
    gt, gh, gw = torch.meshgrid(ts, lat, lon, indexing="ij")
    return torch.stack([gt.reshape(-1), gh.reshape(-1), gw.reshape(-1)], dim=-1)  # (T*h*w,3)
