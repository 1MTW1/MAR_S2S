"""
SST_swin/dataset.py — ERA5 raw skin temperature 로딩·정규화·land0·ocean mask·슬라이딩 윈도우 (instruction §2).

데이터(단일 소스):
  · raw skt(Kelvin) : era5_forcing_00utc.zarr  ["forcing"].sel(channel="skt")  (16071, 180, 360)
  · land-sea mask   : era5_forcing_00utc.zarr  ["land_sea_mask"]               (180, 360, 1=land)
  → anomaly·climatology 는 입력경로에서 쓰지 않는다(평가 ACC 에만 별도 사용).

정규화(§2.2): ★ocean 픽셀만으로 계산한 전역 μ,σ(학습기간 고정 스칼라) z-score → land 픽셀 0 할당(정규화 '후').
  (doy climatology 제거 아님 — 식 7 그대로.)
격자: native(H,W)에서 auto_pad 로 Swin 호환 (Hp,Wp) 자동. 0.25°(720×1440,window5)=pad 0(무패딩).
  (구면 pad 미사용 — relative position bias + 경도-roll 증강으로 경도주기성 처리.) pad·land 는 mask 0 으로 loss 제외.
출력: x(과거 in_len일), y(미래 out_len일) 각각 (T, Hp, Wp).
"""
from __future__ import annotations

import os
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

# ── doy 인덱싱 (366 slot, 윤년 2000 기준 ordinal — compute_climatology 와 공유) ──
_N_SLOTS = 366
_BASE_ORD = date(2000, 1, 1).toordinal()
_FEB29_SLOT = date(2000, 2, 29).toordinal() - _BASE_ORD          # = 59


def doy_slot(times) -> np.ndarray:
    """datetime64(배열/스칼라) → 366-slot doy 인덱스 ∈[0,365]. (month,day)→ordinal(2000). Feb29=59."""
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    memo: dict = {}
    out = np.empty(len(idx), np.int64)
    for i, (m, d) in enumerate(zip(idx.month, idx.day)):
        k = (int(m), int(d))
        s = memo.get(k)
        if s is None:
            s = date(2000, k[0], k[1]).toordinal() - _BASE_ORD; memo[k] = s
        out[i] = s
    return out

def auto_pad(H: int, W: int, patch: int, window: int, n_down: int):
    """native (H,W) → Swin U-Net 호환 (Hp,Wp) 대칭 pad 계산.
    제약: Hp/patch, /2, …(n_down-1회) 가 모두 window 배수·정수 → Hp 가 patch·window·2^(n_down-1) 배수.
    반환 (pad=((t,b),(l,r)), Hp, Wp). 이미 호환이면 pad=0."""
    m = patch * window * (2 ** (n_down - 1))
    Hp = ((H + m - 1) // m) * m
    Wp = ((W + m - 1) // m) * m
    ph, pw = Hp - H, Wp - W
    return ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), Hp, Wp


def _is_oisst(fds) -> bool:
    """데이터 소스 자동감지: OISST(var 'sst' + 'lsmask' 1=ocean, °C, land=NaN) vs ERA5(forcing[skt]+land_sea_mask)."""
    return ("sst" in fds and "lsmask" in fds)


def _field_ocean_lat(fds, start=None, end=None):
    """(da:(time,H,W) lazy, ocean:(H,W) bool, lat:(H,)) — OISST/ERA5 공용. start/end 있으면 시간 슬라이스."""
    if _is_oisst(fds):
        ocean = (fds["lsmask"].values == 1)                        # OISST 규약 1=ocean
        da = fds["sst"]                                            # °C, land=NaN
    else:
        ocean = (fds["land_sea_mask"].values < 0.5)               # ERA5 규약 1=land → ocean=<0.5
        da = fds["forcing"].sel(channel="skt")                     # Kelvin
    if start is not None:
        da = da.sel(time=slice(start, end))
    return da, ocean, fds["lat"].values.astype(np.float32)


def compute_grid_pad(forcing_zarr: str, patch: int, window: int, n_down: int):
    """forcing zarr 의 native (H,W) → (pad, (Hp,Wp), (H,W)). train/eval 이 동일 격자 쓰도록 1곳서 계산."""
    fds = xr.open_zarr(forcing_zarr)
    mask_var = "lsmask" if _is_oisst(fds) else "land_sea_mask"
    H, W = fds[mask_var].shape
    pad, Hp, Wp = auto_pad(int(H), int(W), patch, window, n_down)
    return pad, (Hp, Wp), (int(H), int(W))


def _zero_pad(a: np.ndarray, pad) -> np.ndarray:
    """(...,H,W) → (...,Hp,Wp) zero-pad (pad 영역은 loss mask 0 으로 제외).
    0.25°(720×1440, window5)는 pad=0 → no-op. (구면 pad 미사용: relative position bias + 경도-roll 증강으로 처리.)"""
    p = [(0, 0)] * (a.ndim - 2) + [pad[0], pad[1]]
    return np.pad(a, p, mode="constant", constant_values=0.0)


def _open_field(forcing_zarr, start, end):
    """반환 (da:(time,H,W) ★lazy raw skt DataArray, ocean(H,W) bool, time).
    모든 타깃이 raw skt 를 읽고, anomaly/residual 은 climatology 로 _norm 에서 표준화한다(별도 anomaly zarr 불필요)."""
    fds = xr.open_zarr(forcing_zarr)
    da, ocean, _ = _field_ocean_lat(fds, start, end)               # OISST(sst)/ERA5(forcing[skt]) 자동
    return da, ocean, da["time"].values


def build_static(forcing_zarr: str, pad) -> Tuple[torch.Tensor, torch.Tensor]:
    """ocean_mask(Hp,Wp) {0,1}, lat_weight(Hp,) 반환 (pad 영역 0). pad=((t,b),(l,r))."""
    fds = xr.open_zarr(forcing_zarr)
    _, ocean_b, lat = _field_ocean_lat(fds)                          # OISST/ERA5 자동
    ocean = ocean_b.astype(np.float32)                               # (H,W) ocean=1
    latw = np.cos(np.deg2rad(lat)).clip(min=0)
    ocean_p = _zero_pad(ocean, pad)                                  # ★ mask 는 zero-pad (loss 에서 pad 제외)
    Hp = len(lat) + pad[0][0] + pad[0][1]
    latw_p = np.zeros(Hp, dtype=np.float32); latw_p[pad[0][0]:pad[0][0] + len(lat)] = latw
    return torch.from_numpy(ocean_p), torch.from_numpy(latw_p)


def compute_global_ocean_stats(forcing_zarr, train_start, train_end,
                               cache="SST_swin/skt_global_stats.npz") -> Tuple[float, float]:
    """학습기간 raw skt 의 ★ocean 픽셀만으로 전역 μ,σ(스칼라). ★시간 청크 스트리밍(0.25° 대용량 대응). 캐시 재사용."""
    if os.path.exists(cache):
        d = np.load(cache); return float(d["mu"]), float(d["sigma"])
    da, ocean, _ = _open_field(forcing_zarr, train_start, train_end)
    n = s = ss = 0.0
    for t0 in range(0, da.sizes["time"], 512):                      # 시간 청크로 누적(메모리 절약)
        v = da.isel(time=slice(t0, t0 + 512)).values.astype(np.float32)[:, ocean]   # ocean 픽셀만
        n += v.size; s += float(v.sum()); ss += float((v.astype(np.float64) ** 2).sum())
    mu = s / n; sigma = float(np.sqrt(ss / n - mu ** 2))
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, mu=mu, sigma=sigma)
    print(f"[stats] global OCEAN-only μ={mu:.3f}K σ={sigma:.3f}K (n={int(n)}) → {cache}")
    return mu, sigma


class S2SSwinDataset(Dataset):
    """과거 in_len일 → 미래 out_len일 OISST. 반환 x,y: (T,Hp,Wp) — **anomaly-persistence residual 전용**.
    입출력 = anomaly z = clip((skt−c_mu[doy])/c_sig[doy], ±clip), land=0.
    persistence(a_IC) skip 은 train/eval 의 predict() 에서 처리(여기선 anomaly z 데이터만 제공)."""

    def __init__(self, forcing_zarr, start, end, pad,
                 in_len=14, out_len=14, stride=1, clim_npz=None,
                 augment=False, load_in_memory=True, skt_clip=5.0, return_scale=False):
        assert clim_npz, "residual 타깃은 clim_npz(per-doy c_mu/c_sig) 필요"
        self.in_len, self.out_len, self.stride = in_len, out_len, stride
        self.augment = augment                                  # ★ 경도-roll 증강(train only)
        self.return_scale = return_scale                        # ★ 타깃프레임 물리 std(c_sig[doy]) 반환(val K RMSE)
        self.pad = pad                                          # ((t,b),(l,r))
        self.da, self.ocean, self.time = _open_field(forcing_zarr, start, end)   # 항상 raw skt
        self._mem = self.da.values.astype(np.float32) if load_in_memory else None
        self.H, self.W = self.ocean.shape
        cl = np.load(clim_npz)
        self.c_mu = cl["c_mu"].astype(np.float32); self.c_sig = cl["c_sig"].astype(np.float32)
        self.slots = doy_slot(self.time); self.clip = skt_clip
        n = len(self.time)
        self.starts = list(range(0, n - (in_len + out_len) + 1, stride))
        Hp = self.H + pad[0][0] + pad[0][1]; Wp = self.W + pad[1][0] + pad[1][1]
        print(f"[S2SSwinDataset] {start}~{end} (residual, mem={load_in_memory}): frames {n}, "
              f"windows {len(self.starts)} (in {in_len}/out {out_len}), grid→({Hp},{Wp})")

    def __len__(self):
        return len(self.starts)

    def _slab(self, sl: slice) -> np.ndarray:
        """시간 슬라이스 (len,H,W) — in-memory 면 배열, 아니면 zarr 에서 lazy read."""
        return (self._mem[sl] if self._mem is not None
                else self.da.isel(time=sl).values).astype(np.float32)

    def _norm(self, sl: slice, k: int = 0) -> np.ndarray:
        """raw skt → anomaly z = clip((skt−c_mu[doy])/c_sig[doy], ±clip) → land0 → (roll) → zero-pad."""
        raw = self._slab(sl)
        ss = self.slots[sl]
        z = np.clip((raw - self.c_mu[ss]) / self.c_sig[ss], -self.clip, self.clip)
        z = z.copy(); z[:, ~self.ocean] = 0.0                        # ★ 표준화 후 land=0
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)        # OISST land=NaN·clim NaN 방어(마스킹됨)
        if k:
            z = np.roll(z, k, axis=-1)                               # 경도 roll (land0 도 함께 이동)
        return _zero_pad(z, self.pad)                                # 0.25° 는 pad 0 → no-op

    def _ocean_pad(self, k: int = 0) -> np.ndarray:
        """ocean mask (k 경도roll) → zero-pad. (Hp,Wp). roll 시 field 와 정합 위해 마스크도 같이 이동."""
        o = self.ocean.astype(np.float32)
        if k:
            o = np.roll(o, k, axis=-1)
        return _zero_pad(o, self.pad)

    def _scale(self, sl: slice, k: int) -> np.ndarray:
        """타깃 프레임의 물리 std (z→K 복원용) = c_sig[doy]. (len,Hp,Wp)."""
        sc = self.c_sig[self.slots[sl]]                             # (len,H,W)
        sc = np.nan_to_num(sc, nan=0.0)                              # OISST land c_sig=NaN → 0 (val K RMSE 마스킹됨)
        if k:
            sc = np.roll(sc, k, axis=-1)
        return _zero_pad(sc, self.pad)

    def __getitem__(self, idx):
        s = self.starts[idx]
        k = int(np.random.randint(self.W)) if self.augment else 0    # 경도 shift(0~W-1)
        out_sl = slice(s + self.in_len, s + self.in_len + self.out_len)
        x = self._norm(slice(s, s + self.in_len), k)                 # (in_len,Hp,Wp)
        y = self._norm(out_sl, k)
        m = self._ocean_pad(k)                                       # (Hp,Wp) sample별 ocean mask
        if not self.return_scale:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m),
                torch.from_numpy(self._scale(out_sl, k)))            # +물리 std (val)
