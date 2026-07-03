"""S2S 패키지에서 옮겨온 유틸 (S2S_inject 자립용).
원본: S2S/eval_leadtime.py(lat_weights,lw_mean), S2S/data/s2s_dataset.py(load_latent_stats),
      S2S/plot_korea_forecast.py(region_aavg).
"""
import json
from typing import Tuple

import numpy as np
import torch


def lat_weights(lat_deg: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat_deg.astype(np.float64)))
    w = w / w.mean()
    return w[None, None, :, None]


def lw_mean(x: np.ndarray, lw: np.ndarray) -> np.ndarray:
    """x:(T,C,H,W) → (T,C) lat-weighted 격자 평균."""
    return (x * lw).mean(axis=(-1, -2))


def load_latent_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """era5_latent_stats.json → (mean(C,), std(C,))."""
    with open(stats_path, "r") as f:
        d = json.load(f)
    mean = torch.tensor(d["mean"], dtype=torch.float32)
    std = torch.tensor(d["std"], dtype=torch.float32)
    return mean, std


def region_aavg(fields, lat, lon, lat_lo, lat_hi, lon_lo, lon_hi):
    """fields:(...,H,W) → (...,) 박스 cos-lat 가중 area-average (lon 평균 후 lat cos 가중)."""
    la = np.where((lat >= lat_lo) & (lat <= lat_hi))[0]
    lo = np.where((lon >= lon_lo) & (lon <= lon_hi))[0]
    sub = fields[..., la[:, None], lo[None, :]]
    wlat = np.cos(np.deg2rad(lat[la].astype(np.float64)))
    zonal = sub.mean(axis=-1)
    return (zonal * wlat).sum(axis=-1) / wlat.sum()
