"""
DCAE(Stage 0/2) 용 raw field 데이터셋 (s2s_instruction.md §B3).

LaDCast2 `dataloader/era5_field_dataset.py` 를 이식. `era5_00utc.zarr`
(time, 9, 180, 360, 물리단위) 에서 프레임별 (C, H, W) 를 읽어 per-channel mean/std
로 표준화해 반환한다. DCAE 는 단일 프레임 autoencoder 이므로 시간축 없이 학습.
"""

import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


def load_field_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    with open(stats_path, "r") as f:
        d = json.load(f)
    return (torch.tensor(d["mean"], dtype=torch.float32),
            torch.tensor(d["std"], dtype=torch.float32),
            d.get("channel_names"))


class Era5FieldDataset(Dataset):
    """반환: (field (C,H,W) 표준화, timestamp int YYYYMMDDHH)."""

    def __init__(self, zarr_path, var, stats_path, start_date, end_date,
                 load_in_memory=False):
        ds = xr.open_zarr(zarr_path)
        self.da = ds[var].sel(time=slice(start_date, end_date))
        self.time = self.da["time"].values
        mean, std, _ = load_field_stats(stats_path)
        self.mean, self.std = mean[:, None, None], std[:, None, None]
        self.ts = np.array(
            [pd.Timestamp(t).year * 1000000 + pd.Timestamp(t).month * 10000
             + pd.Timestamp(t).day * 100 + pd.Timestamp(t).hour for t in self.time],
            dtype=np.int64,
        )
        if load_in_memory:
            arr = self.da.values.astype(np.float32)
            arr = (arr - self.mean.numpy()[None]) / self.std.numpy()[None]
            self.data = torch.from_numpy(arr)
            print(f"[Era5FieldDataset] in-memory {tuple(self.data.shape)} "
                  f"(~{self.data.numel()*4/1e9:.1f} GB)")
        else:
            self.data = None

    def __len__(self):
        return self.da.sizes["time"]

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx], self.ts[idx]
        x = torch.from_numpy(self.da.isel(time=idx).values.astype(np.float32))
        return (x - self.mean) / self.std, self.ts[idx]


def prepare_field_dataloader(zarr_path, var, stats_path, start_date, end_date,
                             batch_size=16, shuffle=True, num_workers=8,
                             load_in_memory=False, drop_last=True):
    ds = Era5FieldDataset(zarr_path, var, stats_path, start_date, end_date, load_in_memory)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None,
                      pin_memory=True, drop_last=drop_last)
