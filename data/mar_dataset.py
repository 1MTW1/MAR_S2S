"""S2S_inject 데이터셋 — direct 주입 전용 (대기 latent + 결합 skt 필드 1°).

S2SInjectDirectDataset: 결합 skt(180,360) = where(open-ocean, BC소스, ERA5 skt z) → 18×18 패치.
  ocean_source:
    skt      — open-ocean = ERA5 skt z (GT BC, MAR 학습용·천장)
    oisst    — open-ocean = OISST 1° anomaly z
    refined  — open-ocean = recon(GT OISST) z  (data/skt1deg_refined_unet.zarr)
    forecast — IC 프레임 = refined(관측), 미래 프레임 = transfer 예보 (skt1deg_forecast_bc*.zarr,
               __getitem__ 에서 IC 별 오버레이) — 배포형(미래 GT 미사용)
  mask_seaice: OISST 연중 최저 clim ≤ ice_thresh 해빙권을 visible 에서 제외(생성 대상化).
  반환: latents(window,Cz,h,w), ts, skt_p(window,h,w,P²) ×2(in=tgt), ocean_tok(h,w).
"""

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader, Dataset


def load_latent_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(stats_path) as f:
        d = json.load(f)
    return (torch.tensor(d["mean"], dtype=torch.float32),
            torch.tensor(d["std"], dtype=torch.float32))


def _ts_to_int(t) -> int:
    dt = pd.Timestamp(t)
    return dt.year * 1000000 + dt.month * 10000 + dt.day * 100 + dt.hour


class S2SInjectDirectDataset(Dataset):
    """★ OISST 1° ocean-mean **직접주입**(인코더 없음) 데이터셋.

    결합 skt 필드(180,360) = where(OISST ocean_mask, OISST_1°_anom_z, ERA5_skt_anom_z):
      · ocean 셀(OISST 4×4 ≥min_ocean) = OISST native anomaly z (visible 입력, 예측 안 함)
      · land  셀                        = ERA5 skt anomaly z (=LST; masked→생성 타깃)
    ocean_tok(h,w) = OISST ocean_mask 18×18 블록 ocean비율 ≥ ocean_thresh.

    반환: latents, ts, skt(결합,window,h,w,P²), skt(동일; in=tgt), ocean_tok.
      (해양=visible 입력, 육지 미래=masked 예측 → skt_in=skt_tgt=결합필드로 충분)
    """

    def __init__(
        self, latents_zarr, var, latent_stats, start_date, end_date,
        oisst_1deg_zarr, oisst_1deg_clim_npz, skt_anomaly_zarr,
        future_len=14, input_len=1, stride=1, target_std=None, load_in_memory=True,
        skt_clip=5.0, skt_patch=18, ocean_thresh=0.5, ocean_source="oisst",
        mask_seaice=False, ice_thresh=-1.7, refined_zarr=None, forecast_zarr=None,
    ):
        assert ocean_source in ("oisst", "skt", "refined", "forecast"), ocean_source
        ds = xr.open_zarr(latents_zarr)
        self.da = ds[var].sel(time=slice(start_date, end_date))
        self.time = self.da["time"].values
        self.input_len, self.future_len = input_len, future_len
        self.window = input_len + future_len
        self.stride = stride
        self.target_std = target_std
        self.skt_clip = skt_clip
        self.P = skt_patch

        mean, std = load_latent_stats(latent_stats)
        self.mean, self.std = mean[:, None, None], std[:, None, None]
        self.ts = np.array([_ts_to_int(t) for t in self.time], dtype=np.int64)
        n = self.da.sizes["time"]
        self.starts = list(range(0, n - self.window + 1, stride))
        if not self.starts:
            raise ValueError(f"시퀀스 길이 {self.window} 보다 구간이 짧음 (n={n}).")

        # ── 대기 latent (표준화, 메모리) ──
        if load_in_memory:
            arr = (self.da.values.astype(np.float32) - self.mean.numpy()[None]) / self.std.numpy()[None]
            if target_std is not None:
                arr = arr * target_std
            self.data = torch.from_numpy(arr)
            print(f"[DirectDS] latents in-memory {tuple(self.data.shape)} "
                  f"(~{self.data.numel()*4/1e9:.2f} GB), windows={len(self.starts)}")
        else:
            self.data = None

        # ── ERA5 skt anomaly z (land 타깃) ──
        az = xr.open_zarr(skt_anomaly_zarr)
        era = az["anomaly"].reindex(time=self.time).values.astype(np.float32)
        if np.isnan(era).any():
            raise ValueError("ERA5 skt anomaly NaN — skt_anomaly_zarr 시간범위 부족.")
        era = np.clip(era, -skt_clip, skt_clip)
        self.H, self.W = era.shape[1:]
        self.h, self.w = self.H // self.P, self.W // self.P
        assert self.H % self.P == 0 and self.W % self.P == 0, "패치 정수배 아님"

        # ── ocean_mask (OISST 1°) + 해빙 마스크(OISST 기준) ──
        od = xr.open_zarr(oisst_1deg_zarr)
        ocean_mask = (od["ocean_mask"].values > 0)                            # (180,360) bool
        doy = np.array([min(pd.Timestamp(t).dayofyear, 366) - 1 for t in self.time])
        cl = np.load(oisst_1deg_clim_npz)
        cmu, csig = cl["c_mu"].astype(np.float32), cl["c_sig"].astype(np.float32)
        # ★ 해빙: OISST 연중 최저 climatology SST ≤ ice_thresh → 결빙권(개빙 아님). visible BC 에서 제외.
        if mask_seaice:
            ice = (np.nanmin(cmu, axis=0) <= ice_thresh) & ocean_mask         # (180,360) 해빙권 ocean
            vis = ocean_mask & ~ice                                          # open-ocean = visible BC
        else:
            ice = np.zeros_like(ocean_mask); vis = ocean_mask

        if ocean_source == "oisst":
            ot = od["time"].values
            if self.time[0] < ot[0] or self.time[-1] > ot[-1]:
                raise ValueError(f"OISST 1° 시간범위 {str(ot[0])[:10]}~{str(ot[-1])[:10]} 부족.")
            osst = od["sst"].reindex(time=self.time).values.astype(np.float32)   # (N,180,360) °C, land NaN
            z_o = np.clip((osst - cmu[doy]) / np.clip(csig[doy], 1e-6, None), -skt_clip, skt_clip)
            comb = np.where(vis[None], np.nan_to_num(z_o, nan=0.0), era)      # open-ocean=OISST / (해빙+land)=ERA5(생성)
            src = "OISST1°"
        elif ocean_source in ("refined", "forecast"):
            # refined: 전 프레임 recon(GT OISST) z (make_refined_bc.py)
            # forecast: IC 프레임=refined(관측), 미래 프레임=transfer 예보(__getitem__ 에서 덮어씀)
            assert refined_zarr, f"ocean_source={ocean_source} 는 refined_zarr 필요"
            rz = xr.open_zarr(refined_zarr)["z"].reindex(time=self.time).values.astype(np.float32)
            z_r = np.clip(np.nan_to_num(rz, nan=0.0), -skt_clip, skt_clip)
            comb = np.where(vis[None], z_r, era)
            src = "refined(OISST→UNet)"
            if ocean_source == "forecast":       # ★ 배포형: 미래 open-ocean = transfer(B) 예보 (미래 GT 미사용)
                assert forecast_zarr, "ocean_source=forecast 는 forecast_zarr 필요"
                fz = xr.open_zarr(forecast_zarr)
                assert fz.sizes["lead"] >= future_len, "forecast lead 부족"
                ft = {np.datetime64(t): i for i, t in enumerate(fz["time"].values)}
                self.fc_idx = np.array([ft.get(np.datetime64(t), -1) for t in self.time])
                fc = fz["z"].values.astype(np.float32)[:, :future_len]         # (Nic,T,180,360)
                self.fc = torch.from_numpy(np.clip(np.nan_to_num(fc, nan=0.0), -skt_clip, skt_clip))
                self.vis_t = torch.from_numpy(vis)
                src = "forecast(transfer B)"
        else:                                    # ocean_source == "skt": open-ocean 도 ERA5 skt
            comb = era
            src = "ERA5-skt"

        self.skt = torch.from_numpy(comb.astype(np.float32))                  # (N,180,360)
        print(f"[DirectDS] combined skt (open-ocean={src} / 해빙+land=ERA5생성) {tuple(self.skt.shape)}, "
              f"ocean_source={ocean_source}, mask_seaice={mask_seaice}")

        # ── ocean_tok(visible) = open-ocean 18×18 블록 비율 ≥ thresh ──
        ofrac = vis.astype(np.float32).reshape(self.h, self.P, self.w, self.P).mean(axis=(1, 3))
        self.ocean_tok = torch.from_numpy(ofrac >= ocean_thresh)              # (h,w) bool (해빙 제외 open-ocean)
        print(f"[DirectDS] open-ocean tok {int(self.ocean_tok.sum())}/{self.h*self.w} "
              f"(ocean {ocean_mask.mean():.3f}, 해빙 {float(ice.mean()):.3f}, open {vis.mean():.3f})")

    def __len__(self):
        return len(self.starts)

    def _patchify(self, x):
        win = x.shape[0]
        x = x.reshape(win, self.h, self.P, self.w, self.P)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.reshape(win, self.h, self.w, self.P * self.P)

    def __getitem__(self, idx):
        s = self.starts[idx]
        sl = slice(s, s + self.window)
        if self.data is not None:
            latents = self.data[sl]
        else:
            x = self.da.isel(time=sl).values.astype(np.float32)
            x = (x - self.mean.numpy()) / self.std.numpy()
            latents = torch.from_numpy(x * self.target_std if self.target_std is not None else x)
        ts = torch.from_numpy(self.ts[sl])
        skt_w = self.skt[sl]
        if getattr(self, "fc", None) is not None:                             # ★ forecast: 미래 open-ocean 교체
            j = int(self.fc_idx[s + self.input_len - 1])
            assert j >= 0, f"IC {self.time[s + self.input_len - 1]} 의 forecast BC 없음"
            skt_w = skt_w.clone()
            skt_w[self.input_len:] = torch.where(self.vis_t[None], self.fc[j], skt_w[self.input_len:])
        skt_p = self._patchify(skt_w)                                         # (window,h,w,P²)
        return latents, ts, skt_p, skt_p, self.ocean_tok


def prepare_inject_direct_dataloader(
    latents_zarr, var, latent_stats, start_date, end_date,
    oisst_1deg_zarr, oisst_1deg_clim_npz, skt_anomaly_zarr,
    future_len=14, input_len=1, stride=1, target_std=None,
    batch_size=8, shuffle=True, num_workers=8, load_in_memory=True, drop_last=True,
    skt_clip=5.0, skt_patch=18, ocean_thresh=0.5, ocean_source="oisst",
    mask_seaice=False, ice_thresh=-1.7, refined_zarr=None, forecast_zarr=None,
) -> DataLoader:
    ds = S2SInjectDirectDataset(
        latents_zarr, var, latent_stats, start_date, end_date,
        oisst_1deg_zarr, oisst_1deg_clim_npz, skt_anomaly_zarr,
        future_len=future_len, input_len=input_len, stride=stride, target_std=target_std,
        load_in_memory=load_in_memory, skt_clip=skt_clip, skt_patch=skt_patch,
        ocean_thresh=ocean_thresh, ocean_source=ocean_source,
        mask_seaice=mask_seaice, ice_thresh=ice_thresh, refined_zarr=refined_zarr,
        forecast_zarr=forecast_zarr,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None,
                      pin_memory=True, drop_last=drop_last)
