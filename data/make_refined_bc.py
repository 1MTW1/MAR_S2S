"""OISST 0.25° → recon UNet(refine) → SKT-ocean anomaly 1° BC 사전계산.

파이프라인 (배포형 ② BC):
  OISST 0.25° raw → OISST clim z → UNet(sst_recon, same-time refine) → SKT 0.25° z
  → ×ERA5 0.25° c_sig = 물리 anomaly K → 4×4 ERA5-ocean-mean pool → 1° anomaly K
  → ÷ERA5 1° c_sig = 1° z  (S2SInjectDirectDataset 의 ocean_source="skt" 와 같은 정의역)

실행:
  uv run python -m baseline.data.make_refined_bc \
      --ckpt sst_recon/out_big/ckpt.pt --out data/skt1deg_refined_unet.zarr
검증(2022, vis 셀 corr: refined vs identity):
  uv run python -m baseline.data.make_refined_bc --check-only --out data/skt1deg_refined_unet.zarr
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
import torch
import xarray as xr

import dask
import numcodecs.blosc as _blosc
_blosc.use_threads = False
dask.config.set(scheduler="synchronous")

from baseline.recon import UNet

_BASE = date(2000, 1, 1).toordinal()


def _doy(times):
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    return np.array([date(2000, m, d).toordinal() - _BASE for m, d in zip(idx.month, idx.day)], np.int64)


def build(args, device):
    oz = xr.open_zarr(args.oisst_zarr)
    sst = oz["sst"].sel(time=slice(args.start, args.end))     # ★ isel 은 이 sliced 축 기준
    times = sst["time"].values
    doy = _doy(times)                                          # 0.25° clim (recon 컨벤션: 2000-ordinal)
    doy1 = np.array([min(pd.Timestamp(t).dayofyear, 366) - 1 for t in times])  # 1° clim (dataset 컨벤션)

    oc = np.load(args.oisst_clim)
    ocmu = np.nan_to_num(oc["c_mu"].astype(np.float32))
    ocsig = np.clip(np.nan_to_num(oc["c_sig"].astype(np.float32), nan=1.0), 1e-6, None)
    ec = np.load(args.era5_clim_025)
    ecsig = np.nan_to_num(ec["c_sig"].astype(np.float32))            # land NaN→0 (ocean 밖 기여 차단)
    ez = xr.open_zarr(args.era5_zarr_025)
    ocean_e = (ez["land_sea_mask"].values < 0.5).astype(np.float32)  # (720,1440)
    c1 = np.load(args.era5_clim_1deg)
    csig1 = np.clip(c1["c_sig"].astype(np.float32), 1e-6, None)      # (366,180,360)

    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model = UNet(base=sd["e0.c1.weight"].shape[0]).to(device).eval()
    model.load_state_dict(sd)

    oce_t = torch.from_numpy(ocean_e).to(device)
    den = oce_t.reshape(180, 4, 360, 4).sum((1, 3))                  # (180,360) 블록 내 ERA5-ocean 셀수
    print(f"[refine] {str(times[0])[:10]}~{str(times[-1])[:10]} {len(times)}d, "
          f"UNet base={sd['e0.c1.weight'].shape[0]}, ocean-empty 1° cells={(den == 0).sum().item()}")

    out = np.full((len(times), 180, 360), np.nan, dtype=np.float32)
    B = args.batch
    for s in range(0, len(times), B):
        sl = slice(s, min(s + B, len(times)))
        raw = sst.isel(time=sl).values.astype(np.float32)
        d = doy[sl]
        oa = np.clip((np.nan_to_num(raw) - ocmu[d]) / ocsig[d], -args.clip, args.clip)
        oa_t = torch.from_numpy(np.nan_to_num(oa)).to(device) * oce_t
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pr = model(oa_t[:, None]).float()[:, 0]                  # (B,720,1440) SKT 0.25° z
        anomK = pr * torch.from_numpy(ecsig[d]).to(device) * oce_t   # 물리 anomaly K (ocean만)
        num = anomK.reshape(-1, 180, 4, 360, 4).sum((2, 4))
        a1 = torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, float("nan")))
        z1 = a1.cpu().numpy() / csig1[doy1[sl]]                      # 1° z (ERA5 1° clim 기준)
        out[sl] = np.clip(z1, -args.clip, args.clip)
        if (s // B) % 20 == 0:
            print(f"  {s}/{len(times)}", flush=True)

    lat1 = sst["lat"].values.reshape(180, 4).mean(1) if "lat" in oz else np.arange(180)
    lon1 = sst["lon"].values.reshape(360, 4).mean(1) if "lon" in oz else np.arange(360)
    xr.Dataset({"z": (("time", "lat", "lon"), out)},
               coords={"time": times, "lat": lat1, "lon": lon1},
               attrs={"desc": "refined SKT-ocean anomaly z 1° (OISST→UNet→pool)",
                      "unet_ckpt": args.ckpt}).chunk(
        {"time": 64}).to_zarr(args.out, mode="w")
    print(f"[refine] saved {args.out}")


def check(args):
    """2022 vis(open-ocean) 셀에서 refined vs identity(OISST 1° z) vs GT(ERA5 1° z) corr."""
    rz = xr.open_zarr(args.out)["z"].sel(time=slice(args.check_start, args.check_end))
    times = rz["time"].values
    ref = rz.values.astype(np.float32)
    era = xr.open_zarr("data/era5_skt_anomaly.zarr")["anomaly"].reindex(time=times).values.astype(np.float32)
    od = xr.open_zarr("data/oisst_1deg_oceanmean.zarr")
    osst = od["sst"].reindex(time=times).values.astype(np.float32)
    ocean = od["ocean_mask"].values > 0
    cl = np.load("data/oisst_1deg_climatology.npz")
    cmu, csig = cl["c_mu"].astype(np.float32), cl["c_sig"].astype(np.float32)
    ice = (np.nanmin(cmu, axis=0) <= -1.7) & ocean
    vis = ocean & ~ice
    d = np.array([min(pd.Timestamp(t).dayofyear, 366) - 1 for t in times])
    oid = np.clip((osst - cmu[d]) / np.clip(csig[d], 1e-6, None), -5, 5)

    lat = od["lat"].values if "lat" in od else 89.5 - np.arange(180)
    w = np.cos(np.deg2rad(lat)).clip(0)[:, None] * vis

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        ww = np.broadcast_to(w[None], a.shape) * m
        am = np.nansum(a * ww) / ww.sum(); bm = np.nansum(b * ww) / ww.sum()
        aa = a - am; bb = b - bm
        return float(np.nansum(aa * bb * ww) / np.sqrt(np.nansum(aa**2 * ww) * np.nansum(bb**2 * ww)))

    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        ww = np.broadcast_to(w[None], a.shape) * m
        return float(np.sqrt(np.nansum((a - b) ** 2 * ww) / ww.sum()))

    print(f"[check] {args.check_start}~{args.check_end}, {len(times)}d, vis cells={int(vis.sum())}")
    print(f"  refined  vs ERA5-1°z : corr={corr(ref, era):.4f}  RMSE(z)={rmse(ref, era):.4f}")
    print(f"  identity vs ERA5-1°z : corr={corr(oid, era):.4f}  RMSE(z)={rmse(oid, era):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sst_recon/out_big/ckpt.pt")
    ap.add_argument("--out", default="data/skt1deg_refined_unet.zarr")
    ap.add_argument("--oisst-zarr", default="data/oisst_mean_1982_2023.zarr")
    ap.add_argument("--oisst-clim", default="data/oisst_sst_climatology_025.npz")
    ap.add_argument("--era5-zarr-025", default="data/era5_skt_00utc_0p25.zarr")
    ap.add_argument("--era5-clim-025", default="SST_swin/static/era5_skt_climatology_0p25.npz")
    ap.add_argument("--era5-clim-1deg", default="S2S/static/era5_skt_climatology.npz")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--check-start", default="2022-01-01")
    ap.add_argument("--check-end", default="2022-12-31")
    args = ap.parse_args()
    if not args.check_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        build(args, device)
    check(args)


if __name__ == "__main__":
    main()
