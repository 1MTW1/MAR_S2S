"""배포형 forecast BC 사전계산 — IC별 transfer(B) 예보 → 1° z.

IC(t0)까지의 관측만 사용 (미래 GT 없음):
  과거 14d OISST + 과거 14d skt → [frozen SST_swin + transfer UNet] → 미래 14d skt anom z 0.25°
  → ×ERA5 0.25° c_sig → 4×4 ERA5-ocean-mean pool → ÷ERA5 1° c_sig → (lead 14, 180, 360) z

출력 zarr dims: (time=IC 날짜, lead=1..14, lat, lon). S2SInjectDirectDataset(ocean_source="forecast") 가
윈도의 미래 프레임 open-ocean 을 이 예보로 덮어씀 (IC 프레임은 refined_zarr=recon(GT@IC) 사용).

실행:
  uv run python -m baseline.data.make_forecast_bc \
      --start 2020-01-01 --end 2021-12-31 --out data/skt1deg_forecast_bc.zarr
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
import torch
import xarray as xr

import yaml

from baseline.models.swin import UTransformer
from baseline.transfer import (UNetIO, FcstAnom, load_swin, T,   # (import 로 blosc/dask fork-safe)
                                swin_forecast)

import dask; dask.config.set(scheduler="threads")               # 단일 프로세스 → 읽기 스레드 재활성
import numcodecs.blosc as _blosc; _blosc.use_threads = True

_BASE = date(2000, 1, 1).toordinal()


def _doy(times):
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    return np.array([date(2000, m, d).toordinal() - _BASE for m, d in zip(idx.month, idx.day)], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", default="sst_recon/out_transfer/ckpt.pt")
    ap.add_argument("--swin", default="SST_swin/outputs/sst_swin_0p25_oisst/ckpt_best.pt")
    ap.add_argument("--swin-cfg", default="SST_swin/configs/sst_swin_0p25_oisst.yaml")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--out", default="data/skt1deg_forecast_bc.zarr")
    ap.add_argument("--era5-clim-1deg", default="S2S/static/era5_skt_climatology.npz")
    ap.add_argument("--clip", type=float, default=5.0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    swin = load_swin(args.swin, args.swin_cfg, dev)         # frozen residual SST_swin
    tk = torch.load(args.transfer, map_location="cpu")
    model = UNetIO(base=tk.get("base", 64)).to(dev).eval()
    res_anchor = tk.get("anchor") == "persist"              # transfer 가 persist 앵커면 pred=persist+net
    model.load_state_dict(tk["model"])
    kind = f"transfer-UNet(anchor={tk.get('anchor', 'none')})"

    oz = xr.open_zarr("data/oisst_mean_1982_2023.zarr"); o_sst = oz["sst"]
    ez = xr.open_zarr("data/era5_skt_00utc_0p25.zarr"); e_skt = ez["forcing"].sel(channel="skt")
    ocean = (ez["land_sea_mask"].values < 0.5).astype(np.float32)
    oc = np.load("data/oisst_sst_climatology_025.npz")
    ocmu = np.nan_to_num(oc["c_mu"].astype(np.float32))
    ocsig = np.clip(np.nan_to_num(oc["c_sig"].astype(np.float32), nan=1.0), 1e-6, None)
    ec = np.load("SST_swin/static/era5_skt_climatology_0p25.npz")
    ecmu = np.nan_to_num(ec["c_mu"].astype(np.float32))
    ecsig_n = np.clip(np.nan_to_num(ec["c_sig"].astype(np.float32), nan=1.0), 1e-6, None)  # 입력 anom 용
    ecsig0 = np.nan_to_num(ec["c_sig"].astype(np.float32))                                  # z→K (land 0)
    c1 = np.load(args.era5_clim_1deg)
    csig1 = np.clip(c1["c_sig"].astype(np.float32), 1e-6, None)

    ocean_t = torch.from_numpy(ocean).to(dev)
    fa = FcstAnom(ocean_t, dev)
    den = ocean_t.reshape(180, 4, 360, 4).sum((1, 3))

    o_idx = {np.datetime64(t): i for i, t in enumerate(oz["time"].values)}
    e_idx = {np.datetime64(t): i for i, t in enumerate(ez["time"].values)}
    oset = set(np.asarray(oz["time"].values))
    et = ez["time"].sel(time=slice(args.start, args.end)).values
    ics = np.array([t for t in et if t in oset])
    print(f"[fc-bc] IC {args.start}~{args.end}: {len(ics)}개, refine={kind}")

    out = np.full((len(ics), T, 180, 360), np.nan, dtype=np.float32)
    day = np.timedelta64(1, "D")
    for i, t0 in enumerate(ics):
        ro = o_idx[np.datetime64(t0)]; re = e_idx[np.datetime64(t0)]
        past = np.array([np.datetime64(t0) - (T - 1 - k) * day for k in range(T)])
        fut = np.array([np.datetime64(t0) + (k + 1) * day for k in range(T)])
        dpa = _doy(past); dfo = _doy(fut)
        dfo1 = np.array([min(pd.Timestamp(t).dayofyear, 366) - 1 for t in fut])
        oK = np.nan_to_num(o_sst[ro - T + 1:ro + 1].values).astype(np.float32)
        eK = e_skt[re - T + 1:re + 1].values.astype(np.float32)
        oap = np.clip((oK - ocmu[dpa]) / ocsig[dpa], -args.clip, args.clip) * ocean
        sktp = np.clip((eK - ecmu[dpa]) / ecsig_n[dpa], -args.clip, args.clip) * ocean
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            oap_t = torch.from_numpy(oap)[None].to(dev)
            sktp_t = torch.from_numpy(sktp)[None].to(dev)
            oa_fc = swin_forecast(swin, oap_t, fa, torch.from_numpy(dfo)[None].to(dev), clip=args.clip)
            pr = model(torch.cat([oa_fc, sktp_t, oap_t], 1)).float()[0]         # (T,720,1440)
            if res_anchor:                                                      # residual anchor: persist + net
                pr = pr + sktp_t[0, -1:]
        anomK = pr * torch.from_numpy(ecsig0[dfo]).to(dev) * ocean_t
        num = anomK.reshape(T, 180, 4, 360, 4).sum((2, 4))
        a1 = torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, float("nan")))
        out[i] = np.clip(a1.cpu().numpy() / csig1[dfo1], -args.clip, args.clip)
        if i % 50 == 0:
            print(f"  {i}/{len(ics)}", flush=True)

    od = xr.open_zarr("data/oisst_1deg_oceanmean.zarr")
    lat1 = od["lat"].values if "lat" in od else np.arange(180)
    lon1 = od["lon"].values if "lon" in od else np.arange(360)
    xr.Dataset({"z": (("time", "lead", "lat", "lon"), out)},
               coords={"time": ics, "lead": np.arange(1, T + 1), "lat": lat1, "lon": lon1},
               attrs={"desc": "transfer(B) forecast skt-ocean anomaly z 1° (IC별 미래 14일, 미래 GT 미사용)",
                      "transfer_ckpt": args.transfer, "swin_ckpt": args.swin}).chunk(
        {"time": 32}).to_zarr(args.out, mode="w")
    print(f"[fc-bc] saved {args.out}  ({out.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
