"""S2S_inject ★OISST 1° 직접주입(direct) 평가 — lead-time RMSE/ACC.

direct 모델: 결합 skt(ocean=OISST 1° visible / land=ERA5 LST) 를 skt_proj 직접주입, 대기+land LST 생성.
평가는 **정직한 기준**으로:
  · 대기 RMSE: model vs **persistence**(IC 관측장 유지) vs climatology  (전지구 + ★열대 20S–20N 분리)
  · 대기 ACC : model vs anomaly-persistence
  · LST RMSE : 육지(OISST mask land) 물리 K, vs climatology

실행(멀티-GPU):
  uv run accelerate launch --config_file S2S_inject/configs/accelerate.yaml -m baseline.eval_mar \
      --ckpt S2S_inject/outputs/s2s_inject_14day_oisst1deg/ckpt_best.pt \
      --dcae S2S_SST/outputs/dcae/dcae --ic-stride 15 --ensemble 10 \
      --save-npz S2S_inject/outputs/s2s_inject_14day_oisst1deg/eval.npz
"""
import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr
from accelerate import Accelerator
from tqdm.auto import tqdm

from baseline.utils import lat_weights, load_latent_stats, lw_mean
from baseline.data.field_dataset import load_field_stats
from baseline.data.mar_dataset import S2SInjectDirectDataset
from baseline.data.skt_climatology import SktClimatology
from baseline.eval_utils import decode_sliced, load_inject, unpatch_skt
from baseline.models.dcae import AutoencoderDC


@torch.no_grad()
def sample_ens(model, ic, ts, skt, ot, M, chunk, num_iter, temp, det_atmo, device):
    fa_all, sk_all = [], []
    for i in range(0, M, chunk):
        m = min(chunk, M - i)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fa, sk = model.sample_sst_direct(ic[i:i + m], ts[i:i + m], skt[i:i + m], ot[i:i + m],
                                             num_iter=num_iter, temperature=temp, det_atmo=det_atmo)
        fa_all.append(fa.float()); sk_all.append(sk.float())
    return torch.cat(fa_all, 0), torch.cat(sk_all, 0)


def main():
    ap = argparse.ArgumentParser(description="S2S_inject direct(OISST 1°) leadtime RMSE/ACC")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dcae", required=True)
    ap.add_argument("--ic-stride", type=int, default=15)
    ap.add_argument("--ensemble", type=int, default=10)
    ap.add_argument("--ens-chunk", type=int, default=10)
    ap.add_argument("--num-iter", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--n-eval-ch", type=int, default=9)
    ap.add_argument("--det-atmo", action="store_true")
    ap.add_argument("--trop-lat", type=float, default=20.0, help="열대 분리 위도(±)")
    ap.add_argument("--clim-npz", default="data/climatology.npz")
    ap.add_argument("--field-zarr", default="data/era5_00utc.zarr")
    ap.add_argument("--field-var", default="snapshot")
    ap.add_argument("--field-stats", default="S2S_SST/static/era5_field_stats.json")
    ap.add_argument("--skt-clim-npz", default="S2S/static/era5_skt_climatology.npz")
    ap.add_argument("--val-start", default=None)
    ap.add_argument("--val-end", default=None)
    ap.add_argument("--ocean-source-override", default=None,
                    choices=["oisst", "skt", "refined", "forecast"],
                    help="ckpt 학습설정과 다른 BC 소스로 평가 (예: skt 학습 모델에 refined/forecast 주입)")
    ap.add_argument("--refined-zarr", default="data/skt1deg_refined_unet.zarr")
    ap.add_argument("--forecast-zarr", default="data/skt1deg_forecast_bc.zarr")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    acc = Accelerator()
    device = acc.device
    model, cfg = load_inject(args.ckpt, device, use_ema=not args.no_ema)
    dc, sc = cfg["data"], cfg["sample"]
    assert dc.get("oisst_1deg_zarr"), "이 ckpt 는 direct(OISST 1°) 가 아님."
    cond_len, T = model.cond_len, model.T
    M = args.ensemble
    num_iter = args.num_iter or sc.get("num_iter", T)
    temp = args.temperature if args.temperature is not None else sc.get("temperature", 1.0)
    nch = args.n_eval_ch

    dcae = AutoencoderDC.from_pretrained(args.dcae).to(device).eval()
    lat_mean_t, lat_std_t = load_latent_stats(dc["latent_stats"])
    lat_mean = lat_mean_t[:, None, None].to(device); lat_std = lat_std_t[:, None, None].to(device)
    target_std = dc.get("target_std")
    fmean_t, fstd_t, _ = load_field_stats(args.field_stats)
    fmean = fmean_t[:nch, None, None].to(device); fstd = fstd_t[:nch, None, None].to(device)

    vstart = args.val_start or dc["val"][0]
    vend = args.val_end or dc["val"][1]
    ds = S2SInjectDirectDataset(
        latents_zarr=dc["latents_zarr"], var=dc["var"], latent_stats=dc["latent_stats"],
        start_date=vstart, end_date=vend,
        oisst_1deg_zarr=dc["oisst_1deg_zarr"], oisst_1deg_clim_npz=dc["oisst_1deg_clim_npz"],
        skt_anomaly_zarr=dc["skt_anomaly_zarr"],
        future_len=T, input_len=cond_len, stride=dc["stride"], target_std=target_std,
        load_in_memory=True, skt_clip=dc.get("skt_clip", 5.0), skt_patch=dc.get("skt_patch", 18),
        ocean_thresh=dc.get("skt_ocean_thresh", 0.5),
        ocean_source=args.ocean_source_override or dc.get("ocean_source", "oisst"),
        mask_seaice=dc.get("mask_seaice", False), ice_thresh=dc.get("ice_thresh", -1.7),
        refined_zarr=args.refined_zarr or dc.get("refined_zarr"),
        forecast_zarr=args.forecast_zarr)
    if args.ocean_source_override:
        acc.print(f"[eval-direct] ★ BC override: 학습={dc.get('ocean_source', 'oisst')} "
                  f"→ 평가={args.ocean_source_override}")
    window = ds.window

    fld = xr.open_zarr(args.field_zarr)
    chan = [str(c) for c in fld["channel"].values][:nch]
    lw = lat_weights(fld["lat"].values)                            # (1,1,H,1)
    flat = fld["lat"].values
    c_mu = np.load(args.clim_npz)["c_mu"].astype(np.float32)[:, :nch]
    # 열대 lat-weight (전지구/열대 2세트)
    lw2d = lw[0, 0, :, 0][:, None]                                  # (H,1)
    Wlon = fld.sizes["lon"]
    trop2d = np.broadcast_to((np.abs(flat) <= args.trop_lat)[:, None].astype(np.float32),
                             (len(flat), Wlon))                     # (H,W)
    w_trop = lw2d * trop2d; den_trop = float(w_trop.sum())          # 2D 가중(경도 포함)

    # LST(육지 K) 준비: OISST mask land + ERA5 c_sig
    c_sig = np.asarray(SktClimatology.from_file(args.skt_clim_npz).c_sig, dtype=np.float32)  # (366,H,W)
    ocean_mask = xr.open_zarr(dc["oisst_1deg_zarr"])["ocean_mask"].values > 0                # (H,W)
    land_px = (~ocean_mask).astype(np.float32)
    w2d = lw2d * land_px; den_land = float(w2d.sum())
    Pp, hh, ww = ds.P, ds.h, ds.w

    win_idx = list(range(0, len(ds), args.ic_stride))
    local = win_idx[acc.process_index::acc.num_processes]
    acc.print(f"[eval-direct] {vstart}~{vend}, windows {len(win_idx)} (~{len(local)}/proc), "
              f"ens={M}, num_iter={num_iter}, temp={temp}, ckpt={args.ckpt.split('/')[-2]}")
    torch.manual_seed(args.seed + acc.process_index)

    T_ = T
    se = np.zeros((T_, nch)); se_p = np.zeros((T_, nch)); clim_se = np.zeros((T_, nch))       # 전지구
    se_t = np.zeros((T_, nch)); se_pt = np.zeros((T_, nch)); clim_t = np.zeros((T_, nch))     # 열대
    afo = np.zeros((T_, nch)); aff = np.zeros((T_, nch)); aoo = np.zeros((T_, nch))
    pfo = np.zeros((T_, nch)); pff = np.zeros((T_, nch))
    lst_se = np.zeros(T_); lst_clim = np.zeros(T_)
    n_ic = 0
    for wi in tqdm(local, disable=not acc.is_local_main_process):
        latents, ts, skt, _, otok = ds[wi]
        s0 = ds.starts[wi]
        valid = pd.to_datetime(ds.time[s0 + cond_len: s0 + window])
        truth = fld[args.field_var].sel(time=valid).values.astype(np.float32)[:, :nch]        # (T,C,H,W)
        doy = np.array([min(pd.Timestamp(v).dayofyear, 366) - 1 for v in valid])
        cm = c_mu[doy]
        ic_idx = s0 + cond_len - 1
        ic_truth = fld[args.field_var].sel(time=ds.time[ic_idx]).values.astype(np.float32)[:nch]  # (C,H,W)
        persist = ic_truth[None]                                                              # (1,C,H,W) 지속

        ic = latents[:cond_len][None].to(device).repeat(M, 1, 1, 1, 1)
        tsm = ts[None].to(device).repeat(M, 1)
        sktm = skt[None].to(device).repeat(M, 1, 1, 1, 1)
        ot_m = otok[None].to(device).repeat(M, 1, 1)
        fut_atmo, skt_full = sample_ens(model, ic, tsm, sktm, ot_m, M, args.ens_chunk,
                                        num_iter, temp, args.det_atmo, device)
        ens = decode_sliced(fut_atmo, dcae, lat_mean, lat_std, fmean, fstd, target_std, nch)
        fm = ens.mean(0)                                                                      # (T,C,H,W)

        # 전지구 RMSE (model/persist/clim)
        se += lw_mean((fm - truth) ** 2, lw)
        se_p += lw_mean((persist - truth) ** 2, lw)
        clim_se += lw_mean((cm - truth) ** 2, lw)
        # 열대 RMSE
        se_t += ((fm - truth) ** 2 * w_trop).sum((-1, -2)) / den_trop
        se_pt += ((persist - truth) ** 2 * w_trop).sum((-1, -2)) / den_trop
        clim_t += ((cm - truth) ** 2 * w_trop).sum((-1, -2)) / den_trop
        # ACC (model / anomaly-persist)
        oa = truth - cm; fa = fm - cm
        afo += lw_mean(fa * oa, lw); aff += lw_mean(fa * fa, lw); aoo += lw_mean(oa * oa, lw)
        pa = (ic_truth - c_mu[min(pd.Timestamp(ds.time[ic_idx]).dayofyear, 366) - 1])[None]
        pfo += lw_mean(pa * oa, lw); pff += lw_mean(pa * pa, lw)
        # LST(육지 K)
        pred_zf = unpatch_skt(skt_full[:, cond_len:].mean(0).cpu().numpy(), hh, ww, Pp)       # (T,H,W) z
        gt_zf = ds.skt[s0 + cond_len: s0 + window].numpy()                                    # (T,H,W) z(land=ERA5)
        csig = c_sig[doy]
        lst_se += (((pred_zf - gt_zf) * csig) ** 2 * w2d[None]).sum((1, 2)) / den_land
        lst_clim += (((0.0 - gt_zf) * csig) ** 2 * w2d[None]).sum((1, 2)) / den_land
        n_ic += 1

    def asum(a):
        t = torch.tensor(np.asarray(a, np.float64), device=device)
        return acc.reduce(t, reduction="sum").cpu().numpy()
    se, se_p, clim_se = asum(se), asum(se_p), asum(clim_se)
    se_t, se_pt, clim_t = asum(se_t), asum(se_pt), asum(clim_t)
    afo, aff, aoo, pfo, pff = asum(afo), asum(aff), asum(aoo), asum(pfo), asum(pff)
    lst_se, lst_clim = asum(lst_se), asum(lst_clim)
    n_ic = int(asum(np.array([n_ic]))[0])
    if not acc.is_main_process:
        return

    n = max(n_ic, 1)
    r = np.sqrt(se / n); rp = np.sqrt(se_p / n); rc = np.sqrt(clim_se / n)
    rt = np.sqrt(se_t / n); rpt = np.sqrt(se_pt / n); rct = np.sqrt(clim_t / n)
    accm = afo / np.sqrt(np.clip(aff * aoo, 1e-12, None))
    accp = pfo / np.sqrt(np.clip(pff * aoo, 1e-12, None))
    lr = np.sqrt(lst_se / n); lrc = np.sqrt(lst_clim / n)
    leads = np.arange(1, T + 1)

    def tbl(title, m, p, c, wcol="win vs persist"):
        print(f"\n=== {title} ===")
        print(f"{'lead':>5}{'model':>10}{'persist':>10}{'clim':>10}{'  '+wcol:>16}")
        for i, ld in enumerate(leads):
            w = '✓' if m[i].mean() < p[i].mean() else '✗'
            print(f"+{ld:<4d}{m[i].mean():>10.4f}{p[i].mean():>10.4f}{c[i].mean():>10.4f}{w:>16}")
        print(f"{'mean':>5}{m.mean():>10.4f}{p.mean():>10.4f}{c.mean():>10.4f}")

    print(f"\n[eval-direct] 집계 IC {n_ic}개")
    tbl("대기 RMSE 전지구 (채널평균 lat-weighted)", r, rp, rc)
    tbl(f"대기 RMSE 열대 |lat|≤{args.trop_lat:.0f}", rt, rpt, rct)
    print(f"\n=== 대기 ACC (model vs anomaly-persist) ===")
    print(f"{'lead':>5}{'model':>10}{'persist':>10}{'win?':>7}")
    for i, ld in enumerate(leads):
        print(f"+{ld:<4d}{accm[i].mean():>10.4f}{accp[i].mean():>10.4f}"
              f"{'  ✓' if accm[i].mean() > accp[i].mean() else '  ✗':>7}")
    print(f"{'mean':>5}{accm.mean():>10.4f}{accp.mean():>10.4f}")
    print(f"\n=== LST 육지 RMSE (K) ===")
    print(f"{'lead':>5}{'model':>10}{'clim':>10}{'win?':>7}")
    for i, ld in enumerate(leads):
        print(f"+{ld:<4d}{lr[i]:>10.4f}{lrc[i]:>10.4f}{'  ✓' if lr[i] < lrc[i] else '  ✗':>7}")
    print(f"{'mean':>5}{lr.mean():>10.4f}{lrc.mean():>10.4f}")

    if args.save_npz:
        np.savez(args.save_npz, leads=leads, n_ic=n_ic, chan=chan,
                 atmo_rmse=r, persist_rmse=rp, clim_rmse=rc,
                 atmo_rmse_trop=rt, persist_rmse_trop=rpt, clim_rmse_trop=rct,
                 acc_model=accm, acc_persist=accp, lst_rmse=lr, lst_clim_rmse=lrc)
        print(f"\n[eval-direct] saved {args.save_npz}")


if __name__ == "__main__":
    main()
