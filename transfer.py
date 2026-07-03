"""transfer — 배포형 skt-ocean 예보 (fc BC 의 원천): [frozen SST_swin 예보 + 과거 skt + 과거 OISST]
→ 미래 14일 skt anomaly z (0.25°). make_forecast_bc.py 가 이 ckpt 로 1° forecast BC 생성.

구조:
  frozen SST_swin(--swin/--swin-cfg, target=raw/residual/anomaly 자동): 과거 OISST 14d → 미래 OISST 14d anom z
  transfer UNetIO(학습): 입력 42ch = [예보 OISST 14 | 과거 skt 14 | 과거 OISST 14] → 14ch 미래 skt z
  --anchor persist: pred = skt_z(t0) + net (residual 타깃, head zero-init)
  loss = ocean·lat 가중 MSE. train 1982–2019 / val 2020–2021 / test 2022.

기준선(2022): persist corr 0.828 / perfect+res 상한 0.918 / production B(raw swin) 0.870.

실행:
  uv run accelerate launch --config_file S2S_inject/configs/accelerate.yaml -m baseline.transfer \
      --epochs 8 --stride 3 --out sst_recon/out_transfer
  uv run accelerate launch ... -m baseline.transfer --eval-only --ckpt sst_recon/out_transfer/ckpt.pt
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from baseline.models.swin import UTransformer
from baseline.recon import CB, _cpad, _blosc_off   # noqa: F401 (recon import 가 blosc/dask fork-safe 설정)

_BASE = date(2000, 1, 1).toordinal()
T = 14


def _doy(times):
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    return np.array([date(2000, m, d).toordinal() - _BASE for m, d in zip(idx.month, idx.day)], np.int64)


class TransferDataset(Dataset):
    """윈도(2T 연속일) → 과거 OISST anom(swin 입력)·과거 skt anom·미래 skt anom·미래 doy."""

    def __init__(self, start, end, stride, clip=5.0):
        self.oz = xr.open_zarr("data/oisst_mean_1982_2023.zarr")
        self.ez = xr.open_zarr("data/era5_skt_00utc_0p25.zarr")
        et = self.ez["time"].sel(time=slice(start, end)).values
        oset = set(np.asarray(self.oz["time"].values))
        self.times = np.array([t for t in et if t in oset])
        self.o_idx = {np.datetime64(t): i for i, t in enumerate(self.oz["time"].values)}
        self.e_idx = {np.datetime64(t): i for i, t in enumerate(self.ez["time"].values)}
        self.o_sst = self.oz["sst"]; self.e_skt = self.ez["forcing"].sel(channel="skt")
        self.ocean = (self.ez["land_sea_mask"].values < 0.5).astype(np.float32)
        oc = np.load("data/oisst_sst_climatology_025.npz")
        self.ocmu = np.nan_to_num(oc["c_mu"].astype(np.float32))
        self.ocsig = np.clip(np.nan_to_num(oc["c_sig"].astype(np.float32), nan=1.0), 1e-6, None)
        ec = np.load("SST_swin/static/era5_skt_climatology_0p25.npz")
        self.ecmu = np.nan_to_num(ec["c_mu"].astype(np.float32))
        self.ecsig = np.clip(np.nan_to_num(ec["c_sig"].astype(np.float32), nan=1.0), 1e-6, None)
        self.clip = clip
        self.starts = list(range(0, len(self.times) - 2 * T + 1, stride))
        print(f"[TransferDS] {start}~{end}: {len(self.times)} frames, windows {len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        past = self.times[s:s + T]; fut = self.times[s + T:s + 2 * T]
        dpa = _doy(past); dfo = _doy(fut)
        r0o = self.o_idx[np.datetime64(past[0])]; r0e = self.e_idx[np.datetime64(past[0])]
        oK = np.nan_to_num(self.o_sst[r0o:r0o + T].values).astype(np.float32)        # 과거 OISST K
        eK = self.e_skt[r0e:r0e + 2 * T].values.astype(np.float32)                   # 과거+미래 skt K
        assert oK.shape[0] == T and eK.shape[0] == 2 * T, "날짜 비연속"
        oap = np.clip((oK - self.ocmu[dpa]) / self.ocsig[dpa], -self.clip, self.clip) * self.ocean
        ez_all = np.clip((eK - self.ecmu[np.concatenate([dpa, dfo])]) /
                         self.ecsig[np.concatenate([dpa, dfo])], -self.clip, self.clip) * self.ocean
        return (torch.from_numpy(oap.astype(np.float32)),
                torch.from_numpy(ez_all[:T]), torch.from_numpy(ez_all[T:]),
                torch.from_numpy(dfo))


class UNetIO(nn.Module):
    """recon.UNet 의 다채널 입출력 버전 (42→14)."""

    def __init__(self, cin=3 * T, cout=T, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e0 = CB(cin, c[0]); self.e1 = CB(c[0], c[1]); self.e2 = CB(c[1], c[2]); self.bot = CB(c[2], c[3])
        self.u2 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.d2 = CB(c[3], c[2])
        self.u1 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.d1 = CB(c[2], c[1])
        self.u0 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.d0 = CB(c[1], c[0])
        self.out = nn.Conv2d(c[0], cout, 1)

    def forward(self, x):
        e0 = self.e0(x); e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2)); b = self.bot(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        d0 = self.d0(torch.cat([self.u0(d1), e0], 1))
        return self.out(d0)


def load_swin(path, cfg_path, dev):
    sk = torch.load(path, map_location="cpu", weights_only=False)
    mc = yaml.safe_load(open(cfg_path))["model"]
    swin = UTransformer(img_size=(720, 1440), patch_size=mc["patch_size"], T=T,
                        embed_dim=mc["embed_dim"], depths=tuple(mc["depths"]),
                        depths_up=tuple(mc["depths_up"]), num_heads=tuple(mc["num_heads"]),
                        window_size=mc["window_size"], mlp_ratio=mc.get("mlp_ratio", 4.0),
                        ape=mc.get("ape", False))
    swin.load_state_dict(sk["model"]); swin.to(dev).eval()
    for p in swin.parameters():
        p.requires_grad_(False)
    return swin


def swin_forecast(swin, oap, fa, dfo, clip=5.0):
    """frozen residual SST_swin → 미래 14d OISST anomaly z (B,T,H,W). pred = a_IC + net, clamp·ocean."""
    out = oap[:, -1:] + swin(oap).float()                    # anomaly-persistence residual
    return out.clamp(-clip, clip) * fa.ocean


class FcstAnom:
    """ocean 마스크 보유 (residual swin 경로에선 swin_forecast 가 fa.ocean 만 사용)."""

    def __init__(self, ocean, dev):
        self.ocean = ocean                                                            # (H,W) tensor


@torch.no_grad()
def evaluate(swin, model, dl, fa, W, accelerator, anchor="none"):
    """리드별 RMSE(z)/corr — transfer vs persist (past skt 마지막 프레임 고정)."""
    model.eval()
    dev = accelerator.device
    acc_v = torch.zeros(7, T, device=dev)   # [se_m, se_p, fo_m, ff_m, fo_p, ff_p, oo]
    n = torch.zeros(1, device=dev)
    for oap, sktp, tgt, dfo in dl:
        oap, sktp, tgt = oap.to(dev), sktp.to(dev), tgt.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            oa_fc = swin_forecast(swin, oap, fa, dfo.to(dev))
            pr = model(torch.cat([oa_fc, sktp, oap], 1)).float()
        if anchor == "persist":
            pr = sktp[:, -1:] + pr                            # persistence + residual
        pers = sktp[:, -1:].expand(-1, T, -1, -1)
        for j, p in enumerate((pr, pers)):
            acc_v[0 + j] += (((p - tgt) ** 2) * W).sum((0, 2, 3)) / W.sum()
            acc_v[2 + 2 * j] += ((p * tgt) * W).sum((0, 2, 3))
            acc_v[3 + 2 * j] += ((p ** 2) * W).sum((0, 2, 3))
        acc_v[6] += ((tgt ** 2) * W).sum((0, 2, 3))
        n += oap.shape[0]
    acc_v = accelerator.reduce(acc_v, "sum"); n = accelerator.reduce(n, "sum").item()
    se_m, se_p, fo_m, ff_m, fo_p, ff_p, oo = acc_v
    out = {"rmse_m": (se_m / n).sqrt().cpu().numpy(), "rmse_p": (se_p / n).sqrt().cpu().numpy(),
           "corr_m": (fo_m / (ff_m * oo).clamp(min=1e-12).sqrt()).cpu().numpy(),
           "corr_p": (fo_p / (ff_p * oo).clamp(min=1e-12).sqrt()).cpu().numpy()}
    return out


def print_table(e, ref_npz="sst_recon/resid_lambda.npz"):
    print(f"\n{'lead':>5}{'transfer':>10}{'persist':>10}   |{'c_transfer':>11}{'c_persist':>10}")
    for t in range(T):
        print(f"+{t+1:<4d}{e['rmse_m'][t]:>10.4f}{e['rmse_p'][t]:>10.4f}   |"
              f"{e['corr_m'][t]:>11.4f}{e['corr_p'][t]:>10.4f}")
    print(f"{'mean':>5}{e['rmse_m'].mean():>10.4f}{e['rmse_p'].mean():>10.4f}   |"
          f"{e['corr_m'].mean():>11.4f}{e['corr_p'].mean():>10.4f}")
    if os.path.exists(ref_npz):
        r = np.load(ref_npz)
        print(f"[ref C(2022)] chain+res corr {r['corr_chain+res'].mean():.4f} / "
              f"persist {r['corr_persist'].mean():.4f} / perfect+res {r['corr_perfect+res'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swin", default="SST_swin/outputs/sst_swin_0p25_oisst/ckpt_best.pt")
    ap.add_argument("--swin-cfg", default="SST_swin/configs/sst_swin_0p25_oisst.yaml")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--bs", type=int, default=2)          # per-GPU (42ch 720×1440 무거움)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--out", default="sst_recon/out_transfer")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--anchor", default="none", choices=["none", "persist"],
                    help="persist: pred = skt_z(t0)+net (residual 타깃, head zero-init)")
    ap.add_argument("--train-start", default="1982-01-01")
    ap.add_argument("--train-end", default="2019-12-31")
    ap.add_argument("--val-stride", type=int, default=10)
    ap.add_argument("--test-stride", type=int, default=5)
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    dev = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(args.out, exist_ok=True)

    swin = load_swin(args.swin, args.swin_cfg, dev)
    model = UNetIO(base=args.base)
    if args.anchor == "persist":
        nn.init.zeros_(model.out.weight); nn.init.zeros_(model.out.bias)   # 시작 = skt persistence
    accelerator.print(f"[transfer] UNetIO params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
                      f"procs={accelerator.num_processes}, anchor={args.anchor}")

    test_ds = TransferDataset("2022-01-01", "2022-12-31", args.test_stride)
    test_dl = DataLoader(test_ds, batch_size=args.bs, num_workers=args.nw, worker_init_fn=_blosc_off)
    ocean_t = torch.from_numpy(test_ds.ocean).to(dev)
    lat = xr.open_zarr("data/era5_skt_00utc_0p25.zarr")["lat"].values.astype(np.float32)
    W = (ocean_t * torch.from_numpy(np.cos(np.deg2rad(lat)).clip(0)).to(dev)[:, None])[None, None]
    fa = FcstAnom(ocean_t, dev)

    if args.eval_only:
        tk = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(tk["model"])
        anchor = tk.get("anchor", args.anchor)
        model, test_dl = accelerator.prepare(model, test_dl)
        e = evaluate(swin, accelerator.unwrap_model(model), test_dl, fa, W, accelerator, anchor=anchor)
        if accelerator.is_main_process:
            print_table(e)
        return

    tr = TransferDataset(args.train_start, args.train_end, args.stride)
    va = TransferDataset("2020-01-01", "2021-12-31", args.val_stride)
    tr_dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.nw,
                       worker_init_fn=_blosc_off, persistent_workers=args.nw > 0, drop_last=True)
    va_dl = DataLoader(va, batch_size=args.bs, num_workers=args.nw, worker_init_fn=_blosc_off)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model, opt, tr_dl, va_dl, test_dl = accelerator.prepare(model, opt, tr_dl, va_dl, test_dl)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for oap, sktp, tgt, dfo in tr_dl:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                oa_fc = swin_forecast(swin, oap, fa, dfo.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = model(torch.cat([oa_fc, sktp, oap], 1)).float()
                if args.anchor == "persist":
                    pr = sktp[:, -1:] + pr
                loss = ((pr - tgt) ** 2 * W).sum() / (W.sum() * T * oap.shape[0])
            opt.zero_grad(); accelerator.backward(loss); opt.step()
        e = evaluate(swin, accelerator.unwrap_model(model), va_dl, fa, W, accelerator, anchor=args.anchor)
        rm = e["rmse_m"].mean()
        accelerator.print(f"[ep{ep}] val RMSE(z)={rm:.4f} corr={e['corr_m'].mean():.4f} "
                          f"(persist {e['rmse_p'].mean():.4f}/{e['corr_p'].mean():.4f})")
        if rm < best and accelerator.is_main_process:
            best = rm
            torch.save({"model": accelerator.unwrap_model(model).state_dict(),
                        "base": args.base, "swin": args.swin,
                        "anchor": args.anchor},
                       os.path.join(args.out, "ckpt.pt"))
        accelerator.wait_for_everyone()
    sd = torch.load(os.path.join(args.out, "ckpt.pt"), map_location="cpu")["model"]
    accelerator.unwrap_model(model).load_state_dict(sd)
    e = evaluate(swin, accelerator.unwrap_model(model), test_dl, fa, W, accelerator, anchor=args.anchor)
    if accelerator.is_main_process:
        print(f"\n[TEST 2022] transfer (anchor={args.anchor})")
        print_table(e)


if __name__ == "__main__":
    main()
