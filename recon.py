"""recon — same-time refine UNet: OISST(t) anomaly z → ERA5-skt(t) ocean anomaly z (0.25°).

현 파이프라인 역할:
  · production ckpt `sst_recon/out_big/ckpt.pt` (base 96) → make_refined_bc.py 가
    refined GT BC(`data/skt1deg_refined_unet.zarr`) 생성에 사용.
  · UNet/CB 블록은 transfer.py(UNetIO) 가 재사용.

정규화: 입력 (sst−c_mu_OISST[doy])/c_sig_OISST[doy], 타깃 (skt−c_mu_ERA5[doy])/c_sig_ERA5[doy],
±5 clip, land0. ocean = ERA5 lsm<0.5, loss/eval = ocean·lat 가중.

실행:
  uv run accelerate launch ... -m baseline.recon --epochs 8 --stride 3 --base 96 --out sst_recon/out_big
  uv run ... -m baseline.recon --eval-only --ckpt sst_recon/out_big/ckpt.pt
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
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

# ★ fork 데드락 원천 차단 (부모에서 설정 → DataLoader 워커 fork 시 상속):
#   lazy zarr 를 워커에서 .values 하면 blosc/dask 스레드풀 lock 이 fork 에 깨져 첫 배치서 멈춤.
import dask                                        # noqa: E402
import numcodecs.blosc as _blosc                   # noqa: E402
_blosc.use_threads = False                         # blosc 내부 스레드 off
dask.config.set(scheduler="synchronous")           # dask 스레드풀 미사용(워커 내 단일스레드 read)

_BASE = date(2000, 1, 1).toordinal()


def _doy(times):
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    return np.array([date(2000, m, d).toordinal() - _BASE for m, d in zip(idx.month, idx.day)], np.int64)


def _blosc_off(_):
    try:
        import numcodecs.blosc as b; b.use_threads = False
    except Exception:
        pass


class ReconDataset(Dataset):
    """OISST anom(입력) / ERA5-skt anom(타깃) / ocean(가중) 프레임 페어."""

    def __init__(self, start, end, stride=1, clip=5.0):
        oz = xr.open_zarr("data/oisst_mean_1982_2023.zarr")
        ez = xr.open_zarr("data/era5_skt_00utc_0p25.zarr")
        et = ez["time"].sel(time=slice(start, end)).values
        ot = set(np.asarray(oz["time"].values))
        times = np.array([t for t in et if t in ot])[::stride]        # 공통 날짜(OISST 1982+)
        self.times = times
        self.o_sst = oz["sst"]; self.e_skt = ez["forcing"].sel(channel="skt")
        self.o_idx = {np.datetime64(t): i for i, t in enumerate(oz["time"].values)}
        self.e_idx = {np.datetime64(t): i for i, t in enumerate(ez["time"].values)}
        oc = np.load("data/oisst_sst_climatology_025.npz")
        self.ocmu = np.nan_to_num(oc["c_mu"].astype(np.float32)); self.ocsig = oc["c_sig"].astype(np.float32)
        ec = np.load("SST_swin/static/era5_skt_climatology_0p25.npz")
        self.ecmu = ec["c_mu"].astype(np.float32); self.ecsig = ec["c_sig"].astype(np.float32)
        self.ocean = (ez["land_sea_mask"].values < 0.5).astype(np.float32)   # (H,W) 타깃 ocean
        self.clip = clip
        self.doy = _doy(times)
        print(f"[ReconDataset] {start}~{end} stride{stride}: {len(times)} frames, ocean {self.ocean.mean():.3f}")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, i):
        t = np.datetime64(self.times[i]); d = int(self.doy[i])
        o_raw = self.o_sst[self.o_idx[t]].values.astype(np.float32)
        e_raw = self.e_skt[self.e_idx[t]].values.astype(np.float32)
        oa = np.clip((np.nan_to_num(o_raw) - self.ocmu[d]) / np.clip(self.ocsig[d], 1e-6, None), -self.clip, self.clip)
        ea = np.clip((e_raw - self.ecmu[d]) / np.clip(self.ecsig[d], 1e-6, None), -self.clip, self.clip)
        oa = np.nan_to_num(oa) * self.ocean; ea = np.nan_to_num(ea) * self.ocean
        return (torch.from_numpy(oa[None]), torch.from_numpy(ea[None]),
                torch.from_numpy(self.ocean[None]))


def _cpad(x, p=1):                                       # 경도 circular, 위도 zero
    x = F.pad(x, (p, p, 0, 0), mode="circular")
    return F.pad(x, (0, 0, p, p), mode="constant", value=0.0)


class CB(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3); self.c2 = nn.Conv2d(cout, cout, 3)
        self.n1 = nn.GroupNorm(8, cout); self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(_cpad(x)))); return F.silu(self.n2(self.c2(_cpad(x))))


class UNet(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e0 = CB(1, c[0]); self.e1 = CB(c[0], c[1]); self.e2 = CB(c[1], c[2]); self.bot = CB(c[2], c[3])
        self.u2 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.d2 = CB(c[3], c[2])
        self.u1 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.d1 = CB(c[2], c[1])
        self.u0 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.d0 = CB(c[1], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e0 = self.e0(x); e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2)); b = self.bot(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        d0 = self.d0(torch.cat([self.u0(d1), e0], 1))
        return self.out(d0)


def latw(H, dev):
    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    return torch.tensor(np.cos(np.deg2rad(lat)).clip(0), dtype=torch.float32, device=dev)[None, None, :, None]


@torch.no_grad()
def evaluate(model, dl, accelerator):
    """RMSE(z) + anomaly corr(UNet/identity) 2종:
      · pooled: ocean·lat 가중 공간+시간 corr / · pixel: 픽셀별 시간상관 median(ocean)."""
    model.eval()
    dev = accelerator.device
    pool = torch.zeros(7, device=dev)              # [se, sw, fo, ff, oo, io, ii]
    pix = None; nf = torch.zeros(1, device=dev); ocean = None; lw = None
    for oa, ea, m in dl:
        if lw is None:
            lw = latw(oa.shape[-2], dev)
            H, W = oa.shape[-2], oa.shape[-1]
            pix = torch.zeros(8, H, W, device=dev)  # Σp,Σe,Σo,Σpp,Σee,Σoo,Σpe,Σoe (픽셀별)
            ocean = (m[0, 0] > 0.5)
        w = m * lw
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pr = model(oa).float()
        pool[0] += ((pr - ea) ** 2 * w).sum(); pool[1] += w.sum()
        pool[2] += (pr * ea * w).sum(); pool[3] += (pr * pr * w).sum(); pool[4] += (ea * ea * w).sum()
        pool[5] += (oa * ea * w).sum(); pool[6] += (oa * oa * w).sum()
        p, e, o = pr[:, 0], ea[:, 0], oa[:, 0]      # (B,H,W)
        pix[0] += p.sum(0); pix[1] += e.sum(0); pix[2] += o.sum(0)
        pix[3] += (p * p).sum(0); pix[4] += (e * e).sum(0); pix[5] += (o * o).sum(0)
        pix[6] += (p * e).sum(0); pix[7] += (o * e).sum(0)
        nf += p.shape[0]
    pool = accelerator.reduce(pool, "sum").tolist()
    pix = accelerator.reduce(pix, "sum"); n = accelerator.reduce(nf, "sum").item()
    se, sw, fo, ff, oo, io, ii = pool
    mp, me, mo = pix[0] / n, pix[1] / n, pix[2] / n
    vp = pix[3] / n - mp ** 2; ve = pix[4] / n - me ** 2; vo = pix[5] / n - mo ** 2
    cm_pix = ((pix[6] / n - mp * me) / torch.sqrt((vp * ve).clamp(min=1e-12)))[ocean]
    ci_pix = ((pix[7] / n - mo * me) / torch.sqrt((vo * ve).clamp(min=1e-12)))[ocean]
    return {"rmse": (se / sw) ** 0.5, "cm_pool": fo / (ff * oo) ** 0.5, "ci_pool": io / (ii * oo) ** 0.5,
            "cm_pix": cm_pix.median().item(), "ci_pix": ci_pix.median().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)      # train 시간 서브샘플
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8)          # per-GPU
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--out", default="sst_recon/out")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    dev = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(args.out, exist_ok=True)

    model = UNet(args.base)
    accelerator.print(f"[recon] UNet params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
                      f"procs={accelerator.num_processes}")
    test_ds = ReconDataset("2022-01-01", "2022-12-31", stride=5)
    test_dl = DataLoader(test_ds, batch_size=args.bs, num_workers=args.nw, worker_init_fn=_blosc_off)

    if args.eval_only:
        sd = torch.load(args.ckpt, map_location="cpu")["model"]
        model.load_state_dict(sd)
        model, test_dl = accelerator.prepare(model, test_dl)
        e = evaluate(model, test_dl, accelerator)
        accelerator.print(f"[TEST] RMSE(z)={e['rmse']:.4f}  pooled: UNet={e['cm_pool']:.4f} id={e['ci_pool']:.4f}  "
                          f"pixelwise: UNet={e['cm_pix']:.4f} id={e['ci_pix']:.4f}")
        return

    tr = ReconDataset("1982-01-01", "2019-12-31", stride=args.stride)
    va = ReconDataset("2020-01-01", "2021-12-31", stride=10)
    tr_dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.nw,
                       worker_init_fn=_blosc_off, persistent_workers=args.nw > 0, drop_last=True)
    va_dl = DataLoader(va, batch_size=args.bs, num_workers=args.nw, worker_init_fn=_blosc_off)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model, opt, tr_dl, va_dl, test_dl = accelerator.prepare(model, opt, tr_dl, va_dl, test_dl)
    lw = None; best = 1e9
    for ep in range(args.epochs):
        model.train()
        for oa, ea, m in tr_dl:
            if lw is None:
                lw = latw(oa.shape[-2], dev)
            w = m * lw
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pr = model(oa).float()
                loss = ((pr - ea) ** 2 * w).sum() / w.sum()
            opt.zero_grad(); accelerator.backward(loss); opt.step()
        e = evaluate(model, va_dl, accelerator)
        accelerator.print(f"[ep{ep}] val RMSE(z)={e['rmse']:.4f}  pool UNet={e['cm_pool']:.4f}/id={e['ci_pool']:.4f}  "
                          f"pix UNet={e['cm_pix']:.4f}/id={e['ci_pix']:.4f}")
        if e["rmse"] < best and accelerator.is_main_process:
            best = e["rmse"]
            torch.save({"model": accelerator.unwrap_model(model).state_dict()},
                       os.path.join(args.out, "ckpt.pt"))
    accelerator.wait_for_everyone()
    sd = torch.load(os.path.join(args.out, "ckpt.pt"), map_location="cpu")["model"]
    accelerator.unwrap_model(model).load_state_dict(sd)
    e = evaluate(model, test_dl, accelerator)
    accelerator.print(f"\n[TEST 2022] RMSE(z)={e['rmse']:.4f}")
    accelerator.print(f"  pooled corr : UNet={e['cm_pool']:.4f}  identity={e['ci_pool']:.4f}")
    accelerator.print(f"  pixelwise   : UNet={e['cm_pix']:.4f}  identity={e['ci_pix']:.4f}")
    accelerator.print("→ UNet>identity 면 identity 넘어 배운다는 증거. pixelwise 가 이전 0.70 과 직접 비교값.")


if __name__ == "__main__":
    main()
