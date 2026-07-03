"""coupled 2-cycle 실험 ① — 생성된 LST 를 SKT-ocean BC 재추정에 재사용할 수 있는가.

근거: transfer BC 는 과거 해양·표면 관측만 입력받지만, MAR 은 대기 3D IC 를 조건으로 받는다.
MAR 이 생성한 미래 LST(skill 0.30)에는 대기 IC 정보가 녹아 있으므로, [forecast BC, LST_gen]
→ true skt-ocean 매핑이 forecast BC 단독보다 좋다면 "대기→해양 되먹임" 채널이 존재하는 것.

gen : 200ep MAR + forecast BC 로 val 윈도의 skt 전장(생성 LST 포함) 을 ens-mean 으로 저장
fit : 1° 소형 UNet — 입력 [fc(open-ocean), lst_gen(그 외), vis mask] → true era skt z (open-ocean)
      2020 IC 학습 / 2021 IC 검증. baseline = fc 그대로 (identity).

실행:
  uv run accelerate launch --config_file S2S_inject/configs/accelerate.yaml -m baseline.cycle gen
  uv run python -m baseline.cycle fit
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from accelerate import Accelerator

from baseline.utils import load_latent_stats
from baseline.data.field_dataset import load_field_stats
from baseline.data.mar_dataset import S2SInjectDirectDataset
from baseline.eval_utils import decode_sliced, load_inject, unpatch_skt

OUT = "S2S_inject/outputs/s2s_inject_14day_sktocean_200ep/cycle1_gen.npz"
ATM_CH = list(range(9))                                # 전 대기채널 t/u/v @300/500/850


def build_ds(cfg, start, end, stride):
    dc = cfg["data"]
    return S2SInjectDirectDataset(
        latents_zarr=dc["latents_zarr"], var=dc["var"], latent_stats=dc["latent_stats"],
        start_date=start, end_date=end,
        oisst_1deg_zarr=dc["oisst_1deg_zarr"], oisst_1deg_clim_npz=dc["oisst_1deg_clim_npz"],
        skt_anomaly_zarr=dc["skt_anomaly_zarr"],
        future_len=14, input_len=1, stride=stride, target_std=dc.get("target_std"),
        load_in_memory=True, skt_clip=dc.get("skt_clip", 5.0), skt_patch=dc.get("skt_patch", 18),
        ocean_thresh=dc.get("skt_ocean_thresh", 0.5), ocean_source="forecast",
        mask_seaice=dc.get("mask_seaice", False), ice_thresh=dc.get("ice_thresh", -1.7),
        refined_zarr="data/skt1deg_refined_unet.zarr", forecast_zarr="data/skt1deg_forecast_bc.zarr")


def gen(args):
    acc = Accelerator(); device = acc.device
    model, cfg = load_inject(args.ckpt, device, use_ema=True)
    ds = build_ds(cfg, args.start, args.end, args.stride)
    T, P, hh, ww = 14, ds.P, ds.h, ds.w
    era = xr.open_zarr(cfg["data"]["skt_anomaly_zarr"])["anomaly"].reindex(time=ds.time)\
        .values.astype(np.float32)                                     # true skt z 1°
    dcae = fmean = fstd = lat_mean = lat_std = None
    if args.dcae:                                                      # ★ Atmo 조건: ens-mean latent 디코드
        from baseline.models.dcae import AutoencoderDC
        dcae = AutoencoderDC.from_pretrained(args.dcae).to(device).eval()
        lm, lsd = load_latent_stats(cfg["data"]["latent_stats"])
        lat_mean = lm[:, None, None].to(device); lat_std = lsd[:, None, None].to(device)
        fm, fs, _ = load_field_stats("S2S_SST/static/era5_field_stats.json")
        fmean = fm[:9, None, None].to(device); fstd = fs[:9, None, None].to(device)

    win = list(range(len(ds)))
    local = win[acc.process_index::acc.num_processes]
    acc.print(f"[cycle1-gen] {args.start}~{args.end} windows {len(win)} (~{len(local)}/proc), "
              f"ens={args.ens}, atmo={'on' if args.dcae else 'off'}")
    recs = []
    for wi in local:
        latents, ts, skt, _, otok = ds[wi]
        s0 = ds.starts[wi]
        M = args.ens
        ic = latents[:1][None].to(device).repeat(M, 1, 1, 1, 1)
        tsm = ts[None].to(device).repeat(M, 1)
        sktm = skt[None].to(device).repeat(M, 1, 1, 1, 1)
        ot_m = otok[None].to(device).repeat(M, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fut_atmo, skt_full = model.sample_sst_direct(ic, tsm, sktm, ot_m, num_iter=21, temperature=1.0)
        # ★ per-member skt (ens M개 각각 저장 → updater 데이터 M배). skt_full:(M,window,h,w,P²)
        gen_m = np.stack([unpatch_skt(skt_full[m, 1:].float().cpu().numpy(), hh, ww, P)
                          for m in range(M)], 0).astype(np.float16)                     # (M,T,180,360)
        fc_z = unpatch_skt(skt[1:].numpy(), hh, ww, P)                                  # 입력 BC(=forecast)
        atm = None
        if dcae is not None:                                                            # ens-mean 대기 9ch
            with torch.no_grad():
                dec = decode_sliced(fut_atmo.float().mean(0, keepdim=True), dcae, lat_mean, lat_std,
                                    fmean, fstd, cfg["data"].get("target_std"), 9)      # (1,T,9,180,360)
            atm = np.asarray(dec[0][:, ATM_CH]).astype(np.float16)                      # (T,9,180,360)
        recs.append((ds.ts[s0], fc_z.astype(np.float32), gen_m,
                     era[s0 + 1: s0 + 1 + T], atm))
    # gather via file-per-rank (간단·안전)
    part = {"ics": np.array([r[0] for r in recs]),
            "fc": np.stack([r[1] for r in recs]), "gen": np.stack([r[2] for r in recs]),
            "tru": np.stack([r[3] for r in recs])}
    if args.dcae:
        part["atm"] = np.stack([r[4] for r in recs])
    np.savez(args.out.replace(".npz", f".rank{acc.process_index}.npz"), **part)
    acc.wait_for_everyone()
    if acc.is_main_process:
        parts = [np.load(args.out.replace(".npz", f".rank{r}.npz")) for r in range(acc.num_processes)]
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
        np.savez(args.out, **merged)
        for r in range(acc.num_processes):
            os.remove(args.out.replace(".npz", f".rank{r}.npz"))
        print(f"[cycle1-gen] saved {args.out}  windows={len(merged['ics'])}  fc={merged['fc'].shape}")


class TinyUNet(nn.Module):
    """1°(180×360) 소형 updater — 경도 circular pad."""

    def __init__(self, cin=4, base=32):
        super().__init__()
        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=0), nn.GroupNorm(8, o), nn.SiLU())
        self.e0a, self.e0b = cb(cin, base), cb(base, base)
        self.e1a, self.e1b = cb(base, base * 2), cb(base * 2, base * 2)
        self.u = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d0a, self.d0b = cb(base * 2, base), cb(base, base)
        self.out = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # 시작 = fc 그대로(잔차)

    @staticmethod
    def _p(x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return F.pad(x, (0, 0, 1, 1))

    def forward(self, x):
        e0 = self.e0b(self._p(self.e0a(self._p(x))))
        e1 = self.e1b(self._p(self.e1a(self._p(F.max_pool2d(e0, 2)))))
        d = self.d0b(self._p(self.d0a(self._p(torch.cat([self.u(e1), e0], 1)))))
        return self.out(d)[:, 0]


def fit(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.gen_npz)
    ics = pd.to_datetime([str(t) for t in d["ics"].astype(np.int64)], format="%Y%m%d%H")
    fc, gen_, tru = d["fc"], d["gen"], d["tru"]                       # fc/tru (N,14,H,W), gen (N,M,14,H,W)
    if gen_.ndim == 4:                                               # 구버전(ens-mean) 호환
        gen_ = gen_[:, None]
    M = gen_.shape[1]
    gen_mean = gen_.astype(np.float32).mean(1)                        # (N,14,H,W) 배포/test 용
    atm = d["atm"].astype(np.float32) if (args.use_atmo and "atm" in d.files) else None  # (N,14,9,H,W)
    if args.use_atmo:
        assert atm is not None, "--use-atmo 인데 gen npz 에 atm 없음 (gen --dcae 로 재생성)"
        if args.n_atmo:
            atm = atm[:, :, :args.n_atmo]                            # 앞 N채널만 (9채널 효과 분리)
    od = xr.open_zarr("data/oisst_1deg_oceanmean.zarr"); ocean = od["ocean_mask"].values > 0
    cl = np.load("data/oisst_1deg_climatology.npz")
    vis = ocean & ~((np.nanmin(cl["c_mu"], axis=0) <= -1.7) & ocean)
    lat = od["lat"].values; w = (np.cos(np.deg2rad(lat)).clip(0)[:, None] * vis).astype(np.float32)
    H, Wd = vis.shape
    tr = ics.year == 2020; te = ics.year == 2021
    print(f"[fit] train IC {tr.sum()}×{M}mem={tr.sum()*M*14} samp / test IC {te.sum()} (ens-mean)  (2020/2021)")

    W = torch.from_numpy(w).to(dev)
    lead_ch = (np.arange(14, dtype=np.float32) / 13.0)
    use_gen = not args.no_gen
    if not use_gen:
        print("[fit] ★ 대조군: LST_gen 채널 제거 (fc 후처리 효과만 측정)")
    n_atm = atm.shape[2] if atm is not None else 0
    atm_std = None
    if atm is not None:
        atm_std = atm[tr].reshape(-1, n_atm, H * Wd).std(axis=(0, 2)).astype(np.float32) + 1e-6
        print(f"[fit] atmo {n_atm}ch std={atm_std.round(2)}")

    def batches(samp, bs, shuffle, use_mean):
        """samp: (K,3) rows (i_win,i_mem,i_lead). use_mean 이면 gen=ens-mean(배포 일치)."""
        order = np.random.permutation(len(samp)) if shuffle else np.arange(len(samp))
        for s in range(0, len(order), bs):
            j = samp[order[s:s + bs]]; iw, im, il = j[:, 0], j[:, 1], j[:, 2]
            gsrc = gen_mean[iw, il] if use_mean else gen_[iw, im, il].astype(np.float32)
            g = gsrc * (~vis) if use_gen else np.zeros((len(iw), H, Wd), np.float32)
            ch = [fc[iw, il] * vis, g, np.broadcast_to(vis, (len(iw), H, Wd)),
                  np.broadcast_to(lead_ch[il][:, None, None], (len(iw), H, Wd))]
            if atm is not None:
                ch += list((atm[iw, il] / atm_std[None, :, None, None]).transpose(1, 0, 2, 3))
            x = np.stack(ch, 1)
            yield (torch.from_numpy(x).to(dev), torch.from_numpy(tru[iw, il]).to(dev),
                   torch.from_numpy(fc[iw, il]).to(dev), il)

    if args.train_mean:                                              # ★ 대조: 학습도 ens-mean (test 와 정합)
        samp_tr = np.array([(iw, -1, il) for iw in np.where(tr)[0] for il in range(14)])
        print(f"[fit] ★ train 도 ens-mean ({len(samp_tr)} samp)")
    else:
        samp_tr = np.array([(iw, im, il) for iw in np.where(tr)[0] for im in range(M) for il in range(14)])
    samp_te = np.array([(iw, -1, il) for iw in np.where(te)[0] for il in range(14)])
    cin = 4 + n_atm
    net = TinyUNet(cin=cin).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(args.epochs):
        net.train(); tot = n = 0
        for x, y, f0, _ in batches(samp_tr, args.bs, True, use_mean=args.train_mean):
            pr = f0 + net(x)                                          # fc + 보정
            loss = (((pr - y) ** 2) * W).sum() / (W.sum() * x.shape[0])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        print(f"[ep{ep}] train mse(z)={tot/n:.4f}")

    net.eval()
    acc_v = torch.zeros(7, 14, device=dev)                            # se_u, se_f, uo,uu, fo,ff, oo
    with torch.no_grad():
        for x, y, f0, i_lead in batches(samp_te, args.bs, False, use_mean=True):
            pr = f0 + net(x)
            for b in range(x.shape[0]):
                k = int(i_lead[b])
                yy, pp, ff_ = y[b], pr[b], f0[b]
                acc_v[0, k] += (((pp - yy) ** 2) * W).sum() / W.sum()
                acc_v[1, k] += (((ff_ - yy) ** 2) * W).sum() / W.sum()
                acc_v[2, k] += (pp * yy * W).sum(); acc_v[3, k] += (pp * pp * W).sum()
                acc_v[4, k] += (ff_ * yy * W).sum(); acc_v[5, k] += (ff_ * ff_ * W).sum()
                acc_v[6, k] += (yy * yy * W).sum()
    nk = te.sum()
    a = acc_v.cpu().numpy()
    print(f"\n[2021 test] open-ocean lat-weighted — updated(fc+LST_gen) vs fc 단독")
    print(f"{'lead':>5}{'RMSE_upd':>10}{'RMSE_fc':>10}{'corr_upd':>10}{'corr_fc':>10}")
    ru, rf, cu, cf = [], [], [], []
    for k in range(14):
        ru.append(np.sqrt(a[0, k] / nk)); rf.append(np.sqrt(a[1, k] / nk))
        cu.append(a[2, k] / np.sqrt(a[3, k] * a[6, k])); cf.append(a[4, k] / np.sqrt(a[5, k] * a[6, k]))
        print(f"+{k+1:<4d}{ru[-1]:>10.4f}{rf[-1]:>10.4f}{cu[-1]:>10.4f}{cf[-1]:>10.4f}")
    print(f"{'mean':>5}{np.mean(ru):>10.4f}{np.mean(rf):>10.4f}{np.mean(cu):>10.4f}{np.mean(cf):>10.4f}")
    tag = ("nogen" if args.no_gen else "gen") + ("_atmo" if atm is not None else "")
    torch.save({"model": net.state_dict(), "cin": cin, "use_gen": use_gen,
                "use_atmo": atm is not None, "atm_std": atm_std},
               f"S2S_inject/outputs/s2s_inject_14day_sktocean_200ep/cycle_updater_{tag}.pt")
    print(f"saved cycle_updater_{tag}.pt")


def apply(args):
    """updater 로 BC v2 zarr 생성 → cycle-2 는 eval_leadtime_direct --forecast-zarr <v2> 로."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uk = torch.load(args.updater, map_location="cpu", weights_only=False)
    net = TinyUNet(cin=uk["cin"]).to(dev).eval(); net.load_state_dict(uk["model"])
    d = np.load(args.gen_npz)
    ics = pd.to_datetime([str(t) for t in d["ics"].astype(np.int64)], format="%Y%m%d%H")
    fc, gen_ = d["fc"], d["gen"]
    if gen_.ndim == 5:                                               # (N,M,14,H,W) → ens-mean(배포)
        gen_ = gen_.astype(np.float32).mean(1)
    atm = d["atm"].astype(np.float32) if uk["use_atmo"] else None
    if uk["use_atmo"]:
        assert "atm" in d.files, "updater 가 atmo 조건인데 gen npz 에 atm 없음"
        atm = atm[:, :, :len(uk["atm_std"])]                        # updater 학습 채널 수로 정렬
    od = xr.open_zarr("data/oisst_1deg_oceanmean.zarr"); ocean = od["ocean_mask"].values > 0
    cl = np.load("data/oisst_1deg_climatology.npz")
    vis = ocean & ~((np.nanmin(cl["c_mu"], axis=0) <= -1.7) & ocean)
    H, Wd = vis.shape
    lead_ch = (np.arange(14, dtype=np.float32) / 13.0)[:, None, None]
    out = np.full_like(fc, np.nan)
    with torch.no_grad():
        for i in range(len(ics)):
            ch = [fc[i] * vis, gen_[i] * (~vis) if uk["use_gen"] else np.zeros_like(fc[i]),
                  np.broadcast_to(vis, (14, H, Wd)).astype(np.float32),
                  np.broadcast_to(lead_ch, (14, H, Wd))]
            if atm is not None:
                ch += list((atm[i] / uk["atm_std"][None, :, None, None]).transpose(1, 0, 2, 3))
            x = torch.from_numpy(np.stack(ch, 1).astype(np.float32)).to(dev)
            out[i] = np.clip(fc[i] + net(x).cpu().numpy(), -5, 5)
    xr.Dataset({"z": (("time", "lead", "lat", "lon"), out)},
               coords={"time": ics.values, "lead": np.arange(1, 15),
                       "lat": od["lat"].values, "lon": od["lon"].values},
               attrs={"desc": "cycle-2 updated BC (fc + updater[LST_gen(+atmo)])",
                      "updater": args.updater}).chunk({"time": 32}).to_zarr(args.out_zarr, mode="w")
    print(f"[apply] saved {args.out_zarr}  ICs={len(ics)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "fit", "apply"])
    ap.add_argument("--ckpt", default="S2S_inject/outputs/s2s_inject_14day_sktocean_200ep/ckpt_best.pt")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ens", type=int, default=8)
    ap.add_argument("--dcae", default=None, help="지정 시 Atmo(t850,u850,v850) ens-mean 도 저장")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gen-npz", default=OUT)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--no-gen", action="store_true", help="대조군: LST_gen 채널 0 (fc 후처리만)")
    ap.add_argument("--use-atmo", action="store_true", help="Atmo_gen 조건 추가")
    ap.add_argument("--train-mean", action="store_true", help="학습도 ens-mean (per-member 대신, test 정합)")
    ap.add_argument("--n-atmo", type=int, default=None, help="atmo 채널 앞 N개만 (기본 전체 9)")
    ap.add_argument("--updater", default="S2S_inject/outputs/s2s_inject_14day_sktocean_200ep/cycle_updater_gen.pt")
    ap.add_argument("--out-zarr", default="data/skt1deg_forecast_bc_v2.zarr")
    args = ap.parse_args()
    {"gen": gen, "fit": fit, "apply": apply}[args.mode](args)


if __name__ == "__main__":
    main()
