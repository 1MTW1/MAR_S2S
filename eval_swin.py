"""
SST_swin/evaluate.py — lead-time별 area-weighted RMSE / Bias / ACC (instruction §6) · accelerate · YAML config.

  · 예측 z → Kelvin 복원 후 ocean·위도가중 집계. RMSE/Bias=K. ACC=doy-climatology anomaly 상관(평가 전용).
  · persistence(입력 마지막날) 베이스라인 동시. 윈도우 GPU 분할.
  · target=raw 면 clim 없어도 RMSE/Bias 가능(ACC 는 clim 있을 때만). anomaly/residual 은 clim 필수(z→K 복원).

실행:
  accelerate launch --config_file SST_swin/accelerate.yaml -m SST_swin.evaluate \
      --config SST_swin/configs/sst_swin_1deg.yaml --ckpt SST_swin/outputs/sst_swin_1deg/ckpt_best.pt
"""
import argparse

import numpy as np
import torch
import yaml
from accelerate import Accelerator

from baseline.data.swin_dataset import S2SSwinDataset, build_static, compute_grid_pad, doy_slot
from baseline.models.swin import UTransformer


def _unpad(a, pad, H, W):
    return a[..., pad[0][0]:pad[0][0] + H, pad[1][0]:pad[1][0] + W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ice-thresh", type=float, default=-1.7,
                    help="해빙 판정: 연중 최저 clim SST ≤ thresh → 해빙권. clim 있으면 full/open-ocean 분리 출력")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dc, mc, ec = cfg["data"], cfg["model"], cfg.get("eval", {})

    accelerator = Accelerator(); device = accelerator.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    T = dc["T"]                                                # residual 전용: pred=persist+net, z→K=c_sig[doy]

    pad, (Hp, Wp), (H, W) = compute_grid_pad(dc["forcing_zarr"], mc["patch_size"],
                                             mc["window_size"], len(mc["depths"]))
    model = UTransformer(img_size=(Hp, Wp), patch_size=mc["patch_size"], T=T,
                         embed_dim=mc["embed_dim"], depths=tuple(mc["depths"]),
                         depths_up=tuple(mc["depths_up"]), num_heads=tuple(mc["num_heads"]),
                         window_size=mc["window_size"], mlp_ratio=mc.get("mlp_ratio", 4.0),
                         ape=mc.get("ape", False))
    model.load_state_dict(ck["model"]); model.to(device).eval()

    ds = S2SSwinDataset(dc["forcing_zarr"], dc["test"][0], dc["test"][1], pad,
                        in_len=T, out_len=T, stride=ec.get("stride", dc["stride"]),
                        clim_npz=dc["clim_npz"], load_in_memory=dc.get("load_in_memory", True))
    ocean, latw = build_static(dc["forcing_zarr"], pad)
    ocean_np = _unpad(ocean.numpy(), pad, H, W); latw_np = latw.numpy()[pad[0][0]:pad[0][0] + H]
    w2 = ocean_np * latw_np[:, None]; denom = w2.sum()

    clim_npz = dc["clim_npz"]
    c_mu = c_sig = None
    regions = [("full ocean", w2)]
    if clim_npz:
        cl = np.load(clim_npz)
        # ★ 해빙권(연중 최저 clim SST ≤ thresh) 분리 — nan_to_num 전의 raw clim 으로 판정 (land=all-NaN → 해빙 아님)
        with np.errstate(invalid="ignore"):
            ice = (np.nanmin(cl["c_mu"].astype(np.float32), axis=0) <= args.ice_thresh) & (ocean_np > 0)
        if ice.shape == w2.shape and ice.any():
            regions.append(("open ocean(해빙 제외)", w2 * (~ice)))
        # OISST clim 은 land=NaN → 0 으로(ocean mask w2=0 로 어차피 제외; 0*NaN=NaN 누수 방지 → ACC/anom RMSE nan 방지)
        c_mu = np.nan_to_num(cl["c_mu"].astype(np.float32), nan=0.0)
        c_sig = np.nan_to_num(cl["c_sig"].astype(np.float32), nan=0.0)
    have_acc = c_mu is not None                                  # ACC 가능 여부
    R = len(regions)
    W2 = np.stack([w for _, w in regions]).astype(np.float32)    # (R,H,W)
    denoms = W2.sum((1, 2))                                      # (R,)
    if R > 1:
        accelerator.print(f"[SST_swin eval] 해빙권 비율 {ice.mean():.3f} (ocean 중 {ice.sum()/max((ocean_np>0).sum(),1):.3f})")

    se_m = np.zeros((R, T)); se_p = np.zeros((R, T)); bias_m = np.zeros((R, T))
    fo = np.zeros((R, T)); ff = np.zeros((R, T)); oo = np.zeros((R, T))
    pfo = np.zeros((R, T)); pff = np.zeros((R, T))
    win = list(range(len(ds)))
    local = win[accelerator.process_index::accelerator.num_processes]
    accelerator.print(f"[SST_swin eval] residual clim={'on' if have_acc else 'off(ACC생략)'} "
                      f"windows {len(win)} (~{len(local)}/proc)")
    for i in local:
        x, y, _ = ds[i]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = x[-1:].expand(T, -1, -1) + model(x[None].to(device))[0].float().cpu()   # persist+net
        s = ds.starts[i]
        doy = doy_slot(ds.time[s + T: s + 2 * T])                  # target 프레임 doy slot
        cs = c_sig[doy]; cm = c_mu[doy]                            # z → K: raw = z·c_sig[doy] + c_mu[doy]
        pk = _unpad(out.numpy(), pad, H, W) * cs + cm
        ok = _unpad(y.numpy(), pad, H, W) * cs + cm
        persk = _unpad(x[-1].numpy(), pad, H, W)[None] * cs + cm
        Wr = W2[:, None]                                          # (R,1,H,W)
        se_m += ((pk - ok) ** 2 * Wr).sum((2, 3)) / denoms[:, None]
        se_p += ((persk - ok) ** 2 * Wr).sum((2, 3)) / denoms[:, None]
        bias_m += ((pk - ok) * Wr).sum((2, 3)) / denoms[:, None]
        if have_acc:
            cm = c_mu[doy]; fa = pk - cm; oa = ok - cm; pa = persk - cm
            fo += (fa * oa * Wr).sum((2, 3)); ff += (fa * fa * Wr).sum((2, 3))
            oo += (oa * oa * Wr).sum((2, 3))
            pfo += (pa * oa * Wr).sum((2, 3)); pff += (pa * pa * Wr).sum((2, 3))

    def asum(a):
        t = torch.tensor(np.asarray(a, dtype=np.float64), device=device)
        return accelerator.reduce(t, reduction="sum").cpu().numpy()
    se_m, se_p, bias_m = asum(se_m), asum(se_p), asum(bias_m)
    fo, ff, oo, pfo, pff = asum(fo), asum(ff), asum(oo), asum(pfo), asum(pff)
    n = asum(np.array([len(local)]))[0]
    if not accelerator.is_main_process:
        return

    print(f"\n[SST_swin eval] test {dc['test'][0]}~{dc['test'][1]}, windows {int(n)}, residual\n")
    for r, (rname, _) in enumerate(regions):
        rmse_m = np.sqrt(se_m[r] / n); rmse_p = np.sqrt(se_p[r] / n); bias = bias_m[r] / n
        print(f"=== {rname} ===")
        if have_acc:
            acc_m = fo[r] / np.sqrt(np.clip(ff[r] * oo[r], 1e-12, None))
            acc_p = pfo[r] / np.sqrt(np.clip(pff[r] * oo[r], 1e-12, None))
            print(f"{'lead':>5}{'RMSE_m':>10}{'RMSE_persist':>14}{'Bias_m':>10}{'ACC_m':>9}{'ACC_persist':>13}")
            for t in range(T):
                print(f"+{t+1:<4d}{rmse_m[t]:>10.4f}{rmse_p[t]:>14.4f}{bias[t]:>10.4f}{acc_m[t]:>9.4f}{acc_p[t]:>13.4f}")
            print(f"{'mean':>5}{rmse_m.mean():>10.4f}{rmse_p.mean():>14.4f}{bias.mean():>10.4f}"
                  f"{acc_m.mean():>9.4f}{acc_p.mean():>13.4f}\n")
        else:
            print(f"{'lead':>5}{'RMSE_m':>10}{'RMSE_persist':>14}{'Bias_m':>10}  (ACC 생략: clim 없음)")
            for t in range(T):
                print(f"+{t+1:<4d}{rmse_m[t]:>10.4f}{rmse_p[t]:>14.4f}{bias[t]:>10.4f}")
            print(f"{'mean':>5}{rmse_m.mean():>10.4f}{rmse_p.mean():>14.4f}{bias.mean():>10.4f}\n")


if __name__ == "__main__":
    main()
