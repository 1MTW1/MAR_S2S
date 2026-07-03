"""
SST_swin/train.py — U-Transformer 학습 (instruction §5) · ★accelerate 멀티-GPU · ★YAML config.

  AdamW(0.9,0.95)·cosine LR·ocean·위도가중 L2·best-val·early stop. 과거 14일 → 미래 14일 direct.
  격자(pad/window)는 config 의 model 섹션 + forcing zarr native 해상도에서 자동 계산(1°/0.25° 공용).

실행:
  accelerate launch --config_file SST_swin/accelerate.yaml -m SST_swin.train \
      --config SST_swin/configs/sst_swin_1deg.yaml
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

from baseline.data.swin_dataset import S2SSwinDataset, build_static, compute_grid_pad
from baseline.loss import masked_latweighted_l2
from baseline.models.swin import UTransformer


def make_loader(dc, pad, split, shuffle):
    rng = dc[split]
    ds = S2SSwinDataset(dc["forcing_zarr"], rng[0], rng[1], pad,
                        in_len=dc["T"], out_len=dc["T"], stride=dc["stride"],
                        clim_npz=dc["clim_npz"],
                        augment=(split == "train" and dc.get("augment", False)),
                        load_in_memory=dc.get("load_in_memory", True),
                        return_scale=(split == "val"))               # ★ val 은 물리 std 동봉(K RMSE)
    return DataLoader(ds, batch_size=dc["batch_size"], shuffle=shuffle,
                      num_workers=dc["num_workers"], drop_last=shuffle, pin_memory=True)


def predict(model, x):
    """anomaly-persistence residual: 예측 = persistence(a_IC=마지막 입력일) + net."""
    return x[:, -1:].expand(-1, model.T if hasattr(model, "T") else x.shape[1], -1, -1) + model(x)


@torch.no_grad()
def validate(model, loader, latw, accelerator):
    """★물리 K weighted RMSE. pred=x[-1](persist)+net → (pred−gt)·c_sig[doy]=K 오차."""
    model.eval()
    lw = latw[:, None] if latw.dim() == 1 else latw              # (Hp,1)
    tot = torch.zeros(2, device=accelerator.device)             # [Σ se_K, Σ w]
    for x, y, mask, scale in loader:                             # scale=타깃 물리 std (B,T,Hp,Wp)
        with accelerator.autocast():
            pred = predict(model, x)                             # a_IC + net
        errK = (pred.float() - y.float()) * scale.float()       # 물리 K 오차 ((pred−gt)·c_sig/σ)
        w = (mask.float() * lw)[:, None]                        # (B,1,Hp,Wp) ocean·lat
        tot[0] += (errK ** 2 * w).sum().detach()
        tot[1] += w.expand_as(errK).sum().detach()
    tot = accelerator.reduce(tot, reduction="sum")              # ★ 전 GPU 합산
    return torch.sqrt(tot[0] / tot[1].clamp(min=1)).item()      # 물리 K RMSE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (data/model/train/eval)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]
    dc.setdefault("batch_size", tc["batch_size"]); dc.setdefault("num_workers", tc["num_workers"])

    accelerator = Accelerator(mixed_precision="bf16" if tc.get("amp") else "no")
    torch.manual_seed(tc["seed"]); np.random.seed(tc["seed"])
    if accelerator.is_main_process:
        os.makedirs(tc["out_dir"], exist_ok=True)

    # ── 격자/pad 자동 계산 (forcing native 해상도 + model 설정) ──
    pad, (Hp, Wp), (H, W) = compute_grid_pad(dc["forcing_zarr"], mc["patch_size"],
                                             mc["window_size"], len(mc["depths"]))
    accelerator.print(f"[grid] native {H}×{W} → pad {pad} → {Hp}×{Wp}")

    # anomaly z 는 per-doy climatology 로 표준화되므로 전역 μ,σ 불필요.
    ocean, latw = build_static(dc["forcing_zarr"], pad)
    latw = latw.to(accelerator.device)                           # lat 가중(경도 불변), mask 는 sample별 loader 제공

    train_loader = make_loader(dc, pad, "train", shuffle=True)
    val_loader = make_loader(dc, pad, "val", shuffle=False)

    model = UTransformer(img_size=(Hp, Wp), patch_size=mc["patch_size"], T=dc["T"],
                         embed_dim=mc["embed_dim"], depths=tuple(mc["depths"]),
                         depths_up=tuple(mc["depths_up"]), num_heads=tuple(mc["num_heads"]),
                         window_size=mc["window_size"], mlp_ratio=mc.get("mlp_ratio", 4.0),
                         drop=mc.get("dropout", 0.0), attn_drop=mc.get("dropout", 0.0),
                         ape=mc.get("ape", False))
    nn.init.zeros_(model.head.weight); nn.init.zeros_(model.head.bias)   # ★ head zero-init → 시작=persistence
    accelerator.print(f"[model] residual params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
                      f"(dropout={mc.get('dropout',0.0)}, wd={tc['weight_decay']}, augment={dc.get('augment')})")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.95),
                            weight_decay=tc["weight_decay"])
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])

    best = float("inf"); bad = 0
    for epoch in range(tc["epochs"]):
        model.train(); run = 0.0
        for i, (x, y, mask) in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss = masked_latweighted_l2(predict(model, x), y, mask, latw)
            accelerator.backward(loss); opt.step()
            run += loss.item()
            if i % 50 == 0:
                accelerator.print(f"  e{epoch} [{i}/{len(train_loader)}] loss={loss.item():.4f}")
        vloss = validate(model, val_loader, latw, accelerator)   # ★ 물리 K RMSE
        accelerator.print(f"[epoch {epoch}] train_z={run/max(len(train_loader),1):.4f}  "
                          f"val_RMSE={vloss:.4f}K  lr={sched.get_last_lr()[0]:.2e}")

        if accelerator.is_main_process:
            ck = {"model": accelerator.unwrap_model(model).state_dict(), "config": cfg}
            torch.save(ck, os.path.join(tc["out_dir"], "ckpt_latest.pt"))
            if vloss < best:
                torch.save(ck, os.path.join(tc["out_dir"], "ckpt_best.pt"))
                accelerator.print(f"[epoch {epoch}] ★ new best val_RMSE={vloss:.4f}K → ckpt_best.pt")
        bad = 0 if vloss < best - 1e-5 else bad + 1
        best = min(best, vloss)
        sched.step()
        accelerator.wait_for_everyone()
        if bad >= tc.get("patience", 10 ** 9):
            accelerator.print(f"[early-stop] val {tc['patience']} epoch 미개선 → 종료 (best={best:.4f})")
            break


if __name__ == "__main__":
    main()
