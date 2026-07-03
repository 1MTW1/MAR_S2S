"""
S2S_inject 학습 — dual-stream MM-DiT MAR (대기 latent + 결합 skt 1° direct 주입).

open-ocean BC visible, 대기+육지 LST 생성. accelerate 멀티-GPU + EMA. (RMSE 평가는 eval_leadtime_direct.)
production config: configs/s2s_inject_14day_sktocean_200ep.yaml (GT skt BC, 200ep, PP 전략)

  uv run accelerate launch --config_file S2S_inject/configs/accelerate.yaml -m baseline.train_mar \
      --config S2S_inject/configs/s2s_inject_14day_sktocean_200ep.yaml   [--max-steps 3]
"""
import argparse
import math
import os

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from baseline.data.mar_dataset import prepare_inject_direct_dataloader
from baseline.models.mar import S2SInjectTransformer


def build_model(mc) -> S2SInjectTransformer:
    return S2SInjectTransformer(
        latent_channels=mc["latent_channels"], latent_h=mc["latent_h"], latent_w=mc["latent_w"],
        future_len=mc["__future_len"], cond_len=mc.get("__cond_len", 1),
        skt_patch=mc.get("skt_patch", 18),
        embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        n_dual=mc.get("n_dual", 6), n_single=mc.get("n_single", 6), mlp_ratio=mc.get("mlp_ratio", 4.0),
        diff_width=mc["diff_width"], diff_depth=mc["diff_depth"],
        diff_batch_mul=mc.get("diff_batch_mul", 2),
        num_sampling_timesteps=mc.get("num_sampling_timesteps", 32),
        sigma_data=mc.get("sigma_data", 0.5), mask_ratio_min=mc.get("mask_ratio_min", 0.5),
        skt_recon_weight=mc.get("skt_recon_weight", 1.0), dropout=mc.get("dropout", 0.0),
        det_weight=mc.get("det_weight", 0.5), det_max_frames=mc.get("det_max_frames", 10),
        ocean_proj=mc.get("ocean_proj", False),
    )


def schedule_eps(step, total_steps, cc, floor=None):
    """GT 사용 확률 ε: warmup(=1) → inverse sigmoid ramp → plateau(floor). (instruction §2)
    floor=None 이면 SST용 eps_floor, 명시하면 그 값(대기용 eps_atmo_floor 등)."""
    warm = cc.get("warmup_frac", 0.15) * total_steps
    ramp_end = cc.get("ramp_end_frac", 0.75) * total_steps
    floor = cc.get("eps_floor", 0.1) if floor is None else floor
    k = cc.get("k_sigmoid", 9.0)
    if step < warm:
        return 1.0
    if step >= ramp_end:
        return floor
    t = (step - warm) / max(ramp_end - warm, 1.0)
    s = 1.0 / (1.0 + math.exp(k * (t - 0.5)))                # 1 → 0
    return floor + (1.0 - floor) * s


def make_loader(dc, tc, split, shuffle):
    return prepare_inject_direct_dataloader(
        latents_zarr=dc["latents_zarr"], var=dc["var"], latent_stats=dc["latent_stats"],
        start_date=dc[split][0], end_date=dc[split][1],
        oisst_1deg_zarr=dc["oisst_1deg_zarr"], oisst_1deg_clim_npz=dc["oisst_1deg_clim_npz"],
        skt_anomaly_zarr=dc["skt_anomaly_zarr"],
        future_len=dc["future_len"], input_len=dc.get("input_len", 1), stride=dc["stride"],
        target_std=dc.get("target_std"), batch_size=tc["batch_size"], shuffle=shuffle,
        num_workers=tc["num_workers"], load_in_memory=dc["load_in_memory"], drop_last=shuffle,
        skt_clip=dc.get("skt_clip", 5.0), skt_patch=dc.get("skt_patch", 18),
        ocean_thresh=dc.get("skt_ocean_thresh", 0.5),
        ocean_source=dc.get("ocean_source", "oisst"),
        mask_seaice=dc.get("mask_seaice", False), ice_thresh=dc.get("ice_thresh", -1.7),
        refined_zarr=dc.get("refined_zarr"), forecast_zarr=dc.get("forecast_zarr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                    help="다른 ckpt(.pt)에서 모델 가중치만 warm-start (Stage 3 end-to-end fine-tune용)")
    ap.add_argument("--checkpoints_total_limit", type=int, default=3)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]
    dc.setdefault("skt_patch", mc.get("skt_patch", 18))
    dc.setdefault("skt_ocean_thresh", mc.get("skt_ocean_thresh", 0.5))
    dc.setdefault("skt_land_thresh", mc.get("skt_land_thresh", 0.5))
    mc["__future_len"], mc["__cond_len"] = dc["future_len"], dc.get("input_len", 1)

    coupled_cfg = tc.get("coupled") or {}                    # instruction.md 2-iter unroll (있으면 활성)
    alt = bool(tc.get("alt"))                                # (레거시) 2-iter 교대 학습 (forward_alt)
    sst_given = bool(dc.get("oisst_sst_zarr"))               # ★ SST-given(OISST 0.25° encoder) 경로
    sst_direct = bool(dc.get("oisst_1deg_zarr"))             # ★ OISST 1° 직접주입 경로
    ddp_kw = []
    if coupled_cfg or alt:                                   # 일부 param no_grad 가능 → 허용
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kw = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16" if tc.get("amp", False) else "no",
                              gradient_accumulation_steps=tc.get("grad_accum", 1), kwargs_handlers=ddp_kw)
    set_seed(tc["seed"])
    if accelerator.is_main_process:
        os.makedirs(tc["output_dir"], exist_ok=True)

    loader = make_loader(dc, tc, "train", shuffle=True)
    val_loader = make_loader(dc, tc, "val", shuffle=False) if dc.get("val") else None

    model = build_model(mc)
    init_from = args.init_from or tc.get("init_from")
    if init_from:                                       # Stage 3: 가중치 warm-start (EMA 우선)
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        accelerator.print(f"[init-from] {init_from} (missing={len(miss)}, unexpected={len(unexp)})")
    if accelerator.is_main_process:
        accelerator.print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
                          f"(dual={mc.get('n_dual',6)}, single={mc.get('n_single',6)})")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"],
                            betas=tuple(tc["betas"]))
    steps_per_epoch = math.ceil(len(loader) / accelerator.num_processes /
                                accelerator.gradient_accumulation_steps)
    total_steps = steps_per_epoch * tc["num_epochs"]
    warmup = round(tc.get("warmup_epochs", 0) * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(opt, warmup * accelerator.num_processes,
                                            total_steps * accelerator.num_processes)
    model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    ema_model = None
    if tc.get("use_ema"):
        ema_model = EMAModel(accelerator.unwrap_model(model).parameters(), decay=tc["ema_decay"])
        ema_model.to(accelerator.device)

    gstep, first_epoch = 0, 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = sorted([d for d in os.listdir(tc["output_dir"]) if d.startswith("checkpoint")],
                          key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)
        if path:
            accelerator.load_state(os.path.join(tc["output_dir"], path))
            first_epoch = int(path.split("-")[1]) + 1
            gstep = first_epoch * steps_per_epoch
            accelerator.print(f"[resume] {path}")

    best_val = float("inf")
    model.train()
    progress = tqdm(total=args.max_steps or total_steps, initial=gstep, desc="inject train",
                    disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    for epoch in range(first_epoch, tc["num_epochs"]):
        for latents, ts, skt_in, skt_tgt, ocean_tok in loader:
            eps = schedule_eps(gstep, total_steps, coupled_cfg) if coupled_cfg else None
            eps_atmo = schedule_eps(gstep, total_steps, coupled_cfg,
                                    floor=coupled_cfg.get("eps_atmo_floor", 0.0)) if coupled_cfg else 0.0
            with accelerator.accumulate(model):
                out = model(latents, ts, skt_in, skt_tgt, ocean_tok,
                            eps=eps, eps_atmo=eps_atmo, alt=alt, sst_given=sst_given,
                            sst_direct=sst_direct)
                accelerator.backward(out["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.step(model.parameters())
                gstep += 1; progress.update(1)
                post = {"epoch": epoch, "loss": f"{out['loss'].item():.4f}",
                        "atmo": f"{out['atmo_loss'].item():.4f}", "skt": f"{out['skt_loss'].item():.4f}",
                        "lr": f"{sched.get_last_lr()[0]:.1e}"}
                if "det_loss" in out:
                    post["det"] = f"{out['det_loss'].item():.4f}"
                if "sst_res_loss" in out:
                    post["sstr"] = f"{out['sst_res_loss'].item():.4f}"
                if eps is not None:
                    post["eps"] = f"{eps:.2f}"; post["epsA"] = f"{eps_atmo:.2f}"
                progress.set_postfix(**post)
                if args.max_steps is not None and gstep >= args.max_steps:
                    progress.close()
                    accelerator.print("[smoke] max-steps reached; saving and exiting.")
                    save_ckpt(accelerator, "smoke", model, ema_model, cfg, gstep, tc["output_dir"])
                    return

        last = (epoch == tc["num_epochs"] - 1)
        if val_loader is not None and ((epoch + 1) % tc.get("validate_every_epochs", 5) == 0 or last):
            val_eps = 0.0 if coupled_cfg else None       # coupled: ε=0(추론 동일) 고정 평가 → best 편향 제거
            vres = validate(model, val_loader, accelerator, ema_model, tc.get("val_max_batches"),
                            eps=val_eps, alt=alt, sst_given=sst_given,
                            sst_direct=sst_direct)   # 학습과 동일 경로로 검증
            accelerator.print(f"[val] epoch {epoch}: loss={vres['loss']:.4f} "
                              f"(atmo={vres['atmo']:.4f} skt={vres['skt']:.4f} sstr={vres['sst_res']:.4f})")
            if accelerator.is_main_process:   # ★ val 결과를 파일로 기록(모니터링용)
                import json
                with open(os.path.join(tc["output_dir"], "val_log.jsonl"), "a") as f:
                    f.write(json.dumps({"epoch": epoch, "step": gstep, **vres}) + "\n")
            # ★ best-val 추적 — atmo 손실 기준(예보가 목표). 갱신 시 ckpt_best 저장(과적합 전 모델 보존).
            if vres["atmo"] < best_val:
                best_val = vres["atmo"]
                save_ckpt(accelerator, "best", model, ema_model, cfg, gstep, tc["output_dir"])
                accelerator.print(f"[val] ★ new best (atmo={best_val:.4f}, epoch {epoch}) → ckpt_best.pt")
        if (epoch + 1) % tc.get("save_every_epochs", 10) == 0 or last:
            save_ckpt(accelerator, "latest", model, ema_model, cfg, gstep, tc["output_dir"])
        if (epoch + 1) % tc.get("checkpoint_every_epochs", 10) == 0 or last:
            sp = os.path.join(tc["output_dir"], f"checkpoint-{epoch}")
            accelerator.save_state(sp)
            if accelerator.is_main_process:
                _prune(tc["output_dir"], args.checkpoints_total_limit)
    progress.close()


@torch.no_grad()
def validate(model, val_loader, accelerator, ema_model=None, max_batches=None, eps=None, alt=False,
             sst_given=False, sst_direct=False):
    """반환 dict(loss, atmo, skt, sst_res) — 과적합 진단용 분리 로깅.
    alt=True 면 학습과 동일한 forward_alt(2-iter 교대, SST 가림→예측값만 대기에)로 평가.
    coupled 학습이면 eps=0(추론과 동일: 보정 SST만)으로 ★고정 평가 → ε 난이도 변동에 따른
    best 편향(warmup 쪽으로 쏠림) 제거. alt·eps 모두 없으면 단일 MAR forward."""
    unwrapped = accelerator.unwrap_model(model)
    if ema_model is not None:
        ema_model.store(unwrapped.parameters()); ema_model.copy_to(unwrapped.parameters())
    # ★ diffusion 노이즈까지 고정(재현성) — 학습 RNG 스트림은 보존 후 복원.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(1234)
    was_training = model.training
    model.eval()
    tot = torch.zeros(5, device=accelerator.device)   # [loss, atmo, skt, sst_res, n]
    for bi, (latents, ts, skt_in, skt_tgt, ocean_tok) in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        out = model(latents, ts, skt_in, skt_tgt, ocean_tok,
                    eps=eps, eps_atmo=0.0, alt=alt, sst_given=sst_given, sst_direct=sst_direct)
        b = latents.shape[0]
        tot[0] += out["loss"].detach() * b
        tot[1] += out["atmo_loss"].detach() * b
        tot[2] += out["skt_loss"].detach() * b
        tot[3] += out["sst_res_loss"].detach() * b if "sst_res_loss" in out else 0.0
        tot[4] += b
    tot = accelerator.reduce(tot, reduction="sum")
    if ema_model is not None:
        ema_model.restore(unwrapped.parameters())
    torch.set_rng_state(cpu_rng)                       # 학습 RNG 스트림 복원
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if was_training:
        model.train()
    n = tot[4].clamp(min=1)
    return {"loss": (tot[0] / n).item(), "atmo": (tot[1] / n).item(), "skt": (tot[2] / n).item(),
            "sst_res": (tot[3] / n).item()}


def save_ckpt(accelerator, tag, model, ema_model, cfg, step, out_dir):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    ema_state = None
    if ema_model is not None:
        ema_model.store(unwrapped.parameters()); ema_model.copy_to(unwrapped.parameters())
        ema_state = {k: v.clone() for k, v in unwrapped.state_dict().items()}
        ema_model.restore(unwrapped.parameters())
    torch.save({"model": unwrapped.state_dict(), "ema": ema_state, "config": cfg, "step": step},
               os.path.join(out_dir, f"ckpt_{tag}.pt"))
    accelerator.print(f"[ckpt] saved ckpt_{tag}.pt")


def _prune(out_dir, limit):
    if limit is None:
        return
    import shutil
    ck = sorted([d for d in os.listdir(out_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]))
    for d in ck[:max(0, len(ck) - limit)]:
        shutil.rmtree(os.path.join(out_dir, d))


if __name__ == "__main__":
    main()
