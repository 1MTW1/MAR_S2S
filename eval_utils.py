"""평가 공용 헬퍼 — ckpt 로드 / DCAE 디코드 / skt un-patchify.
(eval_leadtime_direct.py, cycle_lst.py 가 사용)"""
import numpy as np
import torch

from baseline.train_mar import build_model


def unpatch_skt(patch, h, w, P):
    """(T,h,w,P*P) → (T, h*P, w*P) 물리 격자(z)."""
    T = patch.shape[0]
    return patch.reshape(T, h, w, P, P).transpose(0, 1, 3, 2, 4).reshape(T, h * P, w * P)


@torch.no_grad()
def decode_sliced(fut, dcae, lat_mean, lat_std, fmean, fstd, target_std, n_ch):
    """fut:(M,T,Cz,h,w) 표준화 latent → (M,T,n_ch,H,W) 물리. DCAE 14ch decode 후 앞 n_ch 슬라이스."""
    use_bf16 = fut.device.type == "cuda"
    out = []
    for e in range(fut.shape[0]):
        z = fut[e].float()                                # (T,Cz,h,w)
        if target_std is not None:
            z = z / target_std
        z = z * lat_std + lat_mean
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            phys = dcae.decode(z, return_dict=False)[0]   # (T,14,H,W)
        out.append((phys[:, :n_ch].float() * fstd + fmean).cpu().numpy())
    return np.stack(out, 0)


def load_inject(ckpt_path, device, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = dict(cfg["model"])
    mc["__future_len"] = cfg["data"]["future_len"]
    mc["__cond_len"] = cfg["data"].get("input_len", 1)
    model = build_model(mc)
    sd = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg
