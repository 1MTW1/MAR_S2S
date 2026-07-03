"""
SST_swin loss — 위도 가중(latitude-weighted) L2 + ocean mask (instruction §4).

    L2 = Σ_i m_i·cos(θ_i)·(y_i−ŷ_i)²  /  Σ_i m_i·cos(θ_i)

  · m_i  : ocean mask (ocean=1, land/pad=0) — land·padding 픽셀은 loss·역전파에서 완전 제외.
  · cosθ : 위도 면적 가중 (정규 위경도 격자 고위도 과대표집 보정).
  · 분모 = 가중치 합 → 시간·배치·공간 모두 마스킹된 평균.
"""
import torch


def lat_weights(lats_deg: torch.Tensor) -> torch.Tensor:
    """위도(°) → cos(θ) 가중. 평균 1 로 정규화(스케일 안정)."""
    w = torch.cos(torch.deg2rad(lats_deg)).clamp(min=0)
    return w / w.mean().clamp(min=1e-8)


def masked_latweighted_l2(pred: torch.Tensor, target: torch.Tensor,
                          ocean_mask: torch.Tensor, latw: torch.Tensor) -> torch.Tensor:
    """pred,target: (B,T,H,W). ocean_mask: (H,W) 또는 ★(B,H,W)(경도-roll 증강 시 sample별). latw: (H,) 또는 (H,1).
    반환: 스칼라 (가중 평균 MSE)."""
    if latw.dim() == 1:
        latw = latw[:, None]                       # (H,1)
    w = ocean_mask * latw                          # (H,W) 또는 (B,H,W)
    w = w[None, None] if w.dim() == 2 else w[:, None]   # (1,1,H,W) 또는 (B,1,H,W)
    se = (pred - target) ** 2 * w
    return se.sum() / w.expand_as(se).sum().clamp(min=1e-8)
