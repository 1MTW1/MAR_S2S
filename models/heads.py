"""
보조 head (s2s_instruction.md §C1, §D2).

C1. 결정론 head: decoder hidden state z_i 로 토큰을 직접 예측(평균 μ). 가중 MSE 는
    앞 10프레임에만 w(i)=e^{-k} (k=프레임 인덱스).
D2. σ head: 같은 head 가 μ 와 log-variance(s) 를 동시 출력(heteroscedastic).
    loss = ‖z-μ‖²·e^{-s}/2 + s/2.
    - μ 의 MSE 는 앞 10프레임만, σ(분산)는 전체 프레임에서 학습.
    - σ 는 hidden state 에서만 측정되며 diffusion 값 생성과 분리(§8-4).
      예측·평가는 μ 로, σ 는 언마스킹 선택 신호로만 사용(§D3).

Stage 1 에서는 사용하지 않는다(deterministic=False). Stage 3 에서 μ-head,
Stage 6 에서 σ-head 를 활성화한다.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DeterministicHead(nn.Module):
    """z_i -> μ (그리고 옵션으로 log-variance s)."""

    def __init__(self, z_channels: int, target_channels: int, width: int = 1024, predict_sigma: bool = False):
        super().__init__()
        self.predict_sigma = predict_sigma
        out = target_channels * (2 if predict_sigma else 1)
        self.target_channels = target_channels
        self.net = nn.Sequential(
            nn.Linear(z_channels, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, out),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """z: (..., z_channels) -> (mu, logvar|None), 각 (..., target_channels)."""
        out = self.net(z)
        if self.predict_sigma:
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar
        return out, None


def frame_decay_weights(num_frames: int, max_frames: int = 10,
                        tail_hold: bool = False, device=None) -> torch.Tensor:
    """프레임별 지수 가중 w(k)=e^{-k} (OmniCast §A.2, frame=리드타임 단위).
    tail_hold=False: k>=max_frames 는 0 (원본 OmniCast 컷오프 — Stage3 baseline).
    tail_hold=True : k>=max_frames 를 frame max_frames 와 동일 가중 e^{-max_frames} 로 유지
                     (완전 0 대신 약한 신호 → σ head 의 reference μ 발산 방지).
    반환 (num_frames,). 정규화는 호출측(loss)의 w.sum() 으로 처리."""
    idx = torch.arange(num_frames, device=device)
    if tail_hold:
        k = torch.clamp(idx, max=max_frames).float()           # k>=max_frames → max_frames
        w = torch.exp(-k)                                       # → e^{-max_frames} 로 유지
    else:
        w = torch.exp(-idx.float())
        w = torch.where(idx < max_frames, w, torch.zeros_like(w))
    return w


def deterministic_mse_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    frame_idx: torch.Tensor,
    max_frames: int = 10,
    tail_hold: bool = False,
    weighted: bool = True,
) -> torch.Tensor:
    """프레임 가중 MSE. weighted=True 면 앞 max_frames 는 e^{-k}, 그 외는 tail_hold 에 따라
    0(기본·baseline) 또는 e^{-max_frames}(유지). weighted=False 면 모든 프레임 균등 가중(plain MSE)
    — 짧은 horizon(예: 3일) 에선 e^{-k} 감쇠가 불필요하므로 끈다. w.sum() 으로 정규화(=합1과 동치).
    mu/target:(N,D), frame_idx:(N,)."""
    if weighted:
        w_table = frame_decay_weights(int(frame_idx.max()) + 1, max_frames, tail_hold, device=mu.device)
        w = w_table[frame_idx][:, None]  # (N,1)
    else:
        w = torch.ones(frame_idx.shape[0], 1, device=mu.device)  # 균등 가중
    se = ((mu - target) ** 2) * w
    denom = w.sum().clamp(min=1e-8) * mu.shape[-1]
    return se.sum() / denom


def heteroscedastic_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    frame_idx: torch.Tensor,
    mu_max_frames: int = 10,
) -> torch.Tensor:
    """heteroscedastic loss (§D2). ★ 이 함수 하나가 μ·σ 학습을 모두 담당한다 —
    바깥에서 deterministic_mse_loss(가중 MSE)를 따로 더하지 말 것(중복).
        μ 항: 앞 mu_max_frames 는 e^{-k}, 그 외는 frame mu_max_frames 와 동일 가중
              e^{-mu_max_frames}(tail_hold=True). σ 항이 μ 를 detach 하므로 이 항이 μ 의
              '유일한' 학습원 → 후반 μ 를 0 대신 약하게 학습시켜 σ reference 발산을 막는다.
        σ 항: 전체 프레임 full-weight ‖z-μ‖²·e^{-s}/2 + s/2 (μ detach). σ 는 D3(전 44프레임
              언마스킹 선택)용이므로 후반을 downweight 하지 않는다.
    """
    mu_term = deterministic_mse_loss(mu, target, frame_idx, mu_max_frames, tail_hold=True)
    s = logvar
    sigma_term = (((target - mu.detach()) ** 2) * torch.exp(-s) * 0.5 + 0.5 * s).mean()
    return mu_term + sigma_term
