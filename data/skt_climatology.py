"""skt(skin temperature) day-of-year climatology — doy 규칙의 단일 정의처.

S2S BC(boundary forcing) 의 skt 채널에 대해 **doy 별 pixelwise climatology/std** 를
다룬다. 설계는 `code/scripts/compute_climatology.py`(IMPLEMENTATION_SPEC v4 §3) 컨벤션:

  - climatology(c_mu): doy 별 pixelwise 평균장 (366, H, W) — 2단계(raw doy 평균 → ±w일 smoothing).
  - std(c_sig): doy 별 pixelwise 표준편차장 (366, H, W) — 동일 2단계.
  - 표준화 anomaly(= 평년 대비 "몇 std", per-pixel): z = (skt − c_mu[doy]) / c_sig[doy].
    persistence 복원은 이 z 를 고정하고 **각 미래 날의 자기 doy std** 로 되돌린다:
        skt_est(t) = c_mu[doy_t] + z · c_sig[doy_t]
    → 분산의 계절 변화(c_sig[doy_t]≠c_sig[doy_IC])까지 반영.
  - ★ 모델 입력 최종 정규화는 여기 c_sig 가 아니라 forcing_stats 의 **global 단일채널
    (skt−skt_mu)/skt_sig** 로 한다(학습 정규화와 일치). 본 모듈의 c_mu/c_sig 는 오직
    z-score 계산·persistence 복원에만 쓴다.

doy 인덱싱은 기준 윤년 2000 의 (month, day) ordinal 로 매핑한다(평년/윤년이 같은
(월,일)에 같은 slot, Feb 29 → slot 59). 전처리 스크립트·데이터셋이 모두 이 모듈의
`doy_slot` 을 import 해서 동일 규칙을 쓴다.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# 366-slot 테이블의 기준 윤년. (month, day) → 이 해의 ordinal 로 매핑 → 평년/윤년이
# 같은 (월,일)에 같은 slot. Feb 29 → slot 59.
_LEAP_REF_YEAR = 2000
_N_SLOTS = 366
_FEB29_SLOT = (date(_LEAP_REF_YEAR, 2, 29).toordinal()
               - date(_LEAP_REF_YEAR, 1, 1).toordinal())   # = 59


def doy_slot(times) -> np.ndarray:
    """datetime64(스칼라/배열) → 366-slot day-of-year 인덱스 ∈ [0, 365].

    기준 윤년 2000 의 (month, day) ordinal 로 매핑:
        slot = ordinal(2000, m, d) − ordinal(2000, 1, 1)
    → 평년/윤년 모두 같은 (월,일)이면 같은 slot. 평년 Mar 1 은 slot 60(Feb 29 슬롯 건너뜀).
    """
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    base = date(_LEAP_REF_YEAR, 1, 1).toordinal()
    memo: dict[tuple[int, int], int] = {}
    out = np.empty(len(idx), dtype=np.int64)
    for i, (m, d) in enumerate(zip(idx.month, idx.day)):
        key = (int(m), int(d))
        slot = memo.get(key)
        if slot is None:
            slot = date(_LEAP_REF_YEAR, key[0], key[1]).toordinal() - base
            memo[key] = slot
        out[i] = slot
    return out


class SktClimatology:
    """skt doy별 pixelwise climatology(c_mu)/std(c_sig) 보관/적용.

    - c_mu  : (366, H, W) doy 평균장.
    - c_sig : (366, H, W) doy 표준편차장 (sigma_floor 로 하한).
    """

    def __init__(self, c_mu: np.ndarray, c_sig: np.ndarray, sigma_floor: float = 1e-6):
        self.c_mu = np.asarray(c_mu, dtype=np.float32)                       # (366, H, W)
        self.c_sig = np.maximum(np.asarray(c_sig, dtype=np.float32), sigma_floor)
        if self.c_mu.shape != self.c_sig.shape:
            raise ValueError(f"c_mu/c_sig shape mismatch: {self.c_mu.shape} vs {self.c_sig.shape}")
        if self.c_mu.shape[0] != _N_SLOTS:
            raise ValueError(f"c_mu must have {_N_SLOTS} doy slots, got {self.c_mu.shape}")

    @classmethod
    def from_file(cls, path: str) -> "SktClimatology":
        d = np.load(path)
        sf = float(d["sigma_floor"]) if "sigma_floor" in d.files else 1e-6
        return cls(d["c_mu"], d["c_sig"], sigma_floor=sf)

    def clim_for_times(self, times) -> np.ndarray:
        """times → (len, H, W) doy 평균장 c_mu[doy]."""
        return self.c_mu[doy_slot(times)]

    def std_for_times(self, times) -> np.ndarray:
        """times → (len, H, W) doy 표준편차장 c_sig[doy]."""
        return self.c_sig[doy_slot(times)]

    def standardize(self, skt: np.ndarray, times) -> np.ndarray:
        """skt 픽셀장 → 표준화 anomaly(몇 std) z = (skt − c_mu[doy]) / c_sig[doy].  (skt:(N,H,W))"""
        s = doy_slot(times)
        return (skt - self.c_mu[s]) / self.c_sig[s]

    def destandardize(self, z: np.ndarray, times) -> np.ndarray:
        """표준화 anomaly z → skt 픽셀장.  skt = z · c_sig[doy] + c_mu[doy].

        persistence: z 에 IC 시점 z(=몇 std)를 broadcast 로 고정해 넣으면 각 미래 날을
        자기 doy 의 c_sig 로 복원 → skt_est(t) = c_mu[doy_t] + z_IC · c_sig[doy_t]."""
        s = doy_slot(times)
        return z * self.c_sig[s] + self.c_mu[s]
