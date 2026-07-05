# baseline lead-21 결과 — 예보 기간 21일 확장

lead time을 14→21일로 늘려 전 파이프라인 재학습(SST_swin·MAR·transfer·2-cycle; recon/refined는 same-time이라 14일판 재사용).
평가: **2022 test, ensemble 50, IC 24개**. 같은 MAR ckpt에 open-ocean BC만 교체.

## lead-21 사다리

| BC (해양 경계) | 미래 GT | global RMSE | tropics | ACC | ACC+21 | LST(K) |
|---|:---:|---:|---:|---:|---:|---:|
| GT SKT (절대천장, leakage 포함) | ✅ | 5.036 | 3.099 | 0.683 | 0.623 | 2.777 |
| refined GT (관측기반 정직천장) | ✅ | 5.681 | 3.257 | 0.569 | 0.514 | 2.909 |
| **forecast (배포형, 미래 GT 0)** | ❌ | **5.734** | **3.337** | **0.563** | **0.502** | **2.859** |
| climatology | — | 6.931 | 3.665 | 0 | 0 | 3.987 |

곡선: `results/ladder_21.png` · 14 vs 21 비교: `results/compare_14_21.png`

## 14일 vs 21일 (배포형 forecast)

| | 14일 학습 | 21일 학습 |
|---|---|---|
| global RMSE | 5.566 | 5.734 |
| ACC (전 리드 평균) | 0.582 | 0.563 |
| ACC 최장리드 | 0.476 (+14) | **0.502 (+21)** |
| forecast↔refined 갭 (RMSE) | 0.024 | 0.053 |

## 핵심 결과

**배포형이 21일에서도 관측 천장(refined GT)에 붙어 있음.**
- forecast 5.734 vs refined 5.681 — 갭 0.053 (14일 0.024보다 소폭↑, 여전히 매우 작음)
- ACC 0.563 vs 0.569 — 거의 동일. **2-cycle 되먹임이 21일에서도 작동.**

**장리드 학습 효과: 21일로 학습하니 장리드 자체를 더 잘 배움.**
- 최장리드 ACC가 14일의 +14(0.476)보다 21일의 +21(**0.502**)이 높음.
- 전 리드 평균 ACC 하락(0.582→0.563)은 뒤쪽 7일(+15~21)이 더 어려운 구간이라 평균을 낮춘 것.
- climatology(6.931) 대비 RMSE −17% 유지, 열대·LST 강건.

## 학습 설정 (14일판과 동일 + lead 확장)
- LEAD=21 env 로 transfer·cycle·make_forecast_bc 의 T 전파
- MAR batch 24→12 (window 22프레임 OOM 대응), future_len 21
- recon/refined 재사용 (same-time 변환은 lead 무관)
- 2-cycle: ens 50 평균, updater train 2020–2021 / test 2022, 지표면 t/u/v@850 3ch

## 산출물 (`_21` 접미사로 14일판과 분리)
- ckpt: `outputs/swin_21` `outputs/mar_21` `outputs/transfer_21`
- BC: `skt1deg_forecast_bc_21.zarr` `skt1deg_forecast_bc_v2_21.zarr`
- eval: `outputs/mar_21/eval_{skt,refined,forecast}.npz`
