# baseline 최종 결과 — 3-BC 사다리 평가

전 파이프라인을 처음부터 재학습(SST_swin·recon·MAR·transfer·2-cycle updater)한 clean baseline.
평가: **2022 test, ensemble 50, IC 24개**. 같은 MAR ckpt에 open-ocean BC만 교체.

## 최종 사다리

| BC (해양 경계) | 미래 GT | global RMSE | tropics | ACC | ACC+14 | LST(K) |
|---|:---:|---:|---:|---:|---:|---:|
| GT SKT (절대천장, leakage 포함) | ✅ | 4.958 | 3.047 | 0.690 | 0.610 | 2.776 |
| refined GT (관측기반 정직천장) | ✅ | 5.542 | 3.178 | 0.581 | 0.480 | 2.850 |
| **forecast (배포형, 미래 GT 0)** | ❌ | **5.566** | **3.230** | **0.582** | **0.476** | **2.818** |
| climatology | — | 6.934 | 3.677 | 0 | 0 | 4.040 |

리드별 곡선: `results/ladder.png`

## 핵심 결과

**배포형(미래 관측 0)이 관측기반 정직천장(refined GT)에 사실상 도달.**
- global RMSE 5.566 vs refined 5.542 — 갭 **0.024** (거의 동일)
- ACC 0.582 vs refined 0.581 — 배포형이 근소 우위
- LST 2.818 vs refined 2.850 — 배포형이 더 좋음
- climatology(6.934) 대비 RMSE −20%, 전 리드·전 지표 압도

**즉 2-cycle 되먹임이 forecast→refined 갭을 거의 완전히 메웠다.**
updater BC corr = fc 0.849 → **0.872** (+0.023, 이전 세션 0.834 대비 개선).

## 이번 학습 설정 (이전 대비 개선점)
- **2-cycle gen: ens 50 평균** (이전 8) → 깨끗한 LST_gen/Atmo_gen
- **updater train 2020–2021 / test 2022** (이전 2020/2021) → 표본 2배, MAR val과 비겹침
- **지표면 대기 3채널 t/u/v@850** (이전 버그: t300/t500/t850 온도만) → 정정
- ens-mean 학습 (per-member 8배는 분포불일치로 역효과 실증됨)

## 학습된 모델 (val)
- SST_swin (residual): val RMSE 0.458 K
- recon UNet: test 2022 pooled corr 0.868
- MAR (GT skt BC, PP): val atmo 0.480 (200ep 수렴)
- 2-cycle updater: BC test corr 0.872

## 남은 것
- GT SKT↔refined 갭(≈0.58 RMSE / 0.11 ACC)의 상당분은 ERA5 자기일관성 leakage(관측 기반 도달불가).
- refined↔forecast 갭이 사실상 0 → 배포형이 관측 천장 도달. 추가 개선은 SST 예보(SST_swin) 또는 MAR 자체.
