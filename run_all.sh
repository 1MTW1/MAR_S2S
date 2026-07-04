#!/usr/bin/env bash
# baseline 전체 파이프라인 1회 실행 (README [1]~[7]).
# ★ repo 루트에서 실행: bash baseline/run_all.sh   (데이터 경로가 루트 상대이므로)
# 재실행 안전: 각 산출물 존재 시 건너뜀(SKIP=1). 로그는 baseline/outputs/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo 루트로 이동

ACC=baseline/configs/accelerate.yaml
DCAE=S2S_SST/outputs/dcae/dcae
OUT=baseline/outputs
LOG=$OUT/logs; mkdir -p "$LOG"
SWIN=$OUT/swin/ckpt_best.pt
MAR=$OUT/mar/ckpt_best.pt
RECON=$OUT/recon/ckpt.pt
TRANSFER=$OUT/transfer/ckpt.pt
GEN=$OUT/mar/cycle1_gen.npz
UPD=$OUT/mar/cycle_updater_gen_atmo.pt
SKIP=${SKIP:-1}                                 # 1=존재 산출물 건너뜀, 0=항상 재실행

run() { echo -e "\n\033[1;36m[$(date '+%F %T')] $1\033[0m"; }
have() { [ "$SKIP" = 1 ] && [ -e "$1" ]; }

# ── [1] SST_swin (residual, OISST 예보) ──
if have "$SWIN"; then run "SKIP [1] swin ($SWIN 존재)"; else
  run "[1] SST_swin 학습"
  uv run accelerate launch --config_file "$ACC" -m baseline.train_swin \
    --config baseline/configs/swin_residual.yaml 2>&1 | tee "$LOG/1_swin.log"
fi

# ── [2] recon + refined GT BC ──
if have "$RECON"; then run "SKIP [2a] recon"; else
  run "[2a] recon 학습"
  uv run accelerate launch --config_file "$ACC" -m baseline.recon \
    --epochs 8 --stride 3 --base 96 --out "$OUT/recon" 2>&1 | tee "$LOG/2_recon.log"
fi
if have "data/skt1deg_refined_unet.zarr"; then run "SKIP [2b] refined BC"; else
  run "[2b] refined GT BC 생성"
  uv run python -m baseline.data.make_refined_bc --ckpt "$RECON" \
    --out data/skt1deg_refined_unet.zarr 2>&1 | tee "$LOG/2_refined.log"
fi

# ── [3] MAR (GT skt BC 학습, PP) — [1][2]와 독립 ──
if have "$MAR"; then run "SKIP [3] MAR"; else
  run "[3] MAR 학습 (200ep, ~4h)"
  uv run accelerate launch --config_file "$ACC" -m baseline.train_mar \
    --config baseline/configs/mar_sktocean.yaml 2>&1 | tee "$LOG/3_mar.log"
fi

# ── [4] transfer (fc BC 예보; [1] 필요) ──
if have "$TRANSFER"; then run "SKIP [4] transfer"; else
  run "[4] transfer 학습"
  uv run accelerate launch --config_file "$ACC" -m baseline.transfer --epochs 8 --stride 3 \
    --swin "$SWIN" --swin-cfg baseline/configs/swin_residual.yaml \
    --out "$OUT/transfer" 2>&1 | tee "$LOG/4_transfer.log"
fi

# ── [5] forecast BC ([1][2][4] 필요) ──
if have "data/skt1deg_forecast_bc.zarr"; then run "SKIP [5] forecast BC"; else
  run "[5] forecast BC 생성"
  uv run python -m baseline.data.make_forecast_bc --transfer "$TRANSFER" \
    --swin "$SWIN" --swin-cfg baseline/configs/swin_residual.yaml \
    --start 2020-01-01 --end 2022-12-31 --out data/skt1deg_forecast_bc.zarr 2>&1 | tee "$LOG/5_fc.log"
fi

# ── [6] 2-cycle updater ([3][5] 필요; production=ens-mean+3ch) ──
if have "$GEN"; then run "SKIP [6a] cycle gen"; else
  run "[6a] cycle-1 생성 (ens8, ~1h)"
  uv run accelerate launch --config_file "$ACC" -m baseline.cycle gen \
    --ckpt "$MAR" --start 2020-01-01 --end 2022-12-31 --stride 1 --ens 50 --dcae "$DCAE" --out "$GEN" 2>&1 | tee "$LOG/6_gen.log"
fi
run "[6b] updater 학습 (ens-mean+3ch)"
uv run python -m baseline.cycle fit --gen-npz "$GEN" \
  --use-atmo --epochs 8 2>&1 | tee "$LOG/6_fit.log"
run "[6c] BC v2 생성"
uv run python -m baseline.cycle apply --updater "$UPD" --gen-npz "$GEN" \
  --out-zarr data/skt1deg_forecast_bc_v2.zarr 2>&1 | tee "$LOG/6_apply.log"

# ── [7] 평가 (같은 MAR ckpt 에 BC 만 교체; 2021 test, ens50) ──
for SRC in skt refined forecast; do
  run "[7] eval BC=$SRC"
  EXTRA=""; [ "$SRC" = forecast ] && EXTRA="--forecast-zarr data/skt1deg_forecast_bc_v2.zarr"
  uv run accelerate launch --config_file "$ACC" -m baseline.eval_mar --ckpt "$MAR" --dcae "$DCAE" \
    --val-start 2022-01-01 --val-end 2022-12-31 --ic-stride 15 --ensemble 50 \
    --ocean-source-override "$SRC" $EXTRA \
    --save-npz "$OUT/mar/eval_${SRC}.npz" 2>&1 | tee "$LOG/7_eval_${SRC}.log"
done

run "완료 — 전 단계 로그: $LOG/"
