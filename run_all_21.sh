#!/usr/bin/env bash
# baseline 전체 파이프라인 — lead time 21일 버전. ckpt/zarr 는 _21 접미사로 14일판과 분리.
# ★ repo 루트에서: bash baseline/run_all_21.sh   (LEAD=21 env 로 transfer/cycle/make_forecast_bc T 제어)
# recon/refined 는 lead-무관(same-time) → 14일판 재사용.
set -euo pipefail
cd "$(dirname "$0")/.."
export LEAD=21                                       # ★ transfer.T, cycle.LEAD, make_forecast_bc.T 전파
export FORECAST_ZARR=data/skt1deg_forecast_bc_21.zarr   # cycle gen 이 읽을 21일 forecast BC

ACC=baseline/configs/accelerate.yaml
DCAE=S2S_SST/outputs/dcae/dcae
OUT=baseline/outputs
LOG=$OUT/logs_21; mkdir -p "$LOG"
SWIN=$OUT/swin_21/ckpt_best.pt
MAR=$OUT/mar_21/ckpt_best.pt
RECON=$OUT/recon/ckpt.pt                              # ★ 재사용 (lead 무관)
REFINED=data/skt1deg_refined_unet.zarr               # ★ 재사용 (same-time)
TRANSFER=$OUT/transfer_21/ckpt.pt
FC=data/skt1deg_forecast_bc_21.zarr
GEN=$OUT/mar_21/cycle1_gen.npz
UPD=$OUT/mar_21/cycle_updater_gen_atmo.pt
FCV2=data/skt1deg_forecast_bc_v2_21.zarr
SKIP=${SKIP:-1}
run() { echo -e "\n\033[1;36m[$(date '+%F %T')] $1\033[0m"; }
have() { [ "$SKIP" = 1 ] && [ -e "$1" ]; }

# ── [1] SST_swin 21 ──
if have "$SWIN"; then run "SKIP [1] swin_21"; else
  run "[1] SST_swin 학습 (T=21)"
  uv run accelerate launch --config_file "$ACC" -m baseline.train_swin \
    --config baseline/configs/swin_residual_21.yaml 2>&1 | tee "$LOG/1_swin.log"
fi

# ── [2] recon/refined 재사용 (lead 무관) ──
if [ -e "$RECON" ] && [ -e "$REFINED" ]; then run "SKIP [2] recon/refined (14일판 재사용, lead 무관)"; else
  run "[2] recon 필요 — 14일 run_all 먼저 실행 요망"; exit 1
fi

# ── [3] MAR 21 (GT skt BC, ~4h+) ──
if have "$MAR"; then run "SKIP [3] MAR_21"; else
  run "[3] MAR 학습 (future_len=21, 200ep)"
  uv run accelerate launch --config_file "$ACC" -m baseline.train_mar \
    --config baseline/configs/mar_sktocean_21.yaml 2>&1 | tee "$LOG/3_mar.log"
fi

# ── [4] transfer 21 ([1] 필요) ──
if have "$TRANSFER"; then run "SKIP [4] transfer_21"; else
  run "[4] transfer 학습 (T=21)"
  uv run accelerate launch --config_file "$ACC" -m baseline.transfer --epochs 8 --stride 3 \
    --swin "$SWIN" --swin-cfg baseline/configs/swin_residual_21.yaml \
    --out "$OUT/transfer_21" 2>&1 | tee "$LOG/4_transfer.log"
fi

# ── [5] forecast BC 21 (lead 1..21) ──
if have "$FC"; then run "SKIP [5] forecast BC_21"; else
  run "[5] forecast BC 생성 (lead 21)"
  uv run python -m baseline.data.make_forecast_bc --transfer "$TRANSFER" \
    --swin "$SWIN" --swin-cfg baseline/configs/swin_residual_21.yaml \
    --start 2020-01-01 --end 2022-12-31 --out "$FC" 2>&1 | tee "$LOG/5_fc.log"
fi

# ── [6] 2-cycle (ens50, 지표면3ch) ──
if have "$GEN"; then run "SKIP [6a] cycle gen_21"; else
  run "[6a] cycle-1 생성 (ens50, T=21)"
  uv run accelerate launch --config_file "$ACC" -m baseline.cycle gen \
    --ckpt "$MAR" --start 2020-01-01 --end 2022-12-31 --stride 1 --ens 50 --dcae "$DCAE" \
    --out "$GEN" 2>&1 | tee "$LOG/6_gen.log"
fi
run "[6b] updater 학습"
uv run python -m baseline.cycle fit --gen-npz "$GEN" --use-atmo --epochs 8 2>&1 | tee "$LOG/6_fit.log"
run "[6c] BC v2 생성"
uv run python -m baseline.cycle apply --updater "$UPD" --gen-npz "$GEN" \
  --out-zarr "$FCV2" 2>&1 | tee "$LOG/6_apply.log"

# ── [7] 평가 3종 (2022 test, ens50) ──
for SRC in skt refined forecast; do
  run "[7] eval BC=$SRC (T=21)"
  EXTRA="--refined-zarr $REFINED"; [ "$SRC" = forecast ] && EXTRA="$EXTRA --forecast-zarr $FCV2"
  uv run accelerate launch --config_file "$ACC" -m baseline.eval_mar --ckpt "$MAR" --dcae "$DCAE" \
    --val-start 2022-01-01 --val-end 2022-12-31 --ic-stride 15 --ensemble 50 \
    --ocean-source-override "$SRC" $EXTRA --save-npz "$OUT/mar_21/eval_${SRC}.npz" 2>&1 | tee "$LOG/7_eval_${SRC}.log"
done
run "완료 — lead 21 (로그 $LOG/)"
