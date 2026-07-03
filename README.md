# baseline — 깨끗한 재학습용 통합 파이프라인

이번 실험에서 **실제 사용한 코드만** 복사·정리한 self-contained 폴더. 분기 조건문 제거(예: SST_swin=residual 전용).
전체 설계·정규화·성능은 상위 `../PIPELINE.md` 참조.

canonical 선택 (baseline 고정): SST_swin=**residual**, transfer=**UNetIO direct(residual swin 입력)**, MAR=**sktocean GT-BC(PP)**, updater=**ens-mean+atmo3ch**.

## 구조
```
baseline/
  models/   swin · mar(+diffloss,heads,mask_transformer,georope) · dcae(+sphere_conv)
  data/     swin_dataset(residual전용) · mar_dataset · field_dataset · skt_climatology · make_refined_bc · make_forecast_bc
  recon.py      UNet + train(same-time OISST→skt refine)
  transfer.py   UNetIO + train(fc BC 예보; frozen swin 내장추론)
  cycle.py      TinyUNet + gen/fit/apply(2-cycle)
  train_swin.py · train_mar.py · eval_swin.py · eval_mar.py · utils.py · loss.py · eval_utils.py
  configs/  swin_residual.yaml · mar_sktocean.yaml · accelerate.yaml
```
데이터/climatology 경로는 원본 그대로 (`data/*.zarr`, `SST_swin/static/*`, `S2S_SST/static/*`, DCAE `S2S_SST/outputs/dcae/dcae`). 출력은 각 config/`--out` 에서 지정.

## 처음부터 재학습 순서
```bash
ACC=baseline/configs/accelerate.yaml

# [1] SST_swin (residual, OISST 예보)
uv run accelerate launch --config_file $ACC -m baseline.train_swin --config baseline/configs/swin_residual.yaml
#     → <out_dir>/ckpt_best.pt   (검증: -m baseline.eval_swin --config ... --ckpt ...)

# [2] recon (same-time refine; refined GT BC 원천 + fc의 IC프레임)
uv run accelerate launch --config_file $ACC -m baseline.recon --epochs 8 --stride 3 --base 96 --out baseline/outputs/recon
uv run python -m baseline.data.make_refined_bc --ckpt baseline/outputs/recon/ckpt.pt --out data/skt1deg_refined_unet.zarr

# [3] MAR (GT skt BC 학습, PP 전략) — [1][2]와 독립, 병행 가능
uv run accelerate launch --config_file $ACC -m baseline.train_mar --config baseline/configs/mar_sktocean.yaml

# [4] transfer (fc BC 예보기; [1] 필요, frozen swin 내장추론)
uv run accelerate launch --config_file $ACC -m baseline.transfer --epochs 8 --stride 3 \
    --swin <SST_swin ckpt> --swin-cfg baseline/configs/swin_residual.yaml --out baseline/outputs/transfer

# [5] forecast BC ([1][2][4] 필요)
uv run python -m baseline.data.make_forecast_bc --transfer baseline/outputs/transfer/ckpt.pt \
    --swin <SST_swin ckpt> --swin-cfg baseline/configs/swin_residual.yaml \
    --start 2020-01-01 --end 2021-12-31 --out data/skt1deg_forecast_bc.zarr

# [6] 2-cycle updater ([3][5] 필요; production=ens-mean+3ch)
uv run accelerate launch --config_file $ACC -m baseline.cycle gen --stride 1 --ens 8 \
    --dcae S2S_SST/outputs/dcae/dcae --out baseline/outputs/mar/cycle1_gen.npz
uv run python -m baseline.cycle fit   --gen-npz baseline/outputs/mar/cycle1_gen.npz --use-atmo --train-mean --n-atmo 3 --epochs 8
uv run python -m baseline.cycle apply --updater baseline/outputs/mar/cycle_updater_gen_atmo.pt \
    --gen-npz baseline/outputs/mar/cycle1_gen.npz --out-zarr data/skt1deg_forecast_bc_v2.zarr

# [7] 평가 (같은 MAR ckpt 에 BC 만 교체)
uv run accelerate launch --config_file $ACC -m baseline.eval_mar --ckpt <MAR ckpt> --dcae S2S_SST/outputs/dcae/dcae \
    --ic-stride 15 --ensemble 50 --ocean-source-override {skt|refined|forecast} [--forecast-zarr data/skt1deg_forecast_bc_v2.zarr]
```

의존: [1]→[4]→[5]→[6], [2]→refined→[5]의 IC프레임, [3]은 독립. MAR([3])는 상류 재학습과 무관(PP 모듈성).
```
```
