# baseline — 파일별 명세서

각 `.py`가 무엇을 구현하는지. 데이터 흐름 순서(①SST_swin→②transfer→③recon→④MAR→⑤updater)로 묶음.
정규화 규약·성능은 `../PIPELINE.md`, 실행 순서는 `README.md` 참조.

---

## 모델 정의 (`models/`)

### `models/swin.py` — ① SST_swin 본체 (UTransformer, 10.1M)
Swin-Unet(shifted-window) seq2seq. 과거 14일 → 미래 14일 (B,T,H,W)→(B,T,H,W).
- `UTransformer`: encoder 3 stage(PatchMerging) + decoder 2 stage(PatchExpand) + skip concat. `in_chans` 인자로 입력채널 조절(transfer 가 42ch 재사용).
- `SwinBlock`/`WindowAttention`: shifted-window self-attn + relative position bias. `PatchEmbed`(conv patchify), `FinalPatchExpand_X4`(원해상도 복원).
- `window_partition`/`window_reverse`: 윈도 분할·복원.

### `models/mar.py` — ④ MAR 본체 (S2SInjectTransformer, 415M) ★핵심
MM-DiT dual-stream MAR. 대기 latent + 결합 skt 필드 → 미래 대기 latent + 육지 LST 확률생성.
- `S2SInjectTransformer`:
  - `_embed_direct`: 토큰 라우팅 — ocean=`ocean_proj`(visible), 육지=`skt_proj`, 마스크=`mask_token`.
  - `forward_sst_direct`: 학습. γ-MAR 마스킹 → `atmo_diff + 0.3·lst_diff + 0.5·det` 손실.
  - `sample_sst_direct`: 추론. cosine 스케줄 iterative decoding.
- `DualStreamBlock`: 대기/skt 두 스트림 각자 weight + joint attention (MM-DiT).

### `models/dcae.py` — DCAE (frozen, 재학습 안 함)
대기장 ↔ 14ch latent 오토인코더. **가중치는 `S2S_SST/outputs/dcae/dcae`에서 로드**, baseline 은 정의만.
- `AutoencoderDC`(diffusers ModelMixin): `.decode()` 만 평가/2-cycle 에서 사용. `Encoder`/`Decoder`/`EfficientViTBlock` 등 내부 블록.

### MAR 부품
- `models/diffloss.py` — per-token diffusion head. `DiffLoss`(target→z 조건 EDM diffusion), `SimpleMLPAdaLN`(denoiser MLP), `EDMDiffusion`(스케줄/샘플링), `TimestepEmbedder`.
- `models/heads.py` — 보조 결정론 head. `DeterministicHead`(za→μ), `deterministic_mse_loss`(lead별 e^{-k} 가중), `frame_decay_weights`.
- `models/mask_transformer.py` — MAR 가 재사용하는 블록만 유효: `Block`(표준 transformer), `build_3d_sincos_pos_embed`(T×h×w 위치). (그 외 S2SMaskTransformer/AdaLNCausalBlock/georope 경로는 MAR 미사용.)
- `models/georope.py` — geo-RoPE 위치인코딩(mask_transformer lazy import 안전용, MAR 경로 미사용).
- `models/sphere_conv.py` — `SphereConv2d`(경도 circular conv), dcae 가 사용.

---

## 데이터로더 (`data/`)

### `data/swin_dataset.py` — ① SST_swin 데이터셋 (residual 전용)
- `S2SSwinDataset`: 과거/미래 14일 OISST → **anomaly z** = clip((skt−c_mu[doy])/c_sig[doy],±5), land=0.
  반환 (x,y,mask[,scale]). residual 은 train/eval 의 predict 에서 persist+net 처리(여기선 anomaly z 만 제공).
  - `_norm`(anomaly z 표준화+land0+경도roll), `_scale`(z→K 복원용 c_sig[doy]), `__getitem__`(윈도 슬라이스).
- 격자 유틸: `compute_grid_pad`/`auto_pad`(patch·window 배수 pad), `build_static`(ocean mask+lat weight), `_open_field`(OISST/ERA5 자동감지), `doy_slot`(366 doy 인덱스).
- `compute_global_ocean_stats`: raw 타깃 잔재(residual baseline 미사용, 유틸로 잔존).

### `data/mar_dataset.py` — ④ MAR 데이터셋
- `S2SInjectDirectDataset`: 대기 latent IC + **결합 skt 필드**(1°,18×18패치). 반환 latents,ts,skt_p×2,ocean_tok.
  결합 = `where(open-ocean, BC소스, ERA5 skt anom z)`. **ocean_source**: skt(GT,학습)/oisst/refined/**forecast**(IC=refined·미래=예보 오버레이). `mask_seaice`(해빙 visible 제외).
- `prepare_inject_direct_dataloader`(DataLoader 래퍼), `load_latent_stats`, `_ts_to_int`.

### 공용 데이터
- `data/field_dataset.py` — ERA5 대기장 로더. `load_field_stats`(채널 mean/std), `Era5FieldDataset`, `prepare_field_dataloader`. (eval/BC생성 물리 역정규화용.)
- `data/skt_climatology.py` — `SktClimatology`(per-doy c_mu/c_sig 로드·조회), `doy_slot`. (LST 물리 K 복원.)

### BC 생성 (③②산출물 → zarr)
- `data/make_refined_bc.py` — ③ recon(GT OISST(t))→skt z 1° 전 기간 생성 → `skt1deg_refined_unet.zarr`(refined GT BC). `build`(생성)·`check`(vis corr 검증).
- `data/make_forecast_bc.py` — ②transfer 예보 → skt z 1° IC별 미래14일 → `skt1deg_forecast_bc.zarr`(fc BC). frozen swin(`swin_forecast`)+transfer(UNetIO) 내장추론, 4×4 pool.

---

## 학습·추론 스크립트 (top-level)

### `train_swin.py` — ① SST_swin 학습 (residual 전용)
- `main`: anomaly z 로더 → UTransformer(head zero-init=persist 시작) → cosine, K-RMSE val, early-stop.
- `predict`(persist a_IC + net), `validate`(물리 K weighted RMSE), `make_loader`.

### `recon.py` — ③ recon 모델+학습 (same-time refine)
- `UNet`(1→1, 경도 circular pad)+`CB`(conv block, transfer UNetIO 가 재사용). `ReconDataset`(OISST anom z / ERA5-skt anom z 페어).
- `main`: GT OISST→GT skt, ocean·lat MSE. `evaluate`(pooled/pixel corr), `latw`, `_blosc_off`/`_cpad`(fork-safe).

### `transfer.py` — ② transfer 모델+학습 (fc BC 예보)
- `UNetIO`(42ch→14ch, recon.CB 재사용). `TransferDataset`(과거 OISST anom·과거 skt anom·미래 skt anom·doy).
- `load_swin`(frozen residual swin 로드), `swin_forecast`(swin 내장추론=persist+net→anomaly z), `FcstAnom`(ocean mask 보유).
- `main`: [swin예보|과거skt|과거OISST]→미래 skt anom z, ocean·lat MSE. `--anchor persist` 옵션. `evaluate`/`print_table`(리드별 corr/RMSE).

### `train_mar.py` — ④ MAR 학습
- `main`: direct 로더(ocean_source=skt) → S2SInjectTransformer, accelerate 멀티GPU+EMA, cosine+warmup.
- `build_model`(config→모델), `schedule_eps`(GT 사용률 스케줄), `make_loader`, `validate`(val 손실), `save_ckpt`/`_prune`.

### `cycle.py` — ⑤ 2-cycle updater (gen/fit/apply)
- `gen`: frozen MAR 로 IC별 미래14일 생성 → LST_gen(ens멤버)+Atmo_gen(9ch) npz 저장.
- `TinyUNet`(0.1M, 1° updater, out zero-init=fc 시작). `fit`: [fc,LST_gen,vis,lead,Atmo] → true skt 잔차, open-ocean MSE. train 2020/test 2021. `--train-mean`(ens-mean 학습)·`--n-atmo 3`(production).
- `apply`: updater 로 fc → BC v2 zarr. `build_ds`(forecast 모드 데이터셋).

---

## 평가·공용 유틸

- `eval_mar.py` — ④ 리드별 RMSE(전지구/열대)·ACC·LST. `--ocean-source-override {skt|refined|forecast}`로 같은 ckpt 에 BC 교체. `sample_ens`(앙상블 샘플링).
- `eval_swin.py` — ① 리드별 K-RMSE/Bias/ACC(full/open-ocean 분리, 해빙 마스크). residual 전용 z→K.
- `eval_utils.py` — `load_inject`(MAR ckpt→모델), `decode_sliced`(latent→물리장 DCAE decode), `unpatch_skt`(패치→격자).
- `utils.py` — `lat_weights`/`lw_mean`(위도가중), `load_latent_stats`, `region_aavg`(영역평균).
- `loss.py` — `masked_latweighted_l2`(ocean·lat 가중 MSE, swin 학습), `lat_weights`.

---

## 읽기 순서 추천
`models/mar.py`(중심) → `data/mar_dataset.py`(BC 주입) → `transfer.py`(fc 생성) → `cycle.py`(되먹임).
①③은 부품이라 나중. 각 파일 상단 docstring 에 실행 커맨드 있음.
