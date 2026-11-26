# CURRENT STATE (AI Working Memory — 100줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지, 100줄 이내)**. 새 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람/시각화는 절대 금지한다.**

## 1. 현재 Champion & 핵심 자산 위치
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (200회 파편화·가변불량률 스트레스 게이트)
- **Public LB Champion**: **`E20` (`0.926148`)** (`0.5593 → 0.6941 → 0.7167 → 0.8231 → 0.8752 → 0.8846 → 0.926148`)
- **최신 3-Gate 통과 제출 후보 (`E22`, `scripts/run_e22_clean_ensemble.py`)**:
  - **1순위**: `runs/submissions/sub_e22_clean_blend_e20_m22.csv` (`50% E20 SP4 + 50% Clean-14 z_plus_rg`, `rho_E20=0.9891`, `test_rev 0%`)
  - **2순위**: `runs/submissions/sub_e22_clean14_zrg_no_rev.csv` (`OOF 0.9667`, `Mean 0.9729`, `Sim-LB 0.9733 (P10 0.9606)`)
  - **3순위 (Private 방어)**: `runs/submissions/sub_e22_clean_pure_unsup9.csv` (`OOF 0.9613`, 지도학습 `0%`)
- **W&B Artifact (`rlaqudwn123456-dankook-university/CV`)**:
  - `dku-model-cache:latest` (`M01~M22` 26종 `.npz` + `canonical_cids.npz` + `folds_unseen.json` + 앞으로 `*_maps32.npz` 자동 포함)
  - `dku-submissions:latest` (`sub_e20_track_ab_final.csv` 등)

## 2. 앙상블 LOO MVP 순위 & `E21(0.9142)` 하락 교훈
- **앙상블 LOO 순기여 Top-8 (Clean-14 `OOF 0.9667` 기준)**:
  1. `M20_musc_dinov2_l518_v2` (`+0.0025`, `AvgCorr 0.372`)
  2. `M07_vand_clip_l14_laion2b` (`+0.0023`, `AvgCorr 0.293`)
  3. `M16_musc_dinov2_reg4_l` (`+0.0021`, `AvgCorr 0.393`)
  4. `M22_siglip2_512_musc_only` (`+0.0012`, `Single 0.9289`, **1위 가중치 `0.101`**)
  5. `M21_musc_dinov2_b518_v2` (`+0.0008`, `Single 0.9146`)
  6. `M05_siglip_so400m_384` (`+0.0008`, `AvgCorr 0.328`)
  7. `M14_patchmil_convnext_l384` (`+0.0006`, `AvgCorr 0.240`)
  8. `M18_musc_siglip_so400m_384` (`+0.0004`, `Single 0.9261`)
- **제출 전 필수 3-Gate 규칙 (영구 적용)**:
  1. `te_max > 8.0` 이상치 감시 + `z_plus_rg` 상한 클리핑 기본 적용.
  2. 동일 백본 이전 버전 대비 `rho_Champion`이 `-0.05` 이상 급락하는 신규 채널(`test_rev` 등) 즉시 차단.
  3. 직전 LB Champion의 검증된 무기(`SP4_top2` 2D 공간맵)는 동등 이상 2D 맵(`SP_top2`) 대체 전까지 절대 제거 금지.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 `.pt/.npz/.csv/runs/` 커밋
- **`test_rev` 역방향 매칭 (Bank→Query)**: Train OOF만 `0.9714`로 부풀리고 Test 다각도 소형 클러스터에서 노이즈 폭발(`rho 0.30~0.52`, LB `-0.0119`)
- **`ViT-H` / `1B+` 초대형 백본 (`M23 DFN5B-ViT-H`)**: 교내 8 GB RAM 컨테이너 로드 16초 만에 OOMKill Pod 초기화
- E2E 전체 파인튜닝(`M15 0.6781`) / `Farthest-Point` 뱅크 / `k_den=20 P25` 절단

## 4. 교내 Blackwell 서버 즉시 실행 계획 (차세대 Top-5 `<= 320M` SOTA 백본)
- **안전 실행 원칙**: 모델 1개당 **독립 파이썬 프로세스(`for` 루프)** + **`--wandb --auto-sync`** (`*_maps32.npz`까지 W&B Artifact 자동 백업!)
- **스모크 테스트 5/5 통과 완료 (`scripts/run_musc_v2_blackwell.py`)**:
  - `M24_musc_radio_v3_l` (`vit_large_patch16_radio_v3.c_radio`, 304M, 32×32, DINOv2+SigLIP+CLIP+SAM 증류)
  - `M25_musc_dinov3_convnext_l` (`convnext_large.dinov3_lvd1689m`, 198M, 48×48+24×24, LVD-1689M CNN)
  - `M26_musc_pe_core_l336` (`vit_pe_core_large_patch14_336.fb`, 316M, 24×24, Meta Perception Encoder)
  - `M27_musc_aimv2_l448` (`aimv2_large_patch14_448.apple_pt`, 309M, 32×32, Apple 자기회귀 멀티모달)
  - `M28_musc_dinov3_b512` (`vit_base_patch16_dinov3.lvd1689m`, 86M, Native-AR 28×42, LVD-1689M ViT)
- **실행 명령 (독립 프로세스 순차 실행)**:
  ```bash
  for m in M24_musc_radio_v3_l M25_musc_dinov3_convnext_l M26_musc_pe_core_l336 M27_musc_aimv2_l448 M28_musc_dinov3_b512; do
    PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model "$m" --wandb --auto-sync
  done
  ```
- 완료 후 로컬에서 `python scripts/sync_wandb_artifacts.py pull` → `M24~M28`의 `32×32` 2D 공간맵(`SP5_nextgen_top2`) + 1D 앙상블 통합으로 **`0.95+` LB** 정조준!
