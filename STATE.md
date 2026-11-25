# CURRENT STATE (AI Working Memory — 100줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지, 100줄 이내)**. 새 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람/시각화는 절대 금지한다.**

## 1. 현재 Champion & 핵심 자산 위치
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen 제품군 분할, 12개 실제품 `cid=0..12`, 각 정상 500~1000장 + 불량 100장) + **`Sim-LB-800`** (200회 파편화·가변불량률 `pi_c∈[0.08,0.50]` 스트레스 게이트)
- **Public LB Progression**: `0.5593(E04) → 0.6941(E06) → 0.7167(E07) → 0.8231(E12) → 0.8752(E11) → 0.884610(E14) →` **`0.926148(E20, +0.041538!)`**
- **현재 Champion (`E20-TrackAB`, `scripts/run_track_b_v2_final.py`)**:
  - **`E20b_Spatial`** (`LT-Elite9 + M16_svd + M18_opt + SP4_top2`, `z_plus_rg`, NN-BCE `l2=1.0`):
    - **Public LB = `0.926148`** (CV-LB 갭 `-0.0674 → -0.0311` 절반 이상 축소)
    - **Nested 5-Fold OOF = `0.9572`** (`E15 0.9505` 대비 `+0.0067`, `E14 0.9520` 대비 `+0.0052`)
    - **Fold Mean = `0.9660`**: `F0 0.9292 (+0.0222 vs E14)`, `F1 0.9551`, `F2 0.9730`, `F3 0.9856`, `F4 0.9870`
    - **Sim-LB-800 (200 Worlds)**: `0.9635(E15) →` **`0.9684 (+0.0049)`**, P10 하위10% `0.9444 →` **`0.9492 (+0.0048)`**
    - **학습된 12-특성 가중치**: `SP4_top2:0.1233 (1위)`, `M18_opt:0.1162 (2위)`, `M01:0.1073`, `M16_svd:0.1005`, `M06:0.0988`, `M07:0.0908`, `M08:0.0820`, `M05:0.0693`, `M14:0.0641`, `M03:0.0553`, `M04:0.0515`, `M09:0.0407`
  - **로컬 주요 파일**:
    - 검증 제출 CSV: `runs/submissions/sub_e20_track_ab_final.csv` (주력), `runs/submissions/sub_e20_6tower_all19.csv` (백업, OOF `0.9541`)
    - OOF 오답분석용: `runs/track_b_v2/val_oof_e20.json`, 요약: `runs/track_b_v2/track_b_v2_summary.json`
    - 공유 클러스터: `runs/day0_recal/canonical_cids.npz` (`Cluster-v2`, Test Singleton `152/800(19.0%) → 1/800(0.12%)`, 44개 클러스터 복원)
    - 19종 모델 캐시: `runs/queue15/models/M01~M19.npz` + `runs/musc_v2/M16~M19/*_maps32.npz` (`32×32` 2D 공간맵 각 ~19MB)
  - **W&B Artifact 동기화 (`rlaqudwn123456-dankook-university/CV`)**:
    - `dku-model-cache:latest` (`M01~M19.npz` + `canonical_cids.npz` + `folds_unseen.json`, 24 files)
    - `dku-submissions:latest` (`sub_e20_track_ab_final.csv`, `sub_e20_6tower_all19.csv`, `track_b_v2_summary.json`)

## 2. Track A (`M16~M19`) + 32×32 공간맵 실측 정량 요약 (Train-Only)
| 모델 / 특성 | 설정 (`channel_weights`) | 5-Fold OOF | F0 (`1500×1000`) | `F0 cid=2` (n=2200) | `z_musc` / `z_svd` / `z_log` | `ρ(M01)` | 소요/VRAM |
|---|---|---:|---:|---:|---|---:|---|
| `M01` (구형 DINOv2-L) | `dist_thr=0.22`, top-den | `0.9244` | `0.8737` | `0.7846` | _(legacy 19% Test singleton)_ | `1.000` | Kaggle |
| `M16` (DINOv2-Reg4-L) | `Native-AR`, `(0.72/0.18/0.10)` | `0.8996` | `0.8246` (**`svd 0.9016`**) | `m 0.6401` / **`s 0.8517`** | `0.8933` / **`0.9191`** / `0.7545` | `0.775` (`s 0.693`) | 19.3m / ~2.75G |
| `M17` (EVA-02-L MIM) | `448`, `(0.40/0.60/0.00)` | `0.8901` (`opt 0.8991`) | `0.8234` (`F4 0.9731`) | `m 0.6879` / `s 0.7228` | **`0.8983`** / `0.8503` / `0.8061` | **`0.611`** | 12.0m / ~2.87G |
| **`M18` (SigLIP-SO400M)** | `384`, `(0.80/0.20/0.00)` | **`0.9261` (단일 1위!)** | **`0.9171`** (`F4 0.9732`) | **`m 0.8760` / `s 0.8771`** | **`0.9268`** / `0.8888` / `0.8226` | **`0.604`** | 11.8m / ~3.24G |
| `M19` (ConvNeXt-L CNN) | `384(48²+24²)`, `(0.75/0.15/0.10)` | `0.8134` (`opt 0.8408`) | `0.7867` | `m 0.6923` / `s 0.7455` | `0.7969` / **`0.8282`** / `0.7275` | **`0.476`** | 10.2m / ~1.14G |
| **`SP4_top2` (4-백본 32×32)** | 픽셀별 2순위 합의 (`M16~M19`) | **`0.9308` (단일 최고!)** | `0.8895` (`F4 0.9809`) | `F1 0.9414 / F2 0.9660` | 클러스터 Left-Tail 정규화 후 32×32 풀링 | **`0.695`** | CPU 3초 |

- **축적된 모델링 노하우**:
  1. **백본별 채널 분화**: DINOv2-Reg4(`M16`)·ConvNeXt CNN(`M19`)은 저랭크 **`z_svd`** 강세, 비선형 MIM(`M17`)·VLM(`M18`)은 비모수 패치 **`z_musc`** 압도(`+0.04`).
  2. **CLIP 계열 글로벌 논리 채널 유효**: CLIP/VLM 사전학습(`M18`, `M19`)에서는 `z_logical(0.10)` 혼합이 F1~F4에서 `+0.003~+0.007` 일관 순증 (`M18 opt 0.9290`).
  3. **Mid-Level `32×32` 픽셀정렬 합의(`SP4_top2`)의 위력**: 단일 백본의 반사광/경계 FP를 픽셀 좌표 일치 조건으로 소거 → `0.9308`로 모든 단일 모델 추월 + `E20` 1위 가중치.
  4. **비가지치기 합산 주의**: 19-모델 일괄 합산(`Elite-13 0.9491`, `5-Tower 0.9464`)은 저점 모델 희석 발생 → 채널 선택·가지치기 + 공간맵 결합이 필수 (`Legacy-E9 + M16_svd + M18_opt`는 OOF `0.9583`).

## 3. `0.9261 → 0.98+` 공략을 위한 4대 근본 병목 (트러블슈팅 핵심)
1. **[병목 1 — 최대 손실] `M01/M06` 레거시 19% Singleton 잔존**: `E20b_Spatial` 가중치의 **20.6% (`M01 0.107 + M06 0.099`)**가 여전히 구형 캐시(Test 152장/19%가 `cid=-1`로 동료비교 꺼진 채 산출된 raw 점수)를 사용 중 → **`Cluster-v2` 네이티브 DINOv2-L/B MuSc-v2 재추출** 시 즉각 LB 상승 유력.
2. **[병목 2 — 코드 버그] `run_musc_v2_blackwell.py` L387~402 Farthest-Point 뱅크 오염**: `len(pure_order) > max_bank(120)`일 때 `nxt = np.argmax(min_d)`가 정상 코어 대신 top-50% 풀 내 극단치/이상치를 뱅크로 선발해 대형 클러스터(특히 2제품 병합 `F0 cid=2`, n=2,200)에서 `M16_musc=0.6401`, `M17_musc=0.6879`로 급락 → **Query-Adaptive Local Top-Purity Peers** 또는 **Sub-Cluster Stratified Top-Purity Core**로 교정 필요.
3. **[병목 3 — CV-LB 갭 `-0.0311` 원인] 지도학습 33% 가중치 & 글로벌 Rank-Gauss**:
   - `E20b_Spatial`에 지도학습 모델(`M04/M07/M08/M09/M14`) 가중치가 **~33%** 포함되어 Unseen Test 도메인 시프트에 취약 → 순수 비지도 MuSc-v2 + Zero-Shot VLM 중심 스택 vs 지도 포함 스택 정밀 대조 필요.
   - `z_plus_rg`의 글로벌 Rank-Gauss가 파편화된 소형 Test 클러스터의 실제 극단 꼬리 신호를 압축할 위험 → 클러스터 내부 보정(Left-Tail Z / EVT / Power-Transform) 및 메타러너(NN-BCE vs Pairwise AUROC vs Unseen-Fold 일반화 가중치) 전면 재검토 필요.
4. **[병목 4 — 해상도·방법론 확장 & Train 오답 미해결]**:
   - `SP4_top2`의 `32×32` 다운샘플로 초미세 PCB/찍힘 결함 희석 가능성 → `48×48`/Native-AR 공간정렬 검토.
   - 현재 5대 패러다임 외 2026-11-15 이전 공개 오픈소스 기반 **Synthetic Anomaly(SimpleNet/DRAEM) / AnomalyCLIP / PatchCore-Memory / Test-Time Adaptation** 도입 타당성 제로베이스 검토 필요.

## 4. 교내 Blackwell 하드웨어 제약 & Dead Ends (절대 준수)
- **HW 제약**: `RTX PRO 5000 Blackwell` VRAM **`5.50 GiB`**, cgroup RAM **`8.0 GiB` (Swap `0B`)**, CPU 스레드 `2` (`OMP/MKL/OPENBLAS=2`), `num_workers=0`, 1-레이어 `memmap + POSIX_FADV_DONTNEED`. 데이터셋 `data/raw/open/` SHA-256 검증 완료(`11,623` files).
- **Dead Ends**: Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋 / `Sim-LB-800`에서 이상치 점수 기반 클러스터링 금지(반드시 시각 제품 `cids`만 사용) / 비-DINOv2에 `w_svd>=0.50` 과다 배정 / E2E 전체 파인튜닝(`M15 0.6781`) / `k_den=20` P25 일괄 절단.

## 5. 다음 세션(카본 Medium) 즉시 실행 순서
1. `runs/track_b_v2/val_oof_e20.json`으로 **Train-Only FP/FN Top-20 오답 주도 EDA** 실행 (놓치는 결함 유형·제품군 특정).
2. **모델 학습·추론 방법론 + 추천 앙상블 기법 전면 재검토** 후 우선순위 Top-3 확정.
3. 1순위(Farthest-Point 뱅크 오염 수정 + `M01/M06` Cluster-v2 네이티브 재추출)부터 **Fold 0 빠른 검증 → 5-Fold 확정 → `Sim-LB-800` 게이트** 순으로 One-Change 루프 실행.
