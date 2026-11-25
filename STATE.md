# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (200회 파편화·가변불량률 스트레스 게이트)
- **NEW CV + Public LB Champion (`E20-TrackAB`, `scripts/run_track_b_v2_final.py`)**:
  - **Public LB = `0.926148`** (`E14 0.884610` 대비 **`+0.041538` 수직 상승!**, CV-LB 갭 `-0.0674 → -0.0311` 절반 이상 축소)
  - **Nested 5-Fold OOF = `0.9572`** (`Fold Mean 0.9660`: `F0 0.9292`, `F1 0.9551`, `F2 0.9730`, `F3 0.9856`, `F4 0.9870`), **Sim-LB-800 = `0.9684` (`P10 0.9492`)**
  - 제출 CSV: `runs/submissions/sub_e20_track_ab_final.csv` (W&B `dku-submissions:latest`)
- **W&B Artifact 동기화**: `dku-model-cache:latest` (`M01~M19.npz` 등 24개) & `dku-submissions:latest` 최신 반영 완료.

## 2. `0.9261 → 0.98+` 남은 4대 병목 (트러블슈팅 우선순위)
1. **병목 1 (가장 큰 손실 — `M01/M06` 레거시 19% Singleton 잔존)**: `E20`에서 `M01(0.107)`+`M06(0.099)` = **20.6% 가중치**가 여전히 구형 `M01/M06` 캐시(Test 152장 `cid=-1` 동료비교 꺼진 상태의 `oof_raw/test_raw`!)를 사용 중 → **Cluster-v2(`0.12%` singleton) 네이티브 DINOv2-L/B MuSc-v2 재추출** 시 즉각 점프 유력.
2. **병목 2 (`M16/M17` Farthest-Point 뱅크 오염 버그)**: `run_musc_v2_blackwell.py` L391 `argmax(min_d)`가 대형 클러스터(`n>240`)에서 정상 코어 대신 풀 내 극단치/이상치를 뱅크로 선발해 `M16_musc F0 cid=2 0.6401`, `M17_musc 0.6879`로 급락 → **Top-Purity Core + Query-Local kNN Bank** 교정 시 `M16/M17` 및 `SP4_top2` 동반 폭등.
3. **병목 3 (글로벌 Rank-Gauss + 소형 클러스터/불량률 편향)**: `z_plus_rg`가 전체 800장 기준 Rank-Gauss를 섞어 소형 클러스터의 실제 극단 꼬리 신호를 압축하고, `M01~M19` 중 지도학습 모델(`M04/M07/M08/M09/M14`, 총 33% 가중치)이 Unseen 도메인 시프트로 CV-LB 갭(`-0.0311`)의 주원인.
4. **병목 4 (해상도·공간 정밀도 & Train 오답 미해결)**: `SP4_top2`가 `32×32`로 다운샘플되어 초미세 PCB/찍힘 결함이 희석됨 (`runs/track_b_v2/val_oof_e20.json` 기준 Train FP/FN Top-20 오답 주도 EDA 즉시 수행 필요).

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋
- `Sim-LB-800` 월드 생성 시 이상치 점수 기반 클러스터링 금지 / 비-DINOv2에 `w_svd >= 0.50` 과다 배정 / E2E 파인튜닝(`M15`)

## 4. 다음 행동 1개 (카본 Medium 핸드오프 1순위)
1. `runs/track_b_v2/val_oof_e20.json`으로 **Train-Only FP/FN Top-20 오답 분석** 실행 (어떤 제품군·결함 유형이 `0.9572 → 0.98+`를 막는지 확정).
2. **One-Change 1순위**: `run_musc_v2_blackwell.py`의 **Farthest-Point 뱅크 오염(`argmax(min_d)`) 수정** 후 `M16(DINOv2-Reg4)`·`M01-v2(DINOv2-L Cluster-v2 네이티브)` 재추출로 19% Singleton 잔존 + `cid=2` 붕괴 동시 해결.
