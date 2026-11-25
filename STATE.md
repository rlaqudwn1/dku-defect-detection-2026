# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (200회 파편화·가변불량률 스트레스 게이트)
- **Public LB 1위**: **`E14-Q15`** (`Elite-9`, OOF `0.9520` / Public LB `0.884610`)
- **NEW CV + Sim-LB Champion (`E20-TrackAB`, `scripts/run_track_b_v2_final.py`)**:
  - **`E20b_Spatial` (`LT-Elite9 + M16_svd + M18_opt + SP4_top2`, `z_plus_rg`, NN-BCE)**:
    - **Nested 5-Fold OOF = `0.9572`** (`E15 0.9505` 대비 **`+0.0067`**, `E14 0.9520` 대비 **`+0.0052`**)
    - **Fold Mean = `0.9660`** (`F0 0.9292 (+0.0222 vs E14)`, `F1 0.9551`, `F2 0.9730`, `F3 0.9856`, `F4 0.9870`)
    - **Sim-LB-800 (200 Worlds)**: **`0.9635 → 0.9684 (+0.0049)`**, **P10 Worst-10% `0.9444 → 0.9492 (+0.0048)`**
  - **검증 완료 제출 후보 CSV (801행, ID 100% 일치, NaN/Inf 0)**:
    - 주력: `runs/submissions/sub_e20_track_ab_final.csv` (`E20b_Spatial`)
    - 백업(Zero Selection Bias): `runs/submissions/sub_e20_6tower_all19.csv` (`E20d_6Tower_All19`, OOF `0.9541`, Sim-LB `0.9664`)
- **W&B Artifact 동기화**: `dku-model-cache:latest` 최신 (`M01~M19.npz` + `canonical_cids.npz` + `folds_unseen.json`, 24 files).

## 2. Track A (`M16~M19`) + Track B (`E20`) 핵심 돌파구 요약
- **단일 모델 1위**: `M18_musc_siglip_so400m_384` (**`0.9261`**, `F0 0.9171`, `cid=2 0.8760 (+0.0914 vs M01)`, `ρ(M01)=0.604`).
- **Mid-Level `32×32` 공간맵 픽셀정렬 합의 (`SP4_top2`)**: 4종 이종 백본(`DINOv2-Reg4` + `EVA-02` + `SigLIP` + `ConvNeXt-L`)의 클러스터 정규화 `32×32` 맵에서 픽셀별 2순위 합의 추출 → **단일 비지도 특성 `0.9308` (`F4 0.9809`, `ρ(M01)=0.695`)**로 모든 단일 모델 추월!
- **백본별 채널 분화 실증**: `M16(DINOv2-Reg4)`·`M19(ConvNeXt-L)`은 저랭크 `z_svd` 강세(`0.9191`, `0.8282`), `M17(EVA-02)`·`M18(SigLIP)`은 `z_musc` 압도(`0.8983`, `0.9268`), CLIP 계열(`M18`, `M19`)은 글로벌 `z_logical(0.10)` 유효.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋
- `Sim-LB-800` 월드 생성 시 이상치 점수 기반 클러스터링 금지(반드시 시각 제품 `cids`만 사용)
- 비-DINOv2 백본에 `w_svd >= 0.50` 과다 배정 / 저점 모델 비가지치기 등가중 합산

## 4. 다음 행동 1개
- 사용자 승인 후 `runs/submissions/sub_e20_track_ab_final.csv` Public LB 제출 검증 (`E14 0.8846` 대비 실제 LB 점프폭 확인) 및 `runs/track_b_v2/val_oof_e20.json` 기반 Train 오답 Top-20 분석.
