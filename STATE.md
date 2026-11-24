# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`)
- **Public LB 1위 (Champion)**: **`E11` (`0.875246`)** (`E09 MuSc + E10 VAND` — `E07 0.7167` 대비 **`+0.1585`** 폭등!)
- **Train 5-Fold Unseen OOF Champion**: **`E13-W12` (`0.9475`, 10-모델 완전 Wave 1+2, `4_nested_nn_logistic_stack`)**
  - 단일 최고 모델: **`M01_musc_dinov2_l518` (`0.9244`**, `F0 0.8737 | F1 0.9360 | F2 0.9555 | F3 0.9693 | F4 0.9685`, 라벨 0개!)
  - 단일 2위: **`M06_musc_dinov2_b518` (`0.9158`)**, 지도 1위: **`M08_patchmil_dinov2_l518` (`0.8838`)**
  - 검증 완료 제출본: `runs/submissions/sub_e13_wave12_10models.csv` (801줄 무결성 검증 완료)

## 2. 핵심 과학적 발견 (최종 보고서용)
- **정품 `37×37` 비지도 `MuSc`의 압도적 힘**: `L18_r1`(`0.8969`)이 `L18_r3`(`0.8228`)를 `+0.0741` 압도, `M01` 단독 `0.9244` 달성.
- **5대 패러다임 직교 앙상블 시너지**: Spearman 상관계수 `0.16~0.57`(MuSc끼리만 `0.90`)로 **`0.9054 (4모델) → 0.9323 (5모델) → 0.9396 (9모델) → 0.9475 (10모델)`** 계단식 상승.
- **5대 앙상블 기법 Ablation**: `4_nested_nn_logistic_stack (0.9475)` > `1_equal_robust_z (0.9458)` > `3_caruana_greedy_rank (0.9353)` > `2_equal_rank_avg (0.9343)` > `5_power_rank_top_tail (0.9320)`.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- `37×37` 패치 `12×12` 축소 / `top-1 max` 동료 비교 / Min-Max 후 `logit` 재변환 / Center-Crop 라벨 오염
- 서버 로컬에만 코드 두고 Git 미커밋 상태로 실행하기

## 4. 지금 당장 할 다음 행동
1. **Wave 3 (`M11~M15`) 합류**: 교내 Blackwell 서버에서 `Wave 3` 완료 시 백업 zip을 `runs/queue15/models/`에 풀고 `python3 scripts/run_15model_ensemble.py` 실행 → **완전 15-모델 최종 앙상블 (`0.95+` 목표)** 확정.
2. **이종 백본 정품 `MuSc` 확장 (선택·Kaggle 여유 시)**: `EVA-02-Large 448` 또는 `OpenCLIP ViT-L-14-336` 정품 `MuSc` 추가 시 LB 추가 극대화 가능.
3. **최종 제출 & `11/30` 코드 제출 패키징**: `runs/submissions/sub_e13_wave12_10models.csv` (또는 15-모델 완성본) 제출 후, 클린 재현 스크립트 및 라이선스 명세 정리.

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 `defectlab brief`를 실행해서 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
