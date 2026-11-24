# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`)
- **Public LB 1위 (Champion)**: **`E11` (`0.875246`)** (`E09 MuSc + E10 VAND` — `E07 0.7167` 대비 **`+0.1585`** 폭등)
- **Train 5-Fold Unseen OOF Champion**: **`E14-Q15` (`0.9520`, Mean `0.9612`, `7_pruned_elite9_nested_stack`)**
  - Fold별 OOF: **`F0 0.9070 | F1 0.9587 | F2 0.9697 | F3 0.9855 | F4 0.9850`**
  - 무선택편향 5-Tower 대안: **`6_paradigm_5tower_nested_stack` (`0.9476`, 12모델, `A_MuSc 30%` 고정 보호)**
  - 단일 Top-3: **`M01_musc_dinov2_l518` (`0.9244`)**, **`M06_musc_dinov2_b518` (`0.9158`)**, **`M08_patchmil_dinov2_l518` (`0.8838`)**
  - 검증 완료 제출 후보 (`runs/submissions/`): `sub_e14_elite9_nested_stack_oof09520.csv` (1순위) & `sub_e14_5tower_12models_oof09476.csv` (2순위, 무누수 5-Tower)
  - 15/15 모델 통합 백업: `runs/queue15_all15_models_backup.zip` (`1.2 MB`, `M01~M15.npz` + 리포트)

## 2. 핵심 과학적 발견 & Train 오답 관찰 (최종 보고서용)
- **15-모델 Naive Stack(`0.9425`) 하락 원인 규명 & 해결**: 기존 균등축소 `L2=2.0`이 E2E 과적합(`M15 0.6781`)·Fold-1 붕괴(`M12 0.4837`)·중복 모델에 강제 가중치를 주어 비지도 `MuSc` 비중을 `19%`로 희석함 → **Pruning + L2-to-Zero(`0.9520`) / 5-Tower 계층(`0.9476`)**으로 `MuSc` 비중을 `30~34%`로 복원하며 최고점 돌파.
- **5대 패러다임 직교성 + 7-기법 앙상블 Ablation**: Spearman `0.16~0.58`(MuSc끼리만 `0.90`). `Elite-9 (0.9520)` > `5-Tower (0.9476)` > `Naive-15 Stack (0.9425)` > `Equal-Z (0.9387)` > `Power-Rank (0.9340)` > `Equal-Rank (0.9270)` > `Caruana-Rank (0.9255)`.
- **Train OOF Top-20 오답 관찰**: **FN Top-20 중 17/20이 Fold 0(`1500x1000` 고종횡비 미세결함)**, FP Top-20은 F1(`1404x1070`, 7/20)·F4(6/20)·F3(5/20) 자연 질감 편차.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- E2E CNN 전체 파인튜닝(`M15 0.6781` 미학습 제품 과적합) / 균등축소 L2로 약한 모델 강제 혼합
- `37×37` 패치 `12×12` 축소 / `top-1 max` 동료 비교 / Min-Max 후 `logit` 재변환

## 4. 지금 당장 할 다음 행동
1. **오늘 남은 1회 제출 (사용자 승인 후)**: `runs/submissions/sub_e14_elite9_nested_stack_oof09520.csv` (또는 무누수 5-Tower `sub_e14_5tower_12models_oof09476.csv`) 제출로 **`0.90+` Public LB** 실측 검증.
2. **이종 백본 정품 `MuSc` 확장 (다음 단계·Kaggle Free GPU)**: `EVA-02-Large 448` 또는 `OpenCLIP ViT-L-14-336` 정품 `MuSc` 추가 + `1500x1000` 타일링/고해상도 대응으로 Fold 0 FN 추가 공략.
3. **`11/30` 코드 제출 패키징**: 클린 재현 스크립트, 가중치 버전·출처·라이선스 표 정리.

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 `defectlab brief`를 실행해서 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
