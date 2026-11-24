# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`)
- **Public LB 최고점**: **`E07` (`0.7167`)** (`E04 0.5593 → E06 0.6941 → E06-clean 0.6968 → E07 0.7167`); 구출본 `runs/submissions/sub_e11_ensemble_e09_e10.csv` 미제출 대기
- **Train 5-Fold OOF 기준선**: `E08` (`0.7919`, 224px CLIP ZS+L12 `K=12` Logit-Z)
- **투트랙 컴퓨팅 전략**:
  - **Track 1 (Kaggle 무료 GPU — 가동 중)**: `rlaqudwn1/dku-defect-2026-e09-genuine-musc` (`M01_musc_dinov2_l518`, `RUNNING`, 30GB RAM / 영구 저장)
  - **Track 2 (교내 Blackwell 슬라이스)**: `VRAM 5.50 GiB / RAM 8.0 GiB (Swap 0B) / Disk 64 GB / torch 2.15.0+cu132` — `M02~M05` 경량 고속 모델 담당

## 2. 15-모델 Wave Queue & 보고서 자동화 아키텍처
- **Queue 엔진**: `scripts/run_15model_queue.py` (5대 직교 패러다임 × 3백본 = 15종, `Wave 1→2→3` 우선순위, 원자적 체크포인트 `runs/queue15/models/M*.npz` 150KB, `--max-minutes`, `--auto-clean-cache` 64GB 디스크 가드)
- **5대 Nested 앙상블 & 보고서 생성기**: `scripts/run_15model_ensemble.py` (`equal_z`, `equal_rank`, `caruana_greedy`, `nested_nn_logistic_stack`, `power_rank` + Spearman 다양성 행렬 → `queue_report_tables.md` & `sub_queue15_best_ensemble.csv` 자동 갱신)
- **OOM 방어 철칙**: `pip --no-cache-dir`, `DataLoader(num_workers=0)`, `bf16` autocast, 1레이어 `memmap` 즉시 삭제.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- `37×37` 패치 `12×12` 축소 / `top-1 max` 동료 비교 / Min-Max 후 `logit` 재변환 / Center-Crop 라벨 오염
- 서버 로컬에만 코드 두고 Git 미커밋 상태로 장시간 실행하기 (세션 초기화 시 유실)

## 4. 지금 당장 할 다음 행동 (Two-Track 동시 실행)
- **Track 1 (Kaggle — 로컬에서 완료 시 회수)**: `.venv/bin/kaggle kernels status rlaqudwn1/dku-defect-2026-e09-genuine-musc` 확인 → 완료 시 `e09_oof_and_test_cache.npz` 회수해 `runs/queue15/models/M01_musc_dinov2_l518.npz`로 편입.
- **Track 2 (교내 Blackwell 서버 — `M01` 중복 제외하고 `M02~M05` 즉시 실행)**:
  `python3 scripts/run_15model_queue.py --models M02,M03,M04,M05 --auto-clean-cache --max-minutes 180`
- **통합**: `M01` + `M02~M05` 합산 후 `python3 scripts/run_15model_ensemble.py` 실행 → `queue_report_tables.md`·`sub_queue15_best_ensemble.csv` 확정 후 Wave 2/3 확장.

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 `defectlab brief`를 실행해서 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
