# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`, `labels` 키 포함)
- **Public LB 최고점**: **`E07` (`0.7167`)** (`E04 0.5593 → E06 0.6941 → E06-clean 0.6968 → E07 0.7167`)
- **Train 5-Fold Unseen OOF Champion**: **`E12-T2` (`0.9054`, `1_equal_robust_z` / `4_nested_nn_logistic_stack 0.9052`)** — 직전 `E08`(`0.7919`) 대비 **`+0.1135`** 경신!
  - `M02_vand_clip_l336`: **0.8505** (`F0 0.7542 | F1 0.8808 | F2 0.9088 | F3 0.9320 | F4 0.8851`)
  - `M05_siglip_so400m_384`: **0.8400** (`F0 0.8474 | F1 0.6540 | F2 0.8272 | F3 0.9110 | F4 0.9746`)
  - `M03_winclip_l14_laion2b`: **0.8000** (`F0 0.7892 | F1 0.6827 | F2 0.7958 | F3 0.8480 | F4 0.9270`)
  - `M04_patchmil_convnext_b384`: **0.7761** (`F0 0.6637 | F1 0.7346 | F2 0.8202 | F3 0.8935 | F4 0.8873`)
  - 4모델 간 Spearman 상관계수 **`0.22 ~ 0.44`** (패러다임 직교성 입증), 검증 완료 제출본 `runs/queue15/sub_queue15_best_ensemble.csv` (801줄) 생성.

## 2. 투트랙 진행 현황 & 컨테이너 최적화 반영 사항
- **Track 1 (Kaggle 무료 GPU)**: `rlaqudwn1/dku-defect-2026-e09-genuine-musc` (`M01_musc_dinov2_l518`) — 완료 시 `e09_oof_and_test_cache.npz` 회수 대기.
- **Track 2 (Blackwell 5.50 GiB VRAM / 8.0 GiB RAM Swap 0B — 완료)**: `M02~M05` 전부 OOM 없이 완주 (`runs/queue15/models/M02..M05.npz`).
- **적용된 성능·자원 최적화**:
  - `run_e10_vand_adapter.py` & `run_15model_queue.py`: 사전 셔플 memmap + `POSIX_FADV_DONTNEED` + GPU-resident fp16 청크 5-Fold 동시 학습 (`M02` 레이어당 24분→12분, `M04` 전체 5.5분).
  - `run_15model_queue.py`: `WinCLIP/SigLIP` 텍스트 인코더 CPU 오프로드 + `visual bf16` + `M05 batch_size=4` (VRAM `2.14 / 5.63 GiB`), HF 캐시 실용량 `3.5 GB` 가드.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- `37×37` 패치 `12×12` 축소 / `top-1 max` 동료 비교 / Min-Max 후 `logit` 재변환 / Center-Crop 라벨 오염
- `8 GiB` cgroup RAM에서 `13 GB` memmap 75회 랜덤 시크 / `SigLIP-SO400M` fp32 `batch_size=12` (확정 OOM)

## 4. 지금 당장 할 다음 행동
1. **Track 1 (`M01`) 회수 및 5대 패러다임 통합**: Kaggle `e09_oof_and_test_cache.npz` 회수 → `runs/queue15/models/M01_musc_dinov2_l518.npz` 편입 → `python3 scripts/run_15model_ensemble.py` 재실행으로 **5-모델 완전 앙상블(목표 OOF `0.92+`)** 확정.
2. **Wave 2 확장 (선택)**: `python3 scripts/run_15model_queue.py --models M06,M07,M08,M09,M10 --auto-clean-cache --max-minutes 180`.
3. **제출 검증**: 사용자 명시 지시 시 `runs/queue15/sub_queue15_best_ensemble.csv` 제출.

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
