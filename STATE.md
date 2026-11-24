# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`)
- **Public LB 최고점 (Champion)**: **`E07` (`0.7167`)** — `E04(0.5593) → E06(0.6941) → E06-clean(0.6968) → E07(0.7167)`
- **Train 5-Fold OOF 기준선**: `E08` (OpenCLIP `ViT-B-16` 10,821장 전수 OOF, `K=12` Logit-Z `0.7919` / 원시 `0.6209`)
- **주력 실행 환경 (전환 완료)**: 교내 **RTX PRO 5000 Blackwell HAMi 슬라이스** (`VRAM 5.50 GiB`, `컨테이너 RAM 8.0 GiB / Swap 0B`, `CPU 4코어`, `torch 2.15.0+cu132`, `bf16` 고속 지원)

## 2. 핵심 병목 진단 & OOM 방어 철칙
- **`0.7167`에 막혔던 4대 원인**: ① `37×37` 패치를 `12×12`로 축소 ② `E06b`의 `top-1 max` 비교 버그(불량끼리 매칭→거리 `0`) ③ `write_sub` Min-Max 후 `logit` 이중 왜곡 ④ 외곽 결함 자르는 Center-Crop 라벨 오염.
- **Blackwell 8.0 GiB RAM (Swap 0B) OOM 방어 규칙**:
  - `pip install --no-cache-dir`, `DataLoader(num_workers=0~2)`, `bf16` autocast + `batch_size=8~16`, Fold별 체크포인트(`ckpt_*.npz`) + 디스크 `np.memmap` 즉시 삭제.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- `37×37` 패치를 `12×12`로 축소 / 동료 패치 비교 시 `top-1 max` / Min-Max 후 `logit` 재변환
- 5-Fold 전체 Unseen OOF 미검증 모델을 제출에 포함하기 / `8.0 GiB` 컨테이너에서 대형 텐서 RAM 통째 적재

## 4. 지금 당장 할 다음 행동 (15개 모델 Wave-Ordered Queue + 5대 앙상블 자동 루프)
- **정식 Queue 파이프라인 커밋 완료**: `scripts/run_15model_queue.py` (Wave 1~3, 5대 패러다임 × 3종 백본 = 15모델, `--max-minutes`/`--auto-clean-cache`/원자적 체크포인트 이어달리기) + `scripts/run_15model_ensemble.py` (Nested 5-Fold 5대 앙상블 비교 + 보고서용 `queue_report_tables.md` 자동 생성).
- **실행 순서**: 서버에서 `Wave 1 (M01~M05)`부터 실행(`--wave 1 --auto-clean-cache --max-minutes 180`) → 5대 패러다임 대표 5종 확보 후 `Wave 2`·`Wave 3` 확장. (구출된 `runs/submissions/sub_e11_ensemble_e09_e10.csv`도 보존 완료, Kaggle 무료 GPU에서도 `E09` v2 병렬 가동 중.)

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 `defectlab brief`를 실행해서 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
