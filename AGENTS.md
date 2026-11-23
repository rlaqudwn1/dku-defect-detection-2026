# 작업 기준 (Context Engineering & Competition Rules)

## 1. 새 세션 부팅 순서 (토큰 절약 & 문맥 유지)
- 새 세션은 **딱 3개 파일**만 순서대로 읽는다: `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md`.
- 필요하면 `defectlab brief`를 실행해 최신 `git HEAD`와 `runs/` 내 실측 지표를 1초 만에 확인한다.
- 전체 전략과 명령어 레퍼런스가 필요할 때만 `LOOP_PLAN.md`를 읽는다. 과거 요약 문서를 새로 만들지 않는다.

## 2. 실험 루프 & 문맥 갱신 규칙 (Anti-Drift)
- **One-Change Rule**: 한 번의 루프에서 직전 Champion 대비 **변수 딱 1개**만 바꾼다.
- **검증 기준**: Fold-4/Fold-0 빠른 검증 → 통과 시 5-Fold Unseen OOF AUROC 확정. 모든 비교는 동일한 `runs/folds_unseen.json` 기준.
- **오답 주도 EDA (Train 전용)**: 새 Champion이 나오면 반드시 `defectlab errors`로 **Train 검증 Fold의** FP/FN Top-20을 확인하고 다음 가설을 도출한다.
- **루프 종료 30초 의식 (필수)**:
  1. `EXPERIMENTS.md` 표에 결과 1줄 추가 (실패/폐기 포함).
  2. `STATE.md` 덮어쓰기 (Champion, 최신 Train 오답 관찰, Dead Ends, 다음 행동 1개 갱신 — 50줄 이내 유지).
  3. 새로운 보고서·요약 `.md` 파일을 남발하지 않는다.

## 3. 대회 규정 및 엄격한 Test 격리 경계
- `0=정상`, `1=불량`. 제품 수와 관계없이 이미지당 `anomaly_score` 하나다. AUROC는 검증 전체 예측에서 계산한다.
- **Test 완전 격리 (절대 금지)**: `data/raw/open/test/` 이미지의 육안 열람, contact sheet/히트맵/클러스터 갤러리 생성, Test 예측 상위 이미지 시각 점검을 금지한다. 모든 EDA·오답 분석·특징(Representation) 연구·하이퍼파라미터 결정은 **오직 `Train` (`runs/folds_unseen.json`) 내부에서만** 수행한다. Test는 최종 추론(`submission.csv` 생성) 입력으로만 사용한다.
- 원본 이미지/CSV(`data/raw/open`)를 수정하지 않는다. 테스트 정답·외부 원본 데이터/라벨 조회, 외부 이미지 추가 학습, 원격 모델 API 예측을 금지한다.
- 공개 사전학습 가중치는 **2026-11-15 이전 공개 오픈소스**만 사용하며, 최종 제출 시 버전·출처·라이선스를 명시한다.
- 데이터·가중치(`.pt`/`.pth`)·예측 CSV·API key·사용자 절대 경로·비공개 Drive ID·`runs/`를 Git에 넣지 않는다.
- 미관측 값은 추측하지 않고 `null`/`unknown`으로 적는다.
- Markdown은 UTF-8/LF다. 변경 후 `python scripts/check_text.py`, `.venv/bin/ruff check .`, `.venv/bin/pytest`로 확인한다. (커밋 전 `--staged` 포함)
- 대회 제출과 GPU 결제는 사용자의 별도 명시 지시 없이는 진행하지 않는다.
