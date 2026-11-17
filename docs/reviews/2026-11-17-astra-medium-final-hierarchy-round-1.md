# 최종 전체 계층 설계 검토 — 1차

판정: **치명 0 / 중간 1 / 미미 0**. 전체 연구 경로를 무효화하거나 막는 설계 결함은 확인하지 못했다. 사용자 중단 조건에 따라 **검토 반복을 여기서 종료한다. 지금 설계를 수정할 필요는 없다.** 아래 중간 항목은 남은 국소 명확성 문제로 보고한다. 구현·실험 성공을 인증하는 판정은 아니다.

## 대상과 실행 식별

- 기준 commit: `564885ee79bba8d422c6c3cd14fc6c78f64bd1e8`. 읽기 시작 시 작업 트리는 깨끗했다.
- 요청 모델/effort: `gpt-6-astra` / `medium`.
- 호스트 task handle: `/root/astra_medium_final_hierarchy_review`.
- 별도 provider invocation ID·실제 모델/effort 영수증은 제공되지 않았다. 해당 관측값은 null이며 요청값과 task handle을 런타임 증명으로 간주하지 않는다.
- 현행 `AGENTS.md`, `README.md`, `GLOSSARY.md`, `docs/`의 리뷰 외 설계 문서 전부, `reports/WRITING-PLAN.md`, 실험 백로그/카드, 연구 카드/규정/수업 공지/데이터 정의/과거 조사 기록을 읽었다.
- 과거 후속표와 보고서 문맥 리뷰는 현행 문서를 먼저 읽은 뒤 이력으로만 참고했다. 과거 개별 리뷰를 이번 전체 계층 통과의 증거로 쓰지 않았다.
- 검토자가 변경한 파일은 이 보고서 하나다. 설계 수정·새 연구·외부 계정 연결·실행기 구현·학습·데이터 변경은 하지 않았다.

## 실제 계층과 전체 경로

루트 책임은 별도 이름의 서버가 아니라 `MANAGEMENT-HARNESS`의 진행자/결정 책임자와 공통 기록 계층에 있다. 진행자가 계획·범위·대상 버전을 정해 도메인에 요청하고, 실제 응답과 실행 증거를 받아 지적 처분·다음 행동을 결정한다(`docs/MANAGEMENT-HARNESS.md:9`, `:31`, `:39`). 하위 도메인은 자문·산출물 계약, 실행기는 실제 계산·검사의 책임이다. 따라서 단순 도구 목록이나 역할 이름의 나열에 그치지 않는다.

| 루트 계획의 과정 | 담당 하위 책임과 하향 입력 | 상향 결과·다음 연결 |
|---|---|---|
| 문제·가설 | 연구 설계: 규정·질문·관측·근거 카드 | 반증 조건·비교 계획 → 데이터/평가/모델 |
| EDA·카탈로그·특징 | 데이터: manifest·표본·질문·허용 encoder | 관측/미확인·관계 근거·캐시 계보 → 분할/전처리 |
| 분할·평가 설계 | 데이터/평가: 그룹·학습 범위·일반화 대상 | split/protocol·누수 검사·선택 편향 한계 → 학습/비교 |
| 코드·모델·학습 | 코드/재현·모델링: 고정 입력·가중치 근거·설정 | 공통 train/validate/predict·checkpoint·평가 → 후보/운영 |
| HPO·앙상블 | 연구/모델/평가: 사전 후보·예산·같은 holdout | 실패 포함 비교·선택 횟수·정확한 recipe → 최종 선택 |
| 모델 산출물·재현 | 코드/재현·운영: weight/config/code/data/split/recipe | 영속 저장·새 환경 검사·허용오차 결과 → 선택 동결 |
| 보고서·문맥 | 보고서: claim·고정 근거·절 brief·검토 이력 | 실제 주장/표/그림·변경 영향·스냅샷 → 제출본 |
| 시간·Colab | 운영: P000·관측 구간·예약·복구 영수증·마감 | 비용·중단/복구·fallback 제안 → 루트 우선순위 |

이 매핑은 `docs/PLAN.md:167`, `docs/MANAGEMENT-HARNESS.md:19`, 각 하위 설계의 입출력 계약을 종합한 것이다. 모든 도메인에 별도 서버·상시 worker를 추가할 이유는 없다. 미래 연구 하네스의 역할별 모델은 미정이며 현재 구축 작업의 모델 배분을 적용하지 않았다.

## 전체 수명주기 판단

- **권한·결정:** 진행자가 최종 처분을 맡고 자문은 완료나 성공을 선언하지 않는다. 사용자 승인 없이 매 단계 연구를 멈추는 구조가 아니며 실제 대회 제출·결제만 기존 명시 지시 범위다(`docs/MANAGEMENT-HARNESS.md:31`, `:79`).
- **ID·문맥·버전:** task/request/execution/review/decision과 대상 버전, 소유 역할·의존 작업·해시·supersedes가 연결된다. 보고서 문맥은 원본 기록을 복제하지 않고 참조하며 범위 축소를 숨기지 않는다(`docs/harness/RECORDS.md:7`, `:18`; `docs/REPORT-CONTEXT-HARNESS.md:28`, `:36`).
- **독립 검토:** 별도 호출과 고정 대상·실제 응답을 요구한다. unavailable, reviewed, decided, completed, changed_after_review를 구분하므로 자문 자료 준비나 수정만으로 독립 검토 완료를 승격할 수 없다(`docs/MANAGEMENT-HARNESS.md:43`, `:46`, `:70`; `docs/harness/RECORDS.md:59`).
- **과학적 유효성:** 이미지당 불량 확률·0/1 방향, 종속 그룹 분리, fitting 범위·캐시 출처, 미학습 검증 예측, 전체 예측 AUROC, 반복 holdout 선택 한계, 재학습 모델의 점수 분리를 함께 규정한다. 제공 규정의 사전학습 공개일은 2026-11-15까지이며 금지된 외부 데이터·정답 조회·원격 모델 예측으로 확장하지 않는다.
- **실패·회복·상향 보고:** 누수·가중치 출처·재현 실패는 해당 입력의 채택만 막고 다른 조사는 계속한다. 운영자가 예산·중단·복구·fallback을 진행자에게 돌려보낸다. 실행 종료·업로드 요청·영속 저장·새 세션 복구를 구분한다(`docs/MANAGEMENT-HARNESS.md:81`; `docs/RESEARCH-OPERATIONS.md:43`, `:62`).
- **모델·최종 동결:** 모델 명세/run/checkpoint/evaluation/selection을 분리하고 정확한 버전·recipe·재현 증거로 동결한다. 플랫폼 실제 선택은 별도 기록이다. 개인 W&B 인증 없이 전달 파일로 재현하는 경로도 있다(`docs/CODEBASE-AND-MODEL-MANAGEMENT.md:47`, `:71`; `docs/FIRST-RUN-AND-REPRODUCTION.md:34`).
- **보고서 드리프트:** 근거 변경의 영향 범위와 needs_update, 과거 reviewed의 비승계, 원고·claim·run·표/그림·최종 선택·재현 부록의 동시 고정이 있다. 이후 Private 결과를 선택 당시 근거로 소급하지 않는다(`docs/REPORT-CONTEXT-HARNESS.md:74`, `:85`, `:102`).

## FH-01 — 중간 — 루트 단계 상태의 근거와 변경 영향 연결이 덜 명시적임

**현재 근거:** `docs/PLAN.md:167`–`:176`은 A~G의 산출물·진행 조건을 정의한다. `docs/MANAGEMENT-HARNESS.md:97`은 project에 현재 단계를, `:98`은 task에 대상 버전·완료 조건을 둔다. `docs/harness/RECORDS.md:10`은 의존 작업을 연결하고 `:18`은 새 버전/수정 이력을 요구한다. 그러나 이 계약들은 project 단계가 어떤 task/decision/증거 버전 집합으로 성립하는지, 그 입력이 변경됐을 때 루트 단계 표시를 누가 어떻게 재판정하는지를 한 관계로 명시하지 않는다. 보고서에는 별도의 변경 영향 규약이 있다(`docs/REPORT-CONTEXT-HARNESS.md:74`).

**실패 시나리오:** C 기준 실험 완료 뒤 촬영 종속 그룹을 발견해 split v2를 만든다. 해당 새 분할 검토와 보고서 needs_update는 올바르게 남겨도, project의 현재 단계와 기존 C 완료 근거가 split v1을 가리키는지는 즉시 드러나지 않을 수 있다. 다음 진행자가 단계 표시만 읽으면 새 split에서 baseline을 다시 확보해야 한다는 작업을 늦게 발견한다.

**왜 치명이 아닌가:** 고정 split 비교·바뀐 버전의 별도 기록·확인된 누수의 성능 채택 금지·증거 없는 complete 금지가 이미 있다. 기존 결과를 그대로 유효한 최종 결과로 채택하는 것은 다른 명시 계약을 위반한다. 계보도 보존된다. 따라서 이는 전체 경로/과학적 유효성을 무효화하는 필수 계약 부재가 아니라 루트 진행 상황을 정확히 요약하는 규약의 국소 공백이다.

**최소 보완안:** 향후 구체화 시 project 단계의 근거 task/decision/대상 버전을 참조하고, 선행 근거 변경 때 진행자가 영향받는 단계·다음 작업을 재판정한다는 짧은 규약을 연결한다. 별도 엔진·전역 자동 롤백·사용자 승인 게이트는 필요 없다. 이번에는 수정하지 않는다.

**수용 증거:** split v1→v2의 작은 문서 시나리오에서 과거 완료 증거는 보존되고 현재 단계의 유효 근거/미완료 의존 작업/담당자가 드러나며, 영향받지 않는 작업은 계속할 수 있음을 보이면 된다.

## 기준 문서 SHA-256

| 문서 | SHA-256 |
|---|---|
| docs/PLAN.md | `8e9058ecd975e5930cba7c0fa46c7218dbf316bf433e1844a2c092d70327cf44` |
| docs/MANAGEMENT-HARNESS.md | `9cdca19e8c904b5d1dde54a87d7a03efaffad093f7b8d6c04a0913ee2404a65e` |
| docs/harness/RECORDS.md | `8df2fc3d5e701edfe97d2a940b631394cf6a50526f39164466468a030ad43e2b` |
| docs/CODEBASE-AND-MODEL-MANAGEMENT.md | `c304c23e697beb6c53f16f56df302a5123ebbba09454de6bdf9e17ae6c23f9c2` |
| docs/REPORT-CONTEXT-HARNESS.md | `bd18dd77790b87e801e9b2e03c0d94184f5c9a88d39e4c3fbc42485a59989844` |
| docs/HYPOTHESES-AND-DATA.md | `9f12ba53d338657da78c553daea83b247dcbc6e3ba89cf92f5f2e21c1138698f` |
| docs/RESEARCH-OPERATIONS.md | `59f29dcd18d342c38887bd82c6cd0dace2b84041bc243ae1684723bce77c81bd` |
| docs/FIRST-RUN-AND-REPRODUCTION.md | `bf0c583455cbbeec1af7f169bb4fe23f6a20ba7789fa3afdd2a31e4865c30649` |

검사: `python scripts/check_text.py` 결과 `PASS: Markdown UTF-8/LF (worktree)`. 치명이 없으므로 추가 검토 차수와 설계 수정 루프를 시작하지 않는다.
