# FH-01 설계 수정 집중 검토

판정: **FH-01의 설계 공백 해소**. 남은 지적 없음(치명 0 / 중간 0 / 미미 0). 처분은 `design-addressed; implementation/evidence pending`이며, 구현이나 실제 운영 성공을 인증하지 않는다.

검토 범위는 기존 FH-01과 이번 수정 세 문서·수용 시나리오의 대응뿐이다. 전체 설계 재검토와 추가 검토 루프는 수행하지 않았다.

## 확인한 계약

- `MANAGEMENT-HARNESS.md` §5.1은 진행자/결정 책임자가 단계 판정을 관리하며, 완료 조건을 충족한 task·decision·증거의 버전을 참조하도록 한다. `RECORDS.md`의 루트 단계 판정은 ID·버전/hash·판정 시점·이유까지 연결한다.
- 변경 대상에는 데이터·split뿐 아니라 task/decision도 포함된다. 진행자는 의존 관계를 따라 후속 작업·산출물·단계의 영향을 확인하고 먼저 `needs_reassessment`를 표시한다. 관계 미확인은 유효로 자동 처리하지 않는다.
- 변경 근거를 사용하는 다음 단계/선택 전에 재판정하도록 시점이 명시됐다. 재판정 중에는 과거 completed로 현재 완료를 주장할 수 없다.
- 과거 run의 완료 상태·평가 증거·단계 판정을 보존하면서 현재 기준의 근거 유효성을 분리한다. 새 증거의 완료 조건 충족과 supersedes 관계가 필요하므로 버전 번호만 바꾼 완료 승계를 허용하지 않는다.
- 추가 관측·재실행·검토 작업과 담당·미완료 의존성·다음 행동을 연결한다. 영향받지 않는 작업은 계속하며 전역 롤백이나 새 사용자 승인 게이트를 추가하지 않는다.
- `PLAN.md` §11이 단계 진행 조건을 이 계약에 연결한다. split v1→v2 문서 예시는 A2/C와 후속 비교·선택·보고서의 재판정, 새 baseline task와 담당, v1 증거 보존, 독립 작업 지속을 보여 기존 FH-01의 수용 조건을 충족한다.

## 대상 SHA-256

| 문서 | SHA-256 |
|---|---|
| docs/MANAGEMENT-HARNESS.md | `fbe941d6a44e5a86bb999e0f848eeef6bfec629b62417a84cde1f2508221e2ae` |
| docs/harness/RECORDS.md | `f37b10f433c1c8724a65edcf7d170d6d884c2a8a9edfe0592d1ceff6789271d4` |
| docs/PLAN.md | `97638f3f93230174c33bfdd967c8349805bc4bd1fc3a64b9e2d4b1006305f8f6` |
| docs/reviews/2026-11-17-fh-01-design-fix.md | `ac2d3fbcac485fafa1f966ab600cc70d20c8623a7546daeabb87a9c40ce32bd8` |

## 호출·검사 한계

- 요청 모델/effort: `gpt-6-astra` / `medium`. 호스트 task handle: `/root/astra_medium_fh01_fix_review`.
- 별도 provider invocation ID·실제 모델/effort 영수증은 제공되지 않았다. 해당 관측값은 null이며 요청값과 task handle은 런타임 증명이 아니다.
- 검토자는 이 보고서만 작성했다. 설계·코드 수정, 신규 조사, 외부 계정 연결, 학습은 하지 않았다.
- 읽기 전용 텍스트 검사: `python scripts/check_text.py` 결과 `PASS: Markdown UTF-8/LF (worktree)`. 설계 수용 시나리오는 실행 검사가 아니다.
