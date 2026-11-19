# 연구 운영 진입점

진행자는 사용자의 요청을 읽고 작업을 배정하며 근거에 따라 결정한다. 도구는 로컬 실행과 기록을 맡는다. 시작 순서는 **요청·규칙 → 코드/근거 확인 → 도메인 자문 → 독립 검토 → 결정 → 허용된 실행 → 보고·인계**다. 연구 범위는 [PLAN](PLAN.md), 명령 계약은 [RUNBOOK](RUNBOOK.md), 공개 관측은 [현재 상태](CURRENT-STATUS.md)와 [검증 범위](IMPLEMENTATION-VALIDATION.md)를 따른다. 과거 설계 문서의 가상 명령과 JSON은 실행 입력으로 사용하지 않는다.

## 시작할 때 확인할 것

1. README와 AGENTS의 데이터·가중치·원격 API·제출·결제 경계를 확인한다. 기존 구현 허가는 유효하다. 대회 제출·GPU 결제에는 별도 명시 지시가 필요하다.
2. `git rev-parse HEAD`, `git status --short`와 현재 상태 문서의 증거 범위를 대조한다. 합성 CPU 검사와 실제 대회 연구를 구분한다.
3. 이전 인계의 비공개 장부·근거 위치를 확인한다. 이 저장소에는 실제 지속 연구 장부가 포함되어 있지 않다. 먼저 이전 호스트 인계와 로컬 `runs/operations/handoff.md`를 확인한다. 위치를 찾지 못하면 `unknown / evidence unavailable`로 남긴다. 없는 파일에 `status`를 실행하면 빈 요약을 돌려주므로 이를 연구가 없었다는 증거로 사용하지 않는다.
4. 실제 파일 존재를 확인한 뒤 `defectlab status runs/operations/ledger.json`을 실행한다. 명시된 의존성·해시·최신 단계 판정을 읽고 과거 실행 완료와 현재 유효성을 분리한다. stale/missing 증거 또는 새 split/protocol 변경의 영향을 확인하기 전 현재 완료로 승계하지 않는다. 기록되지 않은 관계는 unknown이다.
5. 작업 ID·질문·주 도메인·범위·정확한 commit/설정/입력 hash·완료 조건·다음 행동을 정한다. 필요한 자료만 자문에 제공한다.

`runs/operations/`는 새 운영 세션의 **제안 로컬 위치**이며 Git에서 제외된다. 이전 장부가 다른 위치에 있으면 그 장부를 이어간다. 장부가 정말 없는 것으로 확인되면 새 장부임과 확인하지 못한 과거를 명시하고 시작한다. 장부와 증거를 개인 영속 저장소에 함께 보존하고 목적지 해시를 확인한다. 공개 문서에 사용자 절대 경로·Drive ID·원본·가중치·인증정보를 넣지 않는다.

## Pro 보조 EDA의 단계 재개

이번 Pro 파일럿과 로컬 데이터 조사는 [실행·재개 계획](PRO-ASSISTED-EDA.md) 및 `configs/pro-assisted-eda.json`을 따른다. `defectlab workflow PLAN LEDGER`는 현재 근거와 고정 의존성을 읽어 다음 행동과 병렬 준비 단계를 안내한다. Pro 분석·실행·검토는 호출하지 않는다. 없는 장부는 unknown이며, 패킷 생성은 전송이나 검토 완료가 아니다. 공식 시각 자문 허용 미확인 상태에서도 공개 텍스트 패킷과 로컬 입력 확인은 준비할 수 있다.

## 도메인 배정

매번 7개 역할을 모두 호출하지 않는다. 진행자가 질문에 필요한 주 도메인을 고르고 연관 관점만 추가한다. 아래 경로는 저장소 루트 기준이다.

| 도메인·호출 시점 | 읽을 문서·코드/CLI | 입력 → 기대 산출물 | 진행자의 결정 |
|---|---|---|---|
| 연구 설계: 최초 질문·비교 변경 | `docs/PLAN.md`, `docs/HYPOTHESES-AND-DATA.md` | 규정·관측/가정·이전 결과 → 가설·반증 조건·작은 비교 | 다음 질문·비교 범위 |
| 데이터: EDA·관계·전처리 조사 | `src/defectlab/data.py`, `verify/profile/split` | manifest·카탈로그·표본 ID·원본 → 관계 후보·입력 손실·미확인 | 조사 범위·그룹/분할 후보 |
| 코드·재현: 공통 경로 변경·복구 | `src/defectlab/engine.py`, `tests/`, `docs/FIRST-RUN-AND-REPRODUCTION.md`, `reproduce` | commit·설정·환경·checkpoint/검사 → 재현 절차·실패 조건 | 변경 채택·추가 검사 |
| 모델링: 첫 baseline·새 backbone/loss | `configs/baseline.toml`, `src/defectlab/engine.py`, `train/tune` | 데이터 관측·가설·가중치 규정·예산 → 설정·대안·예상 실패 | 제한된 실험 계획 |
| 평가: split/fitting 변경·모델 선택 | `src/defectlab/engine.py`, `src/defectlab/selection.py`, `evaluate/ensemble/select` | 그룹 근거·예측·선택 이력 → 누수/선택 편향·추정 대상·한계 | 프로토콜·채택 가능한 결론 |
| 연구 운영: 긴 실행 전·GPU 변경 | `docs/COLAB.md`, `docs/RESEARCH-OPERATIONS.md`, `record/status` | 실제 시간·GPU·세션·영속 저장 영수증 → 비용·중단 기준·복구/재현 예약 | 작업 순서·짧은 P000 범위 |
| 보고서: 핵심 주장·최종 원고 | `docs/REPORT-CONTEXT-HARNESS.md`, `src/defectlab/governance.py`, `results/context/freshness/freeze` | 실제 결과·실패·표·선택·재현 → 주장-근거 행·누락·한계 | 주장·수정·최종화 범위 |

독립 페어리뷰는 모든 도메인을 가로지른다. 분할/누수, 새 모델·전처리의 비교 공정성, 긴 실행의 복구, 최종 선택, 핵심 보고서 주장처럼 결과를 바꾸는 판단은 정확한 대상 버전과 근거를 별도 검토자에게 준다. 작성자 자신이나 같은 호출의 역할 변경을 독립 검토로 기록하지 않는다. 기본은 검토 한 차수와 진행자의 처분 한 차수이며 중대한 새 사실이 생기면 해당 변경을 새 버전으로 다룬다.

## 요청에서 실행으로

1. `task`와 `request`를 기록한다. 역할별 런타임 LLM 모델·effort는 미정이다. 명시 요청이 없으면 null을 유지한다. 구축 작업에서 사용한 worker/reviewer 배분을 복사하지 않는다.
2. 진행자가 **현재 호스트가 제공하는 실제 에이전트 호출 도구**로 입력·대상 버전·질문·기대 출력을 전달한다. `defectlab record/status/context`는 호출 도구가 아니다. 호스트 호출 기능이 없으면 unavailable/not_dispatched로 남기고 응답을 만들지 않는다.
3. 실제 호출 handle·원문 응답·입출력 hash·한계를 `execution`으로 남긴다. 요청 모델/effort와 호스트가 확인해 준 observed 메타데이터를 분리한다. 확인되지 않은 observed 값은 null이다. 모델 관측값을 채울 때는 host receipt와 파일 증거가 필요하다.
4. 독립 검토는 별도 실제 호출 후 `review`로 기록한다. 완료된 execution, 서로 다른 author/reviewer, `independent: true`, hash로 고정한 `refs`가 필요하다. 검토자가 작성에 참여했는지도 진행자가 확인한다. 스키마 통과만으로 독립성·과학적 타당성이 입증되지는 않는다.
5. 진행자가 쟁점별 fixed/accepted-risk/deferred/rejected와 이유·증거·남은 한계·다음 행동을 `decision`에 남긴다. fixed는 실제 수정/검사 범위까지만 뜻한다. 수정된 새 hash에 과거 검토를 자동 승계하지 않는다.
6. 결정된 범위의 CLI를 RUNBOOK대로 실행한다. 실제 outcome·실패·시간·저장 영수증을 보존한다. 완료 조건과 파일 근거가 충족된 task/stage만 completed로 기록한다. 결정, 실행 완료, 원격 영속 저장, 새 세션 복구는 별개다.

기록 입력은 `id`, `kind`를 사용한다. 참조는 `refs`/`depends_on`의 `{"id":"기존-ID","sha256":"장부가 계산한 해시"}`다. 입력에 자신의 `sha256`을 넣지 않는다. 파일 증거는 `{"path":"입력 JSON 기준 상대 경로","sha256":"실제 파일 SHA-256"}`다. kind별 세부 검증은 `src/defectlab/governance.py`가 기준이다. [기록 규약](harness/RECORDS.md)의 과거 request_id/execution_id 중심 JSON은 의미 설명용이며 그대로 입력할 수 없다.

## 실행 가능한 작은 장부 예시

아래는 실제 연구와 분리된 smoke다. 새 임시 디렉터리에 **계획만** 기록하며 전문가 호출·독립 검토·대회 실행·단계 완료를 주장하지 않는다. 기본 설치 후 저장소 루트에서 실행한다.

```bash
example_dir=$(mktemp -d)
export EXAMPLE_DIR="$example_dir"
python - <<'PYCODE'
import json
import os
from pathlib import Path
p = Path(os.environ['EXAMPLE_DIR'])
records = [
    {'id': 'EX-TASK', 'kind': 'task', 'owner': 'conductor',
     'status': 'planned', 'question': '정확한 입력으로 자문 준비',
     'next_action': '호스트에서 실제 자문 호출'},
    {'id': 'EX-REQUEST', 'kind': 'request', 'task_id': 'EX-TASK',
     'domain': 'evaluation', 'role': 'advisor', 'status': 'planned',
     'question': '분할이 추정하는 일반화 범위는?',
     'requested': {'model': None, 'reasoning_effort': None}},
    {'id': 'EX-EXECUTION', 'kind': 'execution', 'request_id': 'EX-REQUEST',
     'status': 'not_dispatched', 'host_task_handle': None,
     'observed': {'model': None, 'reasoning_effort': None},
     'limitations': ['실제 호출하지 않은 smoke 예시']},
]
for item in records:
    (p / (item['id'] + '.json')).write_text(
        json.dumps(item, ensure_ascii=False) + '\n', encoding='utf-8')
PYCODE
defectlab record "$example_dir/ledger.json" "$example_dir/EX-TASK.json"
defectlab record "$example_dir/ledger.json" "$example_dir/EX-REQUEST.json"
defectlab record "$example_dir/ledger.json" "$example_dir/EX-EXECUTION.json"
python - <<'PYCODE'
import json
import os
from pathlib import Path
p = Path(os.environ['EXAMPLE_DIR'])
task = json.loads((p / 'ledger.json').read_text())['records'][0]
item = {'id': 'EX-DECISION', 'kind': 'decision', 'owner': 'conductor',
        'refs': [{'id': task['id'], 'sha256': task['sha256']}],
        'choice': '자문 응답 전에는 프로토콜을 확정하지 않음',
        'reason': '실제 응답과 검토 근거 없음',
        'next_action': '호스트 자문 호출 또는 unavailable 인계'}
(p / 'EX-DECISION.json').write_text(
    json.dumps(item, ensure_ascii=False) + '\n', encoding='utf-8')
PYCODE
defectlab record "$example_dir/ledger.json" "$example_dir/EX-DECISION.json"
defectlab status "$example_dir/ledger.json"
```

예상 상태는 records=4, task/request/execution/decision 각 1개, stages={}다. reviewed/completed 기록은 없다. 이 임시 장부를 실제 연구 장부로 복사하지 않는다.

## 루트 단계·근거 변경·보고

진행자가 PLAN의 각 stage 완료 조건을 현재 입력 버전에 대조한다. `stage`에는 stage_id·owner·reason·completion_conditions·next_action을 두고 task/decision 버전을 refs/depends_on으로 고정한다. completed에는 실제 파일 근거와 고정 의존성이 필요하다. 존재하지 않는 장부를 요약 문서에서 거꾸로 완성하지 않는다.

split·protocol·전처리·코드·선택 기준 변경은 새 `change` ID로 targets·old_version·new_version·이유·실제 파일 증거를 남긴다. status는 명시된 의존 후속을 needs_reassessment로 표시한다. 진행자가 누락된 관계도 확인하고 영향받는 후속 단계의 새 작업·담당·다음 행동을 정한다. 이전 run과 판정은 보존한다. 재판정은 새 stage ID와 supersedes/change_id, 새 근거로 남긴다. 버전 번호만 바꿔 완료를 복사하지 않는다.

`results`는 실제 run outcome만 표로 내보낸다. `context`는 절의 purpose·claims·evidence·limitations와 명시 의존성을 묶고 원고를 생성하지 않는다. `freshness`는 파일 변경/누락을 확인한다. `freeze`는 유효한 근거를 검사하며 final brief에는 실제 독립 검토가 필요하다. 명시하지 않은 절 의존성이나 과학적 의미를 자동 추론하지 않는다. 자세한 호출은 RUNBOOK을 따른다.

## 세션 종료와 인계

진행자는 종료 전에 비공개 `runs/operations/handoff.md`를 갱신하고 장부·증거와 함께 개인 영속 저장 위치에 보존한다. 기존 인계 경로가 있으면 이어 쓴다. 복사 요청과 실제 목적지 해시 확인은 구분한다. 다음 세션이 찾을 수 있도록 호스트 인계에 위치를 남긴다.

인계에는 기준 시각·진행자, commit/dirty 상태, 실제 장부 위치와 SHA-256, stage별 최신 record ID/참조 hash/현재 validity, 원본·split·protocol·checkpoint·결과의 버전/해시/위치, 실제 자문·검토·결정 ID, 실패와 미확인 사항, GPU 세션의 마지막 관측/unknown 꼬리, 저장·복구 확인 범위, 다음 최소 작업·주 도메인·완료 조건을 남긴다. 민감한 경로는 비공개 인계에만 둔다. 실제 관측이 없으면 null/unknown으로 적는다.

공개 변경이 있으면 CURRENT-STATUS에 검증한 범위와 한계만 갱신한다. Markdown 변경은 `python scripts/check_text.py`로 검사하고 커밋 전 `--staged`도 검사한다. 문서만 바꾼 세션은 실제 예시 실행과 텍스트 검사를 남기고 합성 CPU 검사 기록을 새 GPU 검증으로 승격하지 않는다.
