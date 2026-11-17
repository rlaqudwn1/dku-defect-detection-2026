# v0.1 독립 구현 리뷰 — 2026-11-17

요청된 검토자: GPT-6 Astra, medium. 실제 모델/effort의 별도 호스트 영수증은 이 문서에서 검증하지 않았다. 검토자는 구현 파일을 수정하지 않았고, 이 리뷰 파일만 작성했다. 구현 담당자가 검토 중 수정한 항목은 최초 발견과 수정 확인을 구분한다.

## 범위와 판정 기준

AGENTS/README/PLAN, 아키텍처·관리·보고서 문맥·FH-01 수정 문서, 실행 가이드, Python 소스, 설정, Colab notebook, 테스트를 확인했다. 기본 Git HEAD는 `4c6c9aac2916fdf97e799e0864d800c8db7180e1`이며 구현은 작업 트리에 있었다. 아래 최종 SHA-256이 실제 검토 대상을 식별한다. 나중 변경은 자동으로 검토된 것으로 간주하지 않는다.

- critical: 잘못된 가중치·현재 무효인 과학적 근거를 정상 완료/최종 산출물로 채택할 수 있는 오류.
- medium: 재현·추적·자동 실행 계약을 깨뜨리는 기능 오류.
- minor: 실행 의미를 오해하게 하는 문서 불일치.

실제 대회 원본이나 정답을 열거나 학습하지 않았다. 재현은 임시 합성 이미지/레코드로 수행했고 결과는 공개 저장소에 넣지 않았다. GPU, AMP의 실제 CUDA 경로, Colab 세션/Drive 복구, W&B cloud 업로드, 새 컴퓨터 재현은 검증하지 않았다.

## 발견과 처분

### IR-01 — critical: 무효화된 과거 검토 결과를 최종 보고서로 동결

최초 `governance.context/freeze`는 파일 해시가 유지되면 명시적 변경에 따른 근거 무효화를 주장에 적용하지 않았다.

재현: 완료 task와 stage, 별도 실행의 독립 review를 기록한다. 원래 증거 파일은 그대로 두고 task를 대상으로 split v1→v2 change를 추가한다. 과거 task/review를 참조하는 `final=true`, `status=reviewed` brief를 만든다. 관측 출력은 `stage_status.C.validity=needs_reassessment`, `freeze(...).frozen=True`였다. FH-01이 보존하려는 과거 완료와 현재 유효성의 구분이 최종 채택에서 사라졌다.

수정 확인: 명시된 claim/ref/review/단계 의존성을 따라 `needs_update`를 지정하고 freeze도 장부를 다시 평가한다. 과거 reviewed 라벨을 직접 복원해도 거절한다. 관계없는 절은 계속 작성할 수 있다. `test_explicit_change_blocks_old_reviewed_final_even_when_old_bytes_are_preserved`와 기존 FH-01 전이 테스트가 통과했다. **해결 확인.**

### IR-02 — critical: 다른 run 디렉터리로 resume하면 best 가중치가 섞임

최초 `train`은 resume가 있을 때 기존 출력의 소유 run을 검사하지 않았다. 합성 모델의 AUROC를 0.5로 고정하고 A/B를 각각 1 epoch 실행한 뒤, A의 last를 B 디렉터리에서 2 epoch까지 재개했다. `status=completed`인데 반환된 best checkpoint의 run ID는 결과의 run ID와 달랐다. 개선이 없는 경우 B의 best를 보존하면서 A의 last/run 결과를 덮어썼다.

수정 확인: 출력 run 및 기존 best/last의 run ID·split·data 출처를 쓰기 전에 검사하고 되감기도 거절한다. 회귀 테스트는 거절 뒤 출력 파일들의 해시가 모두 그대로임을 확인한다. **해결 확인.**

### IR-03 — medium: 경로만 바뀐 동일 앙상블의 재현 실패

최초 recipe 해시는 예측 경로와 증거 파일 해시까지 포함했다. 동일한 검증 예측 두 개를 다른 이름으로 복사해 같은 가중치로 앙상블을 다시 만들면, 확률이 동일해도 `reproduction provenance mismatch: recipe_hash`로 실패했다.

수정 확인: 모델 checkpoint/protocol·가중치·split/data·집계 순서로 구성한 이식 가능한 identity를 증거 경로와 분리했다. 실제 테스트 결합에는 고정한 검증 recipe가 필요하고 구성/가중치 변경은 거절한다. 이동한 예측 파일로 만든 앙상블의 영 허용오차 재현과 잘못된 테스트 recipe 거절 테스트가 통과했다. **해결 확인.**

### IR-04 — medium: 선택 시 현재 checkpoint와 평가된 hash를 대조하지 않음

최초 `select`는 현재 파일의 hash를 새 증거 목록에 넣지만 평가 manifest의 `checkpoint_sha256`와 비교하지 않았다. 평가 후 checkpoint를 교체하면 평가 점수와 다른 bytes를 선택할 수 있는 코드 경로였다. 직접 재현하는 동안 구현 담당자의 수정이 반영되었으므로, 수정 전 성공을 실행 관측했다고 주장하지 않는다.

수정 후 직접 합성 평가를 만들고 checkpoint bytes를 추가 변경한 결과 `selected checkpoint bytes differ from evaluated checkpoint`로 거절됐다. 단일/앙상블의 선언된 checkpoint hash를 모두 검사한다. 교체 checkpoint 거절 회귀 테스트가 추가됐다. **해결 확인.**

### IR-05 — medium: verify 실패가 CLI 성공 종료로 전달됨

최초 `main`은 `passed=false`만 실패 종료로 바꾸고 `verify`의 `ok=false`는 종료 코드 0으로 처리했다. notebook의 `subprocess.run(check=True)` 같은 호출자가 검증 실패를 놓칠 수 있었다.

수정 후 존재하지 않는 합성 입력 디렉터리에 `python -m defectlab verify`를 실행해 `ok=false`, 종료 코드 1을 관측했다. **해결 확인.**

### IR-06 — medium: 실행을 만든 코드 버전이 기록되지 않음

최초 checkpoint/run/model manifest에는 설정·split·data·전처리 설명의 hash가 있었지만 실행 코드 commit/dirty 상태나 실제 source hash가 없었다. 코드가 바뀌어도 같은 설명/설정으로 같은 protocol hash를 만들 수 있어 실행을 정확한 코드로 되돌리는 근거가 부족했다.

수정 확인: checkpoint/run/model manifest가 Git commit·dirty 상태와 패키지 Python 소스별 SHA-256 및 tree digest를 기록한다. 추론 핵심 함수의 source digest가 model spec에 포함되고 로딩 시 현재 구현과 대조한다. 직접 합성 실행에서 checkpoint의 code identity를 확인하고 inference digest를 바꾼 checkpoint가 거절됨을 관측했다. Git 정보가 없는 설치에서는 commit/dirty를 추측하지 않고 null로 둔다. **해결 확인.**

### IR-07 — medium: 실제 run의 protocol이 보고서 표에서 사라짐

`engine`은 `protocol_hash`를 기록했지만 `governance.results`는 `protocol`만 읽어 실제 실행의 protocol 열이 null이 됐다. 수정은 실제 `protocol_hash`를 우선 사용하고 이전 `protocol` 필드를 호환 입력으로 유지한다. 직접 합성 run의 `protocol_hash`가 report-data의 protocol 값으로 보존됨도 실행 확인했다. **해결 확인.**

### IR-08 — minor: verify의 디코드 검사 설명

초기 RUNBOOK은 verify를 디코드 검사까지 하는 명령으로 설명했다. 실제 verify는 파일/CSV/라벨/hash를 검사하고 profile이 픽셀을 디코드한다. 수정 문구가 이 구분을 반영한다. **해결 확인.**

## 확인한 기능과 한계

- 실제 데이터 bytes/라벨을 다시 해시하여 split의 dataset identity와 대조한다. exact duplicate 그룹과 명시적 그룹은 합집합으로 분할하고 양쪽 클래스 존재·ID/그룹 교차를 검사한다. 근접 중복·촬영 종속성을 발견했다고 주장하지 않는다.
- BCE-with-logits와 단일 sigmoid, ID 정렬, 유한한 0~1 확률, 전체 검증 AUROC, checkpoint 실제 hash·학습 ID·검증 ID·라벨·protocol 검사가 소스와 합성 테스트에 연결된다.
- 실제 ResNet18 CPU optimizer/checkpoint/evaluate/predict/CLI 경로를 합성 데이터로 실행하는 통합 테스트가 있다. tiny 모델에서 중단 없는 학습과 epoch 경계 재개의 가중치 일치도 검사한다. 실제 GPU/AMP 재개 동등성은 미검증이다.
- W&B disabled는 로컬 JSONL, 실제 offline SDK는 로컬 run/metric/artifact 흔적을 남긴다. online 연결 테스트는 mock이며 cloud 성공 근거가 아니다. cloud version은 실제 acknowledge 뒤에만 채운다. 원본 이미지·코드 자동 업로드는 설정상 비활성화한다.
- 관리/보고서 모듈은 표준 라이브러리로 동작하고 CLI가 torch를 불러오지 않는 테스트가 있다. Linux 파일 잠금과 원자적 기록을 사용한다. 요청/실행/독립 검토, 불변 ID와 pinned refs, 시간 구간 합집합, 예약 잔량, 미관측 세션 꼬리, 원격 저장/복구 영수증을 구분한다. 영수증의 과학적·운영적 진실까지 자동 증명하지는 않는다.
- Colab notebook은 출력/비밀정보가 없는 실행 안내다. 실제 무료 GPU 가용성·CUDA wheel 설치·Drive 내구성·새 세션 복구가 통과했다고 볼 수 없다. 서버나 자동 대회 제출 경로는 추가하지 않는다.

## 최종 검사 기록

검토 중 최초 suite는 31 passed/3.11s였다. 실제 ResNet 통합/추가 회귀 테스트가 도착한 뒤 suite는 37 passed/29.89s, 집중 검사(governance/common CLI/engine)는 22 passed/4.08s였다. 이 수치는 각 실행 시점의 테스트 집합이며 최종 파일 해시와 다른 중간 버전까지 포괄하는 승인으로 사용하지 않는다.

최종 소스 고정 뒤 전체 suite는 **37 passed/28.94s**였다. 그 suite 수집 이후 추가된 verify 종료 코드 회귀 테스트를 포함한 `tests/test_common_cli.py`는 별도 **4 passed/0.14s**였다. 이 두 실행을 합쳐 검토자 자신의 새 전체 suite 38개 통과로 계산하지 않는다. 구현 진행자는 별도 최종 전체 실행 **38 passed/29.64s**와 notebook의 코드 셀 AST 검사·출력 비어 있음 확인을 전달했다. 이는 진행자의 실행 보고이며 위 검토자 직접 실행과 구분한다. `python scripts/check_text.py`, `.venv/bin/ruff check .`도 통과했다. 별도 직접 검사는 실제 protocol hash 보고서 전달, checkpoint code identity, 변경된 inference digest 거절을 확인했다.

최초 지적은 **critical 2 / medium 5 / minor 1**, 최종 검토 대상에서 미해결 지적은 **critical 0 / medium 0 / minor 0**이다. 이 판정은 가벼운 v0.1의 로컬 기능·기록 계약에 한정한다. 실제 대회 성능, GPU/AMP, 무료 Colab 운용, W&B cloud, 다른 컴퓨터 재현의 준비 완료 판정은 아니다. 추가 설계 확장이나 서버 도입을 요구하지 않는다.

## 최종 검토 대상 SHA-256

검토 중 변경이 병행되어 초기 발견 코드와 아래 최종 수정 코드의 해시는 다르다. 아래 목록은 마지막 읽기 및 집중 검사 대상이다. 리뷰 문서 자체와 다른 새 문서의 변경은 소스 검토로 승계하지 않는다.

```text

546df0323ad546035ae85e26561484c4e72f17813dff3e2e951ddfc9c48e3377  AGENTS.md
9ffa15c96ad949e3998eb7ca8ad22cbe0612cb8caf588ec503dadc525e74266b  README.md
b1597a2f11bc480e962942c72c46c1114ea9ffa45fb02ecb650f64d1e28c4669  docs/PLAN.md
1e3e5c3ed65f476a1a2c0a4b712db0f4ecb1138bc4300a5cb353054ec5b1788c  docs/ARCHITECTURE.md
8c80b269110e161ac87465b111df5243f3aa69bb413d3ee4bb6f5a1d3fd9ffef  docs/MANAGEMENT-HARNESS.md
e87c6eb095be005291f4a15f5710640a006035f99e2bda97b815021a1ab1737d  docs/REPORT-CONTEXT-HARNESS.md
ac2d3fbcac485fafa1f966ab600cc70d20c8623a7546daeabb87a9c40ce32bd8  docs/reviews/2026-11-17-fh-01-design-fix.md
c2bda790d257c083524f92569fd905a96722c2d6c30640aac3a3116f4aea9106  docs/RUNBOOK.md
4a99990cece111de4f00dc6521b57204d7d32a8a33654ca6f4cef4f06b6150bc  docs/COLAB.md
8eae561a3ab11a3b93363eb5b8966f82ca8dded898293763c7902ef10a35f2f0  pyproject.toml
86ac31dc48e34fdaf5fbfb97c1c875b439c66d4a9e7ec2da138dcf4cf776bd00  requirements-verified-cpu.txt
bc75bf1237d102f0560333995f5cd34ed67a06c0e046b7e1a348a8d7b74a4e67  notebooks/colab.ipynb
30e799a76a90f21ec6a3c8ed9a1c21968091babf38756f081117e6d14e9321e9  src/defectlab/__init__.py
935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d  src/defectlab/__main__.py
a6db33dce52edb176b6d10e96d8f1f7dde3c0171997cf0272ae3aeaf135c1cff  src/defectlab/cli.py
625bd437d418d61cb4c0302d2b9eea921a8279d193345617e1713bcc2560f588  src/defectlab/common.py
a62f1f18b0f73e877a4ae7d649fd76bf068988939066b810c6f22d41a1f8bb11  src/defectlab/data.py
e434ef14f0610c7b6d6b1e9f41e20a51be59b3e8c201aa70ad2574db0286ae2d  src/defectlab/engine.py
69413402067087652b623c5455c12f1f2586b02fb7ff63f9188611590f65cf3c  src/defectlab/governance.py
60e8d7eb25ffd3f8c93eeca709a2f4f01dd203df1e453af0c14c4c297cc24975  src/defectlab/selection.py
ff459237c4884207890f85b4173ac555f0eb6f1eaa561eb72f5e8f2c8edf6aeb  src/defectlab/tracking.py
f6d560b0c19d570a3c31122c398d65511b714b8294481fef10d220a2ae829f9a  tests/test_common_cli.py
cde90746386703eb54cadd7cc9b5bf86116780ec42c378a65e73b81d14fc671f  tests/test_data.py
a2d749a49f2d298c84373066431410ed4a610b0a5f876d49f118212ccf08c2b9  tests/test_engine.py
9dc4c59c66df4a7b28697c9a3cf0301d7449524903cc46ee5c702bd00989325c  tests/test_governance.py
2c3e07fd83ef3eabf8cd7d7ed2468d8667a1b242744fbacc06e1671dd2bbc22a  tests/test_integration.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  configs/.gitkeep
2b578d27c4c26c9d65a004d5e58171b01499616b01bd47eab281293a710c9682  configs/baseline.toml
48286916001ed3d34ea3be166c414ffc66f53a3f9ff9295e47fbc1a3e6fe616e  configs/section-example.json
2d904a998285ddfdb7878686d190f648608458ffdcf751609d6b981b0cc2e82a  configs/task-example.json
8b7067089d5509aa1155b9e4b7d1d7cf4541add799c4fd51d34d919e56b2e627  configs/tuning-example.json
f2a0f4449131058d0e974fec4df3a3d15021b40e684096d86376d66176a0d93b  .github/workflows/checks.yml
```
