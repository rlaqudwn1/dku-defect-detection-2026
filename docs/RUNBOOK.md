# v0.1 실행과 복구

Python 3.12 이상을 사용한다. 아래 경로는 저장소 루트 기준이다. 데이터와 실행 결과는 공개하지 않는다. 원본 디렉터리는 읽기 전용 입력이다. 설치·실행은 원격 모델 API를 사용하지 않는다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,vision,tracking,dev]'
python -m defectlab --help
python scripts/check_text.py
ruff check .
pytest
```

CPU 재현 환경은 `requirements-verified-cpu.txt`다. CPU 전용 wheel은 PyTorch CPU index에서 torch/torchvision을 먼저 설치한다. Colab notebook은 기존 GPU torch/torchvision을 먼저 import해 확인하고 재설치하지 않는다. `pip install -e . --no-deps`로 실행기를 설치한 뒤 numpy/Pillow/scikit-learn 버전을 확인하고 실제 `pip freeze`를 보존한다. 필요한 라이브러리가 없다면 해당 패키지만 별도로 설치한다. GPU 종류·실행시간을 보장하지 않는다. 관리와 보고서 CLI는 기본 설치 `pip install -e .`만으로 실행하며 torch를 불러오지 않는다.

## 데이터·분할

`data/raw/open` 아래에 `train.csv`, `train/`, `sample_submission.csv`, `test/`를 둔다. ZIP은 직접 공급한다. 원본 데이터나 비공개 Drive 링크를 저장소에서 다운로드하지 않는다.

```bash
defectlab verify data/raw/open --manifest /path/to/private-manifest.json
defectlab profile data/raw/open runs/eda
defectlab audit-duplicates data/raw/open runs/duplicates --split runs/split.json
defectlab sample-eda runs/eda/profile.json data/raw/open runs/samples --config configs/eda-first-look.json
defectlab compare-transforms data/raw/open runs/transforms --ids train_00001.jpg --sizes 256 384 512
defectlab split runs/eda/profile.json runs/split.json --seed 42 --val-fraction 0.2
```

manifest가 없으면 `verify data/raw/open`으로 CSV/경로/라벨/파일 해시 기본 검사를 한다. 공급자가 전달한 manifest를 사용해야 다운로드 원본과의 바이트 일치까지 검증할 수 있다. profile은 픽셀 디코드 통계와 표본 contact sheet를 만든다. `audit-duplicates`는 파일·픽셀·dHash 후보 쌍을 점검하며 기존 `duplicate_audit.json` 덮어쓰기를 거부한다. `sample-eda`는 calibration·survey·discovery 블라인드 시트를 생성하고 라벨 매핑은 `blind_key.json`에만 둔다(`configs/eda-first-look.json`의 `calibration_review_per_stratum`, `survey_review_per_stratum`은 후속 재판독 메타데이터다). `compare-transforms`는 엔진과 같은 resize+pad 결과를 생성하며 기존 `transform_comparison.json` 덮어쓰기를 거부한다. test 라벨은 null이다. 근접 중복·촬영 관계·제품 의미는 자동 확정하지 않는다. 조사로 확정한 그룹은 `--groups groups.json`의 `{ "groups": {"train_00001.jpg": "capture-1"} }` 형태로 전달할 수 있다.

## 학습·고정 가중치 평가·추론

```bash
defectlab train data/raw/open runs/split.json configs/baseline.toml runs/baseline
defectlab evaluate data/raw/open runs/split.json runs/baseline/best.pt runs/baseline/validation.json
defectlab predict data/raw/open data/raw/open/sample_submission.csv runs/baseline/best.pt runs/baseline/submission.csv
defectlab train data/raw/open runs/split.json configs/baseline.toml runs/baseline --resume runs/baseline/last.pt
```

baseline은 whole-image `resnet18`·1 logit·BCE·단일 sigmoid다. `model`은 `resnet18`, `convnext_tiny`, `swin_t`, `vit_b_16`(단 `image_size`는 16의 배수)을 지원하며 `freeze_backbone`(사전학습 provenance 필수)과 `pos_weight`(양의 유한 실수 또는 null)를 선택할 수 있다. 비율을 보존한 resize/pad와 고정 정규화를 학습 평가/별도 평가/추론에 공유한다. 입력 크기 256은 시작 설정이며 작은 결함 보존 여부는 조사·비교해야 한다. AMP를 요청한 CPU 실행은 거절한다. CPU 설정은 `device="cpu"`, `amp=false`다. CUDA 결정성 실행 시 `_seed()`가 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 기본값을 지정하며 `environment`에 기록한다. AMP에서는 정상적인 `GradScaler` scale backoff를 허용하고 연속 10회 overflow 시에만 실패 처리한다. 임의 pretrained 자동 다운로드가 없으며 추론은 checkpoint만 읽는다. 자체 생성·신뢰하는 checkpoint만 로드한다.

`last.pt`는 optimizer·난수·epoch 상태를 포함한다. epoch 경계에서만 재개한다. 더 긴 재개는 config의 epochs를 늘린다. 다른 split/config를 같은 run에 덮어쓰지 않는다. `best.pt`는 가장 높은 전체 검증 AUROC의 가중치다. 초기 v0.1(다중 모델/동결/pos_weight 확장 이전)에서 만든 checkpoint는 `_model` 해시·`_spec` 키(`freeze_backbone`)·기본 설정 키(`pos_weight`)가 달라 현 버전에서 `resume`/`evaluate`가 거부된다. 로컬/다른 환경의 수치 일치는 별도 검사하며 CPU smoke 결과를 대회 성능으로 해석하지 않는다.

## 작은 튜닝·앙상블·재현

튜닝 JSON은 `max_candidates`, `time_budget_seconds`, `candidates`를 필수로 둔다. 후보는 `{ "config": { ...전체 설정... } }`다. 예산은 후보 시작 전에 검사하며 이미 시작한 epoch를 강제 종료하는 daemon은 없다. 한 후보의 긴 실행은 설정의 epoch 예산으로 제한한다. 실패는 결과에 남는다.

```bash
defectlab tune data/raw/open runs/split.json configs/tuning-example.json runs/tuning
defectlab ensemble runs/mean-validation.json runs/a/validation.json runs/b/validation.json --weights 0.5 0.5 --recipe runs/mean-recipe.json
defectlab reproduce runs/baseline/validation.json runs/reloaded-validation.json --probability-atol 0.000001 --auroc-atol 0.000001
```

검증 recipe를 고정한 뒤 테스트 구성원 sidecar JSON을 `--frozen-recipe runs/mean-recipe.json`으로 전달해 같은 구성/가중치로 결합한다. 각 구성원의 정확한 checkpoint·학습 출처·split·ID·라벨·프로토콜이 맞아야 한다. 같은 검증셋을 반복 선택한 편향은 추가 seed만으로 제거되지 않는다. 재현 허용오차는 최종 선택 전에 정하고 실패 뒤 완화하지 않는다.

## W&B와 영속 저장

`[tracking] mode="disabled"`가 기본이다. offline은 로컬 W&B 파일과 JSONL을 만들며 cloud 업로드·원격 영속 저장 성공이 아니다. online은 사용자가 entity/project와 인증을 제공하고 접근 범위·용량을 확인한 뒤 사용한다. 실제 cloud 경로는 이번 로컬 검증에서 시험하지 않았다. 원본 이미지·코드를 자동 업로드하지 않는다. offline 기록의 Artifact cloud 버전은 null이다. online 업로드의 정확한 버전은 SDK 완료 응답을 받은 뒤에만 events.jsonl 영수증으로 남긴다. 모델 Artifact 업로드는 명시 경로만 사용한다.

Colab 종료 전에 필요한 best/last·설정·분할·결과를 개인 영속 저장 위치에 복사하고 SHA-256을 비교한다. 새 세션에서 다시 내려받아 hash·evaluate·resume를 각각 확인한다. 업로드 요청만으로 복구 가능하다고 기록하지 않는다.

## 관리·보고서

```bash
defectlab record runs/ledger.json configs/task-example.json
defectlab status runs/ledger.json
defectlab results runs/report runs/baseline/run.json
defectlab context runs/ledger.json configs/section-example.json runs/context.json
defectlab freshness runs/context.json
defectlab freeze runs/context.json runs/report-snapshot.json
```

`task-example`은 준비 상태 예시이며 실제 학습 완료를 주장하지 않는다. `section-example`은 계획 주장에 대한 명시 근거가 없으므로 freeze가 거절된다. 실제 결과 파일과 SHA-256을 근거로 채운 뒤 사용한다. 요청·실제 응답·독립 검토·처분은 각각 다른 기록이다. 명령이 전문가를 자동 호출하지 않는다. 미래 연구 하네스의 모델·effort는 미정이다.

기본 추론은 학습·테스트 원본 전체의 hash를 재검증하므로 재현 묶음에 원래의 train.csv/train/ 및 sample_submission.csv/test/를 제공해야 한다. test-only 배포 입력은 v0.1에서 지원하지 않는다.

장부는 단일 진행자와 OS 파일 잠금으로 갱신한다(Linux/Colab). 이전 기록은 새 ID+supersedes로 보존한다. 명시 의존 근거 변경 시 후속 단계의 현재 유효성을 `needs_reassessment`로 분리한다. Colab 잔여 quota·미관측 꼬리 구간은 unknown이다. 보고서 context는 근거 해시를 고정하고 변경/누락을 검사한다. final brief의 freeze는 독립 검토 연결까지 요구한다.

최종 선택은 검증과 테스트 provenance 및 통과한 재현 기록을 연결한다.

```bash
defectlab select runs/baseline/validation.json runs/baseline/submission.csv.json runs/reproduction.json runs/selection.json --reason "선정 근거와 한계"
```

`reproduction.json`에는 `reproduce`의 JSON 출력을 저장한다. 선택 상태는 로컬 선택이며 대회 업로드·새 GPU 환경 재현 완료와 구분한다.

## Pro 보조 EDA 재개와 오프라인 패킷

구체 단계·기록 계약은 [실행·재개 계획](PRO-ASSISTED-EDA.md)을 따른다.

```bash
defectlab workflow configs/pro-assisted-eda.json runs/pro-assisted-eda/ledger.json
python scripts/prepare_pro_packet.py configs/pro-pilot-packet.json runs/pro-assisted-eda/packet-text-001
```

workflow는 읽기 전용이며 없는 연구 이력을 만들지 않는다. 패킷은 새 runs 디렉터리에만 준비하고 전송·Pro 호출·자동 모델 검증을 하지 않는다. 초기 실행은 기존 Python 환경에서 한다.
