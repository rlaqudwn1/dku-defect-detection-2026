# DKU 제품 이미지 불량 탐지 연구 도구

이미지당 불량 확률 하나를 만드는 가벼운 Python 연구 코드다. 라벨은 `0=정상`, `1=불량`이며 전체 검증 예측으로 AUROC를 계산한다. 공개 저장소에는 데이터·가중치·예측·인증정보를 넣지 않는다.

v0.1은 데이터 검사/픽셀 카탈로그/중복 그룹 분할, ResNet18 학습·epoch 경계 재개·고정 가중치 평가·CSV 추론, 작은 순차 튜닝·확률 앙상블·재현 검사, 로컬/W&B 기록, 관리 장부·보고서 근거 묶음을 제공한다. 실제 대회 GPU 학습·성능 측정·제출은 아직 수행하지 않았다. 무료 Colab GPU와 W&B cloud 연결도 미검증이다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,vision,tracking,dev]'
defectlab --help
```

관리·보고서만 필요하면 `python -m pip install -e .`로 설치한다. PyTorch는 실행 환경에 맞는 CPU/CUDA 빌드를 먼저 설치할 수 있다. [정확한 실행·복구 절차](docs/RUNBOOK.md), [Colab notebook](notebooks/colab.ipynb), [환경 기록](requirements-verified-cpu.txt), [실제 검증 범위](docs/IMPLEMENTATION-VALIDATION.md)를 참고한다.

```bash
defectlab verify data/raw/open --manifest /path/to/private-manifest.json
defectlab profile data/raw/open runs/eda
defectlab split runs/eda/profile.json runs/split.json
defectlab train data/raw/open runs/split.json configs/baseline.toml runs/baseline
```

`configs/baseline.toml`은 CUDA 출발 설정이다. CPU에서는 `device="cpu"`, `amp=false`로 별도 설정을 만든다. 256 입력의 적절성·근접 중복·촬영 종속성은 연구로 확인해야 한다. exact duplicate 분리가 독립성을 보증하지 않는다.

원본은 `train.csv`, `train/`, `sample_submission.csv`, `test/`를 포함하는 로컬 디렉터리로 전달한다. 원본을 수정하지 않는다. 검증 manifest는 데이터 소유자가 비공개로 전달하며 기본 공개 manifest로 강제하지 않는다.

규칙: 제공 학습 데이터만 학습하고 테스트 정답·외부 원본/공개 라벨을 조회하지 않는다. 원격 모델 추론 API를 사용하지 않는다. 사전학습은 2026-11-15까지 공개된 가중치의 출처·버전·라이선스·해시를 확인한 로컬 파일만 허용한다. 기본은 무작위 초기화다. 대회 제출과 GPU 결제는 자동 수행하지 않는다.

설계와 연구 책임은 [계획](docs/PLAN.md), [아키텍처](docs/ARCHITECTURE.md), [관리 하네스](docs/MANAGEMENT-HARNESS.md), [보고서 문맥](docs/REPORT-CONTEXT-HARNESS.md)에 보존한다. 이 문서들의 과거 설계 상태 설명은 당시 이력이며 현재 실행 계약은 RUNBOOK과 소스·테스트를 따른다. 미래 연구 하네스의 역할별 LLM 모델·effort는 아직 미정이다.
