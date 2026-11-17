# DKU 제품 이미지 불량 탐지 대회

개인 대회용 비공개 연구 저장소. 실험 기록은 W&B, 첫 실행 환경은 무료 Colab, 원본 데이터와 복구본은 Google Drive를 사용한다.

**현재 단계는 기획 구체화다.** 데이터 인계 자료와 검증 도구, 설계 문서는 준비됐으며 학습·튜닝·앙상블·재현 검사 실행기는 아직 구현하지 않았다. 실제 대회 모델이나 성능 결과는 없다.

## 먼저 읽기

- [구체 기획서](docs/PLAN.md): 필수 기능, 실행 흐름, 완료 기준, 구현 순서.
- [아키텍처 선택 근거](docs/ARCHITECTURE.md): 가벼운 오픈소스 기반 비교와 현재 제안.
- [무료 Colab 판단 계획](docs/COLAB.md): 데이터에서 확인한 사실과 첫 실측 절차.
- [보고서 작성 계획](reports/WRITING-PLAN.md): 실험과 본문·표·그림을 연결하는 방법.
- [과거 대회 준비 조사](research/prior-competition-workflow.md): 10월 27일 조사 이력이며 현재 결정은 기획서를 따른다.
- [대회 규정](research/competition-rules.md), [수업 공지](research/course-notice.md), [라벨·출력 정의](research/data-description.md).

## 확정된 과제

- 라벨: `0=정상`, `1=불량`. 양성 클래스는 불량이다.
- 이미지에 제품이 여러 개 있어도 이미지당 `anomaly_score` 하나를 제출한다.
- 학습 이미지 10,821장(정상 9,621, 불량 1,200), 테스트 800장.
- 제출 컬럼: `image_path,anomaly_score`. UTF-8 CSV이며 예시 파일의 순서를 보존한다.
- 지표: AUROC. 대회 순위는 Private 100%, 수업 프로젝트 평가는 기존 공지 기준 성능 35% + 기술보고서 65%.

## 데이터와 현재 실행 가능한 검증

[Drive 대회 자료](https://drive.google.com/drive/folders/1cmRjMCGqH4WpMo7hUJqVc_fS2H6eBvP4)의 `open.zip`을 내려받아 프로젝트 밖이나 `data/raw/`에 푼다. Git에는 이미지와 가중치를 넣지 않는다. 원본 파일은 수정하지 않는다.

```bash
python -m zipfile -e /path/to/open.zip /path/to/data
python scripts/verify_data.py /path/to/data/open
```

검증 도구는 Python 표준 라이브러리만 사용한다. 데이터 목록·압축파일 해시는 `data/manifests/`에 있다. Drive 원본에는 사용자 계정으로 접근하며 공개 공유를 요구하지 않는다.

## 폴더 역할

```text
configs/                 모델·전처리·학습 설정 (실행기 구현 시 추가)
data/manifests/          데이터 파일 목록·해시·이미지 크기 조사
data/splits/            고정 검증 분할과 분할 해시
src/defectlab/           이후 구현할 공통 학습·평가·추론 코드
scripts/                검증 도구와 이후 보고서 집계 명령
notebooks/              Colab 진입점·탐색 노트북 (이후 구현)
research/               공식 규정·논문·코드의 근거 카드
experiments/            실험 가설·설정 연결과 검토한 결과 기록
models/                 로컬 가중치 캐시 (바이너리는 Git 제외)
submissions/            제출 CSV·제출 선택 기록 (CSV는 Git 제외)
reports/                기술보고서 원고·비교표·그림
docs/                  기획·설계·운영 절차
tests/                 이후 실행기의 누수·재현·정렬 검사
```

CCDS의 폴더 역할을 참고한 구성이다. 템플릿 전체나 별도 실험 서버를 도입하지 않았다. W&B 계정 연결과 Artifact 저장소 생성도 아직 수행하지 않았다.
