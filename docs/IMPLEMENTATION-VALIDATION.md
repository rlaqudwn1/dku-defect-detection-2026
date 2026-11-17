# v0.1 검증 범위

2026-11-17 로컬 CPU 환경에서 구현 경로를 검사했다. 실제 대회 모델 성능·Colab GPU 성능·W&B cloud 성공을 측정한 기록이 아니다.

- 제공 데이터의 manifest 검증을 통과했다(11,623 파일, 8.09초). 원본을 수정하지 않았다.
- 이미지 11,621장 전체 픽셀 디코드를 통과했다(111.65초). 손상 0, RGB 11,621, EXIF 방향 태그 0, SHA-256 바이트 중복 그룹 0이다.
- 조사 카탈로그와 36장 contact sheet는 비공개 runtime 산출물로 유지한다. 제한된 표본에서 train/test 제품·배경 차이를 관측했다. 촬영 관계·픽셀/근접 중복·완전한 제품 분류는 미확인이다.
- seed 42의 20% holdout 후보는 train 정상 7,697/불량 960, validation 정상 1,924/불량 240이다. 이 분할의 대회 대표성은 아직 검증되지 않았다.
- 실제 torchvision ResNet18을 작은 합성 이미지로 optimizer 학습하고 CLI를 통해 저장→별도 평가→동일 가중치 재평가→0 허용오차 재현→제출 형식/순서→앙상블→선택 manifest→결과표→epoch 경계 재개를 수행했다. 첫 통합 검사는 26.50초에 통과했다. 합성 AUROC를 대회 성능으로 보고하지 않는다.
- 별도 작은 결정론적 모델로 중단 없는 실행과 epoch 경계 재개의 model tensor를 동일하게 확인했다. batch 순서·ID·라벨 방향·전처리·누수·원본 바이트 변경·checkpoint 교체·다른 run 복구 혼입을 검사한다.
- 실제 W&B offline SDK로 metrics·Artifact 로컬 기록·finish를 시험했다. online SDK 계약은 mock으로만 검사했다. 실제 계정·Artifact 원격 버전·cloud 접근 범위/용량은 미검증이다.
- 관리/보고서 경로는 torch 없이 실행된다. 실제 응답 없는 검토 승격, 자기 검토, stale 근거와 FH-01 변경 전파, 시간 합집합, 예약 소비, unknown 세션 꼬리, 영속 저장 영수증, 주장 근거/절 문맥/freeze를 검사한다.

최종 검사는 `pytest`, `ruff check .`, `python scripts/check_text.py`로 다시 실행한다. 구현 검토의 발견·수정·최종 대상 해시는 [Astra 독립 구현 리뷰](reviews/2026-11-17-astra-medium-implementation-review.md)에 남긴다. 로컬 고정 가중치 재현과 다른 GPU/새 환경 재현을 구분한다. 제출 플랫폼 업로드·GPU 구매는 수행하지 않았다.
