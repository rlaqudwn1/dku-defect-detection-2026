# 제품 불량 판별 논문: ResNet 이후의 문제와 접근

조사일: 2026-11-17. 목적은 산업 시각 검사 연구를 문제에서 방법으로 읽는 것이다. 발표 연도만 나열한 계보나 최초 발명 목록이 아니다. 아래 방법들은 서로 병렬로 발전했고, 논문을 읽었다는 사실은 구현·채택·성능 검증을 뜻하지 않는다.

## 1. 우리 문제의 출발점

현재 과제는 정상/불량 이미지 라벨을 이용해 이미지당 점수 하나를 만드는 것이다. 위치 정답 없이 이미지 라벨을 쓰는 지도 분류가 직접적인 기준선이다. 결함 위치를 학습하려는 경우에는 같은 라벨이 약한 감독 신호가 된다. 이미지가 불량이라고 모든 작은 영역도 불량인 것은 아니다.

연구의 갈림길은 두 가지 질문이다.

- 학습에서 본 불량과 비슷한 불량을 잘 찾으면 되는가, 보지 못한 결함 종류까지 찾아야 하는가?
- 사진 전체의 의미를 볼 것인가, 작은 표면 패턴을 비교할 것인가?

우리 테스트가 어떤 결함 일반화를 요구하는지는 아직 확인하지 않았다. 정상만 학습하는 논문의 설정을 대회 조건으로 가정하지 않는다.

## 2. 정상 모습을 복원하면 결함이 드러날까?

Autoencoder는 입력을 압축한 뒤 다시 만드는 모델이다. 정상 영상으로 학습하고 입력과 복원 결과의 차이를 이상 단서로 삼는다. 다만 작은 위치 차이도 큰 오차를 만들 수 있고, 픽셀값 차이가 작아도 구조적 결함일 수 있다. 결함까지 복원되면 탐지가 어려워지는 것도 이 접근을 평가할 때 확인할 실패 조건이다.

Bergmann 등의 논문은 단일 픽셀 오차의 문제를 짚고 밝기·대비·구조를 고려하는 SSIM 기반 비교를 제안했다. 이 논문을 복원 기반 결함 연구 전체의 기원으로 단정하지 않는다.

- Paul Bergmann, Sindy Löwe, Michael Fauser, David Sattlegger, Carsten Steger. **Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders**. VISAPP 2019, pp. 372–380; preprint 2018.
- [원문 v3](https://arxiv.org/abs/1807.02011v3), [출판 원문](https://www.scitepress.org/Papers/2019/73645/73645.pdf), DOI: 10.5220/0007364503720380.
- 확인 범위: 저자 초록 및 출판 메타데이터. 구조적 오차의 문제 설명은 원문 근거, 우리 데이터에서의 실패 여부는 미검증이다.

## 3. 정상의 생김새를 특징 공간에서 정의하면 어떨까?

CNN의 마지막 분류 결과 대신 중간 특징을 이용할 수 있다. 여기서 patch 특징은 사진을 실제 파일 조각으로 잘라야만 생기는 것이 아니라, 특징 맵의 각 위치가 보는 국소 영역을 나타낼 수 있다.

**PaDiM**은 정상 영상들의 각 위치에서 CNN 특징의 다변량 Gaussian 분포를 추정한다. 새 특징이 그 위치의 정상 분포에서 얼마나 떨어졌는지를 Mahalanobis 거리로 계산한다. 원문은 이상 맵의 최댓값을 이미지 점수로 사용한다. 제품 위치·배치 변화가 실제로 어떤 영향을 주는지는 별도 평가해야 하며, 위치별 모델이라는 이유만으로 이동에 완전히 실패한다고 단정하지 않는다.

- Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. **PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization**. arXiv:2011.08785v1, 2020.
- [원문](https://arxiv.org/html/2011.08785v1), 특히 III-A–III-C.
- 확인 범위: 본문 방법 절. 본 문서는 확인한 preprint 버전으로 인용하며 워크숍 연도 표기를 임의로 확정하지 않는다.

## 4. 분포 하나로 요약하지 말고 정상 부분들을 기억하면 어떨까?

**PatchCore**는 사전학습 CNN으로 정상 patch 특징을 만들고 대표적인 특징들을 memory bank에 보관한다. 새 이미지의 각 특징을 가까운 정상 특징과 비교한다. 대표 집합인 coreset은 저장량과 검색 부담을 줄이는 장치다. 이는 PaDiM의 위치별 Gaussian 모델과 다른 설계다.

직관은 정상 표면의 사전을 만들고 사전에 잘 맞지 않는 부분을 찾는 것이다. 원문의 이미지 점수는 단순 평균이 아니라 가장 이질적인 patch와 그 이웃을 이용한 재가중을 포함한다. 정상 기준이 충분히 다양한지, 작은 결함이 전처리·특징 해상도에서 살아남는지, 검색 메모리와 시간이 감당 가능한지가 우리 조사 질문이다.

- Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. **Towards Total Recall in Industrial Anomaly Detection**. CVPR 2022, pp. 14318–14328; preprint 2021.
- [원문 v2](https://arxiv.org/html/2106.08265v2), 특히 §3.1–3.3 및 Appendix C.3; [CVF 출판 기록](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).
- 확인 범위: 본문 방법·해상도 관련 절. 논문 내 비교 성능은 저자 평가이며 우리 대회 우수성의 증거로 사용하지 않는다.

## 5. 실제 불량이 부족하다면 학습 문제를 만들 수 있을까?

**CutPaste**는 정상 이미지 일부를 잘라 다른 위치에 붙이는 변형으로 학습 신호를 만든다. 먼저 원본과 변형을 구분하며 표현을 학습하고, 다음에 정상 표현으로 one-class 검출기를 구성한다. 가짜 결함 이진 분류 출력을 그대로 최종 불량 점수로 쓰는 방법으로 축약하지 않는다.

우리에게도 불량 라벨은 있지만 결함 다양성이 충분한지는 별도 문제다. 합성 변형이 실제 결함의 단서를 가르칠지, 접합 경계 같은 인공 흔적만 가르칠지가 비교 질문이다. 외부 이미지·텍스처를 추가 학습에 쓰는 레시피는 대회 규정과 맞지 않으므로 복사하지 않는다.

- Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister. **CutPaste: Self-Supervised Learning for Anomaly Detection and Localization**. CVPR 2021, pp. 9664–9674.
- [저자 원문](https://arxiv.org/abs/2104.04015), [CVF 출판 기록](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html).
- 확인 범위: 원문 초록의 2단계 학습 설정과 출판 기록. 세부 재현 설정은 채택 전에 추가 확인한다.

## 6. 실제 불량 라벨을 쓰면서 새로운 결함도 찾을 수 있을까?

**DRA**는 정상뿐 아니라 일부 알려진 불량을 학습에 사용하면서, 보지 못한 이상 종류에도 대응하려는 open-set supervised 설정을 다룬다. 불량 이미지 전체를 불량 영역으로 간주하지 않고, 이미지 라벨과 top-K patch 집계를 이용한다. 정상/알려진 이상/합성 이상 등에 관련된 표현을 나누어 학습한다.

- Choubo Ding, Guansong Pang, Chunhua Shen. **Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection**. CVPR 2022.
- [원문](https://arxiv.org/html/2203.14506v1), 특히 §3.2와 §4.2; [CVF 출판 원문](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.pdf).
- 확인 범위: 저자 본문과 출판 기록. 논문의 one/ten labeled anomaly 실험이 우리 불량 1,200장 조건과 같지는 않다. 외부 outlier exposure를 포함하는 설정은 우리 규정에서 제외해야 한다. 보지 못한 결함 종류를 평가하려면 그 종류 구분을 먼저 확보해야 하며 현재 미확인이다.

이는 우리에게 특히 중요한 비교 계열이다. 불량 라벨을 버리는 normal-only 방법만 연구하면 주어진 감독 정보를 활용하는 대안을 놓치기 때문이다. DRA가 현재 지도 기준선보다 좋다는 결론은 아니다.

## 7. 더 강한 특징이면 정상 비교도 좋아질까?

**AnomalyDINO**는 DINOv2 patch 표현과 정상 기준 특징의 비교를 결합한다. PatchCore와 완전히 같은 알고리즘은 아니며, few-shot 조건과 전처리·점수 집계도 구별해야 한다. 저자들은 의미적 규칙 위반과 유익한 정상 기준 부족을 실패 조건으로 논의한다.

- Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer. **AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2**. arXiv:2405.14529v2, 2024.
- [확인한 원문 버전](https://arxiv.org/html/2405.14529v2), 특히 §3, Appendix B.
- 확인 범위: 본문 방법 및 실패 조건. 여기서는 확인한 preprint를 인용한다. 가중치 사용 허용·정확 버전·라이선스·해시 확인은 별개이며 다운로드하지 않았다.

## 8. 논문을 우리 실험에 연결하는 기준

| 연구 질문 | 참고 계열 | 우리에게 필요한 관측 |
|---|---|---|
| 주어진 라벨로 구분 가능한가? | 지도 이미지 분류 | 독립적인 분할과 이미지 AUROC |
| 정상과 다른 작은 부분이 있는가? | PaDiM, PatchCore | 위치/배치 변화, 정상 다양성, 결함 크기 |
| 보지 못한 결함에 대응해야 하는가? | open-set supervised 연구 | 결함 종류·분할의 일반화 목표 |
| 합성 학습 신호가 도움이 되는가? | CutPaste | 합성 흔적 의존 여부, 실제 검증 성능 |
| 강한 표현이 비용에 맞는가? | AnomalyDINO | 허용 가중치, 입력 해상도, 시간·메모리 |

MVTec AD는 이 계열을 읽는 공통 평가 맥락이지만 우리 데이터의 출처나 동일성을 뜻하지 않는다. 문헌만 참고하고 외부 데이터·정답은 조회하지 않는다. 논문의 pixel AUROC·AU-PRO와 우리 이미지 AUROC도 구별한다.

최근성 연결: **The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection**, IJCV 2026, [출판 원문](https://link.springer.com/article/10.1007/s11263-026-02743-0)은 극소 결함·조명 변화·겹침 등 어려운 조건을 다룬다. 과거 벤치마크 성능을 우리 성능으로 옮길 수 없다는 점을 읽는 참고다. [기존 최신성 조사](EDA-METHOD-UPDATE-2026-11-17.md)와 연결하며 이 문서는 최신 SOTA 순위를 확정하지 않는다.

## 출처의 역할과 한계

방법 정의는 저자 원문에서 확인한 사실이다. 원문 방법 설명의 신뢰도는 A(해당 방법 자체 설명)이며, preprint로만 버전을 고정한 항목의 출판 상태는 B(venue 별도 확인 필요)로 구분한다. 저자 benchmark 결과는 독립적인 비교 평가가 아니므로 우리 환경의 우수성을 뒷받침하지 않는다. 현재 실제 대회 학습 성능·모델 간 비용 비교는 없다.

보고서에서는 이 목록을 배경/관련 연구로 사용한다. Methods에는 실제 채택하고 실행한 방법·변형·설정만 연결한다. 논문의 아이디어가 설계에 영향을 주었다면 그 사실을 적고, 구현하지 않은 알고리즘을 사용했다고 쓰지 않는다.

검색 범위: industrial visual inspection, visual defect detection, normal-only anomaly detection, open-set supervised anomaly detection을 구분했다. PaDiM/PatchCore의 관련 연구 및 방법 절로 이전 아이디어를 따라갔고, 2026 평가 논문으로 최근 실패 조건을 확인했다. CVF 일부 HTML은 403으로 제한되어 접근 가능한 저자 arXiv 및 출판 메타데이터를 함께 사용했다. 모든 원문 전체를 정독했다는 의미는 아니다.
