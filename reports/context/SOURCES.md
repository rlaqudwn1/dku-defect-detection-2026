# 출처 역할과 확인 범위

기준일: 2026-11-18. 이 목록은 원고v0.1의 초기 출처다. 실제 사용, 구현만 존재, 관련 연구 후보를 구분한다. 원문을 인용했다는 사실은 가중치 사용 허용·실험 재현·우리 성능 우수성의 증거가 아니다.

| ID / BibTeX key | 서지·역할 | 확인 범위·남은 일 |
|---|---|---|
| S01 / He2016ResNet | He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition*. CVPR2016,770–778. [공식 출판 기록](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | 현재 ResNet18 구조의 출처. 공식 서지·초록 재확인. 이 논문 성능을 우리 결과로 사용하지 않음 |
| S02 / Loshchilov2019AdamW | Loshchilov, Hutter. *Decoupled Weight Decay Regularization*. [저자 arXiv](https://arxiv.org/abs/1711.05101) | 현재 AdamW 구현의 방법 출처. 이번 확인은 저자 원문 일부. OpenReview 접근 실패로 이 초기BibTeX는 확인한arXiv 서지로 보존. 최초 등록2017과 확인판2019를 구분 |
| S03 / Fawcett2006ROC | Fawcett. *An introduction to ROC analysis*. PRL27(8),861–874,2006. [DOI](https://doi.org/10.1016/j.patrec.2005.10.010), [출판본 대학 사본](https://www.math.ucdavis.edu/~saito/data/roc/fawcett-roc.pdf) | 기존 원문 조사에서§4/4.1,§7 확인 기록. 이번 DOI/출판사 fetch는 서지 색인 응답이며 본문 추가 확인으로 세지 않음. 온라인 선공개일과 권호연도2006 구분 |
| S04 / Heckert2003EDA | Heckert, Filliben. *NIST/SEMATECH e-Handbook of Statistical Methods, Chapter1: EDA*. [공식 서지](https://www.nist.gov/publications/nistsematech-e-handbook-statistical-methods-chapter-1-exploratory-data-analysis), [개념 원문](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm) | 기존 기초 원문 조사에서 공식 서지와1.1.1 확인. 구조·가정 확인의 근거이며 우리 표본 수·판독 정확도 인증 아님 |

위 서지와 원문 확인 이력의 로컬 증거는 보고서 준비 세션의 비공개 근거에 보존한다. S03/S04의 이전 읽기 기록을 새 원문 정독으로 재기록하지 않는다.

## 이후 원고에 확장할 후보

정상 특징 비교, MIL, DRA/SuperSimpleNet, AnomalyDINO/Dinomaly, 데이터 감사·표본·판독 일치·일반화 평가 문헌은 [논문 안내](../../docs/research/INDUSTRIAL-DEFECT-PAPER-GUIDE.md), [EDA 읽기 안내](../../docs/research/BINARY-CLASSIFICATION-EDA-GUIDE.md)와 기존Pro 응답에서 탐색했다. 실제 인용 문장과 확인한판·절·출판상태를 연결한 뒤BibTeX에 추가한다. 모든 후보를 사용한Methods로 적지 않는다.

외부 방법을 추가하면 다음을 확인한다: 감독/추가데이터, 대상 일반화, 입력·집계·평가지표, 실제 비용 조건, 공식 구현, 실패·반대근거, 우리 규정과의 차이. 가중치 출처·시점·라이선스·전체SHA-256은 논문 citation과 별도 자산 기록이다.
