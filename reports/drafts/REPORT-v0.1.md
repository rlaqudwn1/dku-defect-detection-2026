# 미학습 제품(Unseen-Product) 제조 이미지 이상 탐지를 위한 다중 패러다임 표현 학습 연구

> **원고 버전**: `v0.1` (`2026-11-20`, 진행 중 초안 — Train Unseen 5-Fold 격리 프로토콜 기준)
> **참고 문헌 파일**: [references.bib](../references.bib) · **실험 로그**: [EXPERIMENTS.md](../../EXPERIMENTS.md) · **현재 상태**: [STATE.md](../../STATE.md)

---

## 초록 (진행 중)

본 연구는 DKU 제품 이미지 불량 탐지 대회(Train 10,821장 / Test 800장, 평가지표 AUROC)에서 **학습에 포함되지 않은 미학습 제품(Unseen Product)**으로의 일반화 성능을 극대화하는 표현 학습 및 이상 탐지 파이프라인을 탐구한다. Train 10,821장의 8개 촬영/해상도 그룹을 통째로 분리한 **5-Fold Unseen-Product GroupKFold (`runs/folds_unseen.json`)** 검증 체계를 구축하고, ResNet/PatchCore/MuSc/WinCLIP/DINOv2/AnomalyCLIP 계열 문헌의 원리를 순차적으로 실험했다. 단일 이미지 독립 추론 기반 `WinCLIP ViT-B-16`(`E05`)이 Public LB `~0.77`로 현재 기준선을 형성했으며, Train Fold-4 정밀 분석에서 제품군 간 점수 오프셋(Category Baseline Offset)이 전체 AUROC를 약 `-0.10` 저하시킨다는 사실을 확인했다. 향후 모든 특징(Representation) 연구와 모델 선택은 Test를 완전히 격리한 Train Unseen 5-Fold OOF 기준으로만 수행한다.

---

## 1. 연구 목적과 검증 원칙

1. **과제 정의**: 이미지당 제품 개수와 무관하게 불량 점수 `anomaly_score` 하나(`0=정상`, `1=불량`)를 산출하고 전체 검증 예측의 **AUROC** [Fawcett2006ROC]를 최대화한다.
2. **Train 데이터 요약**: Train 10,821장(정상 9,621 / 불량 1,200, 불량률 11.1%), 디코드 손상 0건, exact 중복 0건. 8개 직사각형 해상도(`1274~1562px`)로 촬영된 고정 제품군(마카로니·껌·캔들·캐슈넛·PCB 등)으로 구성된다.
3. **Test 격리 원칙**: 모든 EDA, 오답 분석(`defectlab errors`), 특징 연구, 하이퍼파라미터 선택은 **오직 Train Unseen 5-Fold (`runs/folds_unseen.json`) 내부에서만** 수행하며, Test는 최종 추론 입력으로만 사용한다. *(참고: 초기 `E04`~`E06` 디버깅 과정에서 무라벨 Test 입력의 예측 상위 표본을 열람했으나 정답 조회나 수동 라벨링은 없었으며, 해당 진단 산출물은 전부 폐기하고 새 연구원 세션부터 완전 격리 프로토콜을 적용했다.)*

---

## 2. 참고 논문 및 방법론적 근거

본 연구의 실험 설계(`E00`~`E07`)는 2026-11-15 이전 공개된 아래 논문들의 원리를 기반으로 한다.

| 참고 논문 | 학회/연도 | 핵심 아이디어 | 본 연구에서의 적용 위치 & 역할 |
|---|---|---|---|
| **ResNet** (`He2016ResNet`) | CVPR 2016 | 잔차 연결 기반 계층적 CNN 표현 | `E00`~`E04` 기준선 및 `layer2+layer3` 멀티-스케일 패치 추출 |
| **AdamW** (`Loshchilov2019AdamW`) | ICLR 2019 | Weight decay 분리 최적화 | `E06a` Patch-MIL 헤드 최적화 |
| **CLIP / OpenCLIP** (`Radford2021CLIP`, `Cherti2023OpenCLIP`) | ICML 2021 / CVPR 2023 | 대규모 이미지-텍스트 대조 사전학습 | `E05` (`ViT-B-16`, `laion2b`), `E06c` (`ViT-L-14-336`, `openai`) 인코더 |
| **PatchCore** (`Roth2022PatchCore`) | CVPR 2022 | 중간층 국소 패치 메모리 뱅크 + 최근접 이웃 거리 | `E04` 및 `E06b` 국소 패치 이상 스코어링 |
| **WinCLIP** (`Jeong2023WinCLIP`) | CVPR 2023 | 다중 상태 텍스트 프롬프트 + 다중 스케일 윈도우 Zero-Shot 이상 탐지 | `E05` (전역+2×2 크롭, Public LB `~0.77`), `E06c` (전역+3×3 10뷰) |
| **DINOv2** (`Oquab2024DINOv2`) | TMLR 2024 | 자기지도학습 기반 고해상도 비전 패치 표현 (`ViT-B/14`, `518px`) | `E06a`/`E06b` 멀티-레이어(`block 7, 11`) 패치 백본 |
| **MuSc** (`Li2024MuSc`) | ICLR 2024 | 무라벨 이미지 간 상호 패치 스코어링 | `E04`/`E06b` 비교 실험 (배치 의존성으로 인한 일반화 한계 확인) |
| **AnomalyCLIP** (`Zhou2024AnomalyCLIP`) | ICLR 2024 | 제품 종류 독립적(Object-Agnostic) 교차-카테고리 결함 개념 학습 | `E06a` Frozen DINOv2 위 경량 Patch-MIL 헤드 학습 근거 |
| **ROC / EDA** (`Fawcett2006ROC`, `Heckert2003EDA`) | PRL 2006 / NIST 2003 | 순위 기반 AUROC 해석 및 탐색적 데이터 분석 | Stage A/B EDA, `defectlab errors`, 제품군 Logit-Z-Score Calibration |

---

## 3. 실험 결과 요약 (`E00` ~ `E07`)

| Exp | 방법론 (참고 논문 기반) | Unseen Fold-4 AUROC | Public LB | 판정 및 Train 기반 핵심 관찰 |
|---|---|---|---|---|
| **E00** | ResNet18 scratch 256px (`He2016ResNet`) | `0.7034` *(holdout)* | - | 폐기 (랜덤 초기화 + 증강 0 + I/O 병목) |
| **E01** | Frozen ResNet18 교차-제품 지도 선형 분류기 | `0.7408` | - | 폐기 (미학습 제품 일반화 한계) |
| **E02** | 라벨 0개 전역 kNN (`k=15`) | `0.7826` | - | Keep (Train 오답 분석으로 제품군별 6배 스케일 편향 규명) |
| **E03** | E02 + 제품군 클러스터별 Robust Z-Score | `0.9004` | - | Keep (`+0.1178` 상승; 전역 풀링의 미세 결함 희석 확인) |
| **E04** | 멀티-스케일 패치 kNN (`PatchCore`/`MuSc`) + 비지도 K-Means | `0.9463` | **`0.5593`** | 폐기 (배치 kNN 단독 방식의 심각한 CV-LB 괴리 확인) |
| **E05** | 단일 이미지 `WinCLIP` (`ViT-B-16`, 전역+2×2, `Jeong2023WinCLIP`) | `0.7879` *(클러스터 Logit-Z `0.8864`)* | **`~0.77`** | **Champion (LB)**: CV-LB 일치 확인. Train Fold-4 내부 AUROC(`0.917/0.863`) 대비 제품군 오프셋으로 `-0.10` 손실 규명 |
| **E06 / E06-clean** | Kaggle GPU `E06a` Patch-MIL + `E06b` MuSc + `E06c` WinCLIP-336 | `0.9614` *(A `0.9127`, C `0.8966`)* | **`0.6941` / `0.6968`** | 폐기 (원시 점수 결합 시 제품군 오프셋 미보정) |
| **E07** | `E05`+`E06a`+`E06c` 제품군 전체 `Logit-Z-Score` 보정 앙상블 | `0.8864` *(E05 Fold-4 검증)* | _미제출_ | 대기 (소규모 이웃 빼기 `k=10`의 `0.6927` 하락을 Train에서 사전 차단 후 전체 클러스터 보정 채택) |

---

## 4. 향후 연구 계획 (Train Unseen 5-Fold 격리 기준)

1. **Train Unseen 5-Fold 특징(Representation) 연구**: DINOv2 / OpenCLIP 각 레이어별 단독 AUROC 프로브 및 `defectlab errors` 기반 오답 해부.
2. **전체 5-Fold Unseen OOF 일관성 확보**: Fold 0~4 전체에서 단일 이미지 독립 추론 성능과 제품군 Calibration 효과를 동시에 검증.
3. **최종 선택 및 재현 검증**: 2026-11-30 10:00 코드 제출 전 Clean 환경 재현 및 모든 사전학습 가중치 출처·라이선스 명기 완료.
