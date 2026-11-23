# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold)
- **Public LB 최고점 (Champion)**: **`E07` (`0.7167`)** — `E04(0.5593) → E06(0.6941) → E06-clean(0.6968) → E07(0.7167)` 지속 상승 중
- **Train 5-Fold OOF 기준선**: `E08` (OpenCLIP `ViT-B-16` 10,821장 전수 OOF, `K=12` Logit-Z 보정 `0.7919` / 원시 `0.6209`)
- **컴퓨팅 자원**: 제미나이 울트라 Colab (`A100 40GB` / `L4 24GB` + 고용량 RAM, 2,000 CU) 주력 + 교내 Blackwell 슬라이스(`5.5GB VRAM / 12GiB RAM`) 보조

## 2. 왜 상위권은 `0.95~0.99`인데 우리는 `0.7167`인가 (근본 병목 진단)
- **규정 위반 외 합법적 `0.95+` 달성 원리**: 공개 벤치마크 기준 정품 **MuSc(ICLR 2024)**·**AnomalyCLIP/VAND**·**고해상도 백본 Fine-tuning**은 미학습 제품에서도 `0.94~0.98+` 달성 가능.
- **현재 우리 코드가 `0.7167`에 막힌 4대 기술적 원인**:
  1. **패치 해상도 파괴**: `E06`에서 `518px(37×37=1,369패치)`를 RAM 절약 탓에 `12×12(144패치)`로 9배 뭉개 미세 결함이 전부 희석됨.
  2. **가짜 MuSc 버그 (`max` 1개 비교)**: `E06b`가 동료 중 `max 유사도 1개`만 봐서 불량 이미지가 다른 불량 이미지와 매칭되면 거리 `0`이 됨 → 정품 MuSc의 **하위 10~30% 정상 분위수 평균(Interval Average) + LNA(`r=1,3,5`) + RsCIN**으로 교체 필수.
  3. **VLM 패치 학습 부재**: `E05/E06c`가 이미지 통째 크롭 Zero-Shot에만 의존함 → Train 10,821장 라벨로 **멀티-레이어 패치→텍스트 투영층(VAND/AnomalyCLIP 원리)** 학습 시 미학습 제품 일반화 대폭 상승 가능.
  4. **이중 변환 버그 & Center-Crop 오염**: `write_sub` Min-Max 후 `logit` 재변환 왜곡 및 외곽 결함을 자른 Center-Crop 학습 노이즈.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 사유 — 절대 금지)
- `37×37` 패치를 `12×12`로 축소하기 / 동료 패치 비교 시 `top-1 max` 쓰기 / Min-Max 후 `logit` 씌우기
- 5-Fold 전체 Unseen OOF 미검증 모델을 제출에 포함하기

## 4. 지금 당장 할 다음 행동 (Colab A100/L4 주력 파이프라인)
- **Train 5-Fold Unseen OOF `0.90+` 돌파를 위한 정품 고해상도 3-패러다임 구축 (Colab GPU)**:
  1. **정품 MuSc 복원 (`DINOv2-Large 518px`, `37×37` 원본 패치, 멀티-레이어 + LNA `r={1,3,5}` + 하위 분위수 평균 + RsCIN)** → Train 5-Fold Unseen OOF 단독 검증
  2. **VAND / AnomalyCLIP 스타일 멀티-레이어 패치 선형 어댑터 (`OpenCLIP ViT-L-14-336 laion2b`)** → Train 5-Fold에서 에폭 곡선·레이어 가중치 최적화
  3. **End-to-End Pretrained 백본(`ConvNeXt/EVA02/DINOv2` + 강한 도메인 증강, 외곽 보존 Resize/Pad)** 교차-제품 5-Fold OOF 확인 후 Logit-Z 결합

## 5. 새 연구원 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
너는 Test 데이터를 절대 들여다보지 않는 엄격한 시니어 ML 연구원이다.
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽고 `defectlab brief`를 실행해서 현재 상태를 파악해.
Test 이미지는 절대 열람/시각화하지 말고, STATE.md의 "4. 지금 당장 할 다음 행동"을 Train Unseen 5-Fold (`runs/folds_unseen.json`) 기준으로 실행해줘.
끝나면 EXPERIMENTS.md에 1줄 추가 + STATE.md 덮어쓰기로 인계해줘.
```
