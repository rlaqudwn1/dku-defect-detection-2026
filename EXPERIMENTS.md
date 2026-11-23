# 실험 로그 (EXPERIMENTS)

> **규칙**: 장부 JSON 쓰지 말고 매 실험 끝날 때마다 여기에 **딱 1줄**만 적는다.
> **기준**: 모든 실험은 동일한 `runs/folds_unseen.json`(Train 미학습 제품군 5-Fold)의 **OOF AUROC**로만 비교한다. **Test 시각 열람 금지.**

| Exp | 변경점 (직전 Champion 대비 1개) | 모델 / 해상도 | Unseen Fold-4 AUROC | 5-Fold OOF AUROC | Public LB | 판정 | Train 오답 관찰 & 다음 가설 |
|---|---|---|---|---|---|---|---|
| E00 | 기존 베이스라인 (scratch, 증강 0, 3ep, holdout) | ResNet18 / 256 | 0.7034 (holdout) | - | - | **폐기** | 랜덤초기화 + 증강 0 + `num_workers=0` I/O 병목 |
| E01 | 교차-제품 지도 선형 분류기 (다른 제품 학습 → Fold-4 평가) | ResNet18 frozen / 224 | 0.7408 | - | - | **Discard** | 미학습 제품 일반화 한계 |
| E02 | 라벨 0개 비지도 kNN 코사인 거리 (`k=15`, 전역 풀링) | ResNet18 frozen / 224 | 0.7826 | - | - | **Keep** | Train `defectlab errors`: FP 20/20=껌, FN 20/20=캔들 → 제품군 스케일 편향 발견 |
| E03 | E02 + 제품군 클러스터별 Robust Z-Score (`(s-med)/IQR`) | ResNet18 frozen / 224 | 0.9004 | - | - | **Keep** | `+0.1178` 상승; 남은 Train FN은 4구 캔들의 미세 찍힘 |
| E04 | E03 + 멀티-스케일 패치 kNN (`layer2+3`, 7×7) + 비지도 K-Means (W&B `ayldkflw`) | ResNet18 frozen / 224 | 0.9463 | - | **0.5593** | **Discard** | CV-LB 괴리 심각(`-0.387`): 배치 kNN 단독 점수화는 미학습 테스트 일반화 실패 |
| E05 | 단일 이미지 독립 추론 WinCLIP Zero-Shot (전역+2×2, W&B `fdxwimp8`) | OpenCLIP `ViT-B-16` / 224 | **0.7879** (`global 0.7933`, 클러스터 Logit-Z `0.8864`) | 0.6209 (원시) | unknown | **Keep** | Fold-4 내부 AUROC `0.917/0.863` 대비 제품군 오프셋으로 전체 합산 시 큰 폭 손실 확인 |
| E06 | Kaggle GPU 3-패러다임: `E06a` DINOv2-B 518 Patch-MIL (`Fold4 0.9127`) + `E06b` MuSc + `E06c` WinCLIP-336 (`Fold4 0.8966`) | DINOv2-B 518 + OpenCLIP `ViT-L-14-336` | 0.9614 | `A+B 0.7893` (A Fold평균 `0.8537`) | **0.6941** | **Discard** | 원시 점수 결합 시 제품군 오프셋 미보정; `E06b` 제거 후 `E06-clean`도 `0.6968` |
| E06-clean | `E06b` 제거 (`0.45*E06a + 0.40*E06c + 0.15*E05`) | DINOv2-B + OpenCLIP `ViT-L/B` | - | - | **0.6968** | **Discard** | 단일 모델 간 제품군 기본 점수 오프셋(Category Offset)이 보정되지 않음 |
| E07 | `E05`+`E06a`+`E06c`에 제품군 전체 `Logit-Z-Score` 보정 적용 (`0.35*E05+0.35*E06a+0.30*E06c`) | OpenCLIP + DINOv2-B | 0.8864 (`E05` Fold-4 검증) | - | **0.7167** | **Champion (LB)** | `0.5593→0.6941→0.6968→0.7167` 최고점 경신! 단 `write_sub` MinMax 후 Logit 왜곡·`12×12` 패치 과도 축소·가짜 MuSc(`max` 1개 비교)가 상한선(`0.95+`)을 막고 있음 |
| E08 | Train 전용 5-Fold Feature/Layer 프로브(OpenCLIP `L3/6/9/12/Final` 10,821장 + DINOv2-B `L3/6/9/11` 2,400장) & 비지도 `K=12` Logit-Z 감사 | OpenCLIP `ViT-B-16` / DINOv2-B 224 | 0.7994 (ZS Fold-4) | **0.7919** (10,821 CLIP ZS+L12) / **0.8421** (2,400 3-way) | - | **Keep (5-Fold CV Champion)** | `K=12`(실루엣 `0.7943`)가 8해상도 내 12제품군 분리→ZS `0.6209→0.7677(+0.1468)`·오라클(`0.7027`) 상회; DINOv2 `L9`(`0.8229`)>`L11`(`0.8186`); Train FN Top-20 중 12/20=PCB 미세결함(224px 다운샘플 소실) → 고해상도 패치(DINOv2 `L9` 518px + WinCLIP-336)+비지도 Logit-Z 5-Fold 검증 필요 |
