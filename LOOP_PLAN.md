# DKU 제품 불량 탐지 대회 — 시니어 ML 엔지니어 실전 루프 계획서

> **대회 마감 (코드 제출)**: `2026-11-30 10:00` (Train 10,821장 / Test 800장 / 평가지표: **AUROC**)
> **핵심 원칙**: *"장부·해시·독립 페어리뷰 20개 쓸 시간에, 오답 40장 보고 실험 표 20줄 채우는 사람이 1등 한다."*

---

## 0. 왜 기존 하네스를 과감히 들어냈는가 (기억해야 할 5가지 교훈)

기존 저장소에는 문서 3,175줄과 관료주의 코드(`governance.py`, `workflow.py`, `packets.py`, `selection.py`, `graft/`, 리뷰 12개)가 있었지만, 정작 모델 성능에 직결되는 기본기가 전부 빠져 있었다.

| 기존 초보자 하네스의 문제 | 이번에 고친 것 |
|---|---|
| Train Data Augmentation **0개** | `augment = "basic" \| "strong"` (Flip/Rotate90/ColorJitter/Crop) 내장 |
| Pretrained 허용 대회에서 `ResNet18` 랜덤 초기화 3 epoch (**AUROC 0.7034**) | `timm:<model_name>` 지원 + [configs/exp01_convnext_s_384.toml](configs/exp01_convnext_s_384.toml) 제공 |
| `num_workers=0` 강제 + `evaluate/predict` CPU 하드코딩 + 매번 11,621장 SHA-256 재해싱 | `num_workers >= 0` 허용, GPU+AMP 평가/추론 + 4x TTA, `--fast` 검증 추가 |
| LR Scheduler 없음(`None`), 단일 80/20 Holdout, 단순 확률 평균 앙상블 | `scheduler = "cosine"` + `warmup_epochs`, **5-Fold StratifiedGroupKFold (`defectlab folds`)**, **Rank/Logit 앙상블** |
| 48장 육안 판독 규약·장부 JSON에 며칠 소비 | **30분 생존 EDA + 매 루프 `defectlab errors` (FP/FN Top-20 자동 시각화) + `EXPERIMENTS.md` 1줄 기록** |

---

## 1. 데이터 분석(EDA) 전략 — 두 번만, 목적 있게

초보자가 빠지는 가장 흔한 함정은 **"분석을 위한 분석"**이다. EDA는 딱 두 단계로 쪼갠다.

### Stage A. 베이스라인 전 — 딱 30분짜리 "생존 EDA"
모델을 돌리기 전에 아래 5개 질문에만 1줄씩 답하고 **즉시 멈춘다**.

```bash
defectlab profile data/raw/open runs/eda
defectlab audit-duplicates data/raw/open runs/duplicates
defectlab folds runs/eda/profile.json runs/folds.json --n-splits 5 --seed 42
```

1. **클래스 비율**: 정상(`0`) 9,621장 vs 불량(`1`) 1,200장 (약 89:11) → Stratified 5-Fold 필수.
2. **해상도·종횡비**: Train과 Test의 이미지 크기/비율이 같은가? → `image_size`(384/512)와 `resize_mode`(`pad` vs `stretch`) 결정.
3. **중복·유사 이미지**: `runs/duplicates/duplicate_audit.json`에서 exact/near-duplicate 그룹 확인 → `folds.json`에 `--groups`로 묶어 **CV 누수 차단**.
4. **정상 20장 vs 불량 20장 육안 대조**: 결함이 전체적인 형태 변형인가, 작은 스크래치/찍힘인가? → 미세할수록 고해상도가 최우선.
5. **Train vs Test 도메인 차이**: 배경·조명·색감 차이가 있는가? → 차이가 있으면 조명/색상 증강 강화.

> **STOP 규칙**: 위 5개 질문에 1줄씩 답했으면 Stage A 종료. 판독 규약 초안이나 통계표 20개 만들기 금지.

### Stage B. 베이스라인 이후 — 매 루프 15분 "오답 주도 EDA"
10,821장을 사람이 다 보지 말고, **모델이 가장 자신 있게 틀린 40장만** 본다.

```bash
defectlab errors data/raw/open runs/exp01/val_fold0.json runs/exp01/errors --top-k 20
```

생성된 `fp_top20.jpg`(정상인데 불량으로 오탐)와 `fn_top20.jpg`(불량인데 정상으로 놓침)를 보고 다음 루프를 기계적으로 결정한다.

| 오답 시트에서 관찰된 현상 | 다음 루프에서 바꿀 변수 (딱 1개) |
|---|---|
| FN에서 384px로 줄였더니 미세 결함이 뭉개져 안 보임 | `image_size = 512` (또는 `768`) |
| FP에서 조명 반사·그림자·배경 노이즈를 불량으로 착각 | `augment = "strong"` (ColorJitter/Crop 강화) |
| 특정 제품 각도·회전 방향에서만 오탐 발생 | 회전/Flip 증강 유지 + `--tta` 활성화 |
| 사람 눈으로 봐도 라벨이 애매하거나 노이즈처럼 보임 | `label_smoothing = 0.05` 테스트 |
| 특정 아키텍처의 한계(텍스처 vs 전역 구조) | 다른 백본 패밀리 추가 (ConvNeXt ↔ Swin/EVA-02 ↔ DINOv2) |

---

## 2. 초보자도 기계적으로 돌리는 5단계 실험 루프

모든 실험은 아래 5단계를 **예외 없이** 반복한다.

1. **One-Change Rule (가설 1줄, 변수 1개)**
   - 직전 Champion 설정에서 **딱 1개**만 바꾼다. 한 번에 3개 바꾸면 무엇이 효과가 있었는지 영원히 모른다.
2. **Fold-0 빠른 검증 (15~30분)**
   ```bash
   defectlab train data/raw/open runs/folds.json configs/exp01_convnext_s_384.toml runs/exp01/fold0 --fold 0 --fast
   defectlab evaluate data/raw/open runs/folds.json runs/exp01/fold0/best.pt runs/exp01/val_fold0.json --fold 0 --fast
   ```
   - 직전 Champion의 Fold-0 AUROC보다 확실히 오르지 않으면(`+0.002` 미만) **미련 없이 즉시 폐기**.
3. **5-Fold OOF 확정 (Fold-0 통과 시에만)**
   - Fold 1~4를 마저 돌려 전체 5-Fold OOF AUROC를 계산한다.
   - **Keep / Discard 판정 기준은 오직 동일한 `runs/folds.json`의 OOF AUROC 하나다.**
4. **오답 40장 확인 (`defectlab errors`)**
   - 새 Champion이 나오면 `fp_top20.jpg` / `fn_top20.jpg`를 갱신해 다음 병목 1개를 뽑는다.
5. **`EXPERIMENTS.md`에 1줄 기록**
   - 성공이든 실패든 [EXPERIMENTS.md](EXPERIMENTS.md) 표에 1줄 남기고 다음 루프로 넘어간다.

---

## 3. 막힐 때 그대로 따라가는 우선순위 사다리 (10일 로드맵)

| 일정 | 단계 | 목표와 구체적 액션 |
|---|---|---|
| **Day 1** | **생존 EDA + 5-Fold 고정** | `profile` → `audit-duplicates` → `folds.json` 확정 (누수 0 확인) |
| **Day 1~2** | **Exp 01: 진짜 베이스라인** | `configs/exp01_convnext_s_384.toml` (Pretrained ConvNeXt-Small 384px, Cosine LR, Basic Aug) 5-Fold 완주 + 첫 제출 CSV 형식 검증 |
| **Day 3~4** | **해상도 & 전처리 루프** | 오답 시트 보고 `384 vs 512`, `pad vs stretch` A/B 테스트 |
| **Day 5~6** | **증강 & 손실함수 루프** | `basic vs strong` 증강, `label_smoothing`, `pos_weight`, `lr`/`epochs` 튜닝 |
| **Day 7~8** | **아키텍처 다양성 + 앙상블** | 동일 5-Fold로 3개 패밀리 확보: ① ConvNeXt ② Swin/EVA-02 ③ DINOv2 → `--tta` + `--method rank` / `logit` 앙상블 |
| **Day 9~10** | **코드 제출 & 재현 검증** | 최종 앙상블 단일 재현 스크립트 정리, Clean 환경에서 `predict` + `reproduce` 통과 후 제출 마감(`11/30 10:00`) |

---

## 4. 대회 규정 체크리스트 (절대 위반 금지)

- [ ] `data/raw/open` 원본 이미지/CSV 절대 수정 금지.
- [ ] 외부 이미지 추가 학습 금지, 테스트 정답/외부 라벨 조회 금지, 원격 API 추론 금지.
- [ ] 사전학습 가중치는 **2026-11-15 이전 공개 오픈소스**만 사용하고, 최종 코드 제출 시 모델명·버전·출처·라이선스 명시.
- [ ] Git에 데이터·가중치(`.pt`/`.pth`)·예측 CSV·API 키·절대 경로 커밋 금지.
- [ ] 대회 제출과 GPU 결제는 명시적 지시 후에만 수행.
