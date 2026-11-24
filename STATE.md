# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 8개 제품군 5-Fold Unseen GroupKFold, `3502/2205/1706/1705/1703`)
- **Public LB & 5-Fold CV 통합 1위 (Champion)**: **`E14-Q15` (`Elite-9`, OOF `0.9520` / Mean `0.9612` / Public LB `0.884610`)**
  - LB 궤적: `E07 0.7167 → E12(MuSc 0%) 0.8231 → E11(MuSc+VAND) 0.8752 → E14(Elite-9, MuSc 34%) 0.8846 (+0.0094)`
  - Fold별 OOF: **`F0 0.9070 | F1 0.9587 | F2 0.9697 | F3 0.9855 | F4 0.9850`**
  - 단일 Top-3: **`M01_musc_dinov2_l518` (`0.9244`)**, **`M06_musc_dinov2_b518` (`0.9158`)**, **`M08_patchmil_dinov2_l518` (`0.8838`)**
  - 15/15 모델 백업: `runs/queue15_all15_models_backup.zip` (`M01~M15.npz` 로컬 보관)

## 2. 핵심 취약점 & 교내 Blackwell 서버 제약 조건
- **CV-LB 갭(`0.9520 vs 0.8846`, `-0.067`) 핵심 원인**: 15모델 중 도메인 시프트 0% 비지도 `MuSc`가 DINOv2 2종(`M01/M06`, 상관 `0.90`)뿐이고 13종은 지도/VLM 모델. 또한 Test 800장 내 다중 제품군 오프셋 보정·TTA·타일링이 미적용됨.
- **Train OOF 오답**: FN Top-20 중 **17/20이 Fold 0 (`1500x1000` 고종횡비 미세결함)** — 정사각 리사이즈 시 가로 결함 소실.
- **교내 Blackwell 컨테이너 제약 (필수 준수)**: `VRAM 5.50 GiB`, `cgroup RAM 8.0 GiB (Swap 0B)`, `Disk 64 GB`, `torch 2.15+cu132` → 반드시 `num_workers=0`, `pip --no-cache-dir`, 1-레이어 스트리밍/memmap(`POSIX_FADV_DONTNEED`), 매 모델 완료 시 `.npz` 저장.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격 — 절대 금지)
- E2E CNN 전체 파인튜닝(`M15 0.6781`) / 균등축소 L2(`l2=2.0`)로 약한·중복 모델 강제 혼합
- `37×37` 패치 `12×12` 축소 / `top-1 max` 동료 비교 / Min-Max 후 `logit` 재변환

## 4. 내일 제출 목표 ROI 순위별 다음 행동 (교내 Blackwell 서버용)
1. **[ROI 1위] 이종 백본 정품 `MuSc` 3종 확장 (`M16~M18`)**: `EVA-02-Large-448 (32×32)`, `SigLIP-SO400M-384 (27×27)` 또는 `OpenCLIP ViT-L-336`, `ConvNeXt-Large-384` 정품 `MuSc` 구현·실행 → 비지도 `MuSc` 타워를 5종 이종 백본(비중 50~60%)으로 확장.
2. **[ROI 2위] 비지도 클러스터별 Robust-Z 보정 + HFlip/타일링 TTA**: DINOv2 전역 임베딩 자동 클러스터링 후 클러스터별 Robust-Z 보정 + `1500x1000` 2-crop 오버랩 타일링(Fold 0 FN `17/20` 공략)을 **Train 5-Fold Unseen CV에서 먼저 검증**.
3. **[ROI 3위] `MuSc` 하이퍼파라미터 튜닝 & 합성 결함(CutPaste/Perlin) 지도 정규화**: `top-k`·`Core-Density` 비율 1-변수 탐색, 정상 이미지 합성 결함 증강으로 `Patch-MIL`/`VAND` 도메인-불변 학습 검증.

## 5. 교내 Blackwell 서버 새 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
먼저 `git pull origin rlaqudwn1/dku-eda-saved-handoff`로 최신 코드를 받고,
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽어줘. (Test 이미지는 절대 열람/시각화 금지!)
교내 Blackwell 컨테이너 제약(VRAM 5.5GiB, RAM 8GiB/Swap 0B, num_workers=0)을 엄수하면서,
STATE.md "4. 내일 제출 목표 ROI 순위별 다음 행동"의 ROI 1위(이종 백본 정품 MuSc M16~M18)부터
One-Change Rule로 Train 5-Fold (`runs/folds_unseen.json`) 검증 및 `runs/queue15/models/M16~M18.npz` 생성을 진행해줘.
```
