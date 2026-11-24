# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (`scripts/sim_lb_800.py`, 200회 파편화·가변불량률 시뮬레이터)
- **Public LB 1위 (Champion)**: **`E14-Q15` (`Elite-9`, OOF `0.9520` / Track B `Top2Mean 0.9533` / Public LB `0.884610`)**
- **Day 0 최신 돌파 (`E15-Recal`)**: `scripts/cluster_v2_and_recal.py` 완료 → **Test Singleton `152/800(19.0%) → 1/800(0.12%)` 박멸**, `Sim-LB-800` **`0.9320 → 0.9375 (+0.0055, P10 +0.0089)`**, **Fold 0 `0.9070 → 0.9181 (+0.0111)`**!
  - Day 1 [제출 1] 검증 완료 후보: `runs/submissions/sub_e15_recal_cluster_v2_left_tail.csv`
  - 공유 표준 클러스터 캐시: `runs/day0_recal/canonical_cids.npz` (`gdrive:dku-defect-2026` 자동 동기화)

## 2. CV-LB 갭(`-0.0674`) 3대 근본 원인 & Blackwell 제약
- **원인 1 (해결 완료)**: 기존 `k_den=20 / P25` 절단으로 Test의 **`19.0%`(152장)가 `cid=-1` Singleton 탈락** → `Cluster-v2`로 `0.12%`로 박멸.
- **원인 2 (해결 완료)**: 불량률 `>25%` 클러스터에서 `IQR(P75-P25)` 폭발 → `Left-Tail Robust-Z (P35, P15~P50)`로 해결.
- **원인 3 (Day 1 Track A 공략)**: `M01/M06`은 DINOv2 단일 가문(`ρ=0.90`) + 151장 동료비교 꺼진 상태였음 → **이종 백본 `MuSc-v2` 4종(`M16~M19`, `lo_i>=1` + `Bank-TTA` + `Cluster-SVD`)** 실행 필수.
- **교내 Blackwell 제약 (필수)**: `VRAM 5.50 GiB`, `cgroup RAM 8.0 GiB (Swap 0B)`, `Disk 64 GB`, `num_workers=0`, 1-레이어 `memmap + POSIX_FADV_DONTNEED`.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격)
- E2E 백본 전체 파인튜닝(`M15 0.6781`) / Diffusion·GAN 학습 / 균등축소 L2(`l2=2.0`)
- `k_den=20` 하위 25% 밀도 일괄 절단 / `IQR(P75-P25)` 단독 보정 / 중복 백본 25+ 양산

## 4. Day 1 즉시 실행 계획 (목표: LB `0.92+` → 최종 `0.95+`)
1. **[교내 Blackwell 서버 — Track A 자동 실행]**: `git pull` + `bash scripts/sync_artifacts.sh pull` 후 아래 순차 실행 (Fold별 체크포인트 복구 + `progress.json` + W&B + `rclone push` + 보고서 자동 갱신 내장):
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M16_musc_dinov2_reg4_l --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M17_musc_eva02_l448 --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M18_musc_siglip_so400m_384 --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M19_musc_convnext_l384 --wandb --auto-sync`
2. **[로컬 PC — Day 1 제출 & Track B]**: `sub_e15_recal_cluster_v2_left_tail.csv` 제출로 `Cluster-v2 + Left-Tail Z` 단독 LB 효과 확인 후, `M16~M19` 도착 즉시 `Unsup 75% : VLM/Sup 25%` 계층 앙상블 통합.

## 5. 교내 Blackwell 서버 새 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
먼저 `git pull origin rlaqudwn1/dku-eda-saved-handoff` 및 `bash scripts/sync_artifacts.sh pull`을 실행하고,
AGENTS.md, STATE.md, EXPERIMENTS.md를 읽어줘. (Test 이미지는 절대 열람/시각화 금지!)
STATE.md "4. Day 1 즉시 실행 계획"의 Track A 자동 러너(`scripts/run_musc_v2_blackwell.py`)로
M16 -> M17 -> M18 -> M19를 순차 실행해줘. (W&B 자동 로깅, Fold별 체크포인트 복구, 완료 시 gdrive 자동 동기화 및 보고서 갱신 포함)
```
