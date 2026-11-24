# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (`scripts/sim_lb_800.py`, 200회 파편화·가변불량률 시뮬레이터)
- **Public LB 1위 (Champion)**: **`E14-Q15` (`Elite-9`, OOF `0.9520` / Track B `Top2Mean 0.9533` / Public LB `0.884610`)**
- **Day 0 최신 돌파 (`E15-Recal`)**: `scripts/cluster_v2_and_recal.py` 완료 → **Test Singleton `152/800(19.0%) → 1/800(0.12%)` 박멸**, `Sim-LB-800` **`0.9320 → 0.9375 (+0.0055, P10 +0.0089)`**, **Fold 0 `0.9070 → 0.9181 (+0.0111)`**!
  - Day 1 [제출 1] 검증 완료 후보: `runs/submissions/sub_e15_recal_cluster_v2_left_tail.csv`
  - 공유 캐시 동기화: **W&B Artifact `dku-model-cache:latest`** (`scripts/sync_wandb_artifacts.py`, Zero-OAuth, `M01~M15.npz` + `canonical_cids.npz` + `folds_unseen.json`)

## 2. CV-LB 갭(`-0.0674`) 3대 근본 원인 & Blackwell 제약
- **원인 1 (해결)**: 기존 `k_den=20 / P25` 절단으로 Test `19.0%`(152장)가 `cid=-1` Singleton 탈락 → `Cluster-v2`로 `0.12%`로 박멸.
- **원인 2 (해결)**: 불량률 `>25%` 클러스터에서 `IQR(P75-P25)` 폭발 → `Left-Tail Robust-Z (P35, P15~P50)`로 해결.
- **원인 3 (교내 서버 단일 트랙 공략)**: `M01/M06`은 DINOv2 단일 가문(`ρ=0.90`) + 151장 동료비교 꺼진 상태였음 → **교내 Blackwell 서버에서 이종 백본 `MuSc-v2` 4종(`M16~M19`, `lo_i>=1` + `Bank-TTA` + `Cluster-SVD`) 순차 완주**.
- **교내 Blackwell 제약 (필수)**: `VRAM 5.50 GiB`, `cgroup RAM 8.0 GiB (Swap 0B)`, `Disk 64 GB`, `num_workers=0`, 1-레이어 `memmap + POSIX_FADV_DONTNEED`.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·원본 라벨 조회 (코드 검증 실격)
- E2E 백본 전체 파인튜닝(`M15 0.6781`) / Diffusion·GAN 학습 / 균등축소 L2(`l2=2.0`)
- `k_den=20` 하위 25% 밀도 일괄 절단 / `IQR(P75-P25)` 단독 보정 / 중복 백본 25+ 양산

## 4. Day 1 교내 Blackwell 서버 단일 트랙 실행 계획 (목표: LB `0.92+` → 최종 `0.95+`)
1. **최신 코드 & W&B Artifact 동기화 (OAuth 불필요, `WANDB_API_KEY` 1개면 완료)**:
   - `git pull origin rlaqudwn1/dku-eda-saved-handoff`
   - `python scripts/sync_wandb_artifacts.py pull`
2. **이종 백본 `MuSc-v2` 4종 순차 실행** (Fold별 자동 체크포인트 복구 + `progress.json` + W&B 실시간 로깅 + 완료 시 W&B Artifact 자동 push 및 보고서 갱신):
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M16_musc_dinov2_reg4_l --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M17_musc_eva02_l448 --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M18_musc_siglip_so400m_384 --wandb --auto-sync`
   - `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M19_musc_convnext_l384 --wandb --auto-sync`

## 5. 교내 Blackwell 서버 새 세션 시작 프롬프트 (복사해서 붙여넣기)
```text
먼저 `git pull origin rlaqudwn1/dku-eda-saved-handoff`를 받고 `python scripts/sync_wandb_artifacts.py pull`로
최신 모델 캐시(M01~M15)와 공유 클러스터(canonical_cids.npz)를 W&B Artifact에서 내려받아줘.
그다음 AGENTS.md, STATE.md, EXPERIMENTS.md를 읽어줘. (Test 이미지는 절대 열람/시각화 금지!)
STATE.md "4. Day 1 교내 Blackwell 서버 단일 트랙 실행 계획"대로 `scripts/run_musc_v2_blackwell.py`로
M16 -> M17 -> M18 -> M19를 `--wandb --auto-sync` 옵션으로 순차 실행해줘.
```
