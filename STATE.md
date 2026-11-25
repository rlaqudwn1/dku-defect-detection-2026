# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (`scripts/sim_lb_800.py`)
- **Public LB 1위 (Champion)**: **`E14-Q15` (`Elite-9`, OOF `0.9520` / Track B `Top2Mean 0.9533` / Public LB `0.884610`)**
- **Day 0 Track B (`E15-Recal`)**: `Cluster-v2` Test Singleton `19.0%→0.12%`, `Sim-LB-800` `0.9320→0.9375`, `Fold 0` `0.9070→0.9181`.
- **W&B Artifact 동기화**: `dku-model-cache:latest` 최신 반영 완료 (`M01~M16.npz` + `canonical_cids.npz` + `folds_unseen.json`, 21 files).

## 2. Track A Blackwell 자가 최적화 루프 실측 노하우 (M16 완료 → M17 진행)
- **M16 (`M16_musc_dinov2_reg4_l`, 19.3분 완주, W&B `iz42czmu`)**:
  - 리소스: 유휴 VRAM `1.13G` / 활성 `~2.75G` (F0 `448×672` 시 일시적 HAMI-core OOM 경고 1회 후 자동 회수, 크래시 없음), RAM `2.33G`, Swap `0B`.
  - 기본 결합 `(0.72/0.18/0.10)` OOF **`0.8996`** (`F0 0.8246`, `F1 0.9016`, `F2 0.9543`, `F3 0.9551`, `F4 0.9690`), `ρ(M01)=0.775`.
  - **3-채널 정량 감사 (Train-Only, 핵심 돌파구)**:
    - `z_svd`: **OOF `0.9191`** (`F0 0.9016`, `M01 0.8737` 대비 **`+0.0279`**, `ρ(M01)=0.693`)! `F0 cid=2`(2제품 병합, n=2200)에서 `0.8517` vs `M01 0.7846` (`+0.0671`).
    - `z_musc`: OOF `0.8933` (`F0 0.7981`, `F0 cid=2 0.6401` 급락 — `max_bank=120` 초과 시 Farthest-Point `argmax(min_d)`가 정상 코어 대신 풀 내 극단치를 뱅크로 선발하는 현상 의심).
    - `z_logical`: OOF `0.7545` (`F0 0.6534`, `cid=2 0.5022` 랜덤 수준) → Nested 5-Fold 전 Fold 최적 가중 `0.00`.
    - 최적 채널 결합 `(0.40, 0.60, 0.00)` 시 M16 단독 `0.9200` (`F0 0.8823`), Elite-10 `0.9491 → 0.9529` (SVD 단독 반영 시 **`0.9548`, F0 `0.9177`**!).
  - 16-모델 앙상블 변화 (기본 M16 기준): `Equal-Z 0.9387→0.9436 (+0.0049)`, `5-Tower 0.9476→0.9470`, `Elite 0.9520→0.9491` (원인: `z_logical` 노이즈 + `z_musc` 과다 비중).

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋
- E2E 전체 파인튜닝(`M15 0.6781`) / `k_den=20` P25 일괄 절단 / `IQR` 단독 보정 / `z_logical` 고정 `0.10` 혼합

## 4. 다음 행동 1개 (One-Change Rule)
- **M17 (`M17_musc_eva02_l448`)에 딱 1개 변수 반영**: `channel_weights: [0.72, 0.18, 0.10] → [0.40, 0.60, 0.00]` (`z_logical` 제거 + `z_svd` 상향; 단 3채널 원본 `.npz`는 모두 보존).
- 더미 텐서 스모크 테스트 통과 후 실행:
  `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M17_musc_eva02_l448 --wandb --auto-sync`
- M17 완료 후 `F0 cid=2`에서 `z_musc` 급락이 재현되면, **M18 1-변수 최적화로 뱅크 선발(Farthest-Point → Top-Purity Core) 교정** 적용 예정.
