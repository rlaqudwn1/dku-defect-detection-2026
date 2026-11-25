# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (`scripts/sim_lb_800.py`)
- **Public LB 1위 (Champion)**: **`E14-Q15` (`Elite-9`, OOF `0.9520` / Track B `Top2Mean 0.9533` / Public LB `0.884610`)**
- **Day 0 Track B (`E15-Recal`)**: `Cluster-v2` Test Singleton `19.0%→0.12%`, `Sim-LB-800` `0.9320→0.9375`, `Fold 0` `0.9070→0.9181`.
- **W&B Artifact 동기화**: `dku-model-cache:latest` 최신 반영 완료 (`M01~M17.npz` + `canonical_cids.npz` + `folds_unseen.json`, 22 files).

## 2. Track A Blackwell 자가 최적화 루프 실측 노하우 (M16·M17 완료 → M18 진행)
- **M16 (`M16_musc_dinov2_reg4_l`, 19.3분, W&B `iz42czmu`)**: 기본 `(0.72/0.18/0.10)` OOF **`0.8996`** (`ρ(M01)=0.775`); **`z_svd 0.9191`** (`F0 0.9016`, `cid=2 0.8517` vs `M01 0.7846`, `ρ=0.693`), `z_musc 0.8933`, `z_log 0.7545`.
- **M17 (`M17_musc_eva02_l448`, 12.0분, OOM 0회, VRAM `1.14~2.87G`, RAM `2.34G`, W&B `6d36k0i2`)**:
  - 1-변수 변경 `(0.40/0.60/0.00)` 적용 결과: OOF **`0.8901`** (`F0 0.8234`, `F1 0.8697`, `F2 0.9189`, `F3 0.9368`, **`F4 0.9731`**), **`ρ(M01)=0.611`, `ρ(M16)=0.618`** (높은 이종 직교성 입증! 동일 백본 지도 `M12 0.7711` 대비 `+0.1190`).
  - **3-채널 감사**: `z_musc 0.8983` >> `z_svd 0.8503` > `z_log 0.8061`; 최적 결합 `(0.85, 0.15, 0.00)` OOF `0.8991`.
  - **2-모델 교차 검증 노하우**:
    1. **`w_logical = 0.00` 2연속 확정**: M16(`0.7545`)·M17(`0.8061`) 모두 `cid=2`에서 `0.50` 랜덤 수준 → 제거 확정.
    2. **비-DINOv2 비선형 공간에서 SVD 과비중 주의**: DINOv2(M16)는 `z_svd` 강세지만, MIM(M17)은 `z_musc(0.8983)`가 `z_svd(0.8503)`를 크게 압도(`F1 cid=4: 0.913 vs 0.703`, `F2 cid=7: 0.905 vs 0.606`) → 이종 백본 기본 비중은 **`(0.80, 0.20, 0.00)`**이 안전.
    3. **17-모델 앙상블 추이**: `Equal-Z 0.9387→0.9451 (+0.0064)`, `RankAvg 0.9270→0.9351 (+0.0081)`, `Caruana 0.9255→0.9331 (+0.0076)`, `PowerRank 0.9340→0.9370`; 단 `Elite(0.9477)`·`5-Tower(0.9463)`는 `F0 cid=2` 약세 + `M16/M17` 우측 꼬리 분산(`std 1.52/1.45` vs `M01 1.14`)으로 일시 하락(`F2 +0.0080`, `F4 +0.0014`는 크게 개선).
    4. **`F0 cid=2` `musc` 급락 2연속 재현**(`M16 0.6401`, `M17 0.6879`): Farthest-Point 뱅크 오염 가설 강화 → M18 확인 후 M19 뱅크 개선 검토.

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋
- `z_logical` 고정 혼합 / 비-DINOv2 백본에 `w_svd >= 0.50` 과다 배정 / E2E 전체 파인튜닝(`M15 0.6781`)

## 4. 다음 행동 1개 (One-Change Rule)
- **M18 (`M18_musc_siglip_so400m_384`)에 딱 1개 변수 반영**: `channel_weights: [0.40, 0.60, 0.00] → [0.80, 0.20, 0.00]` (`w_logical=0.00` 유지 + VLM 비선형 패치 공간에 맞춰 `z_musc` 0.80 복원).
- 더미 텐서 스모크 테스트(`openclip_vit` + VRAM/RAM 확인) 통과 후 실행:
  `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M18_musc_siglip_so400m_384 --wandb --auto-sync`
