# CURRENT STATE (AI Working Memory — 50줄 이내 유지)

> **규칙**: 이 파일은 **항상 덮어쓴다(누적 금지)**. 새 연구원 세션은 `AGENTS.md` → `STATE.md` → `EXPERIMENTS.md` 3개만 읽고 즉시 작업을 이어간다. **Test 이미지 육안 열람은 엄격히 금지한다.**

## 1. 현재 Champion & 검증 기준
- **CV Split**: `runs/folds_unseen.json` (Train 10,821장 / 5-Fold Unseen) + **`Sim-LB-800`** (`scripts/sim_lb_800.py`)
- **Public LB 1위**: **`E14-Q15`** (`Elite-9`, OOF `0.9520` / Public LB `0.884610`)
- **단일 모델 역대 1위 (NEW!)**: **`M18_musc_siglip_so400m_384` (OOF `0.9261`, F0 `0.9171`, `ρ(M01)=0.604`)** — `M01(0.9244)` 추월!
- **CV 앙상블 잠재 신기록**: `Elite-9 + M18` Nested 5-Fold OOF = **`0.9568` (`+0.0048`, F0 `0.9070→0.9195`, 5개 Fold 전원 상승)**!
- **W&B Artifact 동기화**: `dku-model-cache:latest` 최신 반영 완료 (`M01~M18.npz` + `canonical_cids.npz` + `folds_unseen.json`, 23 files).

## 2. Track A Blackwell 자가 최적화 루프 실측 노하우 (M16~M18 완료 → M19 진행)
- **M16 (`DINOv2-Reg4-L`, 19.3분)**: 기본 `(0.72/0.18/0.10)` `0.8996` (`ρ=0.775`); **`z_svd 0.9191`** (`F0 0.9016`, `ρ=0.693`), `z_musc 0.8933`, `z_log 0.7545`.
- **M17 (`EVA-02-L MIM`, 12.0분, OOM 0회)**: `(0.40/0.60/0.00)` `0.8901` (`F4 0.9731`, `ρ=0.611`); `z_musc 0.8983` >> `z_svd 0.8503` > `z_log 0.8061`.
- **M18 (`SigLIP-SO400M-384 VLM`, 11.8분, OOM 0회, VRAM `1.62~3.24G`, RAM `4.58G`, W&B `ud7yd2vi`)**:
  - 1-변수 변경 `(0.80/0.20/0.00)`으로 **`0.9261`** 달성 (M17 비중 유지 시 `0.9169`였으므로 **`+0.0092` 직접 방어**!).
  - **Fold별**: `F0 0.9171` (`cid=2 0.8760` vs `M01 0.7846`, **`+0.0914`**!), `F1 0.8869`, `F2 0.9029` (`cid=6 0.9946`), `F3 0.9667`, `F4 0.9732` (`cid=11 0.9975`).
  - **3-채널 감사**: `z_musc 0.9268` >> `z_svd 0.8888` > `z_log 0.8226`; 최적 `(0.75/0.12/0.12)` `0.9290` (`+0.0029`, F2 `+0.0176`).
  - **핵심 노하우**: (1) 비-DINOv2(MIM/VLM)는 `z_musc` 주력(`0.75~0.80`)이 필수, (2) **CLIP/VLM 사전학습 글로벌 평균 임베딩**에서는 `z_logical(0.8226)`이 F1~F4 논리결함에 유의미하게 기여(`+0.0029`), (3) `18-모델 5-Tower = 0.9497` 역대 최고, `Equal-Z 0.9489 (+0.0102)`, `NN-Stack 0.9500`, `Elite-12 0.9515` (M16/M17 기본 `.npz` 채널 최적화·가지치기 시 `0.957+` 유력).

## 3. 절대 다시 하지 말 것 (Dead Ends)
- Test 이미지 육안 열람 / 외부 데이터·라벨 조회 / Git에 Drive ID·`runs/`·`artifacts/` 커밋
- 비-DINOv2 백본에 `w_svd >= 0.50` 과다 배정 / E2E 전체 파인튜닝(`M15 0.6781`) / `k_den=20` P25 절단

## 4. 다음 행동 1개 (One-Change Rule)
- **최종 4번 모델 M19 (`M19_musc_convnext_l384`, CLIP-LAION2B CNN)에 딱 1개 변수 반영**: `channel_weights: [0.80, 0.20, 0.00] → [0.75, 0.15, 0.10]` (M18에서 입증된 CLIP 계열 `z_logical` 기여 `0.10` 복원 + 비-DINOv2 `z_svd` 소폭 축소).
- 더미 텐서 스모크 테스트(`timm_cnn` Stage-2 `48×48` + Stage-3 `24×24`, VRAM/RAM 확인) 통과 후 실행:
  `PYTHONPATH=scripts python scripts/run_musc_v2_blackwell.py --model M19_musc_convnext_l384 --wandb --auto-sync`
