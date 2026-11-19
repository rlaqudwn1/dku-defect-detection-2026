"""Interactive local W&B API key setup without storing secrets in the repository."""
import argparse
import getpass
import os
from pathlib import Path


def _netrc_has_wandb() -> bool:
    netrc = Path.home() / ".netrc"
    if not netrc.is_file():
        return False
    text = netrc.read_text(encoding="utf-8", errors="ignore")
    return "api.wandb.ai" in text or "api.forge.coreweave.com" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only report credential status")
    args = parser.parse_args()

    has_env = bool(os.environ.get("WANDB_API_KEY", "").strip())
    has_netrc = _netrc_has_wandb()
    print(f"WANDB_API_KEY env: {'SET' if has_env else 'unset'} | ~/.netrc api.wandb.ai: {'SET' if has_netrc else 'unset'}")

    if args.check:
        return 0 if (has_env or has_netrc) else 1

    import wandb

    print("\n1) 열린 브라우저 탭(https://wandb.ai/authorize)에서 API Key를 복사하세요.")
    print("2) 아래에 붙여넣고 Enter를 누르세요 (입력 내용은 화면에 표시되지 않습니다).\n")
    key = getpass.getpass("WANDB_API_KEY: ").strip()
    if not key:
        print("취소되었습니다: 빈 값이 입력되었습니다.")
        return 1

    ok = wandb.login(key=key, relogin=True)
    if not ok:
        print("로그인 실패: API Key를 다시 확인해 주세요.")
        return 1

    viewer = wandb.Api().viewer
    username = getattr(viewer, "username", None)
    entity = getattr(viewer, "entity", None)
    print(f"\nW&B 로그인 완료! (username={username}, entity={entity})")
    print("~/.netrc에 안전하게 저장되었습니다 (Git 저장소 밖).")
    print("\n원격 실행용 참고:")
    print("- Kaggle: 노트북 Add-ons -> Secrets -> Label: WANDB_API_KEY")
    print("- Colab:  왼쪽 열쇠 아이콘(Secrets) -> Name: WANDB_API_KEY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
