"""Read-only UTF-8/LF check for project Markdown in worktree or staging."""
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--staged", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
command = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
           if args.staged else
           ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
raw = subprocess.check_output(command, cwd=root)
errors = []
for item in sorted(set(raw.split(b"\0"))):
    if not item or not item.endswith(b".md"):
        continue
    name = item.decode("utf-8")
    if not args.staged and not (root / name).is_file():
        continue
    data = (subprocess.check_output(["git", "show", ":" + name], cwd=root)
            if args.staged else (root / name).read_bytes())
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(name + ": invalid UTF-8")
    if b"\r" in data or data.startswith(b"\xef\xbb\xbf"):
        errors.append(name + ": expected UTF-8 without BOM and LF")
for error in errors:
    print(error)
print("FAIL" if errors else "PASS: Markdown UTF-8/LF (" +
      ("staged" if args.staged else "worktree") + ")")
raise SystemExit(bool(errors))
