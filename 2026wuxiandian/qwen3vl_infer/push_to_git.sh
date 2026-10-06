#!/usr/bin/env bash
# 把本目录纯推理代码推到 git。
# 账号密码从 git账号密码.txt 读（不要把这个 txt 提交进仓库）。
# 格式任选一种：
#   1) 三行：用户名 / 密码或Token / 仓库 https 地址
#   2) 两行：用户名 / 密码，仓库地址写下面 GIT_URL
#   3) 键值：GIT_USER=...  GIT_PASS=...  GIT_URL=...
set -euo pipefail

# 没有写在 txt 第三行时，改这里
GIT_URL="${GIT_URL:-https://github.com/YOUR_USER/YOUR_REPO.git}"
GIT_EMAIL="${GIT_EMAIL:-}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
sed -i 's/\r$//' "$0" 2>/dev/null || true

CRED=""
for cand in \
  "${CRED_FILE:-}" \
  "$ROOT/git账号密码.txt" \
  "$ROOT/../qwen3vl_sft/git账号密码.txt" \
  "/root/autodl-fs/2026wuxiandian/qwen3vl_sft/git账号密码.txt" \
  "/root/git账号密码.txt"
do
  if [[ -n "${cand}" && -f "$cand" ]]; then
    CRED="$cand"
    break
  fi
done
if [[ -z "$CRED" ]]; then
  echo "[error] 找不到 git账号密码.txt"
  echo "  在本目录放一个，内容："
  echo "    用户名"
  echo "    密码或Token"
  echo "    https://github.com/你的账号/仓库.git"
  exit 1
fi
echo "[info] cred_file=$CRED"

GIT_USER=""
GIT_PASS=""
URL_FROM_FILE=""

python3 - "$CRED" <<'PY'
import os, sys, re, shlex
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
kv = {}
pos = []
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r"^(GIT_USER|USER|USERNAME|GIT_PASS|PASS|PASSWORD|TOKEN|GIT_URL|URL|REPO)\s*[:=]\s*(.+)$", line, re.I)
    if m:
        kv[m.group(1).upper()] = m.group(2).strip().strip('"').strip("'")
    else:
        pos.append(line)
user = kv.get("GIT_USER") or kv.get("USER") or kv.get("USERNAME")
pw = kv.get("GIT_PASS") or kv.get("PASS") or kv.get("PASSWORD") or kv.get("TOKEN")
url = kv.get("GIT_URL") or kv.get("URL") or kv.get("REPO")
if not user and len(pos) >= 1:
    user = pos[0]
if not pw and len(pos) >= 2:
    pw = pos[1]
if not url and len(pos) >= 3:
    url = pos[2]
if not user or not pw:
    raise SystemExit("txt 里缺用户名或密码")
out = Path("/tmp/qwen3vl_infer_git_env")
lines = [f"GIT_USER={shlex.quote(user)}", f"GIT_PASS={shlex.quote(pw)}"]
if url:
    lines.append(f"GIT_URL={shlex.quote(url)}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("user_len", len(user), "pass_len", len(pw), "has_url", bool(url))
PY

set -a
# shellcheck disable=SC1091
source /tmp/qwen3vl_infer_git_env
set +a
rm -f /tmp/qwen3vl_infer_git_env

if [[ -z "${GIT_USER:-}" || -z "${GIT_PASS:-}" ]]; then
  echo "[error] 未能解析账号密码"
  exit 1
fi
if [[ "$GIT_URL" == *"YOUR_USER"* ]]; then
  echo "[error] 还没写仓库地址。在 txt 第三行写 https://...git ，或改本脚本 GIT_URL"
  exit 1
fi
GIT_EMAIL="${GIT_EMAIL:-${GIT_USER}@users.noreply.github.com}"

if ! command -v git >/dev/null 2>&1; then
  echo "[error] git not found"
  exit 1
fi

export GIT_USER GIT_PASS
ENC_USER="$(python3 - <<'PY'
import os, urllib.parse
print(urllib.parse.quote(os.environ["GIT_USER"], safe=""))
PY
)"
ENC_PASS="$(python3 - <<'PY'
import os, urllib.parse
print(urllib.parse.quote(os.environ["GIT_PASS"], safe=""))
PY
)"

HOSTPATH="${GIT_URL#https://}"
HOSTPATH="${HOSTPATH#http://}"
if [[ "$GIT_URL" == https://* ]]; then
  AUTH_URL="https://${ENC_USER}:${ENC_PASS}@${HOSTPATH}"
else
  AUTH_URL="http://${ENC_USER}:${ENC_PASS}@${HOSTPATH}"
fi

if [[ ! -d .git ]]; then
  git init -b main
fi
git config --local user.name "$GIT_USER"
git config --local user.email "$GIT_EMAIL"

git add -- run_infer_submit.py infer.sh requirements.txt README.md .gitignore push_to_git.sh
git status
if git diff --cached --quiet; then
  echo "[info] 没有新的代码改动"
else
  git commit -m "Add Qwen3-VL infer-only scripts"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$AUTH_URL"
else
  git remote add origin "$AUTH_URL"
fi

git push -u origin HEAD
git remote set-url origin "$GIT_URL"
unset GIT_PASS ENC_PASS AUTH_URL
echo "[ok] pushed to $GIT_URL"
echo "[ok] origin 已改回不含密码的地址"
