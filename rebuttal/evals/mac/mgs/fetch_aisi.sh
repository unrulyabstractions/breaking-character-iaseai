#!/usr/bin/env bash
# Pinned checkout of UKGovernmentBEIS/reward-hacking-misalignment at commit 1c0a303 into vendor/, then a byte check.
#
#   ./fetch_aisi.sh            fetch (only if vendor/ is absent), then verify
#   ./fetch_aisi.sh --verify   verify the existing vendor/ only
#
# The check compares the sha256 of every file in SOURCE.sha256 and also refuses any file under
# misalignment-evals/ that SOURCE.sha256 does not list. Exit 0 only when both hold.
# It never overwrites an existing vendor/ folder.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_URL="https://github.com/UKGovernmentBEIS/reward-hacking-misalignment.git"
COMMIT="1c0a3039744bd91444124b8b4e71fe23f17f0dae"
DEST="$HERE/vendor/reward-hacking-misalignment"
MANIFEST="$HERE/SOURCE.sha256"

verify() {
  local root="$1"
  [ -f "$MANIFEST" ] || { echo "FAIL: $MANIFEST missing" >&2; return 1; }
  local n_expected bad=0
  n_expected=$(grep -c . "$MANIFEST")
  [ "$n_expected" -gt 0 ] || { echo "FAIL: empty manifest" >&2; return 1; }
  # 1. every listed file exists with the listed hash
  if ! (cd "$root" && shasum -a 256 -c --strict "$MANIFEST" >/dev/null); then
    (cd "$root" && shasum -a 256 -c --strict "$MANIFEST" 2>&1 | grep -v ': OK$' >&2) || true
    echo "FAIL: hash mismatch under $root" >&2
    return 1
  fi
  # 2. no unlisted file under misalignment-evals/ (a stray module would be imported silently)
  local listed actual extra
  listed=$(awk '{print $2}' "$MANIFEST" | grep '^misalignment-evals/' | sort)
  actual=$(cd "$root" && find misalignment-evals -type f ! -path '*/__pycache__/*' ! -name '*.pyc' \
             ! -path '*.egg-info/*' | sort)
  extra=$(comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$actual") || true)
  if [ -n "$extra" ]; then
    echo "FAIL: files not in SOURCE.sha256:" >&2
    printf '  %s\n' $extra >&2
    bad=1
  fi
  [ "$bad" -eq 0 ] || return 1
  echo "OK: $n_expected files match SOURCE.sha256 under $root (commit $COMMIT)"
}

if [ "${1:-}" = "--verify" ]; then
  verify "$DEST"
  exit $?
fi
if [ "${1:-}" = "--verify-dir" ]; then   # used by tests/ to prove the check can fail on a copy
  verify "${2:?usage: --verify-dir DIR}"
  exit $?
fi

if [ -e "$DEST" ]; then
  echo "vendor/ already present; verifying it instead of fetching"
  verify "$DEST"
  exit $?
fi

mkdir -p "$HERE/vendor"
TMP="$(mktemp -d "$HERE/vendor/.fetch.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
git -C "$TMP" init -q
git -C "$TMP" remote add origin "$REPO_URL"
git -C "$TMP" config core.sparseCheckout true
printf '%s\n' /LICENSE /README.md /pyproject.toml /uv.lock /scripts/run_misalignment_evals.py /misalignment-evals/ \
  > "$TMP/.git/info/sparse-checkout"
git -C "$TMP" fetch -q --depth 1 origin "$COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD
HEAD_SHA="$(git -C "$TMP" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$COMMIT" ]; then
  echo "FAIL: fetched HEAD $HEAD_SHA is not $COMMIT" >&2
  exit 1
fi
verify "$TMP"
# drop the nested .git so the outer worktree does not see an embedded repository
rm -rf "$TMP/.git"
printf 'repo %s\ncommit %s\nfetched %s\nverified against %s\n' "$REPO_URL" "$COMMIT" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "SOURCE.sha256" > "$TMP/CHECKOUT.txt"
chmod 755 "$TMP"
mv "$TMP" "$DEST"
trap - EXIT
verify "$DEST"
