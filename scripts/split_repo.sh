#!/usr/bin/env bash
# Splits the training tooling and the Streamlit dashboard out of Park-Vision
# into their own repos, preserving git history for the files that move.
#
# Why: dataset_tools/, colab/ and streamlit_app.py/calibrate_tool.py don't
# belong in the same deployable unit as the FastAPI service (app/). They
# have different dependencies (training needs a GPU + full ultralytics
# training stack; the dashboard is a dev tool, not part of the API), and
# bundling them means every API deploy drags training/dashboard code and
# every CI run pays for dependencies it doesn't need.
#
# Usage (run from the Park-Vision repo root, on a clean working tree):
#   ./scripts/split_repo.sh training  <path-to-new-empty-repo-dir>
#   ./scripts/split_repo.sh dashboard <path-to-new-empty-repo-dir>
#
# This only prepares local history-preserving branches / a bundle you can
# push -- it does not create anything on GitHub, since that needs your
# own credentials/org.

set -euo pipefail

TARGET="${1:-}"
DEST="${2:-}"

if [[ -z "$TARGET" || -z "$DEST" ]]; then
  echo "Usage: $0 <training|dashboard> <destination-dir>" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree not clean. Commit or stash changes first." >&2
  exit 1
fi

case "$TARGET" in
  training)
    PATHS=(dataset_tools colab)
    NEW_NAME="park-vision-training"
    ;;
  dashboard)
    PATHS=(streamlit_app.py calibrate_tool.py process_video.py)
    NEW_NAME="park-vision-dashboard"
    ;;
  *)
    echo "Unknown target '$TARGET'. Use 'training' or 'dashboard'." >&2
    exit 1
    ;;
esac

echo "==> Extracting ${PATHS[*]} into a standalone branch (history preserved)"

BRANCH="split/${TARGET}"
git branch -D "$BRANCH" 2>/dev/null || true

# git subtree split keeps only the commits that touched these paths.
# For multiple paths we filter with `git filter-repo` if available (more
# reliable for several paths at once); fall back to subtree split for a
# single path.
if command -v git-filter-repo >/dev/null 2>&1; then
  git worktree add /tmp/"${NEW_NAME}"-src HEAD >/dev/null
  pushd /tmp/"${NEW_NAME}"-src >/dev/null
  path_args=()
  for p in "${PATHS[@]}"; do
    path_args+=(--path "$p")
  done
  git filter-repo --force "${path_args[@]}"
  popd >/dev/null
  SRC_DIR="/tmp/${NEW_NAME}-src"
else
  echo "git-filter-repo not found; falling back to 'git subtree split' (first path only: ${PATHS[0]})" >&2
  git subtree split --prefix="${PATHS[0]}" -b "$BRANCH"
  SRC_DIR=""
fi

mkdir -p "$DEST"
if [[ -n "$SRC_DIR" ]]; then
  cp -r "$SRC_DIR"/. "$DEST"/
  rm -rf "$SRC_DIR"
  git worktree prune
else
  git worktree add "$DEST" "$BRANCH"
fi

echo "==> Done. Review $DEST, then:"
echo "    cd $DEST && git remote add origin <new-empty-github-repo-url> && git push -u origin main"
echo ""
echo "After pushing, remove the moved paths from this (Park-Vision) repo with:"
for p in "${PATHS[@]}"; do
  echo "    git rm -r ${p}"
done
