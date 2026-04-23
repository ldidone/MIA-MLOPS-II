#!/usr/bin/env bash
# Build a clean orphan branch dedicated to the Hugging Face Space and push it.
#
# Rationale:
#  * HF Spaces reject binary files that are not tracked by Git LFS/Xet.
#  * Our repo history (on GitHub) contains training intermediates
#    (data/processed/*.parquet) and a raw model pickle which do NOT belong on
#    the Space. This script builds a minimal, LFS-aware history in a throwaway
#    worktree and force-pushes it as `main` to the `hf` remote.
#
# Usage:
#   ./scripts/deploy_hf_space.sh                  # push to remote "hf"
#   HF_REMOTE=hf ./scripts/deploy_hf_space.sh     # override remote name
#   DRY_RUN=1 ./scripts/deploy_hf_space.sh        # build branch, skip push
set -euo pipefail

HF_REMOTE="${HF_REMOTE:-hf}"
BRANCH="${HF_DEPLOY_BRANCH:-hf-deploy}"
WORKTREE_DIR="${HF_DEPLOY_WORKTREE:-.git/hf-deploy-worktree}"

die() { echo "error: $*" >&2; exit 1; }

command -v git   >/dev/null 2>&1 || die "git is required"
command -v git-lfs >/dev/null 2>&1 || die "git-lfs not found. Install with: brew install git-lfs"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git repository"

[ -f "models/model.pkl" ]              || die "models/model.pkl is missing. Run training first."
[ -f "models/feature_metadata.json" ]  || die "models/feature_metadata.json is missing. Run training first."
[ -f "Dockerfile" ]                    || die "root Dockerfile is missing (needed by HF Spaces)."

if ! git remote get-url "$HF_REMOTE" >/dev/null 2>&1; then
    die "git remote '$HF_REMOTE' is not configured. Run: make hf-space-remote"
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo ">> Preparing throwaway worktree at $WORKTREE_DIR"
git worktree remove -f "$WORKTREE_DIR" 2>/dev/null || true
rm -rf "$WORKTREE_DIR"
git worktree prune
# Drop any stale branch from a previous failed run so --orphan can recreate it.
git branch -D "$BRANCH" 2>/dev/null || true
git worktree add --detach "$WORKTREE_DIR" HEAD

pushd "$WORKTREE_DIR" >/dev/null

echo ">> Creating orphan branch $BRANCH with LFS-tracked binaries"
git checkout --orphan "$BRANCH"
git rm -rf --cached . >/dev/null

# Wipe the worktree so we can overlay the *current* working-tree files from the
# main checkout. This way uncommitted changes (e.g. README tweaks) are always
# included in the Space push. Keep the `.git` pointer untouched.
shopt -s dotglob nullglob
for entry in ./*; do
    case "$(basename "$entry")" in
        .git) ;;
        *) rm -rf "$entry" ;;
    esac
done
shopt -u dotglob nullglob

# Whitelist: everything the Space actually needs. Anything not listed here is
# intentionally excluded (data/, airflow/, docker-compose.yml, tests, …).
cp "$REPO_ROOT/Dockerfile"      ./
cp "$REPO_ROOT/README.md"       ./
cp "$REPO_ROOT/pyproject.toml"  ./
cp "$REPO_ROOT/.dockerignore"   ./
cp -R "$REPO_ROOT/conf"         ./
cp -R "$REPO_ROOT/src"          ./
cp -R "$REPO_ROOT/requirements" ./
# Strip any local caches that may have been copied along with `src/`.
find src -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find src -type f -name '*.pyc' -delete 2>/dev/null || true
mkdir -p models
cp "$REPO_ROOT/models/model.pkl"             models/
cp "$REPO_ROOT/models/feature_metadata.json" models/
[ -f "$REPO_ROOT/models/.gitkeep" ] && cp "$REPO_ROOT/models/.gitkeep" models/ || touch models/.gitkeep

git lfs install --local >/dev/null

cat > .gitattributes <<'EOF'
*.pkl     filter=lfs diff=lfs merge=lfs -text
*.joblib  filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
EOF

git add .gitattributes
git add Dockerfile README.md pyproject.toml .dockerignore
git add conf src requirements
git add models/.gitkeep models/model.pkl models/feature_metadata.json

git -c user.name='HF Space deploy' -c user.email='hf-space-deploy@local' \
    commit -m "Deploy California Housing Streamlit Space" >/dev/null

echo
echo ">> Branch $BRANCH ready in $WORKTREE_DIR"
git log --oneline -1
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ">> DRY_RUN=1 set; skipping push. Inspect $WORKTREE_DIR and push manually:"
    echo "   (cd $WORKTREE_DIR && git push $HF_REMOTE $BRANCH:main --force)"
    popd >/dev/null
    exit 0
fi

echo ">> Pushing $BRANCH -> $HF_REMOTE/main (force)"
git push "$HF_REMOTE" "$BRANCH:main" --force

popd >/dev/null

echo ">> Cleaning up throwaway worktree"
git worktree remove -f "$WORKTREE_DIR"

echo
echo "Done. Check your Space:"
git remote get-url "$HF_REMOTE"
