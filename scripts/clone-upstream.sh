#!/usr/bin/env bash
# clone-upstream.sh — shallow-clone a specific upstream commit (with submodules)
# into a target directory. Used by build-libs.yml to fetch
# leejet/stable-diffusion.cpp at the commit pinned in lib/version.txt.
#
# Usage:
#   scripts/clone-upstream.sh <repo-url> <commit-ish> <dest-dir>
set -euo pipefail

url="${1:?usage: clone-upstream.sh <repo-url> <commit> <dest>}"
commit="${2:?missing commit}"
dest="${3:?missing dest}"

mkdir -p "$dest"
cd "$dest"
git init -q
git remote add origin "$url" 2>/dev/null || git remote set-url origin "$url"

# A short SHA can't be fetched directly by every server; fetch it, and if that
# is refused fall back to unshallowing the default branch and checking out.
if git fetch --depth 1 origin "$commit" 2>/dev/null; then
  git checkout -q FETCH_HEAD
else
  echo "clone-upstream: direct shallow fetch of $commit refused; fetching history" >&2
  git fetch origin
  git checkout -q "$commit"
fi

git submodule update --init --recursive --depth 1
echo "clone-upstream: $url @ $(git rev-parse --short HEAD) checked out into $dest"
