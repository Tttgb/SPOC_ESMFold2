#!/usr/bin/env bash
# ============================================================
# download_spoc_db.sh — download & install the SPOC biological
# databases into data/spoc/
# ------------------------------------------------------------
# Fetches the ready-to-use database archives from Zenodo, verifies
# their SHA256 checksums and extracts them into
#   classifier_package/data/spoc/
#
# Usage:
#   bash scripts/download_spoc_db.sh
#
# Requirements: curl, tar, sha256sum
# ============================================================
set -euo pipefail

# ---- Zenodo record (replace XXXXXXX with the real record ID after upload) ----
ZENODO_RECORD="${ZENODO_RECORD:-https://zenodo.org/records/XXXXXXX/files}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPOC_DIR="$REPO_DIR/data/spoc"
mkdir -p "$SPOC_DIR"

FILES=(
  AlphaMissence.tar.gz
  biogrid.tar.gz
  CoexpressDB.tar.gz
  DepMap.tar.gz
  ProtT5_embedding.tar.gz
)

cd "$SPOC_DIR"

echo "=============================================="
echo "SPOC biological database download"
echo "  source: $ZENODO_RECORD"
echo "  target: $SPOC_DIR"
echo "=============================================="

# ---- 1) download SHA256SUMS (checksums of every archive) ----
if [ ! -f SHA256SUMS ] || ! grep -q 'AlphaMissence' SHA256SUMS 2>/dev/null; then
  echo "[1/3] Downloading SHA256SUMS ..."
  curl -fL --retry 5 -o SHA256SUMS "$ZENODO_RECORD/SHA256SUMS"
fi

# ---- 2) download each archive (resumable; skip if already present & valid) ----
for f in "${FILES[@]}"; do
  target="${f%.tar.gz}"                  # directory after extraction
  if [ -d "$target" ] && grep -q " ${f}$" SHA256SUMS && \
     echo "$(grep " ${f}$" SHA256SUMS | awk '{print $1}')  $f" | sha256sum -c - >/dev/null 2>&1; then
    echo "[skip] ${f} already installed & verified"
    continue
  fi
  if [ -f "$f" ] && echo "$(grep " ${f}$" SHA256SUMS | awk '{print $1}')  $f" | sha256sum -c - >/dev/null 2>&1; then
    echo "[skip] ${f} already present & verified"
  else
    echo "[2/3] Downloading ${f} ..."
    curl -fL -C - --retry 5 -o "$f" "$ZENODO_RECORD/$f"
  fi
done

# ---- 3) verify + extract ----
echo "[3/3] Verifying SHA256 checksums ..."
sha256sum -c SHA256SUMS

echo "      Extracting archives ..."
for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "  -> ${f}"
    tar -xzf "$f"
  fi
done

echo ""
echo "Done: databases installed into data/spoc/"
echo "Run 'python scripts/check_spoc_db.py' to verify completeness."

