#!/usr/bin/env bash
# Fetch the Szczepankiewicz-Hoge-Westin 2019 data with a sparse git checkout.
#   scripts/fetch_dib.sh [dest] [--phantoms]
# Default: the subsampled brain data (Boito/QTI+ subsets, ~340 MB).
# --phantoms adds water, oil and liquid crystal with their b-tensor (xps) files (~220 MB).
# DIPY's fetcher downloads the same brain files over raw.githubusercontent.com, which some sandboxes block.
set -euo pipefail
DEST="data/dib"; PHANTOMS=0
for a in "$@"; do case "$a" in --phantoms) PHANTOMS=1 ;; *) DEST="$a" ;; esac; done
[ -d "$DEST/.git" ] || git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019 "$DEST"
SETS="DATA/brain/NII_Boito_SubSamples"
[ "$PHANTOMS" = 1 ] && SETS="$SETS DATA/water/MD-dMRI DATA/oil/MD-dMRI DATA/hex/MD-dMRI"
git -C "$DEST" sparse-checkout set $SETS
echo "export QTILAB_DIB_ROOT=$DEST/DATA/brain/NII_Boito_SubSamples QTILAB_DIB_DATA=$DEST/DATA"
