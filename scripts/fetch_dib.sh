#!/usr/bin/env bash
# Fetch the Szczepankiewicz-Hoge-Westin 2019 subsampled brain data (Boito/QTI+ subsets, ~340 MB)
# with a sparse git checkout. DIPY's fetcher downloads the same files over raw.githubusercontent.com,
# which some sandboxes block.
set -euo pipefail
DEST="${1:-data/dib}"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019 "$DEST"
git -C "$DEST" sparse-checkout set DATA/brain/NII_Boito_SubSamples
echo "export QTILAB_DIB_ROOT=$DEST/DATA/brain/NII_Boito_SubSamples"
