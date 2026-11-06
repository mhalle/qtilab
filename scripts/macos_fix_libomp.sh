#!/bin/sh
# macOS only: make dipy use torch's OpenMP runtime, so a process never loads two copies of libomp.
#
# dipy's macOS wheels bundle their own libomp (dipy/.dylibs), and so does torch (torch/lib). Importing
# dipy.align (used by qtilab.hhs for run alignment) together with torch aborts with
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# Both link it via @loader_path with compatibility version 5.0.0, so dipy can share torch's copy.
# The usual alternative, KMP_DUPLICATE_LIB_OK=TRUE, can silently give wrong results; don't use it.
#
# Idempotent. Re-run after `uv sync` reinstalls or upgrades dipy or torch:
#     sh scripts/macos_fix_libomp.sh
set -eu
[ "$(uname)" = Darwin ] || { echo "not macOS; nothing to do"; exit 0; }
SP=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
DIPY="$SP/dipy/.dylibs/libomp.dylib"
TORCH="$SP/torch/lib/libomp.dylib"
[ -e "$TORCH" ] || { echo "torch's libomp not found at $TORCH"; exit 1; }
if [ -L "$DIPY" ]; then
    echo "already linked: $DIPY -> $(readlink "$DIPY")"
elif [ -f "$DIPY" ]; then
    mv "$DIPY" "$DIPY.dipy-orig"
    ln -s ../../torch/lib/libomp.dylib "$DIPY"
    echo "linked $DIPY -> torch/lib/libomp.dylib (original kept as libomp.dylib.dipy-orig)"
else
    echo "dipy bundles no libomp here; nothing to do"
fi
uv run python -c "import torch, dipy.align; print('torch + dipy.align import OK')"
