#!/usr/bin/env bash
# Cardiff b-tensor data (Afzali et al. 2021; CC BY 4.0; DOI 10.17035/d.2021.0135083057): two healthy
# participants on the 3T Connectom (300 mT/m), 3 mm isotropic, TE 88 ms; LTE and PTE 439 volumes each and
# STE 182, nominal shells b = 0-12 ms/µm² (the dataset description lists fewer), plus a reversed-PE STE b0.
# The repository's web page sits behind a browser check; Figshare's API and file downloader do not.
#
#     scripts/fetch_cardiff.sh [data/cardiff]     # 251.5 MB zip, 480 MB on disk with it extracted
set -euo pipefail
DEST=${1:-data/cardiff}
mkdir -p "$DEST"
ZIP="$DEST/d.2021.0135083057.zip"
[ -f "$ZIP" ] || curl -sL --fail -o "$ZIP" https://ndownloader.figshare.com/files/49921407   # article 27277359
[ "$(md5 -q "$ZIP" 2>/dev/null || md5sum "$ZIP" | cut -d' ' -f1)" = a785949b26b324d1b69712902a75e2b9 ] \
  || { echo "md5 mismatch: $ZIP" >&2; exit 1; }
unzip -q -o "$ZIP" -x "*/PaxHeader/*" -d "$DEST"
ls "$DEST"/d.2021.0135083057/data1 "$DEST"/d.2021.0135083057/data2
