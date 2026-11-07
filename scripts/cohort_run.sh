#!/usr/bin/env bash
# Prepare, segment and map every Hormone Health session (66), skipping any step whose output exists,
# so it can be rerun after an interruption. Three lanes run at once:
#   CPU lanes (2, interleaved sessions):  prepare_session.py (align + constrained fit, float64 CPU, as for
#                                         the first five sessions), then t1_to_dwi.py
#   MPS lane (1, one GPU job at a time):  FastSurfer asegdkt for every T1 (haversack), then, as each
#                                         session's CPU steps finish, fastsurfer_to_dwi.py and the
#                                         whole-brain posterior of the network $TAG (default _v5)
# Logs: work/logs/cohort/; per-session QC images go to work/qc/ (not results/) unless git tracks them. Prints "FAILED <step> <session>" for any failure and carries on.
#
#     scripts/cohort_run.sh                 # 1 h 45 min on the M1 Max (2026-11-06)
#     TAG=_v4long scripts/cohort_run.sh     # another network (FastSurfer and prepare steps are skipped)
set -u
cd "$(dirname "$0")/.."
TAG=${TAG:-_v5}
MODEL=results/posterior_smoke_hhs${TAG}.pt
HAV=../haversack/.venv/bin
LOG=work/logs/cohort
mkdir -p "$LOG"
mkdir -p work/qc
qc() { [ -f "$1" ] && ! git ls-files --error-unmatch "$1" >/dev/null 2>&1 && mv "$1" work/qc/; true; }
SESSIONS=($(ls -d data/hhs/sub-*/ses-* | sed -E 's|.*/sub-([0-9]+)/ses-([0-9]+)$|\1:\2|'))
echo "$(date +%T) ${#SESSIONS[@]} sessions, network $TAG"

cpu_lane() {   # $1 = lane index (0/1), $2 = number of lanes
  local i sub ses s
  for ((i = $1; i < ${#SESSIONS[@]}; i += $2)); do
    sub=${SESSIONS[i]%:*}; ses=${SESSIONS[i]#*:}; s=sub${sub}_ses${ses}
    if [ ! -f "work/hhs_${s}_qti.npz" ]; then
      OMP_NUM_THREADS=5 uv run python scripts/prepare_session.py "$sub" "$ses" > "$LOG/${s}_prepare.log" 2>&1 \
        || { echo "$(date +%T) FAILED prepare $s"; continue; }
    fi
    if [ ! -f "work/hhs_${s}_t1_to_dwi.npz" ]; then
      OMP_NUM_THREADS=5 uv run python scripts/t1_to_dwi.py "$sub" "$ses" > "$LOG/${s}_t1_to_dwi.log" 2>&1 \
        || { echo "$(date +%T) FAILED t1_to_dwi $s"; continue; }
      qc "results/t1_to_dwi_qc_s${sub}${ses}.png"
    fi
    echo "$(date +%T) cpu done $s"
  done
}

cpu_lane 0 2 & P0=$!
cpu_lane 1 2 & P1=$!

for x in "${SESSIONS[@]}"; do      # MPS lane, part 1: FastSurfer needs only the T1
  sub=${x%:*}; ses=${x#*:}; s=sub${sub}_ses${ses}
  [ -f "work/hhs_${s}_asegdkt.duckn.zip" ] && continue
  "$HAV/haversack" segment "data/hhs/sub-$sub/ses-$ses/anat/sub-${sub}_ses-${ses}_T1w.nii" --task fastsurfer:asegdkt \
    --device mps -o "work/hhs_${s}_asegdkt.duckn.zip" > "$LOG/${s}_fastsurfer.log" 2>&1 \
    || echo "$(date +%T) FAILED fastsurfer $s"
done
echo "$(date +%T) fastsurfer done"

for x in "${SESSIONS[@]}"; do      # MPS lane, part 2: as each session's CPU steps finish
  sub=${x%:*}; ses=${x#*:}; s=sub${sub}_ses${ses}
  while [ ! -f "work/hhs_${s}_t1_to_dwi.npz" ] && { kill -0 $P0 2>/dev/null || kill -0 $P1 2>/dev/null; }; do sleep 20; done
  [ -f "work/hhs_${s}_t1_to_dwi.npz" ] || { echo "$(date +%T) SKIPPED $s (cpu steps failed)"; continue; }
  if [ ! -f "work/hhs_${s}_fastsurfer_dwi.npz" ] && [ -f "work/hhs_${s}_asegdkt.duckn.zip" ]; then
    "$HAV/python" scripts/fastsurfer_to_dwi.py "$sub" "$ses" > "$LOG/${s}_fastsurfer_to_dwi.log" 2>&1 \
      || echo "$(date +%T) FAILED fastsurfer_to_dwi $s"
  fi
  if [ ! -f "work/hhs_${s}_posterior${TAG}_full.npz" ]; then
    uv run python scripts/posterior_maps_hhs.py "" "" "$MODEL" "$TAG" --whole-brain --device mps --sub "$sub" --ses "$ses" \
      > "$LOG/${s}_posterior${TAG}.log" 2>&1 || { echo "$(date +%T) FAILED posterior $s"; continue; }
    qc "results/hhs_${s}_posterior_maps${TAG}_full.png"
  fi
  echo "$(date +%T) mps done $s"
done
wait
echo "$(date +%T) cohort done"
