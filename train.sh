#!/bin/bash
#SBATCH --job-name=unet_imperv_fix
#SBATCH --clusters=anansi
#SBATCH --partition=ada_gpu
#SBATCH --output=training_log_%j.txt
#SBATCH --error=training_error_%j.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e   # any step fails → abort the job; don't train on stale inputs.

echo "Starting job and grabbing GPU..."

# 1. LOAD 2025a MODULES
module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load netcdf4-python/1.7.2-foss-2025a
module load zarr/3.1.1-gfbf-2025a
module load rioxarray/0.21.0-foss-2025a

cd $VSC_HOME

export PATH=$PATH:$HOME/.local/bin
export CKPT_DIR="$VSC_SCRATCH/urbclim_checkpoints"

echo "Starting GPU Tracker..."

# =============================================================================
# STAGE A — regenerate per-split statistics with the patched extract_static_maps
# =============================================================================
# Splits are deterministic (SHUFFLE_SEED=42), so we don't rerun pipeline.splits
# or pipeline.build_lookup. zscore.py reads pipeline/extract.py, which now
# (a) finds the 'imperv' key, (b) raises if no imperv variable is present,
# (c) clamps Copernicus 254/255 nodata flags to 0 before stats.

echo "=== [A.1] Regenerating train/val/test statistics ==="
python -u -m pipeline.zscore

echo "=== [A.2] Sanity check — imperviousness stats must be non-zero ==="
python - <<'PY'
import pandas as pd, sys, os
DATA = "/user/brussel/113/vsc11361/data"
for sp in ("train", "val", "test"):
    df = pd.read_csv(os.path.join(DATA, f"{sp}_statistics.csv"))
    imp = df[df["feature"] == "imperviousness"]
    if imp.empty:
        sys.exit(f"  ✗ {sp}: no imperviousness row at all")
    m, s = float(imp.iloc[0]["mean"]), float(imp.iloc[0]["std"])
    print(f"  {sp:5s}  imperv  mean={m:.4f}  std={s:.4f}")
    if abs(m) < 1e-6 or abs(s) < 1e-6:
        sys.exit(f"  ✗ {sp}: imperv mean/std still zero — patched files may not have"
                 f" reached the cluster. Stop the job and re-scp pipeline/extract.py.")
print("  ✓ imperv stats look healthy, proceeding to training")
PY

# =============================================================================
# STAGE B — preserve pre-fix checkpoints so the old results stay reproducible
# =============================================================================
echo "=== [B] Preserving pre-fix checkpoints ==="
for d in urbclim_v2_models urbclim_v2_models_solar; do
    src="$VSC_SCRATCH/$d"
    dst="$VSC_SCRATCH/${d}_pre_imperv_fix"
    if [ -d "$src" ] && [ ! -d "$dst" ]; then
        mv "$src" "$dst"
        echo "  moved $src -> $dst"
    elif [ -d "$dst" ]; then
        echo "  $dst already exists, leaving $src in place if any"
    else
        echo "  no $src to preserve"
    fi
done

# =============================================================================
# STAGE C — retrain both models with the imperviousness channel actually live
# =============================================================================
# Keep NUM_EPOCHS identical in both scripts (hard rule in 12_may.md §9).
# Run sequentially; they share the GPU. ~2–3 h each on the ada_gpu partition.

echo "=== [C.1] Training 23-ch baseline ==="
python -u v2_residual_unet.py

echo "=== [C.2] Training 25-ch solar variant ==="
python -u v2_residual_unet_solar.py

# =============================================================================
# STAGE D — evaluate both checkpoints
# =============================================================================
# test_full.py reads CHECKPOINT_PATH from its top-of-file constant. If you'd
# rather not edit it twice, comment one of these calls out and run a second
# job for the other model.

echo "=== [D.1] Test eval — 23-ch baseline ==="
# Default CHECKPOINT_PATH points at urbclim_v2_models/best_model.pth (baseline).
python -u test_full.py

echo "=== [D.2] Diagnostic data log — 23-ch baseline ==="
python -u diagnose.py

# If you want the solar eval in the same job, swap CHECKPOINT_PATH inside
# test_full.py and diagnose.py to point at urbclim_v2_models_solar/ then
# uncomment the two lines below. Otherwise run them in a follow-up job.
# python -u test_full.py
# python -u diagnose.py

echo "=== [D.3] Breakdown plots ==="
python -u plot_diagnostics.py

# Cleanup
if [ -n "$GPU_MONITOR_PID" ]; then
    kill $GPU_MONITOR_PID
fi
echo "Finished training. GPU monitoring stopped."
