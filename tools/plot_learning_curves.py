#!/usr/bin/env python
# coding: utf-8
"""
plot_learning_curves.py — learning curves for the v2 Residual U-Net run(s),
read straight out of the checkpoint's `history` dict (no TensorBoard needed).

Reads (whichever exist):
  $VSC_SCRATCH/urbclim_v2_models/latest_model.pth        (23-ch baseline)
  $VSC_SCRATCH/urbclim_v2_models_solar/latest_model.pth  (25-ch solar)
  (falls back to best_model.pth if latest_model.pth is missing)

Writes:
  figures/learning_curves.png   — MSE (left) and L1 (right); train solid, val dashed.
  + prints best-val-MSE / epoch / final train–val gap for each run.

Run anywhere a torch import works:
    python plot_learning_curves.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v2_residual_unet import MY_DATA_DIR

SCRATCH = os.environ.get('VSC_SCRATCH', MY_DATA_DIR)
RUNS = {
    "baseline_23ch": os.path.join(SCRATCH, "urbclim_v2_models"),
    "solar_25ch":    os.path.join(SCRATCH, "urbclim_v2_models_solar"),
}
COLORS = {"baseline_23ch": "#1f78b4", "solar_25ch": "#e31a1c"}

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_history(run_dir):
    for name in ("latest_model.pth", "best_model.pth"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location='cpu')
            h = ckpt.get('history') if isinstance(ckpt, dict) else None
            if h and h.get('train_mse'):
                return h, p
    return None, None


def plot_one_run(label, run_dir):
    """Produce one 2-panel figure (MSE | L1) for a single training run.
    Returns True if a figure was written, False if no history was found."""
    h, src = load_history(run_dir)
    if h is None:
        print(f"  [skip] no usable history in {run_dir}")
        return False

    c = COLORS.get(label, '#1f78b4')
    ep = np.arange(1, len(h['train_mse']) + 1)
    best_i = int(np.argmin(h['val_mse']))
    best_val = h['val_mse'][best_i]
    final_train = h['train_mse'][-1]
    final_val = h['val_mse'][-1]
    best_gap = best_val - h['train_mse'][best_i]
    final_gap = final_val - final_train

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # MSE panel
    ax[0].plot(ep, h['train_mse'], color=c, lw=1.8, ls='-',  label='train')
    ax[0].plot(ep, h['val_mse'],   color=c, lw=2.0, ls='--', label='val')
    ax[0].axvline(best_i + 1, color='grey', lw=1.0, ls=':', alpha=0.7)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('MSE  (°C²)')
    ax[0].set_title('MSE')
    ax[0].grid(ls=':', lw=0.4, alpha=0.6)
    ax[0].legend(fontsize=10, loc='upper right', frameon=False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    # L1 / MAE panel
    ax[1].plot(ep, h['train_l1'], color=c, lw=1.8, ls='-',  label='train')
    ax[1].plot(ep, h['val_l1'],   color=c, lw=2.0, ls='--', label='val')
    ax[1].axvline(best_i + 1, color='grey', lw=1.0, ls=':', alpha=0.7)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MAE  (°C)')
    ax[1].set_title('MAE')
    ax[1].grid(ls=':', lw=0.4, alpha=0.6)
    ax[1].legend(fontsize=10, loc='upper right', frameon=False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    fig.suptitle('Learning curves', fontsize=14, fontweight='bold')
    fig.tight_layout()

    out = os.path.join(FIG_DIR, f"learning_curves_{label}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print(f"[{label}]  from {os.path.basename(src)}")
    print(f"   epochs trained   : {len(ep)}")
    print(f"   best val MSE     : {best_val:.4f}  @ epoch {best_i + 1}")
    print(f"   val−train @ best : {best_gap:+.4f}")
    print(f"   final train MSE  : {final_train:.4f}")
    print(f"   final val MSE    : {final_val:.4f}")
    print(f"   final val−train  : {final_gap:+.4f}")
    print(f"   final val L1     : {h['val_l1'][-1]:.4f}")
    print(f"   saved: {out}\n")
    return True


def main():
    n_ok = 0
    for label, run_dir in RUNS.items():
        if plot_one_run(label, run_dir):
            n_ok += 1
    if n_ok == 0:
        print("  nothing plotted — no checkpoints with history found.")


if __name__ == '__main__':
    main()
