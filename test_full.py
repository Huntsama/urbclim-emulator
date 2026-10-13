#!/usr/bin/env python
# coding: utf-8
"""
test_full.py — simple evaluation of the v2 Residual U-Net.

Two phases:
  - GENERAL          : random hours across all months / hours-of-day,
                       one 3-panel PNG per sampled (city, time).
  - UHI CLIMATOLOGY  : every JJA 22:00 UTC hour, averaged per city.
                       One 3-panel PNG per test city showing the long-term
                       summer-night mean.

Each PNG has three panels:
  | True UrbClim T (°C) | Predicted T (°C) | |Pred − Truth| (°C) |

Plus per-sample / per-city metric CSVs.

Edit the CONFIG block below to change the checkpoint path, the number of
random samples per city, or the output directory. No CLI arguments.
"""
import os
import datetime
import calendar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v2_residual_unet import (
    UrbClimUNet,
    MY_DATA_DIR, ERA5_ZARR_PATH, OROG_PATH, SPLITS_CSV,
)
from pipeline.config import TEST_YEARS, TEST_STATS
from test_model import TestingUrbClimDataset


# ============================================================ CONFIG ==========
# Point CHECKPOINT_PATH at whichever model you want to test:
#   .../urbclim_v2_models/best_model.pth        → the 23-channel baseline
#   .../urbclim_v2_models_solar/best_model.pth  → the 25-channel solar variant
# That is the ONLY line you change between the two test runs. The channel count
# is auto-detected from the checkpoint, and the results are written to a
# per-checkpoint subfolder (named after the checkpoint's parent directory) so
# the two test runs never overwrite each other.
CHECKPOINT_PATH = os.path.join(
    os.environ.get('VSC_SCRATCH', MY_DATA_DIR),
    "urbclim_v2_models", "best_model.pth",
)
RUN_TAG                  = os.path.basename(os.path.dirname(CHECKPOINT_PATH))   # e.g. "urbclim_v2_models"
OUTPUT_ROOT              = os.path.join(MY_DATA_DIR, "test_full", RUN_TAG)
GENERAL_SAMPLES_PER_CITY = 24
SEED                     = 0

# Mostly leave these alone.
UHI_HOUR        = 22                  # canonical UHI-peak hour, UTC
SUMMER_MONTHS   = (6, 7, 8)
BATCH_SIZE      = 1
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================================


# ----------------------------------------------------------------- helpers ---

def autodetect_in_ch(checkpoint_path):
    """Read the checkpoint's first-conv weight to learn how many input
    channels the model was trained with. Eliminates the 23-vs-25 mismatch."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    w = state['inc.conv_block.0.weight']
    return int(w.shape[1])


def build_general_indices(num_cities, samples_per_city, seed):
    """Random hours across both test years; same hours used for every city."""
    rs = np.random.RandomState(seed)
    HPY = TestingUrbClimDataset.HOURS_PER_YEAR
    selected = []
    for _ in range(samples_per_city):
        y_ord = int(rs.randint(0, len(TEST_YEARS)))
        h     = int(rs.randint(0, HPY))
        dt    = datetime.datetime(TEST_YEARS[y_ord], 1, 1) + datetime.timedelta(hours=h)
        # UrbClim's valid window starts at 2008-01-01 07:00
        if dt < datetime.datetime(2008, 1, 1, 7):
            dt = datetime.datetime(2008, 1, 1, 7); h = 7; y_ord = 0
        selected.append((y_ord, h, dt))
    indices = []
    for (y_ord, h, _) in selected:
        city_hour_idx = y_ord * HPY + h
        for city_id in range(num_cities):
            indices.append(city_hour_idx * num_cities + city_id)
    return indices, selected


def build_uhi_indices(num_cities):
    """Every JJA day at 22:00 UTC across both test years (no sampling)."""
    HPY = TestingUrbClimDataset.HOURS_PER_YEAR
    selected = []
    for y_ord, year in enumerate(TEST_YEARS):
        start = datetime.datetime(year, 1, 1)
        for h in range(HPY):
            dt = start + datetime.timedelta(hours=h)
            if dt.month in SUMMER_MONTHS and dt.hour == UHI_HOUR:
                selected.append((y_ord, h, dt))
    indices = []
    for (y_ord, h, _) in selected:
        city_hour_idx = y_ord * HPY + h
        for city_id in range(num_cities):
            indices.append(city_hour_idx * num_cities + city_id)
    return indices, selected


def metrics(truth, pred):
    """RMSE / MAE / bias on flat arrays."""
    truth = np.asarray(truth, dtype=np.float64)
    pred  = np.asarray(pred,  dtype=np.float64)
    err   = pred - truth
    return {
        'n':    int(truth.size),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae':  float(np.mean(np.abs(err))),
        'bias': float(err.mean()),
    }


# ----------------------------------------------------------------- plotting --

def save_3panel_plot(out_path, city, exact_time, truth_c, pred_c):
    """3-panel diagnostic — truth, prediction, absolute error.

    Truth and Pred share a colormap range so they're directly comparable.
    The error panel uses a single-direction red scale (white = no error,
    deep red = large error)."""
    temp_min = float(np.nanmin([truth_c, pred_c]))
    temp_max = float(np.nanmax([truth_c, pred_c]))
    abs_err  = np.abs(pred_c - truth_c)
    err_max  = float(np.nanmax(abs_err)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{city} — {exact_time}', fontsize=16, fontweight='bold')

    im0 = axes[0].imshow(truth_c, cmap='coolwarm',
                         vmin=temp_min, vmax=temp_max, origin='lower')
    axes[0].set_title('True UrbClim Temp (°C)'); axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='°C')

    im1 = axes[1].imshow(pred_c, cmap='coolwarm',
                         vmin=temp_min, vmax=temp_max, origin='lower')
    axes[1].set_title('Predicted Temp (°C)'); axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='°C')

    im2 = axes[2].imshow(abs_err, cmap='Reds',
                         vmin=0, vmax=err_max, origin='lower')
    axes[2].set_title('|Pred − Truth| (°C)'); axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='°C')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------- evaluation phases ---

def run_general_phase(model, dataset, indices, output_dir, in_ch):
    """Per-sample evaluation — one PNG and one CSV row per (city, time).

    `in_ch` is the channel count the *checkpoint* expects; we slice
    batch["input"] down to that, which lets older 23-channel models run
    against the newer 25-channel dataset (extra solar features are dropped).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'─' * 64}\nPhase: GENERAL   (samples = {len(indices)})\n{'─' * 64}")

    loader = DataLoader(
        Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    mse_fn = nn.MSELoss()
    l1_fn  = nn.L1Loss()
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs  = batch["input"][:, :in_ch].to(DEVICE)
            tgt_res = batch["target"].to(DEVICE)
            mask    = batch["mask"].to(DEVICE) > 0.5
            base_t  = batch["base_temp"].to(DEVICE)

            pred_res = model(inputs)

            h = batch['orig_h'][0].item()
            w = batch['orig_w'][0].item()
            pr = pred_res[:, :, :h, :w]
            tr = tgt_res [:, :, :h, :w]
            mk = mask    [:, :, :h, :w]
            bt = base_t  [:, :, :h, :w]

            true_abs = (tr + bt)[0, 0]
            pred_abs = (pr + bt)[0, 0]
            msk_2d   = mk[0, 0]

            city  = batch['city_name'][0]
            time_ = batch['exact_time'][0]

            if msk_2d.sum() == 0:
                print(f"  [{i+1:>4}/{len(indices)}] {city:<22} {time_}  — 0 valid px, skip")
                continue

            mse = mse_fn(pred_abs[msk_2d], true_abs[msk_2d]).item()
            mae = l1_fn (pred_abs[msk_2d], true_abs[msk_2d]).item()
            print(f"  [{i+1:>4}/{len(indices)}] {city:<22} {time_}  "
                  f"MAE={mae:.3f} °C   MSE={mse:.3f} °C²")

            rows.append({
                'city': city, 'time': time_,
                'mae_C': mae, 'mse_C2': mse,
                'valid_px': int(msk_2d.sum().item()),
            })

            true_c = true_abs.cpu().numpy() - 273.15
            pred_c = pred_abs.cpu().numpy() - 273.15
            msk_np = msk_2d.cpu().numpy().astype(bool)
            true_plot = np.where(msk_np, true_c, np.nan)
            pred_plot = np.where(msk_np, pred_c, np.nan)

            safe_time = time_.replace(':', '').replace(' ', 'T')[:13]
            save_3panel_plot(
                os.path.join(output_dir, f"eval_{city}_{safe_time}.png"),
                city, time_, true_plot, pred_plot,
            )

    if not rows:
        print("  no valid samples in general phase.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "per_sample_metrics.csv"), index=False)
    return df


def run_uhi_climatology(model, dataset, indices, output_dir, in_ch):
    """Accumulate per-city time-mean fields over every JJA 22:00 UTC hour;
    save one PNG per city + per-city metrics CSV.

    `in_ch` slices batch["input"] to the checkpoint's expected width."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'─' * 64}\nPhase: UHI CLIMATOLOGY   (forward passes = {len(indices)})\n{'─' * 64}")

    loader = DataLoader(
        Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    # accum[city] -> float64 sums of truth / pred + count
    accum = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs  = batch["input"][:, :in_ch].to(DEVICE)
            tgt_res = batch["target"].to(DEVICE)
            mask    = batch["mask"].to(DEVICE) > 0.5
            base_t  = batch["base_temp"].to(DEVICE)

            pred_res = model(inputs)

            h = batch['orig_h'][0].item()
            w = batch['orig_w'][0].item()
            pr = pred_res[:, :, :h, :w]
            tr = tgt_res [:, :, :h, :w]
            mk = mask    [:, :, :h, :w]
            bt = base_t  [:, :, :h, :w]

            true_abs = (tr + bt)[0, 0]
            pred_abs = (pr + bt)[0, 0]
            msk_2d   = mk[0, 0]
            city     = batch['city_name'][0]

            if msk_2d.sum() == 0:
                continue

            true_c = true_abs.cpu().numpy() - 273.15
            pred_c = pred_abs.cpu().numpy() - 273.15
            msk_np = msk_2d.cpu().numpy().astype(bool)

            if city not in accum:
                accum[city] = {
                    'sum_truth': np.zeros((h, w), dtype=np.float64),
                    'sum_pred':  np.zeros((h, w), dtype=np.float64),
                    'count':     np.zeros((h, w), dtype=np.int32),
                }
            a = accum[city]
            a['sum_truth'] += np.where(msk_np, true_c, 0.0)
            a['sum_pred']  += np.where(msk_np, pred_c, 0.0)
            a['count']     += msk_np.astype(np.int32)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1:>5}/{len(indices)}]  {city}")

    rows = []
    for city, a in accum.items():
        cnt   = np.maximum(a['count'], 1)
        valid = a['count'] > 0
        m_true = a['sum_truth'] / cnt
        m_pred = a['sum_pred']  / cnt

        true_plot = np.where(valid, m_true, np.nan)
        pred_plot = np.where(valid, m_pred, np.nan)

        save_3panel_plot(
            os.path.join(output_dir, f"climatology_{city}.png"),
            city, "JJA 22:00 UTC climatology",
            true_plot, pred_plot,
        )

        m = metrics(m_true[valid], m_pred[valid])
        m['city'] = city
        rows.append(m)
        print(f"  {city:<22}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  "
              f"bias={m['bias']:+.3f}  n_px={m['n']:,}")

    if not rows:
        print("  no valid cities accumulated.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)[['city', 'n', 'rmse', 'mae', 'bias']]
    df.to_csv(os.path.join(output_dir, "per_city_metrics.csv"), index=False)
    print(f"\n  saved {len(rows)} per-city climatology PNGs")
    return df


# ========================================================================== ==

def main():
    print(f"\n{'=' * 64}")
    print(f"test_full.py — simple evaluation")
    print(f"  checkpoint : {CHECKPOINT_PATH}")
    print(f"  output     : {OUTPUT_ROOT}")
    print(f"  device     : {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT_PATH}")

    in_ch = autodetect_in_ch(CHECKPOINT_PATH)
    print(f"  in_ch      : {in_ch} (auto-detected from checkpoint)")
    print(f"{'=' * 64}")

    # ---- load model
    model = UrbClimUNet(in_ch=in_ch, out_ch=1).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    print(f"  loaded checkpoint (epoch={epoch})")

    # ---- dataset
    splits = pd.read_csv(SPLITS_CSV)
    test_cities = splits.loc[splits['split'] == 'test', 'city_name'].tolist()
    num_cities  = len(test_cities)
    print(f"  test cities ({num_cities}): {', '.join(test_cities)}")

    test_dataset = TestingUrbClimDataset(
        csv_file=SPLITS_CSV,
        mode='test',
        era5_path=ERA5_ZARR_PATH,
        orog_path=OROG_PATH,
        stats_csv_path=TEST_STATS,
    )

    out_general = os.path.join(OUTPUT_ROOT, "general")
    out_uhi     = os.path.join(OUTPUT_ROOT, "uhi_22h")

    # ---- Phase 1: general random sampling
    g_idx, g_sel = build_general_indices(num_cities, GENERAL_SAMPLES_PER_CITY, SEED)
    print(f"\n  general phase: {len(g_sel)} hours × {num_cities} cities "
          f"= {len(g_idx)} samples")
    g_df = run_general_phase(model, test_dataset, g_idx, out_general, in_ch)

    # ---- Phase 2: JJA 22:00 climatology
    u_idx, u_sel = build_uhi_indices(num_cities)
    print(f"\n  uhi phase    : {len(u_sel)} hours × {num_cities} cities "
          f"= {len(u_idx)} forward passes  (JJA, {UHI_HOUR:02d}:00 UTC)")
    u_df = run_uhi_climatology(model, test_dataset, u_idx, out_uhi, in_ch)

    # ---- Final aggregate
    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    if not g_df.empty:
        print(f"  GENERAL    n_samples = {len(g_df):>4}  "
              f"mean MAE = {g_df['mae_C'].mean():.3f} °C  "
              f"mean MSE = {g_df['mse_C2'].mean():.3f} °C²")
    if not u_df.empty:
        print(f"  UHI climatology  cities = {len(u_df):>2}  "
              f"mean RMSE = {u_df['rmse'].mean():.3f} °C  "
              f"mean MAE = {u_df['mae'].mean():.3f} °C  "
              f"mean bias = {u_df['bias'].mean():+.3f} °C")
    print(f"\n  outputs:")
    print(f"    {out_general}")
    print(f"    {out_uhi}")


if __name__ == '__main__':
    main()
