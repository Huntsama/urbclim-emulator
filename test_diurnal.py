#!/usr/bin/env python
# coding: utf-8
"""
test_diurnal.py — full diurnal cycle, one "best day" per city.

Defense demo: for Antwerp plus a few randomly sampled test cities, find each
city's hottest summer day in the REAL UrbClim data (scan every June/July/
August hour of the test years, take the day whose spatial-mean temperature
peaks highest), then run the v2 Residual U-Net on every hour (00–23 UTC) of
that day and show how the CNN prediction tracks the UrbClim reference
through the day.

Outputs (one subfolder per city under OUTPUT_DIR):
  - 24 three-panel PNGs, one per hour:
      | UrbClim reference (°C) | CNN prediction (°C) | |Prediction − Reference| (°C) |
    All 24 share ONE color scale so the diurnal change is visible across frames.
  - diurnal_grid_ref_<City>_<date>.png   4×6 grid of the 24 reference maps
  - diurnal_grid_pred_<City>_<date>.png  4×6 grid of the 24 predicted maps
  - diurnal_summary_<City>_<date>.png    city-mean curve, UHI-intensity curve,
                                         per-hour RMSE
  - diurnal_metrics_<City>_<date>.csv    per-hour numbers behind the plots
  - diurnal_summary_<City>_<date>.txt    text companion written from the
                                         actual computed values
  - diurnal_<City>_<date>.gif            animated day, 3-panel diagnostic
  - diurnal_forecast_<City>_<date>.gif   forecast-style loop: reference next
                                         to prediction, big hour stamp, no
                                         error panel (frames kept in
                                         forecast_frames/ for slides)
    (GIFs need Pillow; skipped gracefully if unavailable)
  - diurnal_day_summaries.csv            one row per city, at OUTPUT_DIR root

Edit the CONFIG block below. No CLI arguments. Run on VSC:
    python -u test_diurnal.py
"""
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v2_residual_unet import (
    UrbClimUNet,
    MY_DATA_DIR, BASE_DIR, ERA5_ZARR_PATH, OROG_PATH, SPLITS_CSV,
)
from pipeline.config import TEST_YEARS, TEST_STATS
from test_model import TestingUrbClimDataset


# ============================================================ CONFIG ==========
CHECKPOINT_PATH = os.path.join(
    os.environ.get('VSC_SCRATCH', MY_DATA_DIR),
    "urbclim_v2_models", "best_model.pth",
)
RUN_TAG  = os.path.basename(os.path.dirname(CHECKPOINT_PATH))

# Cities: ALWAYS_INCLUDE are run first, then NUM_RANDOM_EXTRA more test
# cities are drawn at random (seeded, so reruns pick the same ones).
ALWAYS_INCLUDE   = ["Antwerp"]
NUM_RANDOM_EXTRA = 2
CITY_SEED        = 0

# Each city gets its own best day, found in the real UrbClim zarr: the
# summer (June, July, August) day of the test years whose spatial-mean
# temperature peaks highest. Set to False to use FIXED_DATE for every city.
AUTO_PICK_HOTTEST_DAY = True
FIXED_DATE            = "2017-06-21"   # only used if AUTO_PICK... is False

OUTPUT_DIR = os.path.join(MY_DATA_DIR, "test_diurnal", RUN_TAG)
BATCH_SIZE = 1              # must be 1 — dynamic per-city padding
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Map colorbar limits (°C). Set FIXED_TEMP_LIMITS to a (vmin, vmax) tuple to
# pin the temperature colorbar to the same values on every frame, day and city
# (defence-friendly: the scale never moves). Set to None to fall back to the
# per-day data min/max. FIXED_ERR_MAX does the same for the error panel.
FIXED_TEMP_LIMITS = (10.0, 36.0)
FIXED_ERR_MAX     = 4.0

# CORINE classes (1..15, see CORINE_CLASS_MAP) used for the UHI-intensity
# curve: urban = Urban + Sub-urban, rural = vegetated natural classes.
URBAN_CLASSES = (1, 2)
RURAL_CLASSES = (7, 8, 9, 10, 11, 12)

# Channel index of the raw CORINE plane inside batch["input"]
# (13 ERA5 planes + elev + ahf + land_mask → index 16).
CORINE_CH = 16
# ==============================================================================


def autodetect_in_ch(checkpoint_path):
    """Read first-conv weight width so 23- and 25-channel checkpoints both run."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    return int(state['inc.conv_block.0.weight'].shape[1])


def pick_hottest_day(city_row):
    """Scan summer (June, July, August) of the test years in the city's
    UrbClim zarr; return the date whose spatial-mean hourly temperature
    reaches the highest value."""
    import xarray as xr
    target_path = city_row.get(
        'target_path',
        os.path.join(BASE_DIR, f"urbclim_zarr/urbclim_tas_{city_row['city_name']}.zarr"),
    )
    ds = xr.open_zarr(target_path)
    tas = ds['tas']
    tas = tas.sel(time=tas['time'].dt.year.isin(list(TEST_YEARS)))
    tas = tas.sel(time=tas['time'].dt.month.isin([6, 7, 8]))
    spatial_dims = [d for d in tas.dims if d != 'time']
    hourly_mean = tas.mean(dim=spatial_dims).compute()
    peak = hourly_mean.idxmax('time').values
    day = pd.Timestamp(peak).date()
    peak_c = float(hourly_mean.max()) - 273.15
    print(f"  hottest summer hour in test years: {pd.Timestamp(peak)} "
          f"(spatial mean {peak_c:.1f} °C) → using {day}")
    return str(day)


def build_day_indices(date, num_cities, city_id):
    """Dataset indices for hours 00..23 UTC of `date` for one city."""
    if date.year not in TEST_YEARS:
        raise ValueError(f"{date} is not in TEST_YEARS {TEST_YEARS}")
    y_ord = list(TEST_YEARS).index(date.year)
    h0 = int((datetime.datetime(date.year, date.month, date.day)
              - datetime.datetime(date.year, 1, 1)).total_seconds() // 3600)
    HPY = TestingUrbClimDataset.HOURS_PER_YEAR
    return [((y_ord * HPY + h0 + hr) * num_cities + city_id) for hr in range(24)]


def save_hour_plot(out_path, city, time_str, ref_c, pred_c,
                   vmin, vmax, err_max):
    """3-panel frame. Color limits are passed in (shared across all 24 h)."""
    abs_err = np.abs(pred_c - ref_c)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{city} — {time_str} UTC', fontsize=16, fontweight='bold')

    im0 = axes[0].imshow(ref_c, cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('UrbClim reference (°C)'); axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='°C')

    im1 = axes[1].imshow(pred_c, cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('CNN prediction (°C)'); axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='°C')

    im2 = axes[2].imshow(abs_err, cmap='Reds', vmin=0, vmax=err_max, origin='lower')
    axes[2].set_title('|Prediction − Reference| (°C)'); axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='°C')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_forecast_frame(out_path, city, date, hr, ref_c, pred_c, vmin, vmax):
    """Clean 2-panel frame (no error panel) for the forecast-style GIF."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'{city} — {date} — {hr:02d}:00 UTC',
                 fontsize=20, fontweight='bold')

    im0 = axes[0].imshow(ref_c, cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('UrbClim reference', fontsize=14); axes[0].axis('off')

    im1 = axes[1].imshow(pred_c, cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('CNN prediction', fontsize=14); axes[1].axis('off')

    fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02, label='°C')
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def save_grid_plot(out_path, title, fields, hours, vmin, vmax):
    """4×6 grid of 24 hourly maps with one shared colorbar."""
    fig, axes = plt.subplots(4, 6, figsize=(22, 14))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    im = None
    for ax, field, hr in zip(axes.flat, fields, hours):
        im = ax.imshow(field, cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'{hr:02d}:00 UTC', fontsize=10)
        ax.axis('off')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='°C')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_summary_plot(out_path, city, date_str, df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{city} — {date_str} — diurnal cycle (UTC)',
                 fontsize=15, fontweight='bold')
    hrs = df['hour']

    axes[0].plot(hrs, df['mean_ref_C'], 'o-', color='black', label='UrbClim reference')
    axes[0].plot(hrs, df['mean_pred_C'], 's--', color='tab:red', label='CNN prediction')
    axes[0].set_xlabel('hour (UTC)'); axes[0].set_ylabel('city-mean T (°C)')
    axes[0].set_title('City-mean temperature'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hrs, df['uhi_ref_C'], 'o-', color='black', label='UrbClim reference')
    axes[1].plot(hrs, df['uhi_pred_C'], 's--', color='tab:red', label='CNN prediction')
    axes[1].set_xlabel('hour (UTC)'); axes[1].set_ylabel('urban − rural T (°C)')
    axes[1].set_title('UHI intensity'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].bar(hrs, df['rmse_C'], color='steelblue')
    axes[2].set_xlabel('hour (UTC)'); axes[2].set_ylabel('RMSE (°C)')
    axes[2].set_title('Per-hour RMSE'); axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_text_companion(out_path, city, date_str, df):
    """Plain-language summary built from the actual per-hour numbers."""
    best  = df.loc[df['rmse_C'].idxmin()]
    worst = df.loc[df['rmse_C'].idxmax()]
    pk_r  = df.loc[df['uhi_ref_C'].idxmax()]
    pk_p  = df.loc[df['uhi_pred_C'].idxmax()]
    lines = [
        f"Diurnal cycle for {city} on {date_str} (all hours UTC).",
        "",
        f"Day-mean RMSE: {df['rmse_C'].mean():.2f} °C. "
        f"Day-mean MAE: {df['mae_C'].mean():.2f} °C. "
        f"Day-mean bias: {df['bias_C'].mean():+.2f} °C.",
        f"Best hour: {int(best['hour']):02d}:00 (RMSE {best['rmse_C']:.2f} °C). "
        f"Worst hour: {int(worst['hour']):02d}:00 (RMSE {worst['rmse_C']:.2f} °C).",
        "",
        f"City-mean temperature in the reference ranges from "
        f"{df['mean_ref_C'].min():.1f} °C (hour "
        f"{int(df.loc[df['mean_ref_C'].idxmin(), 'hour']):02d}:00) to "
        f"{df['mean_ref_C'].max():.1f} °C (hour "
        f"{int(df.loc[df['mean_ref_C'].idxmax(), 'hour']):02d}:00). "
        f"The CNN ranges from {df['mean_pred_C'].min():.1f} °C to "
        f"{df['mean_pred_C'].max():.1f} °C.",
        "",
        f"UHI intensity (urban minus rural pixels) peaks at "
        f"{pk_r['uhi_ref_C']:.2f} °C at {int(pk_r['hour']):02d}:00 in the reference "
        f"and at {pk_p['uhi_pred_C']:.2f} °C at {int(pk_p['hour']):02d}:00 in the "
        f"CNN prediction.",
        "",
        "Per-hour table:",
        df.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


def try_make_gif(png_paths, out_path, ms_per_frame=500):
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not available — skipping GIF.")
        return
    frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in png_paths]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=ms_per_frame, loop=0)
    print(f"  GIF → {out_path}")


# ==============================================================================

def run_city(model, dataset, cities, city_name, in_ch):
    """Find the city's best day in the real UrbClim data, run all 24 hours,
    write plots / CSV / text companion / GIF. Returns a day-summary dict."""
    city_id  = cities.index(city_name)
    city_row = dataset.df.iloc[city_id]

    print(f"\n{'─' * 64}\nCity: {city_name}\n{'─' * 64}")
    date_str = pick_hottest_day(city_row) if AUTO_PICK_HOTTEST_DAY else FIXED_DATE
    date = datetime.date.fromisoformat(date_str)
    print(f"  day: {date}  (UTC)")

    out_dir = os.path.join(OUTPUT_DIR, f"{city_name}_{date}")
    os.makedirs(out_dir, exist_ok=True)

    indices = build_day_indices(date, len(cities), city_id)
    loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=True)

    # ---- pass 1: run all 24 hours, keep fields in memory -----------------
    refs, preds, masks, times, corine = [], [], [], [], None
    with torch.no_grad():
        for batch in loader:
            inputs  = batch["input"][:, :in_ch].to(DEVICE)
            pred_res = model(inputs)

            h = batch['orig_h'][0].item()
            w = batch['orig_w'][0].item()
            bt = batch["base_temp"].to(DEVICE)[:, :, :h, :w]
            tr = batch["target"].to(DEVICE)[:, :, :h, :w]
            mk = (batch["mask"].to(DEVICE) > 0.5)[:, :, :h, :w]
            pr = pred_res[:, :, :h, :w]

            ref_c  = ((tr + bt)[0, 0]).cpu().numpy() - 273.15
            pred_c = ((pr + bt)[0, 0]).cpu().numpy() - 273.15
            msk    = mk[0, 0].cpu().numpy().astype(bool)

            refs.append(np.where(msk, ref_c, np.nan))
            preds.append(np.where(msk, pred_c, np.nan))
            masks.append(msk)
            times.append(batch['exact_time'][0])
            if corine is None:
                corine = batch["input"][0, CORINE_CH, :h, :w].numpy()
            print(f"  ran {batch['exact_time'][0]}")

    if len(refs) != 24:
        print(f"  WARNING: expected 24 hours, got {len(refs)}")

    # ---- color scale: fixed if configured, else per-day data min/max -----
    if FIXED_TEMP_LIMITS is not None:
        vmin, vmax = FIXED_TEMP_LIMITS
    else:
        vmin = float(np.nanmin([np.nanmin(a) for a in refs + preds]))
        vmax = float(np.nanmax([np.nanmax(a) for a in refs + preds]))
    err_max = FIXED_ERR_MAX if FIXED_ERR_MAX is not None else (
        float(np.nanmax([np.nanmax(np.abs(p - r))
                         for p, r in zip(preds, refs)])) or 1.0)
    print(f"\n  shared scale: {vmin:.1f} … {vmax:.1f} °C, err max {err_max:.1f} °C")

    urban_px = np.isin(corine, URBAN_CLASSES)
    rural_px = np.isin(corine, RURAL_CLASSES)
    print(f"  urban px: {int(urban_px.sum()):,}   rural px: {int(rural_px.sum()):,}")

    # ---- pass 2: plots + metrics ------------------------------------------
    frames_dir = os.path.join(out_dir, "forecast_frames")
    os.makedirs(frames_dir, exist_ok=True)
    rows, png_paths, frame_paths = [], [], []
    for hr, (ref_c, pred_c, msk, t) in enumerate(zip(refs, preds, masks, times)):
        png = os.path.join(out_dir, f"diurnal_{city_name}_{date}_{hr:02d}UTC.png")
        save_hour_plot(png, city_name, t, ref_c, pred_c, vmin, vmax, err_max)
        png_paths.append(png)

        frame = os.path.join(frames_dir, f"frame_{hr:02d}.png")
        save_forecast_frame(frame, city_name, date, hr, ref_c, pred_c, vmin, vmax)
        frame_paths.append(frame)

        err = (pred_c - ref_c)[msk]
        u, r = msk & urban_px, msk & rural_px
        rows.append({
            'hour': hr,
            'mean_ref_C':  float(np.nanmean(ref_c[msk])),
            'mean_pred_C': float(np.nanmean(pred_c[msk])),
            'uhi_ref_C':   float(np.nanmean(ref_c[u]) - np.nanmean(ref_c[r])),
            'uhi_pred_C':  float(np.nanmean(pred_c[u]) - np.nanmean(pred_c[r])),
            'rmse_C':      float(np.sqrt(np.mean(err ** 2))),
            'mae_C':       float(np.mean(np.abs(err))),
            'bias_C':      float(err.mean()),
            'valid_px':    int(msk.sum()),
        })
        print(f"  {hr:02d}:00  ref {rows[-1]['mean_ref_C']:5.1f} °C  "
              f"pred {rows[-1]['mean_pred_C']:5.1f} °C  "
              f"RMSE {rows[-1]['rmse_C']:.2f}  "
              f"UHI ref {rows[-1]['uhi_ref_C']:+.2f} / pred {rows[-1]['uhi_pred_C']:+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"diurnal_metrics_{city_name}_{date}.csv"), index=False)

    save_grid_plot(os.path.join(out_dir, f"diurnal_grid_ref_{city_name}_{date}.png"),
                   f'{city_name} — {date} — UrbClim reference (°C)',
                   refs, range(24), vmin, vmax)
    save_grid_plot(os.path.join(out_dir, f"diurnal_grid_pred_{city_name}_{date}.png"),
                   f'{city_name} — {date} — CNN prediction (°C)',
                   preds, range(24), vmin, vmax)
    save_summary_plot(os.path.join(out_dir, f"diurnal_summary_{city_name}_{date}.png"),
                      city_name, str(date), df)
    write_text_companion(os.path.join(out_dir, f"diurnal_summary_{city_name}_{date}.txt"),
                         city_name, str(date), df)
    # 3-panel diagnostic GIF + clean forecast-style GIF (ref | prediction)
    try_make_gif(png_paths,
                 os.path.join(out_dir, f"diurnal_{city_name}_{date}.gif"))
    try_make_gif(frame_paths,
                 os.path.join(out_dir, f"diurnal_forecast_{city_name}_{date}.gif"),
                 ms_per_frame=400)

    summary = {
        'city': city_name, 'date': str(date),
        'rmse_C': df['rmse_C'].mean(), 'mae_C': df['mae_C'].mean(),
        'bias_C': df['bias_C'].mean(),
        'peak_uhi_ref_C': df['uhi_ref_C'].max(),
        'peak_uhi_pred_C': df['uhi_pred_C'].max(),
        'out_dir': out_dir,
    }
    print(f"  {city_name} {date}  day-mean RMSE {summary['rmse_C']:.3f} °C   "
          f"MAE {summary['mae_C']:.3f} °C   bias {summary['bias_C']:+.3f} °C")
    return summary


def main():
    print(f"\n{'=' * 64}")
    print("test_diurnal.py — best day per city, all 24 hours")
    print(f"  checkpoint : {CHECKPOINT_PATH}")
    print(f"  device     : {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT_PATH}")
    in_ch = autodetect_in_ch(CHECKPOINT_PATH)
    print(f"  in_ch      : {in_ch} (auto-detected)")

    model = UrbClimUNet(in_ch=in_ch, out_ch=1).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = TestingUrbClimDataset(
        csv_file=SPLITS_CSV, mode='test',
        era5_path=ERA5_ZARR_PATH, orog_path=OROG_PATH,
        stats_csv_path=TEST_STATS,
    )
    cities = dataset.df['city_name'].tolist()

    missing = [c for c in ALWAYS_INCLUDE if c not in cities]
    if missing:
        raise ValueError(f"{missing} not in test split. Test cities: {cities}")
    pool = [c for c in cities if c not in ALWAYS_INCLUDE]
    rs = np.random.RandomState(CITY_SEED)
    extra = list(rs.choice(pool, size=min(NUM_RANDOM_EXTRA, len(pool)),
                           replace=False)) if NUM_RANDOM_EXTRA else []
    chosen = list(ALWAYS_INCLUDE) + extra
    print(f"  cities     : {', '.join(chosen)} "
          f"({len(extra)} drawn at random with seed {CITY_SEED})")
    print(f"{'=' * 64}")

    summaries = [run_city(model, dataset, cities, c, in_ch) for c in chosen]

    print(f"\n{'=' * 64}\nSUMMARY (one best day per city)\n{'=' * 64}")
    for s in summaries:
        print(f"  {s['city']:<22} {s['date']}  RMSE {s['rmse_C']:.3f} °C  "
              f"MAE {s['mae_C']:.3f} °C  bias {s['bias_C']:+.3f} °C  "
              f"peak UHI ref {s['peak_uhi_ref_C']:+.2f} / pred {s['peak_uhi_pred_C']:+.2f} °C")
    pd.DataFrame(summaries).to_csv(
        os.path.join(OUTPUT_DIR, "diurnal_day_summaries.csv"), index=False)
    print(f"\n  outputs → {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
