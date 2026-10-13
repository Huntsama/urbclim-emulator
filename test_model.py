#!/usr/bin/env python
# coding: utf-8

"""
==============================================================================
V2 RESIDUAL U-NET EVALUATION SCRIPT
==============================================================================

Loads the best checkpoint of the v2 Residual U-Net and evaluates it on the
unseen test split. For each curated sample:

  - predicts the UHI residual,
  - adds the lapse-rate-corrected base T2M back to recover absolute
    temperature in °C,
  - computes per-sample MSE / MAE on the masked (land + non-sea CORINE) area,
  - saves a 4-panel diagnostic figure (land/water, truth, prediction, error)
    with origin='lower' so north is up — the old plots were flipped because
    imshow defaults to origin='upper'.

One random hour from each TEST_YEAR per city is sampled. All prep (splits,
stats, time sync) now comes from pipeline/, matching the training
dataloader byte-for-byte.

Run on VSC:
    python -u test_model.py
==============================================================================
"""
import os
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v2_residual_unet import (
    UrbClimUNet, UrbClimDataset,
    MY_DATA_DIR, BASE_DIR, ERA5_ZARR_PATH, OROG_PATH, SPLITS_CSV,
    LAPSE_RATE, GRAVITY, CORINE_CLASS_MAP,
)
from pipeline.config import TEST_YEARS, target_datetime, TEST_STATS

BATCH_SIZE      = 1   # must be 1 — cities have different native shapes
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR        = os.path.join(os.environ.get('VSC_SCRATCH', MY_DATA_DIR), "urbclim_v2_models")
CHECKPOINT_PATH = os.path.join(CKPT_DIR, "best_model.pth")
OUTPUT_DIR      = os.path.join(MY_DATA_DIR, "test_eval_outputs")
METRICS_CSV     = os.path.join(OUTPUT_DIR, "per_sample_metrics.csv")


# ==============================================================================
# Testing dataset — one sample = (city, exact hour of exact year), dynamic pad.
# ==============================================================================

class TestingUrbClimDataset(UrbClimDataset):
    """
    Indexing convention for reproducible testing:
        city_hour_idx = idx // num_cities
        actual_idx    = idx % num_cities

    city_hour_idx runs through every hour of every TEST_YEAR in order. Two
    TEST_YEARS × 8760 h = 17520 hours per city. We don't enumerate all of
    them; evaluate_model() picks a deterministic subset per city.
    """

    HOURS_PER_YEAR = 8760   # we sample hour 0..8759 within a year

    def __getitem__(self, idx):
        num_cities    = len(self.df)
        city_hour_idx = idx // num_cities
        actual_idx    = idx % num_cities
        row           = self.df.iloc[actual_idx]

        real_lat = row.get('latitude', row.get('real_lat'))
        real_lon = row.get('longitude', row.get('real_lon'))
        era5_x   = int(row['era5_lon_idx'])
        era5_y   = int(row['era5_lat_idx'])

        year_idx = city_hour_idx // self.HOURS_PER_YEAR
        year     = TEST_YEARS[year_idx] if year_idx < len(TEST_YEARS) else TEST_YEARS[-1]
        hour_in_year = city_hour_idx % self.HOURS_PER_YEAR

        # Use the same target_datetime helper the training dataloader uses so
        # ERA5 and UrbClim return the exact same real hour.
        start = datetime.datetime(year, 1, 1)
        exact_time = start + datetime.timedelta(hours=int(hour_in_year))
        # Clamp into UrbClim's valid window (first 7 hours of 2008 are missing).
        urb_start = datetime.datetime(2008, 1, 1, 7)
        target_dt = exact_time if exact_time >= urb_start else urb_start

        month       = target_dt.month             # 1..12
        month0      = month - 1                   # 0..11, used only for the NDVI monthly index

        # Cyclic seasonal encoding on day-of-year (matches training dataloader).
        doy       = target_dt.timetuple().tm_yday
        month_sin = float(np.sin(2.0 * np.pi * doy / 365.25))
        month_cos = float(np.cos(2.0 * np.pi * doy / 365.25))
        # Cyclic diurnal encoding on local mean solar time (UTC + lon/15).
        lst       = (target_dt.hour + float(real_lon) / 15.0) % 24.0
        hour_sin  = float(np.sin(2.0 * np.pi * lst / 24.0))
        hour_cos  = float(np.cos(2.0 * np.pi * lst / 24.0))

        # 1. ERA5 vector
        era5_sample = (self.era5_ds['cell']
                       .sel(time=target_dt, method='nearest')
                       .isel(x=era5_x, y=era5_y)
                       .isel(variable=slice(0, 13)))
        era5_vector = np.copy(era5_sample.values)
        t2m_era5_raw = era5_vector[2]

        # 2. Static maps (from the cache built in UrbClimDataset.__init__)
        city_maps = self.static_cache[actual_idx]
        elev_raw, isa_raw, ahf_raw, land_mask, ndvi_raw, imperv_raw = (
            city_maps['elev'], city_maps['isa'], city_maps['ahf'],
            city_maps['mask'], city_maps['ndvi'], city_maps['imperv'],
        )
        if ndvi_raw.ndim == 3:
            ndvi_raw = ndvi_raw[month0] if ndvi_raw.shape[0] == 12 else ndvi_raw[0]
        elif ndvi_raw.ndim > 3:
            ndvi_raw = np.squeeze(ndvi_raw)[0]

        # 3. Target
        target_path = row.get(
            'target_path',
            os.path.join(BASE_DIR, f"urbclim_zarr/urbclim_tas_{row['city_name']}.zarr"),
        )
        if not hasattr(self, 'target_zarrs'):
            self.target_zarrs = {}
        if target_path not in self.target_zarrs:
            self.target_zarrs[target_path] = xr.open_zarr(target_path)
        t_urb_raw = np.array(
            self.target_zarrs[target_path]['tas'].sel(time=target_dt, method='nearest').values,
            dtype=np.float32,
        )
        valid_mask = ~np.isnan(t_urb_raw)
        t_urb_raw  = np.nan_to_num(t_urb_raw, nan=0.0)

        # 4. CORINE reclassify (15 classes; class 15 Sea is on the input but
        #    masked out of the loss/metrics — matches training dataloader).
        corine_reclass = np.zeros_like(isa_raw, dtype=np.float32)
        for new_class, orig_codes in CORINE_CLASS_MAP.items():
            corine_reclass[np.isin(isa_raw, orig_codes)] = new_class
        valid_mask = valid_mask & (corine_reclass > 0) & (corine_reclass != 15)

        # 5. Lapse-rate residual + base temperature for plotting
        elev_low = float(
            self.orog_global.sel(latitude=real_lat, longitude=real_lon, method='nearest').values
            / GRAVITY
        )
        t2m_era5_msl   = t2m_era5_raw + (elev_low * LAPSE_RATE)
        target_residual = ((t_urb_raw + (elev_raw * LAPSE_RATE)) - t2m_era5_msl) * valid_mask
        base_temp_crop  = t2m_era5_msl - (elev_raw * LAPSE_RATE)   # 2-D base ERA5 T in K

        # 6. Z-score
        norm_era5   = (era5_vector - self.era5_means) / (self.era5_stds + 1e-8)
        norm_elev   = (elev_raw   - self.elev_mean)  / (self.elev_std   + 1e-8)
        norm_ahf    = (ahf_raw    - self.ahf_mean)   / (self.ahf_std    + 1e-8)
        norm_ndvi   = (ndvi_raw   - self.ndvi_mean)  / (self.ndvi_std   + 1e-8)
        norm_imperv = (imperv_raw - self.imperv_mean) / (self.imperv_std + 1e-8)

        planes = [np.full_like(elev_raw, v) for v in norm_era5]

        # 7. Stack — 23-channel input (matches v2 training) + target + mask + base_temp
        full_stack = np.stack(
            planes + [
                norm_elev, norm_ahf, land_mask,          # 13, 14, 15
                corine_reclass,                          # 16
                norm_ndvi,                               # 17
                norm_imperv,                             # 18
                np.full_like(elev_raw, month_sin),       # 19  DOY sin
                np.full_like(elev_raw, month_cos),       # 20  DOY cos
                np.full_like(elev_raw, hour_sin),        # 21  LST sin
                np.full_like(elev_raw, hour_cos),        # 22  LST cos
                target_residual,                         # 23  (target)
                valid_mask.astype(np.float32),           # 24  (mask)
                base_temp_crop,                          # 25  (base T2M, eval only)
            ],
            axis=0,
        ).astype(np.float32)

        # 8. Pad to next multiple of 16 so ANY city size works with 4-level U-Net
        _, h, w = full_stack.shape
        pad_y = (16 - h % 16) % 16
        pad_x = (16 - w % 16) % 16
        padded = np.pad(
            full_stack, ((0, 0), (0, pad_y), (0, pad_x)),
            mode='constant', constant_values=0,
        )

        return {
            "input":      torch.from_numpy(padded[:23]),
            "target":     torch.from_numpy(padded[23:24]),
            "mask":       torch.from_numpy(padded[24:25]),
            "base_temp":  torch.from_numpy(padded[25:26]),
            "city_name":  row['city_name'],
            "exact_time": str(target_dt),
            "orig_h":     h,
            "orig_w":     w,
        }

    def __len__(self):
        return len(self.df) * len(TEST_YEARS) * self.HOURS_PER_YEAR


# ==============================================================================
# Plot — 3 panels (truth / prediction / error), all with origin='lower'
# ==============================================================================

def save_diagnostic_plot(out_path, city, exact_time,
                         truth_c, pred_c, err_c):
    temp_min = float(np.nanmin([truth_c, pred_c]))
    temp_max = float(np.nanmax([truth_c, pred_c]))
    valid_err = err_c[~np.isnan(err_c)]
    max_err = float(valid_err.max()) if valid_err.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'{city} — {exact_time}', fontsize=16, fontweight='bold')

    im0 = axes[0].imshow(truth_c, cmap='coolwarm', vmin=temp_min, vmax=temp_max, origin='lower')
    axes[0].set_title('True UrbClim Temp (°C)', fontsize=14); axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='°C')

    im1 = axes[1].imshow(pred_c, cmap='coolwarm', vmin=temp_min, vmax=temp_max, origin='lower')
    axes[1].set_title('Predicted Temp (°C)', fontsize=14); axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='°C')

    im2 = axes[2].imshow(err_c, cmap='magma', vmin=0, vmax=max_err, origin='lower')
    axes[2].set_title('|Truth − Predicted|  (°C)', fontsize=14); axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='°C')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(samples_per_year=1, seed=0):
    print("=" * 60)
    print(f"V2 Residual U-Net — test-set evaluation  (device: {DEVICE})")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"test years: {TEST_YEARS}")
    print(f"samples   : {samples_per_year} per city per year")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset — uses test stats (computed over TEST_YEARS on the test cities)
    test_dataset = TestingUrbClimDataset(
        csv_file=SPLITS_CSV,
        mode='test',
        era5_path=ERA5_ZARR_PATH,
        orog_path=OROG_PATH,
        stats_csv_path=TEST_STATS,
    )
    num_cities = len(test_dataset.df)
    print(f"test cities: {num_cities}  ({', '.join(test_dataset.df['city_name'].tolist())})")

    # 2. Curated indices — deterministic with `seed`
    rs = np.random.RandomState(seed)
    indices = []
    for y_ord in range(len(TEST_YEARS)):
        for city_id in range(num_cities):
            for _ in range(samples_per_year):
                h = rs.randint(0, TestingUrbClimDataset.HOURS_PER_YEAR)
                city_hour_idx = y_ord * TestingUrbClimDataset.HOURS_PER_YEAR + h
                indices.append(city_hour_idx * num_cities + city_id)

    test_loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=False,
    )
    print(f"total samples to evaluate: {len(indices)}")

    # 3. Model + checkpoint
    model = UrbClimUNet(in_ch=23, out_ch=1).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"loaded epoch {ckpt.get('epoch', '?')} model")
    model.eval()

    mse_fn = nn.MSELoss()
    l1_fn  = nn.L1Loss()

    # 4. Run
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs    = batch["input"].to(DEVICE)
            tgt_res   = batch["target"].to(DEVICE)
            mask      = batch["mask"].to(DEVICE) > 0.5
            base_t    = batch["base_temp"].to(DEVICE)

            pred_res = model(inputs)

            h = batch['orig_h'][0].item()
            w = batch['orig_w'][0].item()
            pr   = pred_res [:, :, :h, :w]
            tr   = tgt_res  [:, :, :h, :w]
            mk   = mask     [:, :, :h, :w]
            bt   = base_t   [:, :, :h, :w]

            # absolute temperature in Kelvin; then convert to Celsius
            true_abs = (tr + bt)[0, 0]
            pred_abs = (pr + bt)[0, 0]
            msk_2d   = mk[0, 0]

            if msk_2d.sum() == 0:
                print(f"  [{i+1}/{len(indices)}] {batch['city_name'][0]} — 0 valid pixels, skipping")
                continue

            mse = mse_fn(pred_abs[msk_2d], true_abs[msk_2d]).item()
            mae = l1_fn (pred_abs[msk_2d], true_abs[msk_2d]).item()

            city  = batch['city_name'][0]
            time_ = batch['exact_time'][0]
            print(f"  [{i+1:>3}/{len(indices)}] {city:<22} {time_}  "
                  f"MAE={mae:.3f} °C   MSE={mse:.3f} °C²")
            rows.append({'city': city, 'time': time_, 'mae_C': mae, 'mse_C2': mse,
                         'valid_px': int(msk_2d.sum().item())})

            # plot
            true_c = true_abs.cpu().numpy() - 273.15
            pred_c = pred_abs.cpu().numpy() - 273.15
            msk_np = msk_2d.cpu().numpy().astype(bool)
            true_c = np.where(msk_np, true_c, np.nan)
            pred_c = np.where(msk_np, pred_c, np.nan)
            err_c  = np.abs(true_c - pred_c)

            safe_time = time_.replace(':', '').replace(' ', 'T')[:13]
            fname = f"eval_{city}_{safe_time}.png"
            save_diagnostic_plot(os.path.join(OUTPUT_DIR, fname),
                                 city, time_,
                                 true_c, pred_c, err_c)

    # 5. Aggregate + per-sample CSV
    if not rows:
        print("\nno valid samples produced — check mask / paths.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 60)
    print("AGGREGATE TEST-SET METRICS  (whole-city, masked pixels)")
    print("=" * 60)
    print(f"  samples       : {len(df)}")
    print(f"  mean  MAE (°C): {df['mae_C'].mean():.3f}")
    print(f"  mean  MSE (°C²): {df['mse_C2'].mean():.3f}")
    print(f"  median MAE (°C): {df['mae_C'].median():.3f}")
    print(f"  worst MAE (°C): {df['mae_C'].max():.3f}  "
          f"({df.loc[df['mae_C'].idxmax(), 'city']})")
    print()
    print("per-city means:")
    per_city = df.groupby('city')[['mae_C', 'mse_C2']].mean().sort_values('mae_C', ascending=False)
    for city, r in per_city.iterrows():
        print(f"  {city:<22} MAE={r['mae_C']:.3f} °C   MSE={r['mse_C2']:.3f} °C²")

    # Summary plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['mae_C'], bins=20, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('per-sample MAE (°C)')
    axes[0].set_ylabel('count')
    axes[0].set_title('Test-set MAE distribution')

    per_city.sort_values('mae_C').plot.barh(y='mae_C', ax=axes[1], color='teal', legend=False)
    axes[1].set_xlabel('mean MAE (°C)')
    axes[1].set_title('Per-city mean MAE')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "_summary_histogram.png"), dpi=130)
    plt.close(fig)

    print(f"\nplots     → {OUTPUT_DIR}")
    print(f"metrics   → {METRICS_CSV}")
    print(f"summary   → {os.path.join(OUTPUT_DIR, '_summary_histogram.png')}")


if __name__ == '__main__':
    evaluate_model(samples_per_year=3, seed=0)
