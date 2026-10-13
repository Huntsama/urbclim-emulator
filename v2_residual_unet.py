#!/usr/bin/env python
# coding: utf-8

"""
==============================================================================
VERSION 2: THE RESIDUAL U-NET (Solving the Vanishing Gradient)
==============================================================================

This script physically evolves the Version 1 baseline by replacing standard 
DoubleConv blocks with Residual Convolutions (ResidualConv).

THE PROBLEM IN V1:
As we stacked deeper convolutional layers to capture long-range geographic 
patterns, the gradients (learning signals) began to "vanish" before reaching 
the earliest layers. The model struggled to propagate sharp temperature boundaries
(e.g., a cold park completely surrounded by hot concrete).

THE ENGINEERING FIX IN V2:
By wrapping our convolutional blocks in a Residual Bypass Connection ("shortcut"), 
we allow the raw spatial data to leapfrog over the convolutions and add itself 
back into the output. 
This guarantees the mathematically clean flow of gradients backward through 
the entire network, permanently preventing vanishing gradients and allowing 
the U-Net to easily learn complex, high-contrast UHI boundaries.

ARCHITECTURE NOTE:
This is pure Res-UNet. No Attention (Squeeze-and-Excitation) has been added yet.
==============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from pipeline.config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS, target_datetime

# ==============================================================================
# BLOCK 1: Imports & Configuration
# ==============================================================================
MY_DATA_DIR = "/user/brussel/113/vsc11361/data"
BASE_DIR = "/data/brussel/vo/000/bvo00029/urbclim_emulator/data/intermediate"
ERA5_ZARR_PATH = os.path.join(BASE_DIR, "era5_2008_2017_v1.zarr")
OROG_PATH = os.path.join(BASE_DIR, "era5_orography_global_WGS84.nc")

SPLITS_CSV = os.path.join(MY_DATA_DIR, "splits_metadata.csv")
TRAIN_STATS = os.path.join(MY_DATA_DIR, "train_statistics.csv")
VAL_STATS = os.path.join(MY_DATA_DIR, "val_statistics.csv")

LAPSE_RATE = 0.0065  
GRAVITY = 9.80665
BATCH_SIZE = 32       
CROP_SIZE = 64        
LEARNING_RATE = 1.5e-5
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 150

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CORINE Land Cover 2012 (100 m, EEA) — 44 raw codes reclassified to 15 classes per Jonas thesis Table 8.1.
# Used as a single integer channel with values 1..15 (not one-hot). 0 = unmapped (dropped from loss).
# Class 15 (Sea) is visible to the convolutions so they can learn coastal context, but it is excluded
# from the loss mask because UrbClim has no meaningful tas over open water.
CORINE_CLASS_MAP = {
    1:  [1],                                 # Urban
    2:  [2],                                 # Sub-urban
    3:  [3, 4, 5, 6],                        # Industrial
    4:  [10, 11],                            # Urban green
    5:  [34],                                # Snow/Ice
    6:  [8, 9, 30, 31, 38, 39],              # Bare soil
    7:  [7, 18, 26, 27, 32, 35, 36, 37],     # Grassland
    8:  [12, 13, 14, 19, 20, 21],            # Cropland
    9:  [15, 28, 29, 33],                    # Shrubland
    10: [16, 17, 22],                        # Woodland
    11: [23, 25],                            # Broadleaf trees
    12: [24],                                # Needleleaf trees
    13: [40],                                # Rivers
    14: [41],                                # Inland water bodies
    15: [42, 43, 44],                        # Sea — input only, masked in loss
}

# ==============================================================================
# BLOCK 2: The Dataloader (UrbClimDataset) - BASELINE 22 CHANNELS
# ==============================================================================
class UrbClimDataset(Dataset):
    def __init__(self, csv_file, mode, era5_path, orog_path, stats_csv_path):
        full_df = pd.read_csv(csv_file)
        self.df = full_df[full_df['split'] == mode].reset_index(drop=True)
        self.mode = mode
        self.years = {'train': TRAIN_YEARS, 'val': VAL_YEARS, 'test': TEST_YEARS}[mode]
        
        self.era5_ds = xr.open_zarr(era5_path)
        with xr.open_dataset(orog_path) as ds:
            self.orog_global = ds['orog'].load()

        stats_df = pd.read_csv(stats_csv_path)
        # ERA5 rows are 'dynamic'; names come from the zarr (u10, t2m, blh, …).
        # Row order in the CSV matches the ERA5 zarr's variable axis.
        era5_rows = stats_df[stats_df['type'] == 'dynamic'].reset_index(drop=True)
        self.era5_means = era5_rows['mean'].values.astype(np.float32)
        self.era5_stds = era5_rows['std'].values.astype(np.float32)

        self.elev_mean = stats_df[stats_df['feature'] == 'elevation']['mean'].values[0].astype(np.float32)
        self.elev_std = stats_df[stats_df['feature'] == 'elevation']['std'].values[0].astype(np.float32)
        
        self.ahf_mean = stats_df[stats_df['feature'] == 'ahf']['mean'].values[0].astype(np.float32)
        self.ahf_std = stats_df[stats_df['feature'] == 'ahf']['std'].values[0].astype(np.float32)
        
        self.ndvi_mean = stats_df[stats_df['feature'] == 'ndvi']['mean'].values[0].astype(np.float32)
        self.ndvi_std = stats_df[stats_df['feature'] == 'ndvi']['std'].values[0].astype(np.float32)

        self.imperv_mean = stats_df[stats_df['feature'] == 'imperviousness']['mean'].values[0].astype(np.float32)
        self.imperv_std  = stats_df[stats_df['feature'] == 'imperviousness']['std'].values[0].astype(np.float32)

        self.static_cache = {}
        for idx, row in self.df.iterrows():
            path_col = 'static_file_path' if 'static_file_path' in row else 'static_path'
            ndvi_path = row.get('ndvi_path', os.path.join(BASE_DIR, f"ndvi_tmp/NDVI_{row['city_name']}.nc"))
            
            with netCDF4.Dataset(row[path_col], 'r') as nc:
                elev = np.array(np.nan_to_num(nc.variables['elevation'][:], nan=0.0), dtype=np.float32)
                isa = np.array(np.nan_to_num(nc.variables['CORINE'][:], nan=0.0), dtype=np.float32)
                ahf = np.array(np.nan_to_num(nc.variables['AHF'][:], nan=0.0), dtype=np.float32)
                mask = np.array(np.nan_to_num(nc.variables['land_water_mask'][:], nan=0.0), dtype=np.float32)
                # Imperviousness — static_features_tmp_v2 stores it under 'imperv';
                # older generations used 'imperviousness' or 'IMPERV'. Raise loudly
                # if none is present — the previous silent-zero fallback masked a
                # months-long bug where this channel was constant 0.
                imperv_key = next((k for k in ('imperv', 'imperviousness', 'IMPERV')
                                   if k in nc.variables), None)
                if imperv_key is None:
                    raise KeyError(f"no imperviousness variable in {row[path_col]}; "
                                   f"found {list(nc.variables)}")
                imperv = np.array(np.nan_to_num(nc.variables[imperv_key][:], nan=0.0),
                                  dtype=np.float32)
                # Copernicus HRL imperv uses 254/255 as nodata; clamp outside [0, 100].
                imperv = np.where((imperv >= 0.0) & (imperv <= 100.0),
                                  imperv, 0.0).astype(np.float32)
            
            with netCDF4.Dataset(ndvi_path, 'r') as nc_ndvi:
                ndvi_key = 'NDVI' if 'NDVI' in nc_ndvi.variables else list(nc_ndvi.variables.keys())[-1]
                ndvi = np.array(np.nan_to_num(nc_ndvi.variables[ndvi_key][:], nan=0.0), dtype=np.float32)
                
            self.static_cache[idx] = {'elev': elev, 'isa': isa, 'ahf': ahf, 'mask': mask, 'ndvi': ndvi, 'imperv': imperv}

    def __len__(self):
        # 100 sweeps per train city = 6000 crops/epoch; each draws a fresh
        # random (year, month, day_hour) every epoch, so over 150 epochs the
        # time sampling is not a coverage bottleneck. Val stays at 5 (seeded).
        return len(self.df) * 100 if self.mode == 'train' else len(self.df) * 5

    def __getitem__(self, idx):
        actual_idx = idx % len(self.df)
        row = self.df.iloc[actual_idx]
        
        real_lat, real_lon = row.get('latitude', row.get('real_lat')), row.get('longitude', row.get('real_lon'))
        era5_x, era5_y = int(row['era5_lon_idx']), int(row['era5_lat_idx'])
        
        if self.mode == 'val':
            rs = np.random.RandomState(idx + 42)
            year, month, day_hour = rs.choice(self.years), rs.randint(0, 12), rs.randint(0, 720)
        else:
            year, month, day_hour = np.random.choice(self.years), np.random.randint(0, 12), np.random.randint(0, 720)
            
        # Single calendar datetime used for both zarr lookups. See
        # pipeline.config.target_datetime for why this is .sel() not .isel()
        # (leap-year drift + the 7-hour ERA5/UrbClim origin gap).
        target_dt = target_datetime(year, month, day_hour)

        # Cyclic seasonal encoding: day-of-year (1..365/6) so March 1 and March 31
        # get distinct values. The earlier integer-month encoding collapsed every
        # hour of a month to a single (sin, cos) pair.
        doy = target_dt.timetuple().tm_yday
        month_sin = float(np.sin(2.0 * np.pi * doy / 365.25))
        month_cos = float(np.cos(2.0 * np.pi * doy / 365.25))
        # Cyclic diurnal encoding on local solar time (UTC + lon/15) so the diurnal
        # phase is aligned across the cohort — Lisbon's solar noon and Bucharest's
        # solar noon now share the same (hour_sin, hour_cos), regardless of UTC.
        lst = (target_dt.hour + float(real_lon) / 15.0) % 24.0
        hour_sin = float(np.sin(2.0 * np.pi * lst / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * lst / 24.0))

        era5_sample = (self.era5_ds['cell']
                       .sel(time=target_dt, method='nearest')
                       .isel(x=era5_x, y=era5_y)
                       .isel(variable=slice(0, 13)))
        era5_vector = np.copy(era5_sample.values)
        t2m_era5_raw = era5_vector[2]

        city_maps = self.static_cache[actual_idx]
        elev_raw, isa_raw, ahf_raw, land_mask, ndvi_raw, imperv_raw = \
            city_maps['elev'], city_maps['isa'], city_maps['ahf'], city_maps['mask'], \
            city_maps['ndvi'], city_maps['imperv']
        
        if ndvi_raw.ndim == 3:
            ndvi_raw = ndvi_raw[month] if ndvi_raw.shape[0] == 12 else ndvi_raw[0]
        elif ndvi_raw.ndim > 3:
            ndvi_raw = np.squeeze(ndvi_raw)[0]

        target_path = row.get('target_path', os.path.join(BASE_DIR, f"urbclim_zarr/urbclim_tas_{row['city_name']}.zarr"))

        if not hasattr(self, 'target_zarrs'): self.target_zarrs = {}
        if target_path not in self.target_zarrs:
            self.target_zarrs[target_path] = xr.open_zarr(target_path)

        # Same target_dt as ERA5 above — both zarrs now return the same real time.
        t_urb_raw = np.array(
            self.target_zarrs[target_path]['tas'].sel(time=target_dt, method='nearest').values,
            dtype=np.float32,
        )
        valid_mask = ~np.isnan(t_urb_raw)
        t_urb_raw = np.nan_to_num(t_urb_raw, nan=0.0)

        # CORINE reclassification — Jonas thesis Table 8.1 (15 classes; sea = 15).
        # Raw CORINE 2012 codes 1–44 → reclassified integer in {1..15}; 0 means unmapped.
        corine_reclass = np.zeros_like(isa_raw, dtype=np.float32)
        for new_class, orig_codes in CORINE_CLASS_MAP.items():
            corine_reclass[np.isin(isa_raw, orig_codes)] = new_class
        # Sea (class 15) and unmapped (class 0) are dropped from the loss but kept in the
        # input channel so the convolutions can see the land/sea boundary.
        valid_mask = valid_mask & (corine_reclass > 0) & (corine_reclass != 15)

        elev_low = float(self.orog_global.sel(latitude=real_lat, longitude=real_lon, method='nearest').values / GRAVITY)
        t2m_era5_msl = t2m_era5_raw + (elev_low * LAPSE_RATE)
        target_residual = ((t_urb_raw + (elev_raw * LAPSE_RATE)) - t2m_era5_msl) * valid_mask

        norm_era5 = (era5_vector - self.era5_means) / (self.era5_stds + 1e-8)

        if self.mode == 'train':
            noise = np.random.normal(0, 0.05, size=norm_era5.shape)
            norm_era5 = norm_era5 + noise

        norm_elev  = (elev_raw  - self.elev_mean)  / (self.elev_std  + 1e-8)
        norm_ahf   = (ahf_raw   - self.ahf_mean)   / (self.ahf_std   + 1e-8)
        norm_ndvi  = (ndvi_raw  - self.ndvi_mean)  / (self.ndvi_std  + 1e-8)
        norm_imperv= (imperv_raw- self.imperv_mean) / (self.imperv_std + 1e-8)

        planes = [np.full_like(elev_raw, val) for val in norm_era5] # 13 ERA5 variables

        # ---> STACK (23 Inputs + 1 Target + 1 Mask = 25 Total Channels) <---
        full_stack = np.stack(
            planes + [
                norm_elev, norm_ahf, land_mask,          # Channels 13, 14, 15: Statics
                corine_reclass,                          # Channel 16: CORINE reclassified integer (1..14)
                norm_ndvi,                               # Channel 17: Vegetation
                norm_imperv,                             # Channel 18: Imperviousness (NEW — v2 static files)
                np.full_like(elev_raw, month_sin),       # Channel 19: Month sin (cyclic seasonal)
                np.full_like(elev_raw, month_cos),       # Channel 20: Month cos (cyclic seasonal)
                np.full_like(elev_raw, hour_sin),        # Channel 21: Hour-of-day sin (cyclic diurnal)
                np.full_like(elev_raw, hour_cos),        # Channel 22: Hour-of-day cos (cyclic diurnal)
                target_residual,                         # Channel 23: Target UHI Temperature
                valid_mask.astype(np.float32)            # Channel 24: Mask
            ],
            axis=0
        ).astype(np.float32)

        _, h, w = full_stack.shape
        if h >= CROP_SIZE and w >= CROP_SIZE:
            if self.mode == 'train':
                y_c, x_c = np.random.randint(0, h - CROP_SIZE + 1), np.random.randint(0, w - CROP_SIZE + 1)
            else:
                rs_crop = np.random.RandomState(idx + 100)
                y_c, x_c = rs_crop.randint(0, h - CROP_SIZE + 1), rs_crop.randint(0, w - CROP_SIZE + 1)
            crop = full_stack[:, y_c:y_c+CROP_SIZE, x_c:x_c+CROP_SIZE].copy()

        else:
            crop = np.zeros((25, CROP_SIZE, CROP_SIZE), dtype=np.float32)
            crop[:, :h, :w] = full_stack

        return {
            "input": torch.from_numpy(crop[:23]),    # Channels 0–22 are Inputs
            "target": torch.from_numpy(crop[23:24]), # Channel 23 is the Target
            "mask": torch.from_numpy(crop[24:25])    # Channel 24 is the Mask
        }

# ==============================================================================
# BLOCK 3: The Architecture (RESIDUAL U-NET)
# ==============================================================================
class ResidualConv(nn.Module):
    """Residual Convolution Block (No Attention)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_ch)
        )
        
        # The Shortcut (Bypass Connection)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.conv_block(x)
        # Add the raw input data back into the processed output!
        return torch.relu(out + self.shortcut(x))

class UrbClimUNet(nn.Module):
    def __init__(self, in_ch=23, out_ch=1):
        super().__init__()
        self.inc = ResidualConv(in_ch, 96)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(96, 192))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(192, 384))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(384, 768))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ResidualConv(768, 1536))
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(1536, 768, kernel_size=1))
        self.dec1 = ResidualConv(1536, 768)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(768, 384, kernel_size=1))
        self.dec2 = ResidualConv(768, 384)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(384, 192, kernel_size=1))
        self.dec3 = ResidualConv(384, 192)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(192, 96, kernel_size=1))
        self.dec4 = ResidualConv(192, 96)
        
        self.dropout = nn.Dropout2d(p=0.05)
        self.out = nn.Conv2d(96, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.dropout(self.bottleneck(x4))
        d1 = self.dropout(self.dec1(torch.cat([x4, self.up1(x5)], dim=1)))
        d2 = self.dropout(self.dec2(torch.cat([x3, self.up2(d1)], dim=1)))
        d3 = self.dropout(self.dec3(torch.cat([x2, self.up3(d2)], dim=1)))
        d4 = self.dec4(torch.cat([x1, self.up4(d3)], dim=1))
        return self.out(d4)

# ==============================================================================
# BLOCK 4: Target Field Visualisation (sanity check)
# ==============================================================================
def plot_target_samples(dataset, out_path, n_samples=6):
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for k in range(n_samples):
        sample = dataset[k * max(1, len(dataset) // n_samples)]
        target = sample["target"].numpy()[0]
        mask = sample["mask"].numpy()[0] > 0.5
        display = np.where(mask, target, np.nan)
        vmax = np.nanmax(np.abs(display)) if np.any(mask) else 1.0
        im = axes[k].imshow(display, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[k].set_title(f"sample {k} — UHI residual [°C]")
        axes[k].set_xticks([]); axes[k].set_yticks([])
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    for ax in axes[n_samples:]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Target field plot saved to {out_path}", flush=True)

# ==============================================================================
# BLOCK 5: The Training & Validation Loop
# ==============================================================================
if __name__ == '__main__':
    train_dataset = UrbClimDataset(SPLITS_CSV, 'train', ERA5_ZARR_PATH, OROG_PATH, TRAIN_STATS)
    val_dataset = UrbClimDataset(SPLITS_CSV, 'val', ERA5_ZARR_PATH, OROG_PATH, VAL_STATS)

    plot_dir = os.path.join(MY_DATA_DIR, "diagnostics")
    os.makedirs(plot_dir, exist_ok=True)
    plot_target_samples(train_dataset, os.path.join(plot_dir, "v2_target_samples.png"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = UrbClimUNet(in_ch=23, out_ch=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    accumulation_steps = 4
    actual_steps = (len(train_loader) + accumulation_steps - 1) // accumulation_steps

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=8e-5, 
        steps_per_epoch=actual_steps, 
        epochs=NUM_EPOCHS, 
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    
    criterion_mse, criterion_l1 = nn.MSELoss(), nn.L1Loss()
    writer = SummaryWriter(log_dir=os.path.join(MY_DATA_DIR, "runs/v2_resunet"))

    CKPT_DIR = os.path.join(os.environ.get('VSC_SCRATCH', MY_DATA_DIR), "urbclim_v2_models")
    os.makedirs(CKPT_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(CKPT_DIR, "latest_model.pth")

    start_epoch, epochs_without_improvement, best_val_mse = 0, 0, float('inf')
    history = {"train_mse": [], "train_l1": [], "val_mse": [], "val_l1": []}

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        t_mse, t_l1 = 0, 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            inputs, targets, masks = batch["input"].to(DEVICE), batch["target"].to(DEVICE), batch["mask"].to(DEVICE) > 0.5
            outputs = model(inputs)
            
            if masks.sum() > 0:
                loss = criterion_mse(outputs[masks], targets[masks]) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                batch_mse = criterion_mse(outputs[masks], targets[masks]).item()
                t_mse += batch_mse
                t_l1 += criterion_l1(outputs[masks], targets[masks]).item()
                
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Step [{i+1}/{len(train_loader)}] | MSE: {batch_mse:.4f}", flush=True)

        model.eval()
        v_mse, v_l1 = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, masks = batch["input"].to(DEVICE), batch["target"].to(DEVICE), batch["mask"].to(DEVICE) > 0.5
                outputs = model(inputs)
                if masks.sum() > 0:
                    v_mse += criterion_mse(outputs[masks], targets[masks]).item()
                    v_l1 += criterion_l1(outputs[masks], targets[masks]).item()
                
        avg_t_mse, avg_t_l1 = t_mse/len(train_loader), t_l1/len(train_loader)
        avg_v_mse, avg_v_l1 = v_mse/len(val_loader), v_l1/len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("="*60)
        print(f"V2 EPOCH {epoch+1}/{NUM_EPOCHS} SUMMARY")
        print(f"TRAIN -> MSE: {avg_t_mse:.4f} | L1 (MAE): {avg_t_l1:.4f}")
        print(f"VAL   -> MSE: {avg_v_mse:.4f} | L1 (MAE): {avg_v_l1:.4f}")
        print(f"LR    -> {current_lr:.2e}", flush=True)
        print("="*60, flush=True)
        
        history["train_mse"].append(avg_t_mse); history["train_l1"].append(avg_t_l1)
        history["val_mse"].append(avg_v_mse); history["val_l1"].append(avg_v_l1)
        
        ckpt_data = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'history': history}
        torch.save(ckpt_data, CHECKPOINT_PATH)
        
        if avg_v_mse < best_val_mse:
            best_val_mse, epochs_without_improvement = avg_v_mse, 0
            torch.save(ckpt_data, os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"*** NEW BEST VAL MSE: {best_val_mse:.4f} — V2 Best model saved! ***", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"Early Stopping Counter: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}", flush=True)
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n>>>>> EARLY STOPPING TRIGGERED AT EPOCH {epoch+1} <<<<<", flush=True)
                break

    writer.close()
