#!/usr/bin/env python
# coding: utf-8
"""
diagnose.py — build a re-usable DIAGNOSTIC DATA LOG for a trained v2 Residual U-Net.

This does NOT make plots and does NOT touch the whole test set. It runs the chosen
checkpoint over a *sample* of test hours (SAMPLES_PER_CITY per city — default 50,
≈ 1000 forward passes, ~10–15 min on a GPU) and writes tidy data files you can
slice and plot however you like, as many times as you like, *without ever running
the model again*:

    data/diag/<checkpoint-folder>/
      pixels.parquet   ← THE artifact: one row per (sub-sampled) valid pixel —
                         columns: city, lat, corine, corine_name, elev_z, ndvi_z,
                         imperv_z, true_uhi_C, true_T_C, pred_T_C, err_C, abs_err_C,
                         year, month, hour_utc, lst_hour
                         (falls back to pixels.csv.gz if pyarrow/fastparquet is missing)
      samples.csv      ← one row per (city, hour) evaluated: mae/rmse/bias, n_valid_px, time fields
      by_corine.csv    ← exact MAE/RMSE/bias per CORINE class (all sampled pixels, not sub-sampled)
      by_city.csv      ← exact MAE/RMSE/bias per city + each city's mean elevation / NDVI / imperv / |UHI|
      summary.json     ← headline numbers

Then:   python plot_diagnostics.py        (reads data/diag/<tag>/ → figures/diag/<tag>/*.png)
   or:  load pixels.parquet in a notebook and `groupby` whatever you want.

"err" = predicted − true air temperature (lapse-corrected residual error; the base
ERA5 temperature cancels). + bias = model too warm.

Pick the model by editing CHECKPOINT_PATH.   python diagnose.py
"""
import os
import json
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from v2_residual_unet import (
    UrbClimUNet, MY_DATA_DIR, ERA5_ZARR_PATH, OROG_PATH, SPLITS_CSV,
)
from pipeline.config import TEST_YEARS, TEST_STATS
from test_model import TestingUrbClimDataset


# ============================================================ CONFIG ==========
# Change this one line to diagnose the other model:
#   .../urbclim_v2_models/best_model.pth        → 23-ch baseline
#   .../urbclim_v2_models_solar/best_model.pth  → 25-ch solar
CHECKPOINT_PATH = os.path.join(
    os.environ.get('VSC_SCRATCH', MY_DATA_DIR),
    "urbclim_v2_models", "best_model.pth",
)
SAMPLES_PER_CITY  = 50      # random hours per test city  (×20 cities ≈ 1000 forward passes)
PIXELS_PER_SAMPLE = 400     # valid pixels kept per sample for pixels.parquet (×passes = total rows)
SEED              = 0
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# channel indices in the (>=23-ch) input stack — same layout in
# v2_residual_unet.py, v2_residual_unet_solar.py, test_model.py
CORINE_IN_CH = 16
ELEV_IN_CH   = 13
NDVI_IN_CH   = 17
IMPERV_IN_CH = 18
# ==============================================================================

CORINE_NAMES = {
    1: "Urban", 2: "Sub-urban", 3: "Industrial", 4: "Urban green", 5: "Snow/Ice",
    6: "Bare soil", 7: "Grassland", 8: "Cropland", 9: "Shrubland", 10: "Woodland",
    11: "Broadleaf trees", 12: "Needleleaf trees", 13: "Rivers", 14: "Inland water",
    15: "Sea",
}
TEST_START = datetime.datetime(2008, 1, 1, 7)   # UrbClim's valid window starts here


def autodetect_in_ch(path):
    ck = torch.load(path, map_location='cpu')
    st = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    return int(st['inc.conv_block.0.weight'].shape[1])


def random_hours(per_city, seed):
    """A reproducible random sample of (year_ord, hour_in_year, datetime) tuples."""
    rs = np.random.RandomState(seed)
    HPY = TestingUrbClimDataset.HOURS_PER_YEAR
    out = []
    while len(out) < per_city:
        y = int(rs.randint(0, len(TEST_YEARS)))
        h = int(rs.randint(0, HPY))
        dt = datetime.datetime(TEST_YEARS[y], 1, 1) + datetime.timedelta(hours=h)
        if dt < TEST_START:
            continue
        out.append((y, h, dt))
    return out


def main():
    tag = os.path.basename(os.path.dirname(CHECKPOINT_PATH))
    out_dir = os.path.join(MY_DATA_DIR, "diag", tag)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT_PATH}")
    in_ch = autodetect_in_ch(CHECKPOINT_PATH)
    print(f"checkpoint : {CHECKPOINT_PATH}")
    print(f"  in_ch={in_ch}  tag={tag}  device={DEVICE}  samples/city={SAMPLES_PER_CITY}")
    print(f"  output   : {out_dir}\n")

    model = UrbClimUNet(in_ch=in_ch, out_ch=1).to(DEVICE)
    ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    st = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    model.load_state_dict(st)
    model.eval()

    splits = pd.read_csv(SPLITS_CSV)
    trows = splits[splits['split'] == 'test'].reset_index(drop=True)
    cities = trows['city_name'].tolist()
    lat_of = dict(zip(trows['city_name'], trows['latitude']))
    lon_of = dict(zip(trows['city_name'], trows['longitude']))
    nc = len(cities)
    print(f"  test cities ({nc}): {', '.join(cities)}")

    ds = TestingUrbClimDataset(csv_file=SPLITS_CSV, mode='test',
                               era5_path=ERA5_ZARR_PATH, orog_path=OROG_PATH,
                               stats_csv_path=TEST_STATS)
    HPY = TestingUrbClimDataset.HOURS_PER_YEAR
    hrs = random_hours(SAMPLES_PER_CITY, SEED)
    idx = []
    for (y, h, _) in hrs:
        chi = y * HPY + h
        for c in range(nc):
            idx.append(chi * nc + c)
    print(f"  {SAMPLES_PER_CITY} hours × {nc} cities = {len(idx)} forward passes\n")
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # exact accumulators (over ALL sampled pixels)
    cor = {}   # corine class -> [sum_ae, sum_e, sum_se, n]
    cty = {}   # city -> [sum_ae, sum_e, sum_se, n, sum_elev, sum_ndvi, sum_imp, sum_absuhi]
    sample_rows = []
    px_rows = []   # sub-sampled per-pixel records (list of small DataFrames)

    with torch.no_grad():
        for i, b in enumerate(loader):
            full_in = b["input"]
            inputs = full_in[:, :in_ch].to(DEVICE)
            tgt = b["target"].to(DEVICE)
            mask = b["mask"].to(DEVICE) > 0.5
            base = b["base_temp"].to(DEVICE)
            pred = model(inputs)

            h, w = b['orig_h'][0].item(), b['orig_w'][0].item()
            err = (pred - tgt)[0, 0, :h, :w].cpu().numpy()       # residual error °C  (= abs-temp error)
            mk  = mask[0, 0, :h, :w].cpu().numpy().astype(bool)
            c   = b['city_name'][0]
            tstr = b['exact_time'][0]
            if mk.sum() == 0:
                continue
            y_ord, h_in_year, dt = hrs[i // nc]

            true_T = (tgt + base)[0, 0, :h, :w].cpu().numpy() - 273.15
            pred_T = (pred + base)[0, 0, :h, :w].cpu().numpy() - 273.15
            uhi  = tgt[0, 0, :h, :w].cpu().numpy()
            cmap_ = full_in[0, CORINE_IN_CH, :h, :w].numpy()
            elev = full_in[0, ELEV_IN_CH,   :h, :w].numpy()
            ndvi = full_in[0, NDVI_IN_CH,   :h, :w].numpy()
            imp  = full_in[0, IMPERV_IN_CH, :h, :w].numpy()

            e = err[mk]; ae = np.abs(e); se = e * e

            s = cty.setdefault(c, [0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0])
            s[0] += ae.sum(); s[1] += e.sum(); s[2] += se.sum(); s[3] += e.size
            s[4] += elev[mk].sum(); s[5] += ndvi[mk].sum(); s[6] += imp[mk].sum(); s[7] += np.abs(uhi[mk]).sum()

            cc = cmap_[mk].astype(int)
            for k in np.unique(cc):
                sel = cc == k
                a = cor.setdefault(int(k), [0.0, 0.0, 0.0, 0])
                a[0] += ae[sel].sum(); a[1] += e[sel].sum(); a[2] += se[sel].sum(); a[3] += int(sel.sum())

            lst = (dt.hour + lon_of.get(c, 0.0) / 15.0) % 24.0
            sample_rows.append({'city': c, 'time': tstr, 'year': dt.year, 'month': dt.month,
                                'hour_utc': dt.hour, 'lst_hour': round(lst, 2), 'n_valid_px': int(e.size),
                                'mae_C': float(ae.mean()), 'rmse_C': float(np.sqrt(se.mean())), 'bias_C': float(e.mean())})

            step = max(1, e.size // PIXELS_PER_SAMPLE)
            sl = slice(None, None, step)
            ei = e[sl]
            px_rows.append(pd.DataFrame({
                'city': c, 'lat': lat_of.get(c, np.nan),
                'corine': cc[sl],
                'corine_name': [CORINE_NAMES.get(int(k), str(int(k))) for k in cc[sl]],
                'elev_z': elev[mk][sl], 'ndvi_z': ndvi[mk][sl], 'imperv_z': imp[mk][sl],
                'true_uhi_C': uhi[mk][sl], 'true_T_C': true_T[mk][sl], 'pred_T_C': pred_T[mk][sl],
                'err_C': ei, 'abs_err_C': np.abs(ei),
                'year': dt.year, 'month': dt.month, 'hour_utc': dt.hour, 'lst_hour': round(lst, 2),
            }))

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(idx)}]  {c}  {tstr}")

    if not cty:
        print("no valid pixels — nothing collected."); return

    # ---- assemble + write -------------------------------------------------
    px = pd.concat(px_rows, ignore_index=True) if px_rows else pd.DataFrame()
    samp = pd.DataFrame(sample_rows)
    by_cor = pd.DataFrame([{'corine': k, 'name': CORINE_NAMES.get(k, str(k)), 'n_px': v[3],
                            'mae_C': v[0]/v[3], 'rmse_C': float(np.sqrt(v[2]/v[3])), 'bias_C': v[1]/v[3]}
                           for k, v in sorted(cor.items())])
    by_city = (pd.DataFrame([{'city': c, 'lat': lat_of.get(c, np.nan), 'n_px': v[3],
                              'mae_C': v[0]/v[3], 'rmse_C': float(np.sqrt(v[2]/v[3])), 'bias_C': v[1]/v[3],
                              'mean_elev_z': v[4]/v[3], 'mean_ndvi_z': v[5]/v[3], 'mean_imperv_z': v[6]/v[3],
                              'mean_absUHI_C': v[7]/v[3]} for c, v in cty.items()])
               .sort_values('mae_C', ascending=False).reset_index(drop=True))

    n_tot = sum(v[3] for v in cty.values())
    summary = {'checkpoint': CHECKPOINT_PATH, 'tag': tag, 'in_ch': in_ch,
               'samples_per_city': SAMPLES_PER_CITY, 'n_forward_passes': len(idx), 'n_cities': nc,
               'n_pixels_total': int(n_tot), 'n_pixel_rows_in_log': int(len(px)),
               'overall': {'mae_C': sum(v[0] for v in cty.values()) / n_tot,
                           'rmse_C': float(np.sqrt(sum(v[2] for v in cty.values()) / n_tot)),
                           'bias_C': sum(v[1] for v in cty.values()) / n_tot}}

    px_path = os.path.join(out_dir, "pixels.parquet")
    try:
        px.to_parquet(px_path, index=False)
    except Exception as ex:
        px_path = os.path.join(out_dir, "pixels.csv.gz")
        px.to_csv(px_path, index=False, compression='gzip')
        print(f"  (parquet unavailable [{type(ex).__name__}] → wrote {px_path} instead)")
    samp.to_csv(os.path.join(out_dir, "samples.csv"), index=False)
    by_cor.to_csv(os.path.join(out_dir, "by_corine.csv"), index=False)
    by_city.to_csv(os.path.join(out_dir, "by_city.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        # numpy scalars (float32 etc.) aren't JSON-serialisable by default
        json.dump(summary, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    print("\nOVERALL:  MAE={mae_C:.3f} °C   RMSE={rmse_C:.3f} °C   bias={bias_C:+.3f} °C".format(**summary['overall']))
    print("\nby CORINE class:\n" + by_cor.to_string(index=False))
    print("\nby city (worst 8):\n" + by_city[['city', 'lat', 'mae_C', 'rmse_C', 'bias_C', 'n_px']].head(8).to_string(index=False))
    print(f"\n  data log : {px_path}   ({len(px):,} pixel rows)")
    print(f"  also     : samples.csv, by_corine.csv, by_city.csv, summary.json   in {out_dir}")
    print(f"\n  → next:  python plot_diagnostics.py     (or load {os.path.basename(px_path)} in a notebook)")


if __name__ == '__main__':
    main()
