"""
zscore.py — compute z-score statistics (means, stds, mins, maxes) per split.

Orchestrates:
   1. splits.build_splits()            → splits_metadata.csv + full df
   2. extract.* readers over each city → static feature arrays
   3. parallel ERA5 year fetch          → dynamic statistics
   4. write TRAIN_STATS / VAL_STATS / TEST_STATS

The dataloader reads the resulting CSV via lookup.StatsLookup.
"""
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr

from pipeline.config import (ERA5_ZARR_PATH,
                             TRAIN_STATS, VAL_STATS, TEST_STATS,
                             TRAIN_YEARS, VAL_YEARS, TEST_YEARS)
from pipeline.extract import extract_static_maps, extract_ndvi
from pipeline.splits  import build_splits


def _static_stats(split_df):
    elevs, ahfs, ndvis, impervs = [], [], [], []
    cor_min, cor_max = float('inf'), float('-inf')

    # splits.py already upgrades the path to static_features_tmp_v2/.
    # Support both the v4 'static_file_path' and the older 'static_path'.
    path_col = 'static_file_path' if 'static_file_path' in split_df.columns else 'static_path'

    for _, row in split_df.iterrows():
        s = extract_static_maps(row[path_col])
        elevs.append(s['elev'])
        ahfs.append(s['ahf'])
        impervs.append(s['imperv'])
        cor_min = min(cor_min, float(s['corine'].min()))
        cor_max = max(cor_max, float(s['corine'].max()))

        ndvis.append(extract_ndvi(row['ndvi_path']))

    def summarize(arrs):
        flat = np.concatenate([a.flatten() for a in arrs])
        return flat.mean(), flat.std(), flat.min(), flat.max()

    return {
        'elevation':      summarize(elevs),
        'ahf':            summarize(ahfs),
        'ndvi':           summarize(ndvis),
        'imperviousness': summarize(impervs),
        '_corine_range':  (cor_min, cor_max),
    }


def _era5_stats(split_df, years):
    ds = xr.open_zarr(ERA5_ZARR_PATH)
    # Real variable names from the zarr's 'variable' coord — u10, v10, t2m,
    # blh, cape, kx, ssr, str, sp, sshf, tcc, tcw, tp — instead of era5_var_0..12.
    var_names = [str(v) for v in ds['variable'].values]

    era5_x = xr.DataArray(split_df['era5_lon_idx'].values, dims='city')
    era5_y = xr.DataArray(split_df['era5_lat_idx'].values, dims='city')
    ds_cities = ds['cell'].isel(x=era5_x, y=era5_y)

    def fetch(year):
        print(f"    → worker pulling {year}…", flush=True)
        return ds_cities.sel(time=str(year)).values

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        per_year = list(pool.map(fetch, years))

    data = np.concatenate(per_year, axis=0)  # (time, variable, city)
    return (var_names,
            np.nanmean(data, axis=(0, 2)),
            np.nanstd (data, axis=(0, 2)),
            np.nanmin (data, axis=(0, 2)),
            np.nanmax (data, axis=(0, 2)))


def generate_split_statistics(split_df, years, output_csv, split_name):
    print(f"\n=== {split_name.upper()} (years={years}) ===")

    statics = _static_stats(split_df)
    print(f"[{split_name}] static features done")

    var_names, era5_means, era5_stds, era5_mins, era5_maxs = _era5_stats(split_df, years)
    print(f"[{split_name}] era5 stats done — variables: {var_names}")

    rows = [{
        'feature':         var_names[i],
        'type':            'dynamic',
        'scaling_method':  'z-score',
        'mean':            era5_means[i],
        'std':             era5_stds[i],
        'min':             era5_mins[i],
        'max':             era5_maxs[i],
    } for i in range(len(var_names))]

    for feat in ('elevation', 'ahf', 'ndvi', 'imperviousness'):
        m, s, lo, hi = statics[feat]
        rows.append({
            'feature':        feat,
            'type':           'static',
            'scaling_method': 'z-score',
            'mean': m, 'std': s, 'min': lo, 'max': hi,
        })

    cor_lo, cor_hi = statics['_corine_range']
    rows.append({
        'feature':        'corine',
        'type':           'categorical',
        'scaling_method': 'min-max',
        'mean': np.nan, 'std': np.nan, 'min': cor_lo, 'max': cor_hi,
    })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[{split_name}] saved → {output_csv}")


if __name__ == '__main__':
    all_cities = build_splits()

    train = all_cities[all_cities['split'] == 'train']
    val   = all_cities[all_cities['split'] == 'val']
    test  = all_cities[all_cities['split'] == 'test']

    generate_split_statistics(train, TRAIN_YEARS, TRAIN_STATS, 'train')
    generate_split_statistics(val,   VAL_YEARS,   VAL_STATS,   'val')
    generate_split_statistics(test,  TEST_YEARS,  TEST_STATS,  'test')
