"""
build_lookup.py — scan VSC for static_features files and build a fresh master city CSV.

Replaces the outdated city_spatial_index_v3.csv (which pointed to an older
zarr generation). This script follows the pattern from data_prep(3).ipynb:

  1. Load ERA5 coordinates.  In the zarr the lon/lat axes are named `x`/`y`
     (not `longitude`/`latitude`) — both are 1-D coordinate arrays.  The v1
     zarr has a known NaN at index 100 (longitude 0.0) which we patch in place.
  2. Scan STATIC_FEATURES_DIR for `static_features_<City>.nc` files.  These
     store their centre in EPSG:3035 metres, so we read the middle (x,y)
     pair with netCDF4 and transform it to lat/lon (EPSG:4326) via pyproj.
  3. Match each city centre to the nearest ERA5 grid cell (argmin abs).
  4. Require that the matching UrbClim zarr and NDVI file both exist; skip
     otherwise with a logged reason.
  5. Emit city_lookup.csv — single source of truth for splits.py.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.config import (STATIC_FEATURES_DIR, STATIC_NC_TMPL,
                             NDVI_NC_TMPL, URBCLIM_ZARR_TMPL,
                             ERA5_ZARR_PATH, CITY_LOOKUP_CSV)


def _load_era5_grid():
    """Return (lats, lons) as 1-D numpy arrays, with the v1 NaN-at-100 bug
    patched.  ERA5 zarr uses `x` for longitudes and `y` for latitudes."""
    with xr.open_zarr(ERA5_ZARR_PATH) as ds:
        lats = ds['y'].values.copy()
        lons = ds['x'].values.copy()
    if np.isnan(lons[100]):
        print("  patching era5_lons[100] NaN → 0.0 (v1 zarr coordinate bug)")
        lons[100] = 0.0
    return lats, lons


def _city_centre_latlon(static_path, transformer):
    """Read the centre-pixel (x, y) from a static_features .nc file
    (EPSG:3035 metres) and transform to (lat, lon) in EPSG:4326."""
    with netCDF4.Dataset(static_path, 'r') as nc:
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        mid_x = float(nc.variables['x'][nx // 2])
        mid_y = float(nc.variables['y'][ny // 2])
    real_lon, real_lat = transformer.transform(mid_x, mid_y)
    return real_lat, real_lon


def build_lookup(output=CITY_LOOKUP_CSV):
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    print("Loading ERA5 grid…")
    era5_lats, era5_lons = _load_era5_grid()
    print(f"  lon range: {era5_lons.min():.2f}…{era5_lons.max():.2f}  "
          f"lat range: {era5_lats.min():.2f}…{era5_lats.max():.2f}")

    pattern = os.path.join(STATIC_FEATURES_DIR, "static_features_*.nc")
    static_files = sorted(glob.glob(pattern))
    print(f"\nScanning {STATIC_FEATURES_DIR}")
    print(f"  found {len(static_files)} static_features files")
    if not static_files:
        raise RuntimeError(
            f"No static_features_*.nc in {STATIC_FEATURES_DIR} — "
            "check STATIC_FEATURES_DIR in pipeline/config.py"
        )

    rows, skipped = [], []
    for f in static_files:
        city = os.path.basename(f).replace("static_features_", "").replace(".nc", "")
        target_path = URBCLIM_ZARR_TMPL.format(city=city)
        ndvi_path   = NDVI_NC_TMPL.format(city=city)

        if not os.path.exists(target_path):
            skipped.append((city, f"urbclim zarr missing: {target_path}"))
            continue
        if not os.path.exists(ndvi_path):
            skipped.append((city, f"ndvi missing: {ndvi_path}"))
            continue

        try:
            real_lat, real_lon = _city_centre_latlon(f, transformer)
        except Exception as e:
            skipped.append((city, f"static coord read failed: {e}"))
            continue

        y_idx = int(np.argmin(np.abs(era5_lats - real_lat)))
        x_idx = int(np.argmin(np.abs(era5_lons - real_lon)))

        rows.append({
            'city_name':    city,
            'latitude':     round(real_lat, 5),
            'longitude':    round(real_lon, 5),
            'era5_lat_idx': y_idx,
            'era5_lon_idx': x_idx,
            'static_path':  f,
            'ndvi_path':    ndvi_path,
            'target_path':  target_path,
        })

    df = pd.DataFrame(rows).sort_values('city_name').reset_index(drop=True)
    df.to_csv(output, index=False)

    print(f"\nKept   : {len(df)} cities")
    print(f"Skipped: {len(skipped)} cities")
    for name, why in skipped[:20]:
        print(f"    - {name}: {why}")
    if len(skipped) > 20:
        print(f"    … and {len(skipped) - 20} more")
    print(f"\nSaved  → {output}")
    return df


if __name__ == '__main__':
    build_lookup()
