"""
extract.py — pure readers for every raw file in the pipeline.

Every function here is a thin wrapper around netCDF4 / xarray that
does nothing beyond "open file, pull array, NaN-fill". Any math
(lapse-rate, z-score, CORINE reclassification) lives elsewhere so
these primitives can be reused by zscore.py, check_pipeline.py, and
any future ablation script.
"""
import numpy as np
import xarray as xr
import netCDF4

from pipeline.config import GRAVITY


# ---------------------------------------------------------------- static

def extract_static_maps(static_path):
    """Open a static_features_tmp_v2 .nc and return the five maps used by
    the model as a dict of float32 arrays."""
    with netCDF4.Dataset(static_path, 'r') as nc:
        elev   = _read(nc, 'elevation')
        corine = _read(nc, 'CORINE')
        ahf    = _read(nc, 'AHF')
        mask   = _read(nc, 'land_water_mask')

        # Imperviousness key varies across file generations. Static_features_tmp_v2
        # uses 'imperv'. Raise loudly if none of the known names is present —
        # the silent zeros_like fallback that lived here previously masked a
        # months-long bug where the imperv channel was constant 0.
        imperv_key = next((k for k in ('imperv', 'imperviousness', 'IMPERV')
                           if k in nc.variables), None)
        if imperv_key is None:
            raise KeyError(f"no imperviousness variable in {static_path}; "
                           f"found {list(nc.variables)}")
        imperv = _read(nc, imperv_key)
        # Copernicus HRL imperv uses 254/255 as nodata flags; clamp anything
        # outside [0, 100] (% impervious surface) to 0 before z-scoring.
        imperv = np.where((imperv >= 0.0) & (imperv <= 100.0),
                          imperv, 0.0).astype(np.float32)

    return {'elev': elev, 'corine': corine, 'ahf': ahf,
            'mask': mask, 'imperv': imperv}


def extract_ndvi(ndvi_path, month=None):
    """Return NDVI as a 2D map, slicing to `month` (0..11) when the file
    stores a 12-month stack."""
    with netCDF4.Dataset(ndvi_path, 'r') as nc:
        key = 'NDVI' if 'NDVI' in nc.variables else list(nc.variables.keys())[-1]
        ndvi = _read(nc, key)

    if month is None:
        return ndvi
    if ndvi.ndim == 3:
        return ndvi[month] if ndvi.shape[0] == 12 else ndvi[0]
    if ndvi.ndim > 3:
        return np.squeeze(ndvi)[0]
    return ndvi


# ---------------------------------------------------------------- dynamic

def extract_era5_vector(era5_ds, target_dt, era5_x, era5_y, n_vars=13):
    """Pull a single (time, x, y) column of the first `n_vars` ERA5 variables
    at the given calendar datetime. Uses `.sel(method='nearest')` so it's
    immune to leap-year math and the ERA5/UrbClim zarr-origin mismatch.
    Returns (values_1d, timestamp_scalar)."""
    sample = (era5_ds['cell']
              .sel(time=target_dt, method='nearest')
              .isel(x=era5_x, y=era5_y)
              .isel(variable=slice(0, n_vars)))
    return np.copy(sample.values), sample['time'].values


def extract_target(zarr_path, target_dt, cache=None):
    """Return (tas_map, timestamp) from a UrbClim per-city zarr at the given
    calendar datetime. Same `.sel(method='nearest')` approach as ERA5 — both
    zarrs hand back the same real hour even though UrbClim's time[0] is 7h
    later than ERA5's. Optional `cache` dict is keyed by path so the worker
    doesn't re-open the zarr each call."""
    if cache is not None:
        if zarr_path not in cache:
            cache[zarr_path] = xr.open_zarr(zarr_path)
        ds = cache[zarr_path]
    else:
        ds = xr.open_zarr(zarr_path)
    slab = ds['tas'].sel(time=target_dt, method='nearest')
    return np.array(slab.values, dtype=np.float32), slab['time'].values


def load_orography(orog_path):
    """Return the global ERA5 orography DataArray (geopotential, m^2/s^2)
    loaded into memory once per process."""
    with xr.open_dataset(orog_path) as ds:
        return ds['orog'].load()


def elev_low_at(orog_global, lat, lon):
    """Look up coarse ERA5 elevation (meters) at a city centre."""
    return float(orog_global.sel(latitude=lat, longitude=lon,
                                 method='nearest').values / GRAVITY)


# ---------------------------------------------------------------- utils

def _read(nc, key):
    return np.array(np.nan_to_num(nc.variables[key][:], nan=0.0),
                    dtype=np.float32)
