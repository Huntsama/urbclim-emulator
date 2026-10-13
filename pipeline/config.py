"""
Paths, constants, and lookup tables shared by every pipeline module.
Edit this file when the data layout on VSC changes.
"""
import os
from datetime import datetime as _dt, timedelta as _td

# --- VSC paths -----------------------------------------------------------
MY_DATA_DIR = "/user/brussel/113/vsc11361/data"
BASE_DIR    = "/data/brussel/vo/000/bvo00029/urbclim_emulator/data/intermediate"

# Raw inputs
ERA5_ZARR_PATH      = os.path.join(BASE_DIR, "era5_2008_2017_v1.zarr")
OROG_PATH           = os.path.join(BASE_DIR, "era5_orography_global_WGS84.nc")

# Directories scanned by build_lookup.py. These are the authoritative
# source of truth — if VSC moves the data, change these two lines and
# everything downstream (splits, stats, dataloader) follows.
URBCLIM_ZARR_DIR    = os.path.join(BASE_DIR, "urbclim_zarr")
URBCLIM_ZARR_PATTERN = "urbclim_tas_*.zarr"
STATIC_FEATURES_DIR = os.path.join(BASE_DIR, "static_features_tmp_v2")
NDVI_DIR            = os.path.join(BASE_DIR, "ndvi_tmp")

# Filename templates used for path-building (city is a bare name like "Szeged")
URBCLIM_ZARR_TMPL = os.path.join(URBCLIM_ZARR_DIR,    "urbclim_tas_{city}.zarr")
STATIC_NC_TMPL    = os.path.join(STATIC_FEATURES_DIR, "static_features_{city}.nc")
NDVI_NC_TMPL      = os.path.join(NDVI_DIR,            "NDVI_{city}.nc")

# Generated artefacts — all live under MY_DATA_DIR so the repo stays clean.
#   city_lookup.csv      : one row per usable city (build_lookup.py)
#   splits_metadata.csv  : same cities + 'split' column in col 2 (splits.py)
#   *_statistics.csv     : μ/σ/min/max per split (zscore.py). The split
#                          name in the filename makes per-split city lists
#                          redundant, so we no longer emit train_cities.csv etc.
CITY_LOOKUP_CSV = os.path.join(MY_DATA_DIR, "city_lookup.csv")
SPLITS_CSV      = os.path.join(MY_DATA_DIR, "splits_metadata.csv")
TRAIN_STATS     = os.path.join(MY_DATA_DIR, "train_statistics.csv")
VAL_STATS       = os.path.join(MY_DATA_DIR, "val_statistics.csv")
TEST_STATS      = os.path.join(MY_DATA_DIR, "test_statistics.csv")

# --- Split sizes ---------------------------------------------------------
# city_spatial_index_v3.csv holds 100 cities. 60/20/20 (set on supervisor's
# instruction, May 2026) gives larger held-out pools so per-city averages
# on val and test have tighter confidence intervals at the cost of fewer
# training cities.
TRAIN_N = 60
VAL_N   = 20
TEST_N  = 20
SHUFFLE_SEED = 42

# --- Split years ---------------------------------------------------------
# Train takes the six middle years; val/test hold out the inner and outer
# bookends of the 2008-2017 decade so temporal generalisation is tested
# both one step away (val) and at the extreme edges (test).
TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015]
VAL_YEARS   = [2009, 2016]
TEST_YEARS  = [2008, 2017]

# --- Physics constants ---------------------------------------------------
LAPSE_RATE = 0.0065     # K / m dry-adiabatic reduction to mean sea level
GRAVITY    = 9.80665    # m / s^2 (used to convert ERA5 geopotential to meters)

# --- Time sync ----------------------------------------------------------
# The UrbClim zarr starts at 2008-01-01 07:00 — 7 hours after ERA5.
# Any sample that would fall in the first 7 hours of 2008 gets clamped
# forward so both zarrs return the same real timestamp under
# `.sel(time=target_dt, method='nearest')`.
URB_START = _dt(2008, 1, 1, 7)


def target_datetime(year, month, day_hour):
    """Calendar datetime the dataloader and every debug script use.

    Single source of truth for the time-sync fix: built from the same
    (year, month, day_hour) tuple the dataloader samples, then clamped
    into the UrbClim-valid window.  Use this with
    `ds.sel(time=target_dt, method='nearest')` on BOTH zarrs and they
    will always return the same real hour.
    """
    dt = _dt(year, 1, 1) + _td(hours=month * 720 + day_hour)
    return dt if dt >= URB_START else URB_START

# --- CORINE reclassification (Jonas thesis Table 8.1) --------------------
# Raw CORINE 2012 codes (1..44) -> 15 classes. Class 15 "Sea" (raw 42/43/44)
# is fed to the convolutions so they can see the land/sea boundary, but the
# dataloader excludes it from the loss mask (no UrbClim tas over open water).
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
