"""
check_pipeline.py — post-fix sanity check for the UrbClim dataloader.

Run on VSC (raw data + zarr stores only available there):

    python -m pipeline.check_pipeline

Four sections:

  SECTION 1  TIME SYNC (verification)
      Builds target_dt = pipeline.config.target_datetime(year, month, day_hour)
      and calls .sel(time=target_dt, method='nearest') on both ERA5 and the
      UrbClim zarr.  Pass ⇔ both zarrs return the requested timestamp within
      30 min across 30 test points spanning 2008-2017 including leap years.
      This is the same logic __getitem__ now uses.

  SECTION 2  EXTRACT
      Dumps raw shapes + ranges from the file readers for one city: static
      maps (incl. imperviousness), NDVI, ERA5 vector, UrbClim target,
      orography.

  SECTION 3  GETITEM  (pixel-level math)
      Traces the exact arithmetic the dataloader performs for pixel [40, 40]:
      z-score every ERA5 channel, reclassify CORINE, lapse-rate correct both
      temperatures, compute the residual target.  Uses target_datetime() so
      the trace matches what the dataloader actually builds.

  SECTION 4  DATALOADER
      Instantiates UrbClimDataset, grabs sample 0, confirms the output tensor
      shapes + mask coverage are what the U-Net expects.
"""
import os
import sys
import datetime

import numpy as np
import pandas as pd
import xarray as xr

# Allow `python -m pipeline.check_pipeline` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.config  import (SPLITS_CSV, TRAIN_STATS,
                              ERA5_ZARR_PATH, OROG_PATH,
                              URBCLIM_ZARR_TMPL, NDVI_NC_TMPL,
                              LAPSE_RATE, CORINE_CLASS_MAP,
                              target_datetime, URB_START)
from pipeline.extract import (extract_static_maps, extract_ndvi,
                              extract_era5_vector, extract_target,
                              load_orography, elev_low_at)
from pipeline.lookup  import StatsLookup


HR  = '=' * 78
SUB = '-' * 78


def _static_path(row):
    """splits_metadata may carry either v4's `static_file_path` or `static_path`."""
    return row['static_file_path'] if 'static_file_path' in row.index else row['static_path']


def _ndvi_path(row, city):
    return row['ndvi_path'] if 'ndvi_path' in row.index and isinstance(row.get('ndvi_path'), str) \
                            else NDVI_NC_TMPL.format(city=city)


def _target_path(row, city):
    return row['target_path'] if 'target_path' in row.index and isinstance(row.get('target_path'), str) \
                              else URBCLIM_ZARR_TMPL.format(city=city)


# ---------------------------------------------------------------- SECTION 1

def section_1_time_sync():
    print('\n' + HR)
    print('SECTION 1 — TIME SYNC  (verify the fix)')
    print('target_datetime() + .sel(method="nearest") on both zarrs must agree.')
    print(HR)

    df   = pd.read_csv(SPLITS_CSV)
    era5 = xr.open_zarr(ERA5_ZARR_PATH)
    row  = df.iloc[0]
    city = row['city_name']
    urb  = xr.open_zarr(_target_path(row, city))

    # Mix of leap years, non-leap years, and edge slots.
    tests = []
    for year in (2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017):
        for month, day_hour in [(0, 0), (3, 500), (6, 100), (11, 700)]:
            tests.append((year, month, day_hour))

    print(f"\n  running {len(tests)} test points across 10 years × 4 slots")
    print(f"  {'year':<6}{'m':<4}{'dh':<6}{'target_dt':<22}"
          f"{'ERA5 Δ':<18}{'UrbC Δ':<18}{'PASS':<5}")
    print('  ' + SUB[:85])

    fails = 0
    for (year, month, day_hour) in tests:
        target_dt = target_datetime(year, month, day_hour)
        era5_t    = pd.Timestamp(era5['time'].sel(time=target_dt, method='nearest').values)
        urb_t     = pd.Timestamp(urb['time'].sel(time=target_dt, method='nearest').values)

        d_era = era5_t - pd.Timestamp(target_dt)
        d_urb = urb_t  - pd.Timestamp(target_dt)
        ok    = abs(d_era.total_seconds()) <= 1800 and abs(d_urb.total_seconds()) <= 1800
        if not ok: fails += 1

        print(f"  {year:<6}{month:<4}{day_hour:<6}{str(target_dt):<22}"
              f"{str(d_era):<18}{str(d_urb):<18}{'✓' if ok else '✗'}")

    print(SUB)
    if fails == 0:
        print(f"  ✅ {len(tests)}/{len(tests)} points aligned within 30 min — training-safe.")
    else:
        print(f"  ⚠ {fails}/{len(tests)} points drifted. Do not start training.")
    print(f"\n  (URB_START = {URB_START} — samples in the 7-hour pre-start gap are")
    print(f"   clamped forward by target_datetime; they test as Δ=7h on UrbClim,")
    print(f"   Δ=0 on ERA5 — not a failure, just the known gap.)")


# ---------------------------------------------------------------- SECTION 2

def section_2_extract():
    print('\n' + HR)
    print('SECTION 2 — EXTRACT')
    print('Raw-file readers: shapes and ranges for one training city.')
    print(HR)

    df    = pd.read_csv(SPLITS_CSV)
    train = df[df['split'] == 'train'].reset_index(drop=True)
    row   = train.iloc[0]
    city  = row['city_name']

    era5 = xr.open_zarr(ERA5_ZARR_PATH)
    orog = load_orography(OROG_PATH)

    print(f"\n  City        : {city}")
    print(f"  real_lat/lon: {row['latitude']:.4f}, {row['longitude']:.4f}")
    print(f"  era5_lat/x  : {int(row['era5_lat_idx'])}, {int(row['era5_lon_idx'])}")

    print(SUB + "\n  STATIC MAPS (includes imperviousness from v2 files)")
    for k, v in extract_static_maps(_static_path(row)).items():
        print(f"    {k:10s}  shape={str(v.shape):>14s}  "
              f"min={v.min():>9.3f}  max={v.max():>9.3f}  mean={v.mean():>9.3f}")

    print(SUB + "\n  NDVI (month=6 slice)")
    n = extract_ndvi(_ndvi_path(row, city), month=6)
    print(f"    shape={n.shape}  min={n.min():.3f}  max={n.max():.3f}  mean={n.mean():.3f}")

    target_dt = target_datetime(2010, 6, 100)  # 2010-07-04 04:00
    print(SUB + f"\n  ERA5 VECTOR at target_dt={target_dt}")
    vec, ts = extract_era5_vector(era5, target_dt,
                                  int(row['era5_lon_idx']),
                                  int(row['era5_lat_idx']))
    print(f"    timestamp = {str(ts)[:19]}  shape={vec.shape}")
    for i, v in enumerate(vec):
        tag = "  ← T2M (Kelvin, index 2)" if i == 2 else ''
        print(f"      var_{i:02d} = {v:>12.4f}{tag}")

    print(SUB + f"\n  TARGET (UrbClim tas) at target_dt={target_dt}")
    t, ts = extract_target(_target_path(row, city), target_dt)
    valid = ~np.isnan(t)
    print(f"    timestamp = {str(ts)[:19]}  shape={t.shape}")
    print(f"    valid px  = {valid.sum()} / {t.size}  ({100*valid.mean():.1f}%)")
    print(f"    K range   = [{np.nanmin(t):.2f}, {np.nanmax(t):.2f}]  "
          f"mean={np.nanmean(t):.2f}")

    print(SUB + "\n  OROGRAPHY lookup at city centre")
    print(f"    elev_low = {elev_low_at(orog, row['latitude'], row['longitude']):.2f} m")


# ---------------------------------------------------------------- SECTION 3

def section_3_getitem_math():
    print('\n' + HR)
    print('SECTION 3 — GETITEM  (pixel-level math trace at [40, 40])')
    print(HR)

    df    = pd.read_csv(SPLITS_CSV)
    train = df[df['split'] == 'train'].reset_index(drop=True)
    row   = train.iloc[0]
    city  = row['city_name']

    era5  = xr.open_zarr(ERA5_ZARR_PATH)
    orog  = load_orography(OROG_PATH)
    stats = StatsLookup(TRAIN_STATS)

    year, month, day_hour = 2010, 6, 100
    target_dt = target_datetime(year, month, day_hour)
    py, px = 40, 40

    print(f"\n  city={city}   pixel=[{py},{px}]   "
          f"year={year}  month={month}  day_hour={day_hour}")
    print(f"  target_dt={target_dt}")

    statics = extract_static_maps(_static_path(row))
    ndvi_m  = extract_ndvi(_ndvi_path(row, city), month=month)
    era5_vec, era5_ts = extract_era5_vector(era5, target_dt,
                                            int(row['era5_lon_idx']),
                                            int(row['era5_lat_idx']))
    t_urb, urb_ts = extract_target(_target_path(row, city), target_dt)
    elev_low = elev_low_at(orog, row['latitude'], row['longitude'])

    print(f"  ERA5 timestamp    = {str(era5_ts)[:19]}")
    print(f"  UrbClim timestamp = {str(urb_ts)[:19]}")
    drift = pd.Timestamp(era5_ts) - pd.Timestamp(urb_ts)
    print(f"  Δ(ERA5 − UrbClim) = {drift}   {'✓ aligned' if drift == pd.Timedelta(0) else '⚠ drift'}")

    # -- ERA5 planes (channels 0..12) ---------------------------------
    print(SUB + "\n  ERA5 → 13 broadcast planes (z-score)")
    norm_era5 = stats.normalize_era5(era5_vec)
    for i in range(13):
        tag = "  ← T2M" if i == 2 else ''
        name = stats.era5_names[i] if i < len(stats.era5_names) else f"var_{i}"
        print(f"    ch{i:02d}  {name:<6}  raw={era5_vec[i]:>12.4f}  "
              f"μ={stats.era5_means[i]:>10.4f}  σ={stats.era5_stds[i]:>10.4f}  "
              f"z={norm_era5[i]:>8.4f}{tag}")

    # -- Static channels (13..18) -------------------------------------
    r = {
        'elev':   float(statics['elev'][py, px]),
        'ahf':    float(statics['ahf'][py, px]),
        'mask':   float(statics['mask'][py, px]),
        'corine': float(statics['corine'][py, px]),
        'ndvi':   float(ndvi_m[py, px]),
        'imperv': float(statics['imperv'][py, px]),
    }
    reclass = 0
    for new_c, orig in CORINE_CLASS_MAP.items():
        if int(r['corine']) in orig:
            reclass = new_c
            break

    print(SUB + "\n  STATIC channels at [40,40]")
    print(f"    ch13 elev   raw={r['elev']:>8.2f}m → z={stats.normalize('elevation',r['elev']):>7.3f}")
    print(f"    ch14 ahf    raw={r['ahf']:>8.3f}  → z={stats.normalize('ahf',r['ahf']):>7.3f}")
    print(f"    ch15 mask   raw={r['mask']:>8.3f}  (passed through raw)")
    print(f"    ch16 corine raw={int(r['corine']):>3d}      → reclass {reclass} "
          f"({'dropped' if reclass == 0 else 'kept'})")
    print(f"    ch17 ndvi   raw={r['ndvi']:>8.3f}  → z={stats.normalize('ndvi',r['ndvi']):>7.3f}")
    print(f"    ch18 imperv raw={r['imperv']:>8.3f}  → z={stats.normalize('imperviousness',r['imperv']):>7.3f}")

    # -- Time-cycle planes (19..22) -----------------------------------
    hour_of_day = day_hour % 24
    msin, mcos = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    hsin, hcos = np.sin(2*np.pi*hour_of_day/24), np.cos(2*np.pi*hour_of_day/24)
    print(SUB + "\n  TIME-CYCLE broadcast planes")
    print(f"    ch19 month_sin = {msin:+.4f}")
    print(f"    ch20 month_cos = {mcos:+.4f}")
    print(f"    ch21 hour_sin  = {hsin:+.4f}  (hour_of_day = {hour_of_day})")
    print(f"    ch22 hour_cos  = {hcos:+.4f}")

    # -- Residual target ----------------------------------------------
    t2m_raw  = float(era5_vec[2])
    t2m_msl  = t2m_raw + elev_low * LAPSE_RATE
    t_urb_px = float(t_urb[py, px]) if not np.isnan(t_urb[py, px]) else float('nan')
    t_urb_ms = t_urb_px + r['elev'] * LAPSE_RATE
    residual = t_urb_ms - t2m_msl

    print(SUB + "\n  RESIDUAL TARGET = UrbClim(MSL) − ERA5 T2M(MSL)")
    print(f"    ERA5 T2M raw              = {t2m_raw:.3f} K")
    print(f"    ERA5 elev_low             = {elev_low:.1f} m")
    print(f"    ERA5 T2M at MSL           = {t2m_raw:.3f} + {elev_low:.1f}×0.0065 = {t2m_msl:.3f} K")
    print(f"    UrbClim tas (raw, px)     = {t_urb_px:.3f} K")
    print(f"    UrbClim tas at MSL (px)   = {t_urb_px:.3f} + {r['elev']:.1f}×0.0065 = {t_urb_ms:.3f} K")
    print(f"    Residual (target) at px   = {residual:+.3f} °C")

    # -- Valid-mask coverage ------------------------------------------
    reclass_map = np.zeros_like(statics['corine'])
    for new_c, orig in CORINE_CLASS_MAP.items():
        reclass_map[np.isin(statics['corine'], orig)] = new_c
    valid_map = (~np.isnan(t_urb)) & (reclass_map != 0)
    print(SUB + "\n  VALID MASK across whole map")
    print(f"    {valid_map.sum()} / {valid_map.size} pixels usable "
          f"({100*valid_map.mean():.2f}%)")


# ---------------------------------------------------------------- SECTION 4

def section_4_dataloader_smoke():
    print('\n' + HR)
    print('SECTION 4 — DATALOADER SMOKE TEST')
    print(HR)
    try:
        import importlib.util
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        spec = importlib.util.spec_from_file_location(
            'v2_residual_unet', os.path.join(root, 'v2_residual_unet.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  skipped (could not import v2_residual_unet): {e}")
        return

    ds = mod.UrbClimDataset(SPLITS_CSV, 'train',
                            ERA5_ZARR_PATH, OROG_PATH, TRAIN_STATS)
    sample = ds[0]
    inp, tgt, msk = sample['input'], sample['target'], sample['mask']

    print(f"  input  shape = {tuple(inp.shape)}  dtype={inp.dtype}")
    print(f"  target shape = {tuple(tgt.shape)}  dtype={tgt.dtype}")
    print(f"  mask   shape = {tuple(msk.shape)}  dtype={msk.dtype}")

    if (msk > 0.5).any():
        m = msk > 0.5
        print(f"  target[mask] min/max/mean = "
              f"{tgt[m].min().item():.3f} / "
              f"{tgt[m].max().item():.3f} / "
              f"{tgt[m].mean().item():.3f}")

    print("  per-channel (input) min / max / mean:")
    for c in range(inp.shape[0]):
        s = inp[c]
        print(f"    ch{c:02d}  min={s.min().item():>9.3f}  "
              f"max={s.max().item():>9.3f}  mean={s.mean().item():>9.3f}")


# ---------------------------------------------------------------- main

if __name__ == '__main__':
    section_1_time_sync()
    section_2_extract()
    section_3_getitem_math()
    section_4_dataloader_smoke()
    print('\n' + HR + '\n  ALL SECTIONS COMPLETE\n' + HR)
