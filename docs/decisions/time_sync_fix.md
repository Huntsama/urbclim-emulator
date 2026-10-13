# Time Sync Fix — ERA5 ↔ UrbClim

Documentation for a silent data-alignment bug in the UrbClim U-Net
dataloader, how we diagnosed it from the debug logs, the code fix, and
the verification that confirms it is closed.

---

## 1. What was wrong

The original `__getitem__` in `v2_residual_unet.py` fetched each training
pair with positional indices (`.isel(time=N)`):

```python
hour_in_year    = month * 720 + day_hour
era5_year_ds    = self.era5_ds['cell'].sel(time=str(year))
era5_sample     = era5_year_ds.isel(time=hour_in_year, x=era5_x, y=era5_y)
# ...
global_time_idx = (year - 2008) * 8760 + hour_in_year
t_urb_raw       = self.target_zarrs[target_path]['tas'].isel(time=global_time_idx).values
```

That looks innocent but it silently broke training in two independent ways.

### Bug 1 — leap years ignored by `(year - 2008) * 8760`

8760 is the number of hours in a non-leap year. 2008, 2012, 2016 are
leap years (8784 hours). The formula thus drifts by +24 h per past
leap year:

| year | `(y-2008)*8760` | true hours-since-2008 | drift |
|------|-----------------|-----------------------|------:|
| 2008 | 0     | 0     | +0 h  |
| 2009 | 8760  | 8784  | +24 h |
| 2012 | 35040 | 35064 | +24 h |
| 2013 | 43800 | 43848 | +48 h |
| 2016 | 70080 | 70128 | +48 h |
| 2017 | 78840 | 78912 | +72 h |

Consequence: the UrbClim `.isel(global_time_idx)` read lands on a
timestamp up to **72 hours later** than what we thought we asked for.

### Bug 2 — the UrbClim zarr starts 7 hours after ERA5

Our verification script dumped `time[0]` for both stores:

```
ERA5    : 2008-01-01T00:00:00  →  2017-12-31T23:00:00   len = 87672
UrbClim : 2008-01-01T07:00:00  →  2017-12-31T23:00:00   len = 87665
```

The first 7 hours of 2008 simply aren't in the UrbClim zarr. So even
with the leap fix, every `.isel(time=N)` on UrbClim returns ERA5's
`time[N+7]` — a constant **+7 h offset** baked into the zarr metadata.

### Combined effect on training

For a sample in 2017 with month=6, day_hour=100:

| | ERA5 returned | UrbClim returned |
|-|---|---|
| requested (calendar) | 2017-07-04 04:00 | 2017-07-04 04:00 |
| old `.isel(naive_idx)` | 2017-07-01 04:00 | 2017-07-01 11:00 |
| drift | −3 d 0 h | −2 d 17 h |

The model was learning to pair **ERA5 weather from day N** with
**UrbClim temperature from day N+3**. Not a nowcast; partly a 3-day
forecast of the wrong physical system.

Cyclic encodings (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`) did
**not** save us: they are derived from the `(month, day_hour)` the
dataloader *intended* to sample. The encoding told the model "14:00 in
July" while the target underneath was 3 days later. A feature encoding
cannot repair a mismatched pair.

---

## 2. How we diagnosed it

1. `pipeline/check_pipeline.py` section 1 flagged the drift symbolically,
   comparing `.isel(naive)` vs `.isel(leap-aware)` vs `.sel(dt)`.
2. `debug_time_sync.py` iteration 1 proved both halves:
   - Section 1 dumped `time[0]` on both zarrs → showed the 7-hour gap.
   - Section 2 tabulated the `(year-2008)*8760` drift.
   - Section 4 tested 50 `.sel(time=calendar_dt, method='nearest')`
     calls against the zarrs — 49/50 returned Δ = 0. The single failure
     was `2008-01-01 00:00` (in the 7-hour gap) and snapped forward to
     07:00, which is expected.

---

## 3. The fix

One function per lookup, one calendar datetime per sample. Drop every
`.isel(time=...)` on the time axis and use `.sel(time=target_dt,
method='nearest')` instead. `.sel` looks up by the zarr's native time
coordinate, so it ignores both zarr starting offsets and leap-year math.

The patched block in `v2_residual_unet.py:__getitem__`:

```python
hour_in_year = (month * 720) + day_hour

# Build the single calendar datetime both zarrs will be queried with.
# Looking up by time coord (`.sel`) instead of position (`.isel`)
# removes two separate data-alignment bugs the index math had:
#   1. `(year-2008)*8760` ignored leap days → +24 h drift per past leap year.
#   2. The UrbClim zarr starts at 2008-01-01 07:00, not 00:00, so `.isel`
#      indices were off by a constant 7 h vs ERA5 (which starts at 00:00).
# UrbClim has no data before 2008-01-01 07:00 — clamp forward if we sample
# that 7-hour gap so the two zarrs return the same real timestamp.
target_dt = datetime.datetime(year, 1, 1) + datetime.timedelta(hours=hour_in_year)
URB_START = datetime.datetime(2008, 1, 1, 7)
if target_dt < URB_START:
    target_dt = URB_START

era5_sample = (self.era5_ds['cell']
               .sel(time=target_dt, method='nearest')
               .isel(x=era5_x, y=era5_y)
               .isel(variable=slice(0, 13)))
# ...
t_urb_raw = np.array(
    self.target_zarrs[target_path]['tas'].sel(time=target_dt, method='nearest').values,
    dtype=np.float32,
)
```

Notes:

- `target_dt` is built exactly from the `(year, month, day_hour)` tuple
  the dataloader is already sampling — no other behaviour changes.
- The clamp to `2008-01-01 07:00` is only needed when the sample lands
  in the 7-hour UrbClim gap (month = 0, day_hour < 7 of 2008). Without
  it, `.sel(method='nearest')` would snap forward anyway, but the two
  zarrs might snap differently; clamping keeps them lock-step.
- `global_time_idx` is deleted. The `.sel(time=str(year))` pre-slice on
  ERA5 is also deleted — `.sel(time=target_dt)` on the full store is
  equivalent and simpler.

The same patch was applied to `inspect_all_inputs.py` (three places:
ERA5 v1, ERA5 v2, UrbClim target) so diagnostic figures use the same
aligned logic.

---

## 4. Verification — `time_sync_debug_2.log`

Run of `debug_time_sync.py` **after** the patch:

### Section 1 — origins still differ, as expected

```
ERA5    : 2008-01-01T00:00:00 → 2017-12-31T23:00:00  (len=87672)
UrbClim : 2008-01-01T07:00:00 → 2017-12-31T23:00:00  (len=87665)
UrbClim[0] − ERA5[0] = 0 days 07:00:00
```

The gap is a property of the data and hasn't moved — the fix handles
it transparently via `.sel`.

### Section 3 — before/after across six test years

```
year  requested             OLD UrbC .isel[naive]    NEW UrbC .sel(dt)
2008  2008-07-03 04:00:00   2008-07-03 11:00:00      2008-07-03 04:00:00   ✓
2009  2009-07-04 04:00:00   2009-07-03 11:00:00      2009-07-04 04:00:00   ✓
2010  2010-07-04 04:00:00   2010-07-03 11:00:00      2010-07-04 04:00:00   ✓
2012  2012-07-03 04:00:00   2012-07-02 11:00:00      2012-07-03 04:00:00   ✓
2013  2013-07-04 04:00:00   2013-07-02 11:00:00      2013-07-04 04:00:00   ✓
2016  2016-07-03 04:00:00   2016-07-01 11:00:00      2016-07-03 04:00:00   ✓
```

Every tested year — including leap years 2008, 2012, 2016 — now returns
the exact requested timestamp. The OLD column shows what the model used
to train on; notice the drift growing with each leap year and the
constant 7-hour offset on top.

### Section 4 — 500-sample stress test

```
Pass (|ERA5 − UrbClim| ≤ 30 min): 500/500
Worst |ERA5 − UrbClim| drift   : 0.00 h
Worst |ERA5 − requested|       : 0.00 h
Worst |UrbClim − requested|    : 0.00 h
✓ All 500 samples return the same real hour in both zarrs.
```

500 random `(year, month, day_hour)` triples drawn from the same
distribution the dataloader uses. Zero drift on every single one.

### Section 5 — paired values look physical

```
winter (DJF) mean T2M = 274.68 K  (+1.5 °C)
summer (JJA) mean T2M = 292.99 K  (+19.8 °C)
✓ summer warmer than winter (expected)
```

20 random samples pulled ERA5 T2M and UrbClim tas together. Summer
averages 18 °C warmer than winter, which is what Alicante does —
confirmation that the two zarrs are now cross-wired correctly.

### Section 7 — performance

```
            .isel (ms)  .sel (ms)  ratio
ERA5           3.17        3.35    1.06x
UrbClim        1.53        1.76    1.15x
```

`.sel` is 6–15 % slower than `.isel`. Dataloader runs with 8 workers and
`prefetch_factor=4`, so this is absorbed; no bottleneck.

### VERDICT block

```
✅ TIME SYNC FIX VERIFIED
   ERA5 and UrbClim return the same hour across 500 random samples
   and every edge-case year tested. Worst cross-zarr drift: 0.0 min.
   Safe to proceed with training.
```

Plots written alongside the log:

| File | What it shows |
|---|---|
| `01_alignment_scatter.png` | drift (h) of ERA5 and UrbClim returns vs the requested datetime — flat line on 0 |
| `02_drift_histogram.png` | cross-zarr drift distribution — single spike at 0 |
| `03_seasonal_cycle.png` | monthly T2M at Alicante across training years — summer peak visible |
| `04_diurnal_cycle.png` | hourly T2M on July 15 across years — afternoon peak around 14:00 UTC |

---

## 5. Files changed

| File | Change |
|---|---|
| `v2_residual_unet.py` | Added `import datetime`; built `target_dt` in `__getitem__`; replaced two `.isel(time=…)` with `.sel(time=target_dt, method='nearest')`; removed `global_time_idx`. |
| `inspect_all_inputs.py` | Same pattern in three places (ERA5 v1, ERA5 v2, UrbClim). Also fixed the stale `stats_df[...].startswith('era5_var')` filter → `stats_df['type'] == 'dynamic'`, which broke when `zscore.py` started writing real ERA5 variable names. |
| `debug_time_sync.py` | Rewritten as a 500-sample stress test that simulates the dataloader and produces alignment plots. |
| `pipeline/check_pipeline.py` | No change — Section 1 already validates this and will now print ✓ on every test year. |
| `pipeline/zscore.py` | No change — it slices by whole year via `.sel(time=str(y))` and was never affected. |

---

## 6. Before kicking off training

1. **Re-run `pipeline.zscore`.** The stats CSV from `training_log_50029015.txt`
   has the `era5_var_*` alphabetical-sort bug, independent of this fix. Every
   ERA5 channel there was z-scored against the wrong μ / σ. Code is already
   fixed; only the on-disk CSV is stale.
2. Submit `sbatch train.sh`. Step [4/5] (`check_pipeline`) will re-verify
   time sync on the current zarrs; step [5/5] trains with aligned pairs.
