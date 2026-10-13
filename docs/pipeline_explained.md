# The Full Pipeline, Explained Feature by Feature

> Companion to `time_and_features_explained.md`. That file covers the
> time/cyclic encoding side in detail; **this** file walks the whole
> pipeline end-to-end and treats every other input feature as its own
> standalone explanation. Read it once before the talk and you should
> be able to answer any "how does X work?" question without going back
> to the code.
>
> All code pointers are to `v2_residual_unet.py` unless said otherwise.

---

## 0 · What this thesis is doing, in one sentence

We train a CNN that takes a coarse 31 km ERA5 weather pixel + a stack of
high-resolution static maps for one city and outputs the **lapse-rate
corrected residual** between UrbClim's 100 m ground-truth temperature
and ERA5 — i.e. only the local Urban Heat Island anomaly that local
geography can plausibly explain.

Everything below is in service of that one goal.

---

## 1 · The pipeline in one diagram

```
   ┌──────────────────────────────────┐
   │ pipeline/build_lookup.py         │   walk static + zarr trees,
   │                                  │   align city names with files
   └──────────────────────────────────┘  →  city_lookup.csv
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ pipeline/splits.py (seed=42)     │   70/15/15 disjoint cities,
   │                                  │   train years 2010–15
   └──────────────────────────────────┘  →  splits_metadata.csv
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ pipeline/zscore.py               │   compute μ/σ per channel
   │                                  │   on the train cohort only
   └──────────────────────────────────┘  →  train/val/test_statistics.csv
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ pipeline/check_pipeline.py       │   40 random checks: time
   │                                  │   sync, stats, NaN coverage
   └──────────────────────────────────┘  →  go / no-go verdict
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ v2_residual_unet.py              │   UrbClimDataset assembles
   │                                  │   23 inputs + target + mask;
   │                                  │   Residual U-Net trains on
   │                                  │   masked MSE loss
   └──────────────────────────────────┘  →  best_model.pth
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ test_full.py                     │   GENERAL + UHI-PEAK phases,
   │                                  │   per-city / per-CORINE
   │                                  │   metrics + plots
   └──────────────────────────────────┘  →  test_full/{general,uhi_22h}/
```

`train.sh` runs steps 1–5 in sequence under SLURM with `set -e`, so any
failure aborts before training can read a stale lookup or wrong stats.

---

## 2 · The 23 input channels — one explanation per channel

For every channel I list **what it is**, **why we have it**, **how it
is encoded for the model**, and (where relevant) **the gotcha**. All
broadcast to `(64, 64)` planes during stacking so every channel has the
same shape.

### 2.1 Channels 0–12 — the 13 ERA5 variables

| ch | name | physical meaning |
|---:|---|---|
| 0  | `u10` | 10 m zonal wind |
| 1  | `v10` | 10 m meridional wind |
| 2  | `t2m` | 2 m air temperature |
| 3  | `blh` | boundary-layer height |
| 4  | `cape` | convective available potential energy |
| 5  | `kx` | K-index (storm potential) |
| 6  | `ssr` | surface net solar radiation |
| 7  | `str` | surface net thermal radiation |
| 8  | `sp` | surface pressure |
| 9  | `sshf` | surface sensible heat flux |
| 10 | `tcc` | total cloud cover |
| 11 | `tcw` | total column water |
| 12 | `tp` | total precipitation |

- **What it is.** Hourly reanalysis from ECMWF on a ~31 km grid. We
  pull *one* coarse cell per city — the cell that contains the city
  centre — at the chosen timestamp.
- **Why.** They describe the synoptic state of the atmosphere. ERA5
  already knows the bulk weather; the CNN doesn't have to relearn it.
- **How encoded.**
  1. Pull the 13-vector with `era5_ds.sel(time=target_dt, method='nearest').isel(x, y).isel(variable=:13)` (`v2_residual_unet.py:178-182`).
  2. Z-score each variable using μ/σ from `train_statistics.csv`.
  3. Add Gaussian noise σ=0.05 (training only) for regularisation.
  4. Broadcast each scalar to a flat `(64, 64)` plane.
- **Gotcha.** `train_statistics.csv` row order must match the zarr's
  `variable` axis. The historic bug — pandas alphabetised
  `era5_var_0..12` as `0,1,10,11,12,2,3,…` — would silently z-score
  `t2m` against `tcc`'s μ/σ. Fixed by writing real ERA5 names to the CSV
  (`u10, v10, t2m, …`); checked by `pipeline/check_pipeline.py`.

### 2.2 Channel 13 — Elevation

- **What.** A 100 m DEM raster shipped per city in the static .nc file
  (`elevation` variable, read at `v2_residual_unet.py:128`).
- **Why.** Air gets colder with altitude (lapse rate ~0.65 K per 100 m).
  Without it the U-Net would have to learn that from scratch. Note
  elevation enters the system **twice**: as a feature *and* as part of
  the lapse-rate correction that builds the target (§3). That is fine —
  they serve different roles.
- **How encoded.** z-score with `elev_mean`, `elev_std` from the train
  stats CSV.
- **Gotcha.** The orography ERA5 uses (`OROG_PATH`, units of
  geopotential m²/s²) is divided by `GRAVITY = 9.80665` to recover
  metres before lapse-correcting ERA5's coarse `t2m`.

### 2.3 Channel 14 — AHF (anthropogenic heat flux)

- **What.** Static 100 m raster of city-emitted heat (buildings, traffic,
  AC), in W/m². Read from the static .nc as variable `AHF`.
- **Why.** Direct driver of UHI in dense urban cores, especially at
  night when natural radiative cooling is weak.
- **How encoded.** z-score.
- **Caveat.** AHF is a single annual climatology — it has no diurnal or
  seasonal signal. The hour-of-day cycle is left to the time channels.

### 2.4 Channel 15 — Land/water mask (binary)

- **What.** Coarse 0/1 raster, read from the `land_water_mask` variable
  in the static .nc file at `v2_residual_unet.py:131`.
- **Why.** Quick "is this pixel water?" hint so the network doesn't have
  to derive it from CORINE.
- **How encoded.** Fed *raw* — the only non-z-scored continuous-looking
  channel.
- **The gotcha (important).** This mask is **coarser than CORINE**.
  Rivers and harbours are often labelled "land" by `land_water_mask`
  while CORINE classes 13 (Rivers) / 14 (Inland water) say "water". The
  two channels disagree on a few hundred pixels per city. We have not
  fixed this — we let the network learn to live with the inconsistency.
  This is part of the motivation for the v3 SE-attention experiment
  (which can learn to deprioritise the noisier channel).
- **It is NOT used to mask the loss.** The training loss mask is
  built from `~isnan(t_urb_raw) & corine_reclass != 0`. So a pixel
  that `land_mask` calls "water" can still receive a gradient if
  UrbClim has a valid temperature there.

### 2.5 Channel 16 — CORINE land cover (integer 1..14)

This is the most subtle channel; explain it slowly.

- **What.** Static 100 m raster of land-use class (`CORINE` variable in
  the static .nc), originally with 44 raw codes from CORINE 2012.
- **Why.** Different land covers have wildly different thermal
  behaviours — concrete heats up and re-radiates at night, forests
  evapo-transpire, water has high heat capacity, etc. CORINE is the
  master "what is this pixel made of" channel.
- **The reclassification (Jonas Table 8.1).** 44 raw codes are folded
  into **14 active classes** by `CORINE_CLASS_MAP`
  (`v2_residual_unet.py:72-87`). A loop at lines 211–213 does the
  rewrite:
  ```python
  corine_reclass = np.zeros_like(isa_raw, dtype=np.float32)
  for new_class, orig_codes in CORINE_CLASS_MAP.items():
      corine_reclass[np.isin(isa_raw, orig_codes)] = new_class
  ```
  Result: every pixel becomes 0..14 where 0 = unmapped (mostly sea, raw
  codes 42/43/44, plus any garbage NaN).

  | id | name |
  |---:|---|
  | 1 | Urban |
  | 2 | Sub-urban |
  | 3 | Industrial |
  | 4 | Urban green |
  | 5 | Snow / Ice |
  | 6 | Bare soil |
  | 7 | Grassland |
  | 8 | Cropland |
  | 9 | Shrubland |
  | 10 | Woodland |
  | 11 | Broadleaf trees |
  | 12 | Needleleaf trees |
  | 13 | Rivers |
  | 14 | Inland water bodies |
  | (15) | Sea — **excluded** from training, see below |

- **Sea is dropped from the loss mask, not the input.** Two lines do
  this (`v2_residual_unet.py:215-216`):
  ```python
  drop_mask  = (corine_reclass == 0)
  valid_mask = valid_mask & (~drop_mask)
  ```
  Rationale (full write-up in `docs/changelogs/2026-04-25_corine_sea_decision.md`):
  UrbClim's `tas` is meaningless over open water — there is no ground
  truth there. Excluding sea is exactly the choice Jonas made. The cost
  is some coastal context.
- **How encoded.** Fed as a single integer channel of values in
  {1..14}, **not one-hot, not z-scored, not embedded.** This was an
  explicit choice the user made.
- **The honest caveat for the talk.** A CNN reads an integer channel as
  ordinal — it implicitly believes "Inland water (14) > Urban (1)" as
  magnitudes. Convolution kernels are linear; there is no reason
  "Industrial = 3" should be midway between "Urban = 1" and "Urban
  green = 4" in any thermal sense. The standard upgrades are:
  - **One-hot:** 14 binary channels. Removes ordinality completely; in
    exchange you add 13 channels and a linear `Conv1×1` at the input
    can collapse them back if useful.
  - **WorldCover continuous fractions:** 4 channels (BUILT, TREE,
    BAREANDVEG, WATER) of pixel-level class fractions in [0, 1]. This
    is what `next.md` flags as the recommended upgrade — Jonas uses
    both CORINE and WorldCover in his final feature set.
  Both are documented as on-roadmap.

### 2.6 Channel 17 — NDVI (vegetation index)

- **What.** PROBA-V monthly NDVI composite at 100 m. Stored as a
  `(12, H, W)` array in a separate per-city .nc file.
- **Why.** Vegetation cools through evapotranspiration; NDVI is the
  proxy for "how green is this pixel right now". It is the only static
  raster that varies through the year.
- **How encoded.**
  1. The dataset slices `ndvi_raw[month]` to pick the current
     month's composite (`v2_residual_unet.py:190-193`).
  2. z-score with `ndvi_mean`, `ndvi_std`.
- **Gotcha.** Some files come as `(12, 1, H, W)` or weirder shapes;
  the dataloader has a defensive `np.squeeze(...)[0]` for that case.

### 2.7 Channel 18 — Imperviousness

- **What.** Static fraction of sealed surface (concrete, asphalt, roof)
  per pixel in [0, 1]. Stored in `static_features_tmp_v2/*.nc` (newer
  static-features build).
- **Why.** Strong UHI predictor: more sealed surface = less
  evapotranspiration + more daytime heat storage = stronger nocturnal
  re-radiation. Top-tier SHAP feature in Lauwaet et al. 2024.
- **How encoded.** z-score.
- **Loading note.** The dataloader checks both `imperviousness` and
  `IMPERV` variable names; if neither is present it falls back to a
  zero plane (`v2_residual_unet.py:133-136`). The zero fallback exists
  only as a defensive measure for older static files; it is verified
  populated for 100/100 cities in the current build.

### 2.8 Channels 19, 20 — month sin / cos (cyclic seasonal encoding)

- **What.** Two scalars derived from the sampled month integer:
  ```python
  month_sin = sin(2π · month / 12)
  month_cos = cos(2π · month / 12)
  ```
- **Why a circle.** A linear `month/12` would teach the model that Dec
  is maximally far from Jan. Climatically Dec and Jan are *adjacent*.
  Putting `month` on a circle makes that topology correct.
- **Why both sin AND cos.** Either alone is many-to-one. `sin(March) =
  sin(September)` — the model can't tell summer-into-autumn from
  winter-into-spring with sin alone. The pair `(sin, cos)` gives a
  unique point on the unit circle for each month — no collisions.
- **How encoded.** Two flat `(64, 64)` planes, one per scalar
  (`v2_residual_unet.py:242-243`). Same value at every pixel — time
  is not a spatial feature.
- **Numbers the model actually sees.** January → (0.000, 1.000),
  April → (1.000, 0.000), July → (0.000, −1.000), October → (−1.000,
  0.000), back to January smoothly.
- **Caveat.** `month` is integer, so March 1 and March 31 produce
  identical sin/cos even though they're ~30 days apart. Upgrade is
  day-of-year sin/cos: `doy_sin = sin(2π·doy/365.25)`. Same channel
  count, strictly more informative.

### 2.9 Channels 21, 22 — hour sin / cos (cyclic diurnal encoding)

- **What.** Two scalars derived from the hour-of-day:
  ```python
  hour_of_day = day_hour % 24
  hour_sin    = sin(2π · hour_of_day / 24)
  hour_cos    = cos(2π · hour_of_day / 24)
  ```
- **Why both sin AND cos.** Same reason as month: cos alone collapses
  06:00 with 18:00 (both have cos = 0). The pair distinguishes them:
  06:00 → (1, 0), 18:00 → (−1, 0).
- **How encoded.** Two flat planes (`v2_residual_unet.py:244-245`).
- **Caveat for the talk.** `hour_of_day` is **UTC**, not local solar
  time. Lisbon's 14:00 UTC ≈ solar noon; Bucharest's 14:00 UTC is
  mid-afternoon (~2 h further along the day). The model has to learn a
  per-longitude offset. The fix is one line:
  `local_hour = (utc_hour + lon/15) mod 24`, then sin/cos of that. On
  the roadmap; documented in `docs/roadmap/simple_improvements.md`.
- **Bigger upgrade.** Replace the diurnal cyclic pair with a real
  `cos(zenith_clipped)` channel computed via `pvlib` — that encodes
  *how much sun is hitting the ground right now* as a function of
  latitude × declination × hour angle. Top-5 SHAP feature in Blancke;
  full physics derivation in `docs/roadmap/solar_zenith_feature.md`.

---

## 3 · The residual target — the math, slowly

This is the trick that makes the problem learnable. It deserves its
own slide. Code is at `v2_residual_unet.py:218-220`.

We have two temperatures at the chosen timestamp:

- `t2m_era5_raw` — ERA5's 2 m temperature at the coarse city pixel,
  taken from above the orography height `elev_low = orog(lat, lon) / g`.
- `t_urb_raw` — UrbClim's 100 m 2 m temperature at every pixel of the
  city, taken at the local elevation `elev_pixel`.

Both are real 2 m temperatures, but at different reference heights.
We can't subtract them directly without first putting them on the same
altitude.

**Step 1: Lapse-correct both up to mean sea level.** With Γ = 0.0065 K/m:

```
t2m_era5_msl = t2m_era5_raw  + elev_low   * Γ        # K, sea-level value of ERA5
t_urb_msl    = t_urb_raw     + elev_pixel * Γ        # K, sea-level value at every pixel
```

**Step 2: Subtract.**

```
target_residual = (t_urb_msl − t2m_era5_msl) * valid_mask
```

This number is the part of the local temperature that the lapse rate
cannot account for — i.e. the local UHI anomaly plus all the residual
land-cover effects. ERA5 already knows the bulk weather; the lapse rate
handles altitude. Whatever is left is what the CNN has to learn.

At inference, the absolute prediction in °C is reconstructed by adding
the base temperature back:

```
T_pred(lat,lon,h,t) = (residual_pred + base_T) − 273.15
```

where `base_T` is the lapse-rate-corrected ERA5 t2m back at the pixel's
own altitude. `test_full.py` does this at line 309-310 (`pred_abs = pr +
bt`) before computing MAE/MSE, so the reported numbers are in absolute
°C — directly comparable to Jonas Blancke's RF baseline.

---

## 4 · The loss mask, in full

The `valid_mask` controls what the model is graded on. Pixels with
`mask == 0` are excluded from the MSE / MAE loss completely
(`v2_residual_unet.py:408-432`).

A pixel is **invalid** (mask = 0) if **any** of:

1. UrbClim has NaN there at that timestamp (`np.isnan(t_urb_raw)`).
2. The reclassified CORINE is 0 — meaning unmapped or sea (raw 42/43/44).
3. (Implicit) The pixel falls outside a city's footprint after the
   64×64 random crop (those rows/cols are zero-padded with mask = 0).

Notably **NOT** in the mask: the binary `land_water_mask`. That is
deliberate — it disagrees with CORINE on rivers, and CORINE is treated
as the source of truth for the loss.

---

## 5 · The model — Residual U-Net

Architecture is at `v2_residual_unet.py:274-330`. Building block first.

### 5.1 The `ResidualConv` block

```
main:     Conv3×3 → BN → ReLU → Conv3×3 → BN
shortcut: identity                  if in_ch == out_ch
        : Conv1×1 → BN              if in_ch != out_ch
output:   ReLU(main(x) + shortcut(x))
```

The shortcut is the **gradient highway**. With 4 pool levels and
10+ convs in v1, gradients vanished before reaching the early encoder
layers — sharp temperature boundaries blurred. The 1×1 shortcut
guarantees a clean linear path for the gradient to flow backwards even
if the convolutional path saturates.

### 5.2 The full U-Net

| stage | op | channels | spatial |
|---|---|---|---|
| `inc` | ResidualConv | 23 → 96 | 64 × 64 |
| `down1` | MaxPool + ResConv | 96 → 192 | 32 × 32 |
| `down2` | MaxPool + ResConv | 192 → 384 | 16 × 16 |
| `down3` | MaxPool + ResConv | 384 → 768 | 8 × 8 |
| `bottleneck` | MaxPool + ResConv + Dropout2d(0.05) | 768 → 1536 | 4 × 4 |
| `up1 + dec1` | Bilinear↑ + Conv1×1 + cat + ResConv + Dropout | 1536 → 768 | 8 × 8 |
| `up2 + dec2` | … | 768 → 384 | 16 × 16 |
| `up3 + dec3` | … | 384 → 192 | 32 × 32 |
| `up4 + dec4` | Bilinear↑ + Conv1×1 + cat + ResConv (no dropout) | 192 → 96 | 64 × 64 |
| `out` | Conv1×1 | 96 → 1 | 64 × 64 |

- **~26 M params total.** The bottleneck ResidualConv (768 → 1536) is
  about 16 M of those — ~60 % of the model — which is why the only
  Dropout2d(0.05) sits right after it (and on three decoder stages).
- **`align_corners=True`** in upsampling avoids the half-pixel offset
  that would soften sharp temperature boundaries.
- **Why 4 pool levels.** Each pool doubles the receptive field. At 4
  levels the bottleneck "sees" most of a 64×64 crop in one filter
  application, which is roughly the scale at which advection plumes
  (cool air dragged from green corridors into hot streets) operate.

### 5.3 Why a U-Net at all?

Two structural reasons:

1. **Skip connections preserve high-resolution detail.** The encoder
   downsamples to capture context; the decoder reuses encoder feature
   maps to reproduce sharp edges. Without skips, decoder upsampling
   alone produces blurry maps.
2. **Encoder–decoder spans multiple scales.** UHI is a multi-scale
   phenomenon: city-wide trends (10 km) and street-edge gradients
   (100 m) coexist. A plain CNN at one resolution can't capture both.

Compared to Blancke's per-pixel Random Forest, the structural advantage
is the same: the U-Net sees a **neighbourhood** of context (~6.4 km at
64×64 crops × 100 m) when predicting any given pixel, which is exactly
where street-canyon trapping and wind advection live. RF has no
neighbourhood — every pixel is independent.

---

## 6 · The sanity check — `plot_target_samples`

`v2_residual_unet.py:335-358`, called once at startup
(`v2_residual_unet.py:369`).

What it does:

1. Pick 6 evenly-spaced indices through the dataset.
2. For each, pull `target` and `mask`. `np.where(mask, target, NaN)` —
   only valid land pixels are rendered; sea / dropped pixels appear
   white.
3. Symmetric `RdBu_r` colormap, `vmax = max(|residual|)` per sample, so
   positive (red) and negative (blue) residuals are equally visible
   around 0.
4. Saves to `MY_DATA_DIR/diagnostics/v2_target_samples.png`.

What a healthy plot looks like:

- Cities are red blobs (UHI > 0) surrounded by cooler rural blue.
- Numbers in the [-5, +5] °C ballpark — typical UHI magnitudes after
  lapse correction.
- White (NaN / sea / unmapped) regions over open water and bodies.
- No residual signal correlated with elevation gradient — if you see
  elevation contours bleeding through, the lapse-rate correction has
  the wrong sign somewhere.

This is the cheapest possible "data is sane before I burn 6 GPU hours"
plot, and it's what would have caught the historic water bug (where the
target tensor *contained* the CORINE water mask) and the leap-year
drift (where target patterns didn't match the requested season) before
they wasted training runs.

---

## 7 · Training mechanics

`v2_residual_unet.py:399-460`. The bullets that matter:

- **Optimiser:** AdamW, base LR 1.5e-5, weight_decay 1e-4. Decoupled
  L2 — better for ResNet-style nets than vanilla Adam's coupled L2.
- **Schedule:** OneCycleLR, max LR 8e-5, pct_start 0.1, cosine anneal.
  10 % warm-up to peak, then anneal across the remaining 90 %.
- **Batch:** 32 raw, 128 effective via 4-step gradient accumulation.
  Lets us fit the 1536-ch bottleneck on one GPU.
- **Gradient clip:** max-norm 2.0. Stability against ERA5 outliers
  (CAPE can be huge).
- **Loss:** masked MSE on `mask > 0.5`. L1 (MAE) is computed alongside
  for monitoring only.
- **150 epochs**, early stopping patience set to 150 — i.e. effectively
  disabled. The number is a safety net only.
- **Train samples per "epoch":** 70 cities × 50 = 3500. Val: 15 × 5 =
  75 (deterministic). Note this redefines "epoch" — it's not a pass
  over a fixed dataset. Useful to mention in the thesis methodology.
- **Resume:** `latest_model.pth` saves model + optimizer state every
  epoch. Limitation: scheduler state is **not** saved, so an
  interrupted run resumes the optimiser correctly but restarts
  OneCycleLR at step 0. Worth fixing if a run gets killed.

### 7.1 What is monitored vs. logged

- **Per step:** MSE printed to stdout every batch.
- **Per epoch:** train/val MSE/L1 + LR printed; appended to `history`
  dict; saved into the checkpoint.
- **TensorBoard:** `SummaryWriter` is created
  (`v2_residual_unet.py:390`) but **never written to**. If you want TB
  curves, four `writer.add_scalar(...)` calls per epoch is enough.

---

## 8 · The test pipeline (`test_full.py`)

Two phases, on the 15 disjoint test cities, years 2008 and 2017:

- **GENERAL phase** (`test_full.py:109-130`): random hours across both
  years, every month, every hour-of-day. 24 hours/city by default.
- **UHI-PEAK phase** (`test_full.py:133-157`): JJA, 22:00 UTC only —
  the canonical UHI window.

Per sample (`test_full.py:274-375`):

1. Forward pass → `pred_residual`.
2. Add the base ERA5 temperature back: `pred_abs = pred_res + base_t`.
   **Metrics are reported in absolute °C, not in residual space**, so
   they are directly comparable with Jonas's RF.
3. Compute MAE / MSE on masked pixels only.
4. Save a 3-panel triptych: `Truth | Pred | |error|`.
5. Subsample 1000 random valid pixels, store
   `(pred, truth, corine_id, hour, month, city)` for downstream
   per-class / per-hour analysis.

Aggregations (`test_full.py:378-441`):

- Global RMSE / MAE / bias / R².
- Per-city table.
- Per-CORINE-class table + boxplot (Jonas Fig 5.8 style).
- Diurnal bias plot (general phase only — UHI phase is single hour).
- Monthly bias plot.
- MAE histogram + per-city horizontal bar chart.

Outputs land at `MY_DATA_DIR/test_full/{general,uhi_22h}/`.

### 8.1 Headline numbers (current best checkpoint, epoch 15)

| phase | RMSE | MAE | bias | R² |
|---|---:|---:|---:|---:|
| GENERAL | **1.31 °C** | 0.99 | +0.43 | **0.970** |
| UHI-PEAK (JJA, 22:00) | 1.65 | 1.27 | +0.72 | 0.878 |

For comparison: Blancke 2024 RF reports RMSE ≈ 1.30 °C, bias ≈ −0.11 °C
on his disjoint-city test cohort. UrbClim itself has RMSE 0.77 °C
against weather stations — that is the upper bound nobody can beat.

### 8.2 Where the error lives

- **Cluster pattern.** Northern / maritime cities (Copenhagen, Utrecht,
  Newcastle, Dublin, Antwerp) → best (MAE 0.5–0.8). Mediterranean /
  continental (Malaga, Tirana, Marseille, Prague) → worst (MAE
  1.1–1.5). Same 3-cluster effect Blancke identifies.
- **UHI peak collapse.** Malaga at JJA 22:00 UTC: MAE 2.79, R² 0.19.
  Diagnosis is **one phenomenon, not two**: Mediterranean coast +
  summer night = ERA5 sea-pixel leak meeting Mediterranean shrubland.
  Both are explicitly named in Jonas's §6 as the RF's weak regime.
- **Per-CORINE pattern at UHI peak.** Worst classes are Shrubland (MAE
  2.11), Woodland (1.90), Needleleaf (1.52) — all Mediterranean rural.
  Best are Rivers (0.98), Sub-urban (1.05), Urban green (1.05).

---

## 9 · Honest limitations (what to put in the "what's not working" slide)

In rough order of severity:

1. **+0.43 °C systematic warm bias.** Grows to +0.72 at UHI peak.
   Random-error component looks fine; this is a uniform shift,
   correctable for free.
2. **Mediterranean coastal cities collapse at the UHI peak** (Malaga
   R² 0.19). Caused by ERA5 sea-pixel leak + shrubland behaviour.
3. **Ordinal CORINE channel.** A 14-class integer is mathematically
   ordinal to a CNN. Standard upgrade: WorldCover continuous fractions.
4. **No solar geometry feature.** `month_sin/cos + hour_sin/cos` is a
   weak proxy for Jonas's SOLAR_ELEV / DECL (top-tier in his Gini
   plot).
5. **UTC, not local solar time** for the diurnal feature.
6. **Coarse `land_mask` disagrees with CORINE** on rivers/harbours.
   v3 SE attention is supposed to learn around this.
7. **Missing urban-morphology static channels.** `next.md` flags
   COAST (km to coast), BUILT_FRAC (building footprint), HEIGHT (mean
   building height), POP (population density) — all available as 100 m
   rasters in Jonas's preprocessing pipeline, none currently in our
   stack.
8. **TensorBoard writer never written to.** Cosmetic, four lines.
9. **Scheduler state not saved on checkpoint.** OneCycleLR restarts at
   step 0 if the SLURM job is killed and resumed.
10. **Static cache holds all city rasters in RAM.** Fine for 70 train
    cities; would not scale beyond ~200.

---

## 10 · Improvement roadmap (consolidated from `MASTER.md` §6 and `next.md`)

### Tier 0 — free, no retraining

- **Per-city bias correction at inference.** Subtract per-city mean
  residual measured on the validation split. Drops bias from +0.43 to
  ~0 in one afternoon.

### Tier 1 — cheap (1–2 h each)

- Wire the **TensorBoard writer** (4 `add_scalar` calls per epoch).
- **Save scheduler state** in the checkpoint dict.
- **Local solar time** for `hour_sin/cos`: shift UTC by `lon/15`.
- **Day-of-year sin/cos** instead of month-integer sin/cos.
- **Distance-to-coast** static channel (one raster ingest).
- **Per-city median imputation** for any NaN ERA5 channel; CBH.fillna(0).

### Tier 2 — medium (~1 day each)

- **`cos_zenith_clipped`** via `pvlib`. Top-5 SHAP feature in Blancke.
  Full derivation in `docs/roadmap/solar_zenith_feature.md`.
- **WorldCover continuous fractions** (4 channels: BUILT, TREE,
  BAREANDVEG, WATER) alongside or instead of integer CORINE.
- **Re-introduce CORINE class 15 (Sea) on the input side** (keep the
  loss-mask). Closes the coastal-context gap. Step-by-step in
  `docs/changelogs/2026-04-25_corine_sea_decision.md` §7.
- **Add missing ERA5 vars** if not yet present: T_SK (skin temperature),
  SM (soil moisture swvl1), CBH (cloud base height), RH (relative
  humidity). T_SK and SM in particular explain a chunk of UHI residual
  variance through radiative forcing and Bowen ratio.
- **Weather-regime evaluation** — 12 buckets (rainy/dry × cloudy/clear
  × windy/calm × day/night). Blancke §5.4.3.

### Tier 3 — bigger experiments

- **V3 = V2 + Squeeze-and-Excitation channel attention.** Already
  sketched in `res_unet_se.py`. Hypothesis: SE will deprioritise the
  noisier `land_mask` channel where CORINE disagrees.
- **Larger `CROP_SIZE`** (96 or 128). Gives the model more advection
  context. Costs GPU memory; reduce batch.
- **Weighted MSE on extreme residuals** for heatwave peaks.
- **ERA5-Land** instead of ERA5 for coastal cities (Jonas's own §6
  recommendation).

---

## 11 · Where to look in the code (one-line cheats)

```
v2_residual_unet.py:72-87       CORINE_CLASS_MAP (Jonas Table 8.1)
v2_residual_unet.py:122-142     static cache built at __init__
v2_residual_unet.py:147-269     __getitem__ — the whole sample lifecycle
v2_residual_unet.py:170-176     the four cyclic time scalars
v2_residual_unet.py:211-216     CORINE reclass + sea drop
v2_residual_unet.py:218-220     residual target math
v2_residual_unet.py:236-250     channel-stack assembly
v2_residual_unet.py:274-297     ResidualConv block
v2_residual_unet.py:299-330     UrbClimUNet (the full model)
v2_residual_unet.py:335-358     plot_target_samples (sanity check)
v2_residual_unet.py:399-460     training loop + checkpointing

pipeline/config.py              URB_START, target_datetime, year-splits
pipeline/build_lookup.py        city_lookup.csv builder
pipeline/splits.py              70/15/15 deterministic split (seed=42)
pipeline/zscore.py              μ/σ on the train cohort only
pipeline/check_pipeline.py      40-point pre-flight verifier
pipeline/extract.py             helper to pull per-city zarr slices
pipeline/lookup.py              path resolver

test_full.py                    GENERAL + UHI-PEAK test driver
test_model.py                   TestingUrbClimDataset + per-sample plot
train.sh                        SLURM driver, 5 steps, set -e

docs/changelogs/time_sync_fix.md           leap-year + 7-h gap fix
docs/changelogs/2026-04-25_corine_sea_decision.md   sea exclusion rationale
docs/changelogs/2026-04-25_test_results.md          full v2 test write-up
docs/reference/time_and_features_explained.md       cyclic-encoding deep dive
docs/roadmap/solar_zenith_feature.md                next-step solar feature
docs/roadmap/simple_improvements.md                 day-of-year, local time, etc.
docs/roadmap/next.md                                Top-5 priorities
```

---

## 12 · One-paragraph summary (use as the abstract)

This thesis builds a deep-learning emulator of VITO's UrbClim
urban-climate model. A 23-channel Residual U-Net takes ECMWF's coarse
31 km ERA5 reanalysis at the city's containing pixel (13 meteorological
variables), plus high-resolution static descriptors of the city
(elevation, AHF, land/water mask, CORINE 14-class integer,
imperviousness, monthly NDVI), plus a cyclic encoding of the calendar
time (sin/cos of month and hour-of-day), and predicts the
**lapse-rate-corrected residual** between UrbClim's 100 m
ground-truth temperature and ERA5 — i.e. only the local Urban Heat
Island anomaly. Training uses 70 European cities (2010–2015), with
disjoint validation (15 cities, 2009/2016) and disjoint test cohorts
(15 cities, 2008/2017). On the general test the model achieves
**RMSE 1.31 °C, MAE 0.99 °C, R² 0.97** — matching Blancke's 2024
Random-Forest baseline on cities never seen in training. The error
concentrates at the canonical UHI peak (JJA, 22:00 UTC) over
Mediterranean coastal cities — a regime explained by the ERA5
sea-pixel leak meeting Mediterranean shrubland behaviour, both of
which Blancke's thesis names as the RF's weak regime too. Once
trained, the emulator predicts a full city-hour in under a second,
against weeks for the underlying numerical model.
