# My model — end-to-end walkthrough

> A complete, in-order explanation of what the code actually does — from the moment a sample is drawn from disk to the moment a weight is updated. Every choice is annotated with what it does and why it has to be there. Read top-to-bottom before the defence; everything you might be asked is in here.

**Files this document covers:** `pipeline/config.py`, `pipeline/extract.py`, `pipeline/splits.py`, `pipeline/zscore.py`, `v2_residual_unet.py`.

---

## Table of contents

1. [The dataset (`UrbClimDataset`)](#1-the-dataset-urbclimdataset)
2. [The model (`UrbClimUNet`)](#2-the-model-urbclimunet)
3. [The training loop](#3-the-training-loop)
4. [Hyperparameter cheat sheet](#4-hyperparameter-cheat-sheet)
5. [End-to-end one-sentence summary](#5-end-to-end-one-sentence-summary)

---

## 1. The dataset (`UrbClimDataset`)

The dataset is what turns raw disk files into a tensor the network can consume. Every call to `dataset[idx]` produces one fully-formed training sample. Here is everything that happens, in order.

### 1.1 What a single sample is

| Property | Value |
|---|---|
| One sample | one city, one random timestamp, one 64×64 spatial crop |
| Tensor shape on disk | `(25, 64, 64)` — 23 inputs + target + valid mask |
| Tensor shape the model sees | `(23, 64, 64)` — `crop[:23]` |
| Train multiplier | `__len__ = n_cities × 100` → **6000 samples / train epoch** |
| Val multiplier | `__len__ = n_cities × 5` → **100 samples / val epoch** |

**Why this design:** a single forward pass does not need a whole city tile — it needs a representative chunk with both urban and rural pixels so the loss has signal from both regimes. 64×64 covers about 6.4 km × 6.4 km at 100 m resolution, which is a typical European-city extent.

### 1.2 The static cache

For each city, the dataset loads the per-city 100 m static rasters **once** and caches them in memory: elevation, AHF, CORINE, NDVI, imperviousness, land/water mask. They never change with time, so re-reading them on every `__getitem__` would be wasteful.

**Why it matters:** a single training run draws ~900 000 samples; if every sample re-read the static NetCDF from disk the run would be I/O-bound on the cluster. Caching turns this into a memory-bound problem an A100 node easily handles.

### 1.3 Picking the timestamp

| Mode | Logic |
|---|---|
| **train** | `np.random.choice(years)`, `randint(0, 12)`, `randint(0, 720)` — fresh on every call |
| **val** | `RandomState(idx + 42)` — deterministic so val set is identical every epoch |

**Why:** training needs i.i.d. samples across the full hour distribution so the network learns the diurnal and seasonal cycles. Validation must be exactly the same set every epoch so the val-loss curve is comparable across epochs.

### 1.4 Loading the dynamic data

| Source | What is loaded | Resolution |
|---|---|---|
| ERA5 zarr (global) | 13-element vector at the city's coarse cell | ~31 km, hourly |
| UrbClim zarr (per city) | 100 m `tas` raster for that timestamp | 100 m, hourly |
| NDVI NetCDF (per city) | 100 m NDVI raster for the right month | 100 m, monthly |

The ERA5 variables, in fixed order, are: `u10, v10, t2m, blh, cape, kx, ssr, str, sp, sshf, tcc, tcw, tp`.

**Why three separate stores:** ERA5 is global so one shared zarr is most efficient; UrbClim is per-city so one zarr per city is the natural layout; NDVI varies monthly so it has its own temporal axis.

### 1.5 Building the valid-pixel mask

```python
valid_mask  = ~np.isnan(t_urb_raw)                              # drop NaN UrbClim pixels
valid_mask &= (corine_reclass > 0) & (corine_reclass != 15)     # drop unmapped + sea
```

Three pixel types are excluded from the loss:

| Type | Reason for exclusion |
|---|---|
| **Sea (CORINE 15)** | UrbClim's `tas` is undefined over open water |
| **Unmapped (CORINE 0)** | raw codes that did not match the 1–14 remap |
| **NaN in UrbClim** | typically edges and undefined regions |

**Important nuance:** the CORINE integer 15 **stays in the input** channel so the convolutions can still see the coastline. Only the *loss* drops sea pixels. The network is therefore "aware" of where the sea is but is never trained or scored on it.

### 1.6 Building the lapse-rate-corrected residual target

```python
elev_low        = orog_global(city_lat, city_lon) / GRAVITY        # ERA5 elevation in metres
t2m_era5_msl    = t2m_era5_raw + elev_low * LAPSE_RATE             # ERA5 reduced to sea level
target_residual = (t_urb_raw + elev_raw * LAPSE_RATE) - t2m_era5_msl
target_residual *= valid_mask                                       # zero where invalid
```

This is the most important methodological choice in the thesis. The network does **not** predict UrbClim's raw 2 m temperature. It predicts the difference between UrbClim and ERA5 after both have been reduced to mean sea level using the dry-adiabatic standard lapse rate Γ = 0.0065 K m⁻¹.

**Why this target:**

- The dominant variation in 2 m air temperature is the atmospheric column profile, which ERA5 already captures perfectly well.
- Reducing both to mean sea level strips out the bulk altitude signal.
- What is left is the *local urban anomaly* — the part driven by concrete, vegetation, water, anthropogenic heat.
- That is the part where the static rasters matter and where the network's capacity should go.
- Blancke (2024) reports the lapse correction alone dropped his RMSE from 1.90 to 1.71 °C; we adopt the choice verbatim so the comparison is apples-to-apples.

At inference time the residual is added back to the lapse-corrected ERA5 baseline to recover an absolute temperature.

### 1.7 Normalising the inputs

```python
norm_era5    = (era5_vector - era5_means) / (era5_stds + 1e-8)              # z-score
if mode == 'train':
    noise    = np.random.normal(0, 0.05, size=norm_era5.shape)
    norm_era5 += noise                                                       # Bishop input-noise reg.
norm_elev    = (elev_raw    - elev_mean)    / (elev_std    + 1e-8)
norm_ahf     = (ahf_raw     - ahf_mean)     / (ahf_std     + 1e-8)
norm_ndvi    = (ndvi_raw    - ndvi_mean)    / (ndvi_std    + 1e-8)
norm_imperv  = (imperv_raw  - imperv_mean)  / (imperv_std  + 1e-8)
```

Every continuous channel is z-scored using **training-cohort** statistics so no information leaks across splits. The 13 ERA5 channels also get a Gaussian-noise perturbation during training only.

| Operation | Purpose |
|---|---|
| Z-score | Put all inputs on a common scale (~0 mean, ~1 std) so the optimiser does not need a separate effective learning rate per channel |
| Gaussian noise σ=0.05 on ERA5, train only | Bishop 1995 input-noise regularisation; ERA5 itself has uncertainty (~0.5–1 °C), so injecting noise discourages the network from over-fitting any one realisation of the forcing |
| No noise on the statics | Elevation, AHF, NDVI, imperviousness, CORINE are deterministic features of the city — they have no observational uncertainty, so noising them would hurt rather than help |

### 1.8 Stacking the input tensor

The on-disk crop has **25 channels**. The network sees the first 23.

| Ch | Content | Encoding |
|---:|---|---|
| 0–12 | 13 ERA5 variables broadcast as flat planes | z-score + Gaussian noise σ=0.05 (train only) |
| 13 | Elevation (100 m DEM) | z-score |
| 14 | Anthropogenic heat flux (Flanner 2009) | z-score |
| 15 | Coarse land/water mask | binary 0/1 |
| 16 | **CORINE integer 1..15** (15 = Sea, input only, masked in loss) | raw integer |
| 17 | NDVI (monthly, PROBA-V) | z-score |
| 18 | Imperviousness (Copernicus HRL) | z-score |
| 19 | `sin(2π · doy / 365.25)` | raw cyclic |
| 20 | `cos(2π · doy / 365.25)` | raw cyclic |
| 21 | `sin(2π · lst / 24)` where `lst = (UTC + lon/15) mod 24` | raw cyclic, local solar time |
| 22 | `cos(2π · lst / 24)` | raw cyclic, local solar time |
| 23 | Target residual (°C) | — |
| 24 | Valid mask (0/1) | — |

**Why 23 inputs, not 25:** the target and the mask are labels and supervision metadata — not features the model is allowed to peek at. They are sliced off after cropping.

**Why cyclic time encodings (sin/cos):** a naive linear encoding such as `(month+1)/12` would tell the model that December and January are at *opposite* ends of the unit interval. The seasonal cycle is periodic, so we put months on the unit circle through the (sin, cos) pair: December and January then sit at adjacent points and the network sees them as neighbours. The pair is required because a single sinusoid is two-to-one (sin alone collapses March and September, and 06:00 with 18:00). The same logic gives the hour-of-day pair.

### 1.9 Cropping

```python
if mode == 'train':
    y_c, x_c = np.random.randint(0, h - 64 + 1), np.random.randint(0, w - 64 + 1)
else:
    rs_crop  = np.random.RandomState(idx + 100)               # seeded
    y_c, x_c = rs_crop.randint(...), rs_crop.randint(...)
crop = full_stack[:, y_c : y_c + 64, x_c : x_c + 64]
```

| Mode | Crop position |
|---|---|
| train | random — different chunk of every city every time |
| val   | deterministic — same crop per sample every epoch |

**Why random crops on train:** the network is a fully convolutional U-Net with a ~64-pixel receptive field at the bottleneck. Random crops mean each epoch the network sees a different chunk of every city, giving much more sample diversity than fixed crops.

**Why 64×64:** divisible by 16 (four pool levels each divide by 2), and ~6.4 km is a sensible city-extent size.

**Why seeded validation crop:** the val set must be deterministic so the val-loss curve is comparable across epochs.

### 1.10 Padding for small cities

```python
crop = np.zeros((25, 64, 64), dtype=np.float32)
crop[:, :h, :w] = full_stack                                   # paste the (small) city
```

When a city tile is smaller than 64×64, the crop is a zero array with the city pasted into the top-left. The padded pixels have `valid_mask = 0`, so they are excluded from the loss.

**Why this matters:** skip-connection alignment in a four-pool U-Net **requires** H and W divisible by 16. Padding ensures every sample is compliant without polluting the loss.

---

## 2. The model (`UrbClimUNet`)

A 4-level Residual U-Net with ~26 M parameters. Every piece is here for a reason.

### 2.1 The building block — `ResidualConv`

```
main(x)      = BN( Conv3×3( ReLU( BN( Conv3×3(x) ) ) ) )
shortcut(x)  = x                       if C_in == C_out
             = BN( Conv1×1(x) )        otherwise
output       = ReLU( main(x) + shortcut(x) )
```

A two-convolution main path plus a parallel shortcut; the output adds them together before the final ReLU.

**Why a residual block instead of plain Conv-BN-ReLU:**

- In deep networks the gradient vanishes through long convolutional chains.
- The upsampling path tends to blur sharp boundaries.
- The shortcut connection is a **gradient highway** — even if the convolutional path learns nothing, the raw signal flows unaltered, so the gradient with respect to the input always has a direct path back from the loss.
- He et al. (2015) introduced this for ResNet; it has become the standard mitigation of vanishing gradients in deep dense regression.

**Why the 1×1 projection on the shortcut:** only used when channel count changes (e.g. 96 → 192 across stages). It is the cheapest way to make the residual addition shape-compatible.

### 2.2 The 4-level topology

| Stage | Operation | Channels | Spatial |
|---|---|---:|:---:|
| `inc`        | `ResidualConv`                                              | 23 → 96    | 64×64 |
| `down1`      | MaxPool + `ResidualConv`                                    | 96 → 192   | 32×32 |
| `down2`      | MaxPool + `ResidualConv`                                    | 192 → 384  | 16×16 |
| `down3`      | MaxPool + `ResidualConv`                                    | 384 → 768  | 8×8 |
| `bottleneck` | MaxPool + `ResidualConv` + `Dropout2d(0.05)`                | 768 → **1536** | **4×4** |
| `up1 + dec1` | Bilinear↑ + Conv1×1 + cat skip + `ResConv` + `Dropout2d`    | 1536 → 768 | 8×8 |
| `up2 + dec2` | same                                                        | 768 → 384  | 16×16 |
| `up3 + dec3` | same                                                        | 384 → 192  | 32×32 |
| `up4 + dec4` | same (**no dropout**)                                       | 192 → 96   | 64×64 |
| `out`        | Conv1×1                                                     | 96 → 1     | 64×64 |

- The encoder halves the spatial size and doubles the channel count at every stage: features get smaller in space but richer in semantics.
- The **bottleneck** is the richest representation (1536 channels at 4×4). About 16 M of the 26 M parameters live here.
- The decoder upsamples back to 64×64 with bilinear interpolation, applies a 1×1 conv to halve channel count, **concatenates the matching encoder skip**, and runs another `ResidualConv`.
- The final 1×1 conv reduces 96 channels to a single channel — the residual prediction at every pixel.

**Why a U-Net at all:** the skip connections give the decoder direct access to the high-resolution detail that the bottleneck has compressed away. Without them you cannot recover sharp park / concrete / coastline boundaries.

**Why bilinear upsampling with `align_corners=True`:** keeps the encoder skips and the decoder upsamples on the same pixel grid and avoids the half-pixel offset that softens sharp temperature boundaries. Transposed convolutions, the alternative, tend to produce checkerboard artefacts.

**Why 4 levels:** deep enough that the bottleneck receptive field covers an interesting spatial region (~64 pixels ≈ 6.4 km); shallow enough that 4× pooling on a 64×64 input still gives a 4×4 bottleneck instead of collapsing to 1×1.

### 2.3 Dropout2d in the bottleneck and decoder

```python
self.dropout = nn.Dropout2d(p=0.05)
...
x5 = self.dropout( self.bottleneck(x4) )                       # in bottleneck
d1 = self.dropout( self.dec1( cat[ x4, up1(x5) ] ) )           # in dec1
d2 = self.dropout( self.dec2( cat[ x3, up2(d1) ] ) )           # in dec2
d3 = self.dropout( self.dec3( cat[ x2, up3(d2) ] ) )           # in dec3
# d4 (final decoder stage) is NOT dropped
```

`Dropout2d` zeroes **whole feature channels** (not individual pixels) with probability *p*, training-mode only.

| Question | Answer |
|---|---|
| Why dropout? | Regularisation — with 26 M parameters the network could memorise the training cohort. |
| Why *p* = 0.05 (small)? | Too much dropout in dense regression destroys high-frequency content the decoder is meant to recover. 5 % breaks parameter co-adaptation without wiping out information. |
| Why bottleneck + first three decoder stages? | The bottleneck holds ~16 M of 26 M parameters — highest over-fitting risk. `dec4` is left dropout-free because at the highest spatial resolution dropout removes the very high-frequency content the model is trying to reconstruct. |
| Why `Dropout2d` and not `Dropout`? | `Dropout2d` drops entire feature channels, which is the correct granularity for convolutional features. Plain `Dropout` drops independent pixels and leaks information across channels. |

---

## 3. The training loop

### 3.1 What one epoch does

```python
for epoch in range(NUM_EPOCHS):                                # 150
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):                   # 188 batches
        outputs = model(inputs)                                # 32 samples
        loss    = MSE(outputs[masks], targets[masks]) / 4       # accumulation
        loss.backward()
        if (i + 1) % 4 == 0 or (i + 1) == len(train_loader):
            clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()                                   # weight update
            scheduler.step()                                   # LR update
            optimizer.zero_grad()
    # then validation pass
```

Per epoch: **6000 samples → 188 mini-batches → 47 optimizer steps → val pass → log**.

### 3.2 Data volumes per training run

| Quantity | Train | Val |
|---|---:|---:|
| Cities in pool | 60 | 20 |
| Sample multiplier | 100 | 5 |
| Samples / epoch | **6 000** | 100 |
| Batch size | 32 | 32 |
| Batches / epoch | **188** | 4 |
| Gradient accumulation | 4 | n/a |
| Optimizer steps / epoch | **47** | n/a |
| Effective batch | **128** | — |
| Pixels / sample | 4 096 | 4 096 |
| Raw pixels / epoch | ≈ 24.6 M | ≈ 410 k |
| Valid pixels / epoch (≈ 80 % land) | ≈ 19.7 M | ≈ 330 k |

**Over the full 150-epoch run:**
- ~900 000 random 64×64 crops drawn from the 60-city pool.
- ~28 200 forward + backward passes on a 32-sample batch.
- ~7 050 optimizer steps — actual weight updates.
- ~3.7 billion raw pixels seen, of which ~3 billion valid pixels are scored against UrbClim truth.
- Each training city visited ~15 000 times.

### 3.3 Masked MSE loss

```python
loss = criterion_mse(outputs[masks], targets[masks])
```

The boolean `masks` is the channel-24 valid mask. Only valid land pixels (CORINE 1–14, non-padded, non-NaN) contribute to the loss.

| Question | Answer |
|---|---|
| Why masked? | Training on physically meaningless targets (sea pixels with no `tas`) would drag the network toward predicting noise and pollute every gradient. |
| Why MSE and not MAE? | MSE is differentiable everywhere, penalises large outliers quadratically (forces the network to close the worst gaps), and gives gradients proportional to the residual (big errors push harder). MAE's constant-magnitude gradient slows learning. |
| Why not SmoothL1 / Huber? | Tried earlier and reverted — pure MSE gave cleaner gradients in this regime. |

### 3.4 Gradient accumulation

```python
loss = criterion_mse(...) / accumulation_steps                # 4
loss.backward()
if (i + 1) % 4 == 0 or last batch:
    optimizer.step()
```

Every 4 mini-batches' gradients are summed and only then does the optimiser step. Dividing the loss by 4 keeps the magnitude of the accumulated gradient correct.

| Question | Answer |
|---|---|
| Why? | You want the smoothness of a large effective batch (128 samples) but the 1536-channel bottleneck only fits 32 samples in GPU memory at a time. Gradient accumulation gives effective batch 128 without OOM. |
| Why effective batch 128? | Large batches give smoother gradient estimates (less noisy directions). 128 is a sweet spot for U-Net-scale regression — smooth enough to converge stably, small enough that 47 weight updates per epoch is enough. |

### 3.5 Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

Before every optimiser step, the global gradient norm across all parameters is capped at 2.0. If the norm is larger, every gradient is scaled down proportionally.

**Why:** an occasional unlucky batch (a few large-error samples) can produce a huge gradient that throws the optimiser off course. Clipping prevents that single-batch instability from blowing up training.

**Why 2.0:** looser than the 1.0 typical for language models (we *want* the network to react strongly to sharp boundaries) but tight enough to catch true outliers.

### 3.6 OneCycleLR schedule

```python
scheduler = OneCycleLR(
    optimizer, max_lr=8e-5,
    steps_per_epoch=47,                                        # = actual_steps after accum
    epochs=150,
    pct_start=0.1, anneal_strategy='cos',
)
```

The learning rate is a single bell-shaped curve over the full 7050-step run:

1. **Warm-up:** linear from base 1.5×10⁻⁵ to peak 8×10⁻⁵ over the first 10 % of steps.
2. **Cosine anneal:** from peak back down to ~peak / 10⁴ over the remaining 90 %.

| Question | Answer |
|---|---|
| Why this schedule (Smith 2018)? | Warm-up: small LR while the network sets up its feature hierarchy. Peak: aggressive exploration mid-training. Anneal: settle into a sharp minimum. |
| Why `pct_start = 0.1`? | 10 % is enough warm-up for a regression task on z-scored inputs. Vision-classification tasks often use 0.3. |
| Why `max_lr / base_lr ≈ 5×`? | Standard ratio. Lower wastes the schedule; higher destabilises training. |
| Why step the scheduler once per *optimiser* step (47/epoch) instead of per batch (188/epoch)? | The scheduler is told `steps_per_epoch = actual_steps`, which respects gradient accumulation. OneCycleLR is parameterised in *total steps*; only the optimiser-step boundary advances the weight state. |

### 3.7 Optimiser — AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=1e-4)
```

AdamW is Adam with **decoupled weight decay**. Standard Adam mixes weight decay into the gradient (where adaptive learning rates can over- or under-shrink different parameters); AdamW applies weight decay as an explicit shrinkage on the weight itself, separately from the gradient update.

| Question | Answer |
|---|---|
| Why AdamW and not SGD? | Regression with mixed-scale inputs (ERA5, NDVI, integer CORINE, cyclic time) benefits from Adam's per-parameter adaptive learning rates. SGD would require a much more careful manual LR schedule and per-channel scaling. |
| Why `weight_decay = 1e-4`? | Standard L2 regularisation strength for a network of this size. Strong enough to discourage huge weight magnitudes, weak enough not to dominate the optimisation. |

### 3.8 Checkpointing

After every epoch the latest weights are saved as `latest_model.pth`. If validation MSE improves, the weights are *also* saved as `best_model.pth`. The model used at evaluation is **always** `best_model.pth`.

| Question | Answer |
|---|---|
| Why both? | `latest_model.pth` lets training resume after a job timeout. `best_model.pth` is what is evaluated. |
| Why no actual early stopping? | `EARLY_STOPPING_PATIENCE = 150` equals the total epoch count, which effectively disables it. We want the full learning curve for the thesis figure. |

---

## 4. Hyperparameter cheat sheet

Every value in one place. What it does. Why it has that value.

| Hyperparameter | Value | What it does | Why this value |
|---|---|---|---|
| `BATCH_SIZE` | 32 | Samples per forward pass | Largest batch that fits at 1536-channel bottleneck |
| `accumulation_steps` | 4 | Mini-batches summed before optimiser step | Effective batch 128 without OOM |
| Effective batch | 128 | What the optimiser actually sees | Smooth gradient estimates, typical for U-Net |
| `LEARNING_RATE` | 1.5e-5 | Base / warm-up LR | Conservative start; OneCycle ramps it up |
| `max_lr` | 8e-5 | Peak LR mid-training | ~5× base, standard ratio |
| `pct_start` | 0.1 | Fraction of budget spent ramping up | 10 % warm-up is enough for z-scored regression |
| `anneal_strategy` | `cos` | Cosine LR anneal after peak | Smooth landing into a sharp minimum |
| `weight_decay` | 1e-4 | L2 shrinkage in AdamW | Standard regularisation strength |
| `max_norm` (grad clip) | 2.0 | Cap on global gradient norm | Catch outliers without dampening normal signal |
| Loss | Masked MSE | Penalty on prediction errors | Differentiable, outlier-sensitive, gradients scale with error |
| `NUM_EPOCHS` | 150 | Total training-set passes | Long enough for OneCycle to complete its curve |
| `EARLY_STOPPING_PATIENCE` | 150 | Disabled | Want the full learning curve |
| `CROP_SIZE` | 64 | Training crop side length (pixels) | Divisible by 16, covers ~6.4 km |
| Train multiplier | 100 | `__len__ = n_cities × 100` | 6000 samples / epoch from 60 cities |
| Val multiplier | 5 | `__len__ = n_cities × 5` | 100 samples / epoch from 20 cities |
| Gaussian noise σ | 0.05 | Std of N(0, σ²) on z-scored ERA5 | Small enough not to overpower signal, big enough to regularise |
| `Dropout2d p` | 0.05 | Channel-dropout probability | Mild; preserves high-frequency decoder content |
| `LAPSE_RATE` Γ | 0.0065 K m⁻¹ | Lapse correction strength | Standard dry-adiabatic value |
| `GRAVITY` g | 9.80665 m s⁻² | Converts ERA5 geopotential to metres | Physical constant |
| `SHUFFLE_SEED` | 42 | Reproducible city split | Fixed for the whole thesis |
| `TRAIN_N / VAL_N / TEST_N` | 60 / 20 / 20 | Cities per split | 60-train ratio; 20+20 large enough for honest evaluation |
| `TRAIN_YEARS` | 2010–2015 | Six training years | Bulk of the decade |
| `VAL_YEARS` | 2009, 2016 | Inner-bookend years | Closest to train but disjoint |
| `TEST_YEARS` | 2008, 2017 | Outer-bookend years | Furthest from train — hardest generalisation test |

---

## 5. End-to-end one-sentence summary

> Each training sample is a 64×64 crop of one city at one random hour with 23 input channels — 13 z-scored ERA5 variables with Gaussian noise (σ = 0.05), 6 z-scored or binary static rasters (elevation, AHF, land/water mask, CORINE integer 1–15, NDVI, imperviousness), and 4 cyclic sin/cos time channels — and a lapse-rate-corrected residual target masked to valid land pixels; an effective batch of 128 such samples (32 raw × 4 gradient-accumulation steps) is fed through a 4-level Residual U-Net with bilinear-skip upsampling and Dropout2d 0.05 in the bottleneck and first three decoder stages; AdamW with weight decay 1×10⁻⁴, OneCycleLR (warm-up to 8×10⁻⁵ then cosine anneal) and gradient clipping at max-norm 2.0 takes 47 optimiser steps per epoch for 150 epochs, with the lowest-validation-MSE checkpoint kept for evaluation on the 20 unseen test cities.
