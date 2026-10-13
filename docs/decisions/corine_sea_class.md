# CORINE land cover & the Sea-class decision — 2026-04-25

A standalone README explaining how CORINE land cover is handled in this
pipeline, why the **Sea** class is excluded from training, what Jonas did
in his MSc thesis, why our setup matches his, and what we keep open for
later if we ever want the U-Net to see coastal context explicitly.

This file is the single source of truth for the discussion that led up to
`debug_corine.py` and the question *"should we redo the pipeline to
include sea?"* — answer summarised in §6.

---

## 1. TL;DR

- The CORINE 2012 raster has 44 raw codes. Following Jonas Blancke's MSc
  thesis (Table 8.1) we **reclassify them into 15 UrbClim classes**.
- **Class 15 = Sea (raw 42 / 43 / 44).** Jonas drops it. Our production
  pipeline drops it too — so the model never trains on sea pixels.
- Jonas explicitly flags the resulting **coastal-city blind spot** as a
  known weakness (Sections 5.2 and 6 of his thesis).
- Our diagnostic script `debug_corine.py` *re-introduces* the sea class
  for visualisation only, so coastal cities don't look like they have
  "missing chunks" on the maps. This visualisation map (`CORINE_CLASS_MAP_VIS`)
  is **strictly for plotting** — `pipeline/config.py` is unchanged.
- Net result: training behaviour is identical to Jonas. We have a clear
  upgrade path documented in §7 if we want to deviate.

---

## 2. What CORINE is, and where it sits in our pipeline

CORINE Land Cover (CLC) is a 100 m, pan-European raster published by the
European Environment Agency (EEA, 2012). The 2012 vintage is the one
ingested by the VITO UrbClim model itself, so using the same dataset on
the emulator side keeps the inputs aligned with the target.

In our pipeline a single CORINE raster ships per city as the variable
`CORINE` inside `static_features_<city>.nc`. It enters the U-Net as one
of the 23 input channels (channel 22), normalised by the per-split min /
max stored in `train_statistics.csv` etc.

```
static_features_<city>.nc
  ├── CORINE              # int raster, raw 1..44
  ├── elevation           # m
  ├── ahf                 # anthropogenic heat flux
  ├── imperviousness      # 0..1
  └── …
```

Reclassification happens at extraction time via the `CORINE_CLASS_MAP`
constant in `pipeline/config.py`. Every consumer (extractor, dataloader,
debug script) imports it from the same place — no duplication.

---

## 3. Jonas's reclassification (Table 8.1, MSc thesis)

| Reclass id | Class name             | Raw CORINE codes                  |
|-----------:|------------------------|-----------------------------------|
| 1          | Urban                  | 1                                 |
| 2          | Sub-urban              | 2                                 |
| 3          | Industrial             | 3, 4, 5, 6                        |
| 4          | Urban green            | 10, 11                            |
| 5          | Snow / Ice             | 34                                |
| 6          | Bare soil              | 8, 9, 30, 31, 38, 39              |
| 7          | Grassland              | 7, 18, 26, 27, 32, 35, 36, 37     |
| 8          | Cropland               | 12, 13, 14, 19, 20, 21            |
| 9          | Shrubland              | 15, 28, 29, 33                    |
| 10         | Woodland               | 16, 17, 22                        |
| 11         | Broadleaf trees        | 23, 25                            |
| 12         | Needleleaf trees       | 24                                |
| 13         | Rivers                 | 40                                |
| 14         | Inland water bodies    | 41                                |
| **15**     | **Sea**                | **42, 43, 44**                    |

The table is reproduced verbatim in `pipeline/config.py` for classes
1..14. **Class 15 is intentionally absent from the production map.**

---

## 4. The decision: exclude Sea from training

`pipeline/config.py` (today):

```python
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
    # 15: [42, 43, 44],   ← Sea, deliberately omitted (Jonas-aligned)
}
```

Anything that doesn't appear in the map (i.e. raw 42/43/44, plus any
unmapped codes) becomes `0` after `reclassify_corine()`. Downstream:

1. The `valid_mask` used in the loss is built from `(reclass > 0)`.
2. Sea pixels therefore contribute **zero gradient** to the U-Net.
3. The `CORINE` input channel still contains a `0` at those pixels, so
   the convolutions *do* read them — but as the same "no class" value
   used for any other unmapped pixel.

This is the part of the decision that the upgrade path in §7 would
change, but only on the input side — never on the loss side.

---

## 5. Why Jonas excluded Sea — direct quotes

The decision is documented in three places in
`papers/Masterproef_Jonas_Blancke 1.md`:

**§4 — Methodology (line 730):**

> "Although UrbClim data is also available for sea areas, this study
> focuses only on the UHI which takes part over land. For this reason
> sea pixels have been excluded."

**§5.2 — Figure 5.8 caption (line 1121):**

> "Boxplots per CORINE land cover class illustrating the UrbClim minus
> ERA5 temperature for the test cities. … Note that no data is available
> for the sea land cover, as these grid cells are eliminated from the
> dataset."

**§5.4 — Coastal cities, ERA5 leak (line 1130):**

> "[…] in the coastal cities, where the coarse ERA5 pixels include both
> land and sea. As only land pixels are preserved, the presence of lower
> daytime and higher nighttime sea surface temperatures contribute to
> underestimation and overestimation of ERA5, respectively."

**§6 — Future work (line 1522):**

> "[…] the ERA5 sea pixel effect, which impacts the performance of the
> emulator for coastal cities, can be mitigated by using higher
> resolution datasets that focus exclusively on land areas. An example
> of such a dataset include the ERA5-Land dataset."

So the picture from the thesis is consistent: Sea is excluded because
the project's scope is the UHI, which is a *land* phenomenon — but the
exclusion has a known cost on the **input side** for coastal cities,
because ERA5 cells at the coastline mix land and water and Jonas can't
disentangle them at his 31 km input resolution.

---

## 6. Why we match Jonas in production

We're emulating UrbClim's `tas` (2 m air temperature) over urban
environments. UrbClim itself does not produce a meaningful land-surface
temperature over open water — those grid cells are unphysical for our
target variable. Two consequences fall out of that:

1. **Loss side — non-negotiable.** The model must not be penalised for
   what it predicts over sea, because there is no ground truth there.
   This is true regardless of whether we expose Sea as an input class.
   The current `valid_mask = (reclass > 0)` does exactly this.

2. **Input side — defensible default, but a choice.** Currently sea
   pixels enter the network with the same `CORINE = 0` as any other
   unmapped pixel, so the convolutions cannot distinguish "coast next
   to Atlantic" from "missing data". Jonas's random forest doesn't
   suffer from this because it operates pixel-wise — a dropped pixel is
   literally never seen. A U-Net is not so lucky.

So the production pipeline is **Jonas-faithful**, and that's the safe
default we're sticking with for this iteration of training. The
upgrade path is described in §7 below.

---

## 7. The available upgrade path (held in reserve)

If at any point we decide the U-Net needs explicit coastal context, the
change is small and surgical — *not* a "redo the whole pipeline":

1. **`pipeline/config.py`** — uncomment the Sea row:

   ```python
   CORINE_CLASS_MAP = {
       …
       14: [41],
       15: [42, 43, 44],     # ← Sea, kept on the input side only
   }
   ```

2. **Regenerate the three statistics CSVs** — `train_statistics.csv`,
   `val_statistics.csv`, `test_statistics.csv`. The CORINE row's `min`
   stays at 0 but `max` becomes 15 instead of 14, so the per-channel
   normalisation shifts slightly. This is step `[3/5]` of `train.sh`
   and takes a few minutes.

3. **Leave the loss mask alone.** The valid mask should change from
   `reclass > 0` to `(reclass > 0) & (reclass != 15)` so that Sea
   pixels are visible to the convolutions but excluded from the loss
   exactly as before.

   In `v2_residual_unet.py` the relevant line is in the dataloader's
   target-mask construction. A one-line change.

4. **Sanity check.** `pipeline/check_pipeline.py` will pick up the new
   max value automatically. `debug_corine.py` already plots Sea as a
   first-class colour (deep navy `#08306b`), so a coastal city's map
   should look identical before and after the change — only the model
   sees a difference.

**No model architecture change is required.** The CORINE channel is a
single normalised numeric value; only its scale changes.

**Cost / risk:**
- Cost: one short re-run of `pipeline.zscore`, plus one fresh training
  run because all CORINE-channel z-scores shift. Existing checkpoints
  become incompatible.
- Risk: zero on the loss side (mask still excludes sea). Upside is
  bounded: it only matters for coastal cities, where the convolution
  windows currently confuse "open water" with "no data".

We're holding this in reserve until we have a baseline trained
end-to-end on the Jonas-faithful pipeline. Then we can A/B it cleanly.

---

## 8. The diagnostic-only sea class (`debug_corine.py`)

Because the production map drops sea, every coastal city in our cohort
*looks* on a plain CORINE plot like it has a huge missing region. This
is misleading — those pixels are present in the raster, they just hit
codes 42/43/44 and got collapsed to `0` ("Sea / dropped").

`debug_corine.py` therefore defines a **visualisation-only** extension:

```python
from pipeline.config import CORINE_CLASS_MAP
CORINE_CLASS_MAP_VIS = dict(CORINE_CLASS_MAP)
CORINE_CLASS_MAP_VIS[15] = [42, 43, 44]      # Sea — for plotting only
```

In the debug plots:
- Class **0** is now reserved for **truly unmapped** pixels (raw codes
  that aren't in any map entry — should be empty for healthy data).
- Class **15** is **Sea**, drawn in deep navy.
- Classes **1..14** are unchanged from the production map.

This change is *strictly local* to `debug_corine.py`. No production
code, no statistics CSV, no model input is affected — the only thing
it changes is what coastal cities look like in the diagnostic PNGs.

If you ever see a non-empty class 0 in `corine_class_coverage.csv`,
that's a *real* gap worth investigating: it means the static file
contains a CORINE code that isn't in Jonas's reclassification table at
all.

---

## 9. What `debug_corine.py` produces

All artefacts land under `MY_DATA_DIR` (= `/user/brussel/113/vsc11361/data`
on VSC):

```
corine_debug.log                          full log
corine_class_coverage.csv                 100 rows × 18 cols
                                            city_name, shape, n_pixels,
                                            pct_class_00 … pct_class_15,
                                            n_classes_present, dominant_class,
                                            pct_unmapped, pct_sea
corine_plots/
  00_class_glossary.png                   colour legend for the 16 classes
  01_summary_stacked.png                  one stacked bar per city,
                                            sorted by urban total (desc)
  02_summary_heatmap.png                  cities × active-classes heatmap
  city_00_<City>.png                      per-city map + coverage bar
  city_01_<City>.png
  …                                       100 plots in total
```

How to read each:

- **00_class_glossary.png** — the rosetta stone. Open this side-by-side
  with the per-city plots so you can match colours to names without
  guessing. Labels for class 0 ("Unmapped") and class 15 ("Sea") tell
  you which is dropped at training and which is just unrendered.
- **01_summary_stacked.png** — coastal cities (Lisbon, Barcelona,
  Athens, Valencia, etc.) appear as bars with a substantial deep-navy
  segment. If a coastal city *doesn't* have that segment, its raster
  may be land-only.
- **02_summary_heatmap.png** — read **columns**: a near-empty column
  means a class is rare across the cohort and the model will see few
  examples. Snow/Ice (5) is expected to be near-empty for most years.
- **city_NN_<City>.png** — the workhorse. Left panel = the actual
  reclassified raster. Right panel = horizontal bar of class % coverage,
  dominant class on top.

A clean cohort should show:
- `pct_unmapped == 0` (or essentially zero) for all 100 cities,
- `pct_sea > 0` for the coastal subset only,
- `n_classes_present` typically between 6 and 12 per city,
- `dominant_class` usually 8 (Cropland) or 7 (Grassland) for most of
  Europe; 1/2/3 in dense urban cores (rare at our domain sizes).

---

## 10. Decision register

| Date       | Decision                                                                 | Owner | Status        |
|------------|--------------------------------------------------------------------------|-------|---------------|
| 2026-04-24 | Production pipeline excludes Sea from CORINE map (Jonas-aligned)         | me    | shipped       |
| 2026-04-25 | Add Sea (class 15) to `debug_corine.py` for visualisation only           | me    | shipped       |
| 2026-04-25 | **Do not** add Sea to production until baseline run completes            | me    | deferred      |
| TBD        | Re-evaluate after baseline: A/B with Sea on the input side, mask in loss | me    | open          |
| TBD        | Investigate ERA5-Land as the input dataset (Jonas's §6 recommendation)   | me    | future-work   |

---

## 11. Where to look in the code

| File                                | Role re: CORINE                                            |
|-------------------------------------|------------------------------------------------------------|
| `pipeline/config.py`                | `CORINE_CLASS_MAP` — production reclassification, no Sea   |
| `pipeline/extract.py`               | Reads `CORINE` from the static `.nc`, applies the map      |
| `pipeline/zscore.py`                | Computes per-split min/max for the CORINE channel          |
| `pipeline/check_pipeline.py`        | Sanity-checks that CORINE is loaded and normalised         |
| `v2_residual_unet.py`               | Consumes the normalised CORINE channel as channel index 22 |
| `debug_corine.py`                   | Visualises ALL 100 cities, includes Sea via `CORINE_CLASS_MAP_VIS` |
| `2026-04-24_pipeline_work.md`       | Yesterday's wider pipeline log (CORINE map referenced there) |
| `papers/Masterproef_Jonas_Blancke 1.md` | The thesis — Table 8.1 at line 2025, sea exclusion rationale at lines 730 and 1121 |

---

## 12. One-line status

Sea is excluded from training to mirror Jonas. We see it on the
diagnostic maps so coastal cities don't look broken. Whether to feed
it to the U-Net as a 15th input class is a single, reversible config
edit that we will A/B against the Jonas-faithful baseline once we have
one to compare against.
