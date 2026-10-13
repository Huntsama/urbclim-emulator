# UrbClim emulator: a Residual U-Net for 100 m urban temperature maps

Code for my MSc thesis at Vrije Universiteit Brussel (2026), *Deep Learning for Urban Climate Modelling: emulating UrbClim across European cities with a Residual U-Net*. Promotors Prof. Lesley De Cruz and Prof. Sara Top, supervisor Sara Speelman.

One convolutional network stands in for VITO's UrbClim urban climate model. It takes ERA5 reanalysis on a ~31 km grid plus static maps of the city (elevation, land cover, vegetation, imperviousness, anthropogenic heat) and returns the hourly 2 m air temperature at 100 m resolution. One model, 100 European cities, trained on 60 of them and tested on 20 it never saw.

| | RMSE | MAE | Bias | Pixels |
|---|---:|---:|---:|---:|
| Per frame, 50 random hours per test city | **1.23 °C** | 0.92 °C | +0.36 °C | 55 M |
| Elevation-corrected ERA5 baseline, same pixels | 2.00 °C | | | |
| Summer 22:00 climatology (urban heat island) | 0.83 °C | 0.70 °C | +0.30 °C | per city |

A city-year of hourly maps takes 46 seconds on one GPU. UrbClim needs roughly 200,000 CPU-core-hours for the same.

This repository holds the code only. The data (about 1 TB of Zarr stores) and the trained checkpoints live on the VUB cluster and are not included. See *Data* below for where each input comes from.

## How it works

The target is not temperature. The network predicts the **residual** between the UrbClim temperature and a lapse-rate-corrected ERA5 temperature, so it only has to learn the part of the signal that local geography explains. Everything else is in the pipeline.

```
ERA5 zarr (31 km, 13 variables)  ─┐
UrbClim zarrs (100 m, per city)   ─┤  pipeline/        →  city_lookup.csv, splits_metadata.csv,
static features .nc (per city)    ─┤  build_lookup,       train/val/test_statistics.csv
NDVI .nc (per city)               ─┘  splits, zscore

                                      v2_residual_unet.py
                                      UrbClimDataset  →  (23, 64, 64) crops, z-scored, masked
                                      UrbClimUNet     →  residual (K), 4 levels, ~68 M parameters
                                      training loop   →  masked MSE, AdamW, OneCycleLR, 150 epochs

                                      test_full.py / test_model.py / test_diurnal.py
                                      full-tile inference on the 20 held-out cities
```

### 1. Pipeline: from raw files to a city table and normalisation statistics (`pipeline/`)

| Module | What it does |
|---|---|
| `config.py` | Every path, constant and lookup table in one place: cluster paths, the 60/20/20 split sizes, shuffle seed 42, train years 2010 to 2015, validation 2009 and 2016, test 2008 and 2017, lapse rate 6.5 K/km, the CORINE reclassification into 15 classes, and `target_datetime()` which maps a year, month and day-hour index to a timestamp (UrbClim's time axis starts 7 hours after ERA5's, and leap years matter). |
| `extract.py` | Pure readers. Open a file, pull an array, fill NaNs. No arithmetic, so every other module reads data the same way. |
| `build_lookup.py` | Scans the cluster for every city that has a static-features file, an UrbClim store and an NDVI file, reads each city centre (EPSG:3035), converts it to lat/lon and finds the nearest ERA5 grid cell. Writes `city_lookup.csv`, the single source of truth. |
| `splits.py` | Shuffles the lookup with a fixed seed and slices it into 60 train, 20 validation and 20 test cities. Writes `splits_metadata.csv`. |
| `zscore.py` | Per split, computes mean, standard deviation, min and max for every static feature and every ERA5 variable. Statistics are computed on the inputs only, never on the target, so nothing leaks. |
| `lookup.py` | `StatsLookup`, a small class that reads a statistics CSV and hands the dataloader the right mean and standard deviation by name. |
| `check_pipeline.py` | Sanity check to run on the cluster: time sync between ERA5 and UrbClim across 30 timestamps, raw shapes and ranges for one city, and the exact arithmetic the dataloader applies to one pixel. |

The CSVs this stage produced for the thesis run are in `metadata/`, so the split and the statistics can be inspected without the cluster.

### 2. Dataset, model and training (`v2_residual_unet.py`)

**`UrbClimDataset`** builds one sample: one city, one random timestamp from the split's years, one 64 × 64 crop. For that sample it

1. reads the 13 ERA5 variables at the city's grid cell and z-scores them with the split's statistics,
2. reads the static maps (elevation, CORINE class, anthropogenic heat flux, imperviousness, NDVI) and z-scores the continuous ones,
3. encodes the time of day and the day of year as sine and cosine pairs,
4. lapse-rate-corrects both the ERA5 and the UrbClim temperature to the 100 m elevation and takes the difference as the target,
5. builds a valid-pixel mask (land, and not the Sea class, see `docs/decisions/corine_sea_class.md`),
6. stacks the 23 input channels, the target and the mask, and crops.

**`UrbClimUNet`** is a 4-level U-Net whose double convolutions are replaced by residual blocks. Input 23 channels at 64 × 64, bottleneck 1536 channels at 4 × 4, output one channel: the predicted residual in kelvin. About 68 M parameters, most of them in the bottleneck. Dropout2d in the bottleneck and decoder.

**Training**: masked MSE on valid pixels only, AdamW with weight decay 1e-4, OneCycleLR peaking at 8e-5, batch 32 with 4-step gradient accumulation (effective 128), gradient clipping at 2.0, 150 epochs of 6,000 samples. Checkpoints every epoch plus the best validation epoch. `train.sh` is the Slurm job that ran it.

The complete, line-by-line account is in `docs/model_walkthrough.md`.

### 3. Evaluation

| Script | What it measures |
|---|---|
| `test_full.py` | The headline numbers. Runs the best checkpoint on the full tile of each test city (zero-padded to a multiple of 16, no cropping), 50 random hours per city, and reports RMSE, MAE and bias against UrbClim and against the lapse-rate-corrected ERA5 baseline on identical pixels. A second phase averages summer 22:00 fields to score the urban heat island itself. |
| `test_model.py` | One random hour per test year per city, with a 4-panel figure (land/water, truth, prediction, error) for each. Also defines `TestingUrbClimDataset`, which the other evaluation scripts reuse. |
| `test_diurnal.py` | 24 consecutive hours over one city, for the diurnal cycle and the animation on [jwds.me](https://www.jwds.me). |
| `diagnose.py` | Per-city and per-land-class error tables, to see where the model is wrong. |
| `tools/` | Learning curves from the checkpoint history, and the diurnal figure and GIF. |

## Data

Nothing in this repository is data. All three sources are open.

- **UrbClim output**: *Climate variables for cities in Europe from 2008 to 2017*, Copernicus Climate Change Service (C3S) Climate Data Store. Hourly 2 m air temperature at 100 m for 100 cities. Lauwaet et al., 2024.
- **ERA5 reanalysis**: C3S Climate Data Store, hourly single-level variables, 2008 to 2017.
- **Static maps**: CORINE Land Cover and Copernicus imperviousness (Copernicus Land Monitoring Service), PROBA-V NDVI (Copernicus Global Land Service), EU-DEM elevation, and an anthropogenic heat flux map.

The pipeline expects them converted to Zarr and NetCDF under the paths in `pipeline/config.py`. That conversion was done once on the cluster; change the paths in `config.py` for another layout.

## Running it

Everything ran on the VUB/VSC cluster. From the repository root:

```bash
python -m pipeline.build_lookup        # city table
python -m pipeline.zscore              # splits + statistics
python -m pipeline.check_pipeline      # sanity check, no training
sbatch train.sh                        # training (or: python -u v2_residual_unet.py)
python -u test_full.py                 # headline evaluation
```

`requirements.txt` lists the Python packages. The exact cluster setup is below.

## What it ran on

Hardware and software, exactly as used for the thesis run.

| | |
|---|---|
| Machine | `anansi` cluster of the VSC (Flemish Supercomputer Center), the Tier-2 system hosted at VUB |
| Partition | `ada_gpu`, one node, 8 CPU cores, 32 GB RAM |
| GPU | one shared NVIDIA L40S (46 GB), requested as 4 shards (`--gres=shard:4`) |
| Scheduler | Slurm, 12 h wall-time limit per job (`train.sh`) |
| One training run | 150 epochs, 2 to 3 hours. The checkpoint kept is epoch 45, validation RMSE 1.37 °C |
| Toolchain | foss-2025a modules |
| PyTorch | 2.9.1, CUDA 12.8.0 |
| netCDF4 (Python) | 1.7.2 |
| zarr | 3.1.1 |
| rioxarray | 0.21.0 |
| Storage | data and checkpoints on `$VSC_SCRATCH`, code in `$VSC_HOME` |

The thesis itself is LaTeX, built with `latexmk` and the VUB `vub` package ([texlive-vub](https://gitlab.com/rubdos/texlive-vub)), which sets the house style: TeX Gyre Adventor for titles and the VUB blue and orange. The defence slides use the same package's Beamer theme. Neither is in this repository.

The computational resources and services used in this work were provided by the VSC, funded by the Research Foundation - Flanders (FWO) and the Flemish Government.

## Docs

- `docs/model_walkthrough.md`: what the code does from the moment a sample is read to the moment a weight is updated, with the reason for each choice.
- `docs/pipeline_explained.md`: the data pipeline in depth, including what each input looks like.
- `docs/decisions/time_sync_fix.md`: the 7-hour offset between the ERA5 and UrbClim time axes and the leap-year bug it hid.
- `docs/decisions/corine_sea_class.md`: why sea pixels are an input class but excluded from the loss.

## Lineage

The data preparation follows the approach of Jonas Blancke's thesis on the same dataset (Ghent University), which used per-pixel models. The network, the residual target, the pipeline rewrite and the evaluation here are new.

## Licence

MIT. If you use this code, cite the thesis: Shaissah, J. (2026). *Deep Learning for Urban Climate Modelling: emulating UrbClim across European cities with a Residual U-Net*. MSc thesis, Vrije Universiteit Brussel.
