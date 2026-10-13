"""
lookup.py — read *_statistics.csv and expose clean accessors.

The dataloader currently inlines a dozen `stats_df[stats_df['feature']==...]`
lookups. This module hides that behind one class so every consumer
(training loop, check_pipeline, ablation scripts) accesses stats the
same way and we can't silently miss a feature.
"""
import numpy as np
import pandas as pd


STATIC_FEATURES = ('elevation', 'ahf', 'ndvi', 'imperviousness')
EPS = 1e-8


class StatsLookup:
    """Load a single *_statistics.csv and offer mean/std lookups by name."""

    def __init__(self, stats_csv):
        df = pd.read_csv(stats_csv)
        self._df = df

        # ERA5 rows are the 'dynamic' ones — names are u10/t2m/blh/…, so the
        # old era5_var_* prefix filter no longer works. Row order in the CSV
        # is the ERA5 zarr's native variable order; preserve it.
        era5 = df[df['type'] == 'dynamic'].reset_index(drop=True)
        self.era5_names = era5['feature'].tolist()
        self.era5_means = era5['mean'].values.astype(np.float32)
        self.era5_stds  = era5['std'].values.astype(np.float32)

        self._static_mean = {}
        self._static_std  = {}
        for feat in STATIC_FEATURES:
            row = df[df['feature'] == feat]
            if len(row):
                self._static_mean[feat] = float(row['mean'].values[0])
                self._static_std[feat]  = float(row['std'].values[0])

        cor = df[df['feature'] == 'corine']
        self.corine_min = float(cor['min'].values[0]) if len(cor) else float('nan')
        self.corine_max = float(cor['max'].values[0]) if len(cor) else float('nan')

    # --- static features ------------------------------------------------
    def mean(self, feature):
        return self._static_mean[feature]

    def std(self, feature):
        return self._static_std[feature]

    def normalize(self, feature, value):
        return (value - self._static_mean[feature]) / (self._static_std[feature] + EPS)

    # --- ERA5 -----------------------------------------------------------
    def normalize_era5(self, era5_vector):
        return (era5_vector - self.era5_means) / (self.era5_stds + EPS)

    # --- debug ----------------------------------------------------------
    def summary(self):
        return self._df.to_string(index=False)
