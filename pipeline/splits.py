"""
splits.py — shuffle the master city lookup and slice into train/val/test.

Reads city_lookup.csv (produced by build_lookup.py), shuffles deterministically
with config.SHUFFLE_SEED, takes the first TRAIN_N / next VAL_N / next TEST_N
rows, and writes splits_metadata.csv. 'split' is the second column so you
can eyeball which city is in which set without scrolling right.
"""
import pandas as pd

from pipeline.config import (CITY_LOOKUP_CSV, SPLITS_CSV,
                             TRAIN_N, VAL_N, TEST_N, SHUFFLE_SEED)


# Order matters — 'split' in col 2 lets you `head splits_metadata.csv`
# and immediately see the assignment.
DATALOADER_COLS = [
    'city_name', 'split',
    'latitude', 'longitude',
    'era5_lat_idx', 'era5_lon_idx',
    'static_path', 'ndvi_path', 'target_path',
]


def build_splits(lookup_csv=CITY_LOOKUP_CSV, output=SPLITS_CSV,
                 train_n=TRAIN_N, val_n=VAL_N, test_n=TEST_N,
                 seed=SHUFFLE_SEED):
    df = pd.read_csv(lookup_csv)
    total = train_n + val_n + test_n
    if len(df) < total:
        raise ValueError(
            f"Lookup has {len(df)} cities but split requires {total}. "
            f"Either lower TRAIN/VAL/TEST_N in config.py or rerun build_lookup."
        )

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled = shuffled.iloc[:total].copy()

    shuffled.loc[0                : train_n - 1,         'split'] = 'train'
    shuffled.loc[train_n          : train_n + val_n - 1, 'split'] = 'val'
    shuffled.loc[train_n + val_n  : total - 1,           'split'] = 'test'

    keep = [c for c in DATALOADER_COLS if c in shuffled.columns]
    shuffled[keep].to_csv(output, index=False)
    return shuffled


if __name__ == '__main__':
    df = build_splits()
    print(f"Saved splits → {SPLITS_CSV}")
    print(f"  train: {(df['split'] == 'train').sum()} cities")
    print(f"  val  : {(df['split'] == 'val'  ).sum()} cities")
    print(f"  test : {(df['split'] == 'test' ).sum()} cities")
    print("\nTrain:", ', '.join(df[df['split'] == 'train']['city_name'].tolist()))
    print("Val  :", ', '.join(df[df['split'] == 'val'  ]['city_name'].tolist()))
    print("Test :", ', '.join(df[df['split'] == 'test' ]['city_name'].tolist()))
