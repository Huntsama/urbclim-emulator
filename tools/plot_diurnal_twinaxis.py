#!/usr/bin/env python
# coding: utf-8
"""
plot_diurnal_twinaxis.py — one clean twin-axis figure from a diurnal CSV.

The professor's "time x heat + how it changes across the day" side plot:
  x  = hour of day (UTC)
  left  y = city-mean air temperature (°C): UrbClim reference vs CNN prediction
  right y = absolute hourly error |pred - ref| (°C), grey bars

Reads the diurnal_metrics_<City>_<date>.csv written by test_diurnal.py and
emits diurnal_twinaxis_<City>_<date>.png next to it. No model run needed.

    python plot_diurnal_twinaxis.py <path/to/diurnal_metrics_*.csv>
    python plot_diurnal_twinaxis.py            # defaults to the Antwerp CSV
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fixed right-axis range (°C) for the absolute-difference bars, so the scale is
# identical on every plot regardless of the day/city data.
DIFF_AXIS_MAX = 2.5

DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_diurnal_results",
    "urbclim_v2_models", "Antwerp_2017-06-22",
    "diurnal_metrics_Antwerp_2017-06-22.csv",
)


def plot(csv_path):
    df = pd.read_csv(csv_path)
    base = os.path.basename(csv_path)
    # diurnal_metrics_<City>_<date>.csv  ->  <City>, <date>
    stem = base[len("diurnal_metrics_"):-len(".csv")]
    city, date = stem.rsplit("_", 1)
    out_path = os.path.join(os.path.dirname(csv_path),
                            f"diurnal_twinaxis_{city}_{date}.png")

    fig, ax_t = plt.subplots(figsize=(10, 6))
    ax_e = ax_t.twinx()

    # right axis first so the temperature lines sit on top of the bars
    ax_e.bar(df['hour'], df['mae_C'], width=0.7, color='0.8',
             label='|prediction − reference|', zorder=1)
    ax_e.set_ylabel('absolute difference (°C)', color='0.4')
    ax_e.tick_params(axis='y', colors='0.4')
    ax_e.set_ylim(0, DIFF_AXIS_MAX)

    ax_t.plot(df['hour'], df['mean_ref_C'], 'o-', color='black', lw=2,
              label='UrbClim reference', zorder=3)
    ax_t.plot(df['hour'], df['mean_pred_C'], 's--', color='tab:red', lw=2,
              label='CNN prediction', zorder=3)
    ax_t.set_xlabel('hour of day (UTC)')
    ax_t.set_ylabel('city-mean air temperature (°C)')
    ax_t.set_xticks(range(0, 24, 2))
    ax_t.set_xlim(-0.5, 23.5)
    ax_t.grid(alpha=0.3)
    ax_t.set_zorder(ax_e.get_zorder() + 1)   # lines above bars
    ax_t.patch.set_visible(False)            # but let the bars show through

    ax_t.set_title(f'{city} — {date} — diurnal cycle',
                   fontsize=14, fontweight='bold')

    lines = ax_t.get_lines() + [ax_e.patches[0]]
    ax_t.legend(lines, [l.get_label() for l in ax_t.get_lines()]
                + ['|prediction − reference|'], loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(DEFAULT_CSV)
    plot(csv)
