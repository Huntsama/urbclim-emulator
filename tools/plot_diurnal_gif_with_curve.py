#!/usr/bin/env python
# coding: utf-8
"""
plot_diurnal_gif_with_curve.py — animated maps with the diurnal curve underneath.

Stacks each existing forecast frame (UrbClim reference | CNN prediction maps)
on top of the time x heat curve, with a vertical line sweeping to the current
hour as it plays. One combined GIF for the defence.

Reuses forecast_frames/frame_HH.png + diurnal_metrics_*.csv already written by
test_diurnal.py — no model run.

    python plot_diurnal_gif_with_curve.py <city_date_dir>
    python plot_diurnal_gif_with_curve.py     # defaults to Antwerp_2017-06-22
"""
import os
import sys
import glob
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Fixed right-axis range (°C) for the absolute-difference bars.
DIFF_AXIS_MAX = 2.5

DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_diurnal_results",
    "urbclim_v2_models", "Antwerp_2017-06-22",
)


def render_curve(df, hour, width_px):
    """Twin-axis diurnal curve with a vertical marker at `hour`, as a PIL image
    exactly `width_px` wide."""
    dpi = 100
    fig, ax_t = plt.subplots(figsize=(width_px / dpi, 3.2), dpi=dpi)
    ax_e = ax_t.twinx()

    ax_e.bar(df['hour'], df['mae_C'], width=0.7, color='0.85', zorder=1)
    ax_e.set_ylabel('|pred − ref| (°C)', color='0.4', fontsize=9)
    ax_e.tick_params(axis='y', colors='0.4', labelsize=8)
    ax_e.set_ylim(0, DIFF_AXIS_MAX)

    ax_t.plot(df['hour'], df['mean_ref_C'], 'o-', color='black', lw=1.8,
              ms=4, label='UrbClim reference', zorder=3)
    ax_t.plot(df['hour'], df['mean_pred_C'], 's--', color='tab:red', lw=1.8,
              ms=4, label='CNN prediction', zorder=3)
    ax_t.axvline(hour, color='tab:blue', lw=2, alpha=0.7, zorder=4)
    ax_t.set_xlabel('hour of day (UTC)', fontsize=9)
    ax_t.set_ylabel('city-mean T (°C)', fontsize=9)
    ax_t.set_xticks(range(0, 24, 2))
    ax_t.set_xlim(-0.5, 23.5)
    ax_t.tick_params(labelsize=8)
    ax_t.grid(alpha=0.3)
    ax_t.set_zorder(ax_e.get_zorder() + 1)
    ax_t.patch.set_visible(False)
    ax_t.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def build(city_dir, ms_per_frame=450):
    city_dir = os.path.normpath(city_dir)
    csv = glob.glob(os.path.join(city_dir, "diurnal_metrics_*.csv"))[0]
    df = pd.read_csv(csv)
    stem = os.path.basename(csv)[len("diurnal_metrics_"):-len(".csv")]
    frames_dir = os.path.join(city_dir, "forecast_frames")

    canvases = []
    for hr in range(24):
        top = Image.open(os.path.join(frames_dir, f"frame_{hr:02d}.png")).convert('RGB')
        bottom = render_curve(df, hr, top.width)
        canvas = Image.new('RGB', (top.width, top.height + bottom.height), 'white')
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, top.height))
        canvases.append(canvas)

    # ONE shared palette for every frame, else per-frame adaptive palettes
    # recolor the fixed colorbar and it appears to flicker. Derive the palette
    # from all frames stacked, then quantize each frame to it.
    ref = Image.new('RGB', (canvases[0].width, canvases[0].height * len(canvases)))
    for i, c in enumerate(canvases):
        ref.paste(c, (0, i * c.height))
    ref = ref.convert('P', palette=Image.ADAPTIVE, colors=256)
    combined = [c.quantize(palette=ref, dither=Image.NONE) for c in canvases]

    out = os.path.join(city_dir, f"diurnal_maps_with_curve_{stem}.gif")
    combined[0].save(out, save_all=True, append_images=combined[1:],
                     duration=ms_per_frame, loop=0)
    print(f"  → {out}")


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
