"""The two scales in one figure: docs/two-scales.png.

    uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib \
        python -m stlnav.demo_scales --cache data/fields_20261117.npz --ice data/ice_hist/sic_2026-02-15.nc

(a) Speed of the icebreaker in level ice: the hull-scale runs, the route-scale model that is fitted to
    them, and Lindqvist's formula alone, which the route-scale model used before.
(b) What that changes on a voyage: Les Escoumins to Sept-Îles at route scale, with and without the fit.
(c) The voyage on the map, with one place marked.
(d) That place at hull scale: the simulator of icebreaker_jax.py run in the ice found there.

The ice is the same hybrid as in demo.py: a past winter's chart concentration with ONE ASSUMED THICKNESS.
"""
from __future__ import annotations

import argparse
import math
import os

import jax
import numpy as np

from . import demo, ships
from . import env as E
from . import fields as F
from . import hullscale as H
from . import routes as RT

BLUE, VERMIL, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/fields_20261117.npz")
    ap.add_argument("--ice", default="data/ice_hist/sic_2026-02-15.nc")
    ap.add_argument("--thick", type=float, default=0.3, help="assumed ice thickness, m")
    ap.add_argument("--out", default="docs/two-scales.png")
    a = ap.parse_args()

    f = F.with_ice(F.load(a.cache), F.read_chart_concentration(a.ice, F.load(a.cache)), a.thick)
    env, world = RT.world(f, ice_route=True)
    _, world_formula = E.make_world(f, RT.snapped(f, RT.specs(True)), [ships.ICEBREAKER_FORMULA] + RT.CLASSES[1:], RT.CLASS_NAMES)
    log, info, _ = demo.sail(env, world, f, 0, jax.random.key(0))
    log_f, info_f, _ = demo.sail(env, world_formula, f, 0, jax.random.key(0))
    hours, hours_f = (log[-1, 0] - log[0, 0]) / 3600, (log_f[-1, 0] - log_f[0, 0]) / 3600
    print(f"voyage with the hull-scale curves: {hours:.1f} h ({'arrived' if info['arrived'] else 'did not arrive'}); with Lindqvist's formula alone: {hours_f:.1f} h "
          f"({'arrived' if info_f['arrived'] else 'did not arrive'})")

    k = int(np.argmax(log[:, 4]))                                   # the place of the voyage with the most ice
    z = H.zoom(env, world, [log[k, 1]], [log[k, 2]], [log[k, 0]], hvar=0.15, seconds=100.0)
    pic = H.picture(z["cfg"], z["state"])
    print(f"place marked: {log[k, 2]:.2f} N, {-log[k, 1]:.2f} W after {log[k, 0] / 3600:.1f} h; concentration {z['conc'][0]:.2f}, thickness {z['h'][0]:.2f} m; "
          f"hull scale {z['hull_speed'][0]:.2f} m/s, route scale in level ice {z['route_speed'][0]:.2f} m/s, route scale on the voyage {log[k, 3]:.2f} m/s; usable = {bool(z['usable'][0])}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.3, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1, 1.08], hspace=0.38, wspace=0.24)

    # (a) speed in level ice
    ax = fig.add_subplot(gs[0, 0])
    tab = ships.load_table()
    fr = {k: np.asarray(v) for k, v in tab["free_runs"].items()}
    own = jax.tree.map(lambda v: v[0], ships.stack([ships.ICEBREAKER]))
    formula = jax.tree.map(lambda v: v[0], ships.stack([ships.ICEBREAKER_FORMULA]))
    hl = np.linspace(0.02, 1.3, 400)
    for thr, col, mk, name in ((1.0, BLUE, "o", "full throttle"), (0.7, GREEN, "s", "70%"), (0.4, VERMIL, "^", "40%")):
        ax.plot(hl, ships.attainable_speed(formula, thr, hl, 1.0), color=col, ls=(0, (4, 2)), lw=0.9, alpha=0.75)
        ax.plot(hl, ships.attainable_speed(own, thr, hl, 1.0), color=col)
        for h in tab["h"]:
            m = np.isclose(fr["h"], h) & np.isclose(fr["thr"], thr)
            sp = np.where(fr["final_speed"][m] > 0.5, fr["speed"][m], 0.0)
            ax.errorbar(h, sp.mean(), yerr=sp.std(ddof=1), fmt=mk, color=col, ms=3.6, mfc="white" if sp.mean() == 0 else col, elinewidth=0.8, capsize=1.5, zorder=3)
        ax.text(0.04, float(ships.attainable_speed(own, thr, 0.04, 1.0)) + 0.18, name, color=col, fontsize=7.5, va="bottom")
    ax.text(1.31, 0.62, "stopped", fontsize=7.5, color=GREY, ha="right", va="bottom")
    ax.plot([], [], color=GREY, label="route scale, fitted to the hull-scale runs")
    ax.plot([], [], color=GREY, ls=(0, (4, 2)), lw=0.9, label="route scale, Lindqvist's formula alone")
    ax.errorbar([], [], yerr=[], fmt="o", color=GREY, ms=3.6, label="hull-scale runs (mean of 3, bars: s.d.)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=2.2, borderaxespad=0.2)
    ax.set_xlabel("thickness of level ice (m)")
    ax.set_ylabel("speed held (m/s)")
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.25, 9.2)
    ax.text(-0.14, 1.34, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # (b) the voyage at route scale, with and without the fit
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(log_f[:, 0] / 3600, log_f[:, 3], color=GREY, ls=(0, (4, 2)), lw=0.9)
    ax.plot(log[:, 0] / 3600, log[:, 3], color=BLUE)
    ax.plot(log[k, 0] / 3600, log[k, 3], "o", color=VERMIL, ms=5, mec="white", mew=0.8, zorder=4)
    ax.annotate("place marked in c", xy=(log[k, 0] / 3600, log[k, 3]), xytext=(log[k, 0] / 3600 + 0.6, log[k, 3] - 2.0), fontsize=7.5, color=VERMIL,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=VERMIL))
    ax.text(0.98, 0.20, "open water at first, then ice", color="k", transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.text(0.98, 0.12, f"fitted to the hull-scale runs: arrives after {hours:.1f} h", color=BLUE, transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.text(0.98, 0.04, f"Lindqvist's formula alone: {hours_f:.1f} h", color=GREY, transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.set_xlabel("hours after leaving Les Escoumins")
    ax.set_ylabel("speed through the water (m/s)")
    ax.set_ylim(0, 9.2)
    ax.set_title(f"Icebreaker to Sept-Îles, full throttle, route scale\nice chart of {os.path.basename(a.ice)[4:14]}, thickness assumed {a.thick} m", fontsize=8, loc="left")
    ax.text(-0.13, 1.34, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # (c) the map
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#d9d4c7")
    im = ax.pcolormesh(f.lon, f.lat, np.ma.masked_where(f.land > 0.5, f.data[0, F.CONC]), cmap="Blues_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.17, fraction=0.05, aspect=30)
    cb.set_label("ice concentration", fontsize=7.5)
    ax.plot(log[:, 1], log[:, 2], color=BLUE, lw=1.6)
    ax.plot(log[k, 1], log[k, 2], "o", color=VERMIL, ms=6, mec="white", mew=0.8, zorder=5)
    halo = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
    ax.text(log[0, 1] + 0.12, log[0, 2] - 0.17, "Les Escoumins", fontsize=7.5, bbox=halo)
    ax.text(log[-1, 1] - 0.12, log[-1, 2] + 0.02, "Sept-Îles", fontsize=7.5, ha="right", bbox=halo)
    ax.set_xlim(-69.9, -65.2)
    ax.set_ylim(48.1, 50.45)
    ax.set_aspect(1 / math.cos(math.radians(49.2)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_title("Route scale: 2 km grid, 60 s steps", fontsize=8, loc="left")
    ax.text(-0.14, 1.05, "c", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # (d) the marked place at hull scale
    ax = fig.add_subplot(gs[1, 1])
    x0, x1, _, y1 = pic["extent"]
    rgb = np.empty(pic["ice"].shape + (3,))
    rgb[:] = (0.09, 0.22, 0.36)                                     # open water
    rgb[pic["brash"] > 0] = (0.36, 0.50, 0.62)                      # ice broken by the hull
    ice = pic["ice"] > 0
    shade = np.clip(0.80 + 0.5 * (pic["ice"] - 1.0), 0.62, 1.0)     # thicker ice is lighter
    rgb[ice] = shade[ice][:, None] * np.ones(3)
    ax.imshow(rgb, origin="lower", extent=pic["extent"], interpolation="nearest")
    ax.fill(pic["hull_x"], pic["hull_y"], color=VERMIL, zorder=3)
    ax.plot([x0 + 20, x0 + 120], [22, 22], color="k", lw=1.5)
    ax.text(x0 + 70, 27, "100 m", ha="center", fontsize=7.5)
    xs = float(np.mean(pic["hull_x"]))
    ax.annotate("channel of broken ice", xy=(xs - 150, y1 / 2 + 4), xytext=(xs - 190, y1 - 30), fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.text(x1 - 8, 14, f"level ice, {z['h'][0]:.2f} m on average\n(lighter = thicker)", ha="right", va="bottom", fontsize=7.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_title(f"Hull scale at the marked place: 0.5 m grid, 1/360 s steps\n76 m icebreaker at {z['hull_speed'][0]:.1f} m/s, ice force {z['ice'][0] / 1e3:.0f} kN", fontsize=8, loc="left")
    ax.text(-0.07, 1.16, "d", transform=ax.transAxes, fontsize=11, fontweight="bold")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
