"""Demonstration: two ships sail from Les Escoumins to Sept-Îles through real fields and generated traffic.

    uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib \
        python -m stlnav.demo --archive data/archive --start 2026-11-17T00 --ice data/ice_hist/sic_2026-02-15.nc

The ice is a hybrid until the archive holds a winter: concentration from the chart-based daily product of
a past winter day, with ONE ASSUMED THICKNESS everywhere (--thick), laid over the archived wind, current
and waves of the --start day. Without --ice the archived CIOPS ice is used as it is.

The own ship is steered by a simple rule (head for the next waypoint, 1.5 km to the right of the Coast
Guard ice route), at full throttle. It does not avoid traffic; the traffic applies its collision rules
to it. Steered down the route line itself, the faster ship runs into the ships it overtakes.
"""
from __future__ import annotations

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import fields as F
from . import routes as RT
from . import traffic
from .traffic import KY

PATH = [RT.PLACES["escoumins"], (48.483, -69.000), (48.733, -68.800), (49.250, -67.300), (49.85, -66.45), RT.PLACES["sept_iles"]]


def sail(env, world, f, cls, key, path=PATH, reach_km=6.0, right_km=1.5):
    path = [RT.nearest_water(f, *path[0])] + list(path[1:-1]) + [RT.nearest_water(f, *path[-1])]
    # keep the intermediate waypoints right_km to the right of the route, clear of the ships that sail on it
    for i in range(1, len(path) - 1):
        (la0, lo0), (la1, lo1) = path[i - 1], path[i + 1]
        hd = math.atan2((la1 - la0) * KY, (lo1 - lo0) * KY * math.cos(math.radians(path[i][0])))
        path[i] = (path[i][0] - right_km * math.cos(hd) / KY, path[i][1] + right_km * math.sin(hd) / (KY * math.cos(math.radians(path[i][0]))))
    p = RT.env.params(path[0][1], path[0][0], path[-1][1], path[-1][0], cls=cls)
    step = jax.jit(env.step)
    _, s = jax.jit(env.reset)(key, p, world)
    k, log = 1, []
    turn_rate, uow = float(world.classes.turn[cls]), float(world.classes.uow[cls])
    for _ in range(int(env.cfg.t_max / env.cfg.dt)):
        lat, lon = float(s.lat), float(s.lon)
        dx, dy = (path[k][1] - lon) * KY * math.cos(math.radians(lat)), (path[k][0] - lat) * KY
        if math.hypot(dx, dy) < reach_km and k < len(path) - 1:
            k += 1
            continue
        err = (math.atan2(dy, dx) - float(s.th) + math.pi) % (2 * math.pi) - math.pi
        per_step = turn_rate * min(max(float(s.v) / uow, 0.2), 1.0) * env.cfg.dt        # largest heading change in one step
        _, s, r, done, info = step(s, jnp.array([1.0, float(np.clip(err / per_step, -1, 1))]), p, world)
        here = env.sample(world, s.lon, s.lat, s.t)
        log.append((float(s.t), float(s.lon), float(s.lat), float(s.v), float(here[F.CONC]), float(here[F.THICK]), float(info["closest_km"]), bool(info["in_track"])))
        if bool(done):
            break
    return np.array(log, dtype=float), {k: float(v) for k, v in info.items()}, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/archive")
    ap.add_argument("--start", default="2026-11-17T00")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--ice", default=None, help="chart concentration file (ERDDAP mpoDailySatelliteDataSic)")
    ap.add_argument("--thick", type=float, default=0.3, help="assumed ice thickness with --ice, m")
    ap.add_argument("--cache", default=None, help="fields .npz to reuse or create")
    ap.add_argument("--out", default="docs/stlnav-demo.png")
    a = ap.parse_args()

    if a.cache and os.path.exists(a.cache):
        f = F.load(a.cache)
    else:
        f = F.load_window(a.archive, a.start, a.hours, step_h=3)
        if a.cache:
            F.save(f, a.cache)
    ice_label = f"archived CIOPS ice, {a.start[:10]}"
    if a.ice:
        f = F.with_ice(f, F.read_chart_concentration(a.ice, f), a.thick)
        ice_label = f"ice chart of {os.path.basename(a.ice)[4:14]}, thickness assumed {a.thick} m everywhere"
    env, world = RT.world(f, ice_route=True)

    runs = {}
    for name in ("icebreaker", "tanker"):
        log, info, s = sail(env, world, f, RT.CLASS_NAMES.index(name), jax.random.key(0))
        runs[name] = (log, info, s)
        h = (log[-1, 0] - log[0, 0]) / 3600
        end = "arrived" if info["arrived"] else "collided" if info["collision"] else "ran aground" if info["aground"] else "ran out of time"
        print(f"{name}: {end} after {h:.1f} h, mean speed {log[:, 3].mean():.2f} m/s ({log[:, 3].mean() / 0.5144:.1f} kn), "
              f"lowest {log[:, 3].min():.2f} m/s, closest ship {log[:, 6].min():.2f} km, {100 * log[:, 7].mean():.0f}% of the time in another ship's track")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1], hspace=0.22, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    conc = np.ma.masked_where(f.land > 0.5, f.data[0, F.CONC])
    ax.set_facecolor("#d9d4c7")
    im = ax.pcolormesh(f.lon, f.lat, conc, cmap="Blues_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8, label="ice concentration")
    rt = world.routes
    for r in range(rt.length.shape[0]):
        q = jnp.linspace(0.0, rt.length[r], 200)
        lo, la, _ = traffic.place(rt, jnp.full(200, r), q, jnp.zeros(200))
        ax.plot(lo, la, color="0.25", lw=0.6, alpha=0.6, label="routes" if r == 0 else None)
    s = runs["icebreaker"][2]
    lo, la, hd, _, _ = traffic.kinematics(s.traffic, s.t, lambda x, y, t: env.sample(world, x, y, t), rt)
    act, cls = np.array(s.traffic.active), np.array(s.traffic.cls)
    marks = {"icebreaker": ("^", "#d62728"), "bulk": ("o", "#222222"), "tanker": ("o", "#ff7f0e"), "container": ("o", "#9467bd"), "ferry": ("s", "#2ca02c"), "ropax": ("s", "#2ca02c")}
    for c, name in enumerate(RT.CLASS_NAMES):
        m = act & (cls == c)
        if m.any():
            ax.scatter(np.array(lo)[m], np.array(la)[m], s=26, marker=marks[name][0], color=marks[name][1], edgecolor="w", lw=0.5, zorder=4, label=f"traffic: {name} ({m.sum()})")
    for name, col in (("icebreaker", "#d62728"), ("tanker", "#ff7f0e")):
        log = runs[name][0]
        ax.plot(log[:, 1], log[:, 2], color=col, lw=2.2, zorder=5, label=f"own ship: {name}")
    ax.set_xlim(-70.2, -58.5)
    ax.set_ylim(45.8, 51.0)
    ax.set_aspect(1 / math.cos(math.radians(48.5)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"Les Escoumins to Sept-Îles on the Coast Guard ice route of 12 Feb 2026\n{ice_label}; wind, current and waves of {a.start[:10]}; traffic after {(float(s.t)) / 3600:.0f} h", fontsize=10)

    a1, a2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for name, col in (("icebreaker", "#d62728"), ("tanker", "#ff7f0e")):
        log = runs[name][0]
        a1.plot(log[:, 0] / 3600, log[:, 3] / 0.5144, color=col, label=name)
        a2.plot(log[:, 0] / 3600, log[:, 4], color=col, label=name)
    a1.set_ylabel("speed through the water (kn)")
    a2.set_ylabel("ice concentration at the ship")
    for b in (a1, a2):
        b.set_xlabel("hours after departure")
        b.grid(alpha=0.3)
        b.set_ylim(bottom=0)
        b.legend(fontsize=8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
