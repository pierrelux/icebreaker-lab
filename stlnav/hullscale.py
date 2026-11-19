"""The link between the two scales: the hull-scale simulator (icebreaker_jax.py) and the route-scale one (stlnav).

Up, from hull to route: `measure` runs the hull-scale ship in uniform level ice over a grid of thicknesses
and throttles, and records the speed it settles at, the throttle below which it stops, and its speed in
the channel of another ship. `reduce` turns those runs into a correction to Lindqvist's formula, so that
at route scale (ships.py) the icebreaker holds the speed the hull-scale simulator gives. The result is a
small JSON file, stlnav/tables/icebreaker.json, so that route-scale runs do not need the hull-scale
simulator.

    uv run --no-project --python 3.12 --with jax --with numpy python -m stlnav.hullscale        # about 25 minutes on a laptop

Down, from route to hull: `zoom` takes a place and time of a route-scale episode, reads the ice there, and
runs the hull-scale simulator in that ice at the same throttle.

Limits. The hull-scale JAX simulator is a continuous sheet: it has no floes, so it says nothing about how
resistance falls with concentration. It models one hull, the 76 m icebreaker, so the other ship classes
keep Lindqvist's formula as it is. Below `H_MIN` the cusps are smaller than three grid cells and the
hull-scale results are not used.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from icebreaker_jax import Config, EnvParams, Icebreaker

TABLE = pathlib.Path(__file__).parent / "tables" / "icebreaker.json"
# constants of the hull-scale Config that the measured speeds depend on; the table records them
PHYSICS = ("L", "B", "T", "Cb", "bow_frac", "stern_frac", "bow_pow", "stern_taper", "phi_mid", "phi_bow", "phi_stem", "phi_stern", "TB", "uow",
           "sigC", "mu", "E", "nu", "rhoW", "rhoI", "g", "Cf", "Cl", "Cv", "cover_gain", "cell", "dt", "inner", "zcap", "kb", "kc")


def run_config(**kw) -> Config:
    """Hull-scale settings for measurement runs: a long uniform sheet, no ridges, steering at 60 Hz."""
    return Config(**{**dict(W=900.0, H=200.0, outer=1, n_ridges=0, t_max=1e9), **kw})


def track_hold(s, y0):
    """Rudder command that keeps the ship on the line y = y0, heading +x: the heading hold of the browser
    version, aimed back at the line by up to 0.15 rad."""
    heading = jnp.clip(-0.01 * (s.y - y0), -0.15, 0.15)
    return jnp.clip(2.2 * (heading - s.th) - 9.0 * s.r, -1.0, 1.0)


def steady(cfg: Config, h, thr, sigF=0.55e6, u0=None, seed=0, seconds=150.0, settle=60.0, run_in=0.0, chunk=56, state=None, tow=False, hvar=0.0, log=None):
    """Speed and mean ice force of the hull-scale ship once it has settled, in uniform level ice.

    h, thr, sigF, u0 broadcast to one-dimensional arrays, one run each. The ship starts with its bow at
    the ice edge at speed u0; a start near the final speed shortens the transient. The average is taken
    after `settle` seconds and after the ship has covered the share `run_in` of the sheet, while the bow
    is still well inside the sheet. `state` (batched) replaces the
    fresh ice sheet, which is how a second ship is sent down the channel of the first. With `tow` the
    surge speed is held at u0 whatever the forces, as in a towing tank, and `thr` has no effect on it.
    `hvar` is the amplitude of a smooth thickness variation, as a fraction of h (0 = uniform ice).

    Returns a dict of arrays [n]: speed (m/s), ice (N), drift (the trend of the speed over the averaging
    window, m/s; near zero when the ship has settled), samples (number of steps averaged), and the final
    batched state.
    """
    h, thr, sigF = np.broadcast_arrays(np.asarray(h, np.float32), np.asarray(thr, np.float32), np.asarray(sigF, np.float32))
    h, thr, sigF = h.ravel(), thr.ravel(), sigF.ravel()
    n = h.size
    u0 = np.asarray(4.0 if u0 is None else u0, np.float32)
    u0 = u0.ravel() if u0.size == n else np.broadcast_to(u0, (n,))
    env = Icebreaker(cfg)
    step_s = cfg.outer * cfg.inner * cfg.dt
    n_steps = int(round(seconds / step_s))
    x_start = 150.0 - cfg.L / 2 - 5.0                            # the sheet starts at x = 150
    x_end = cfg.W - 3.0 - cfg.L / 2 - 10.0                       # the bow is still well inside the sheet

    def rollout(s, p, th, uu):
        def f(c, _):
            s, acc = c
            _, s, _, _, info = env.step(s, jnp.stack([th, track_hold(s, cfg.H / 2)]), p)
            s = s._replace(u=jnp.where(tow, uu, s.u))
            use = ((s.t > settle) & (s.x > x_start + run_in * (x_end - x_start)) & (s.x < x_end)).astype(jnp.float32)
            tt = s.t - settle
            return (s, acc + use * jnp.stack([1.0, s.u, info["ice_resistance"], tt, tt * tt, tt * s.u])), None
        (s, acc), _ = lax.scan(f, (s, jnp.zeros(6, jnp.float32)), None, length=n_steps)
        k = jnp.maximum(acc[0], 1.0)
        mu, mf, mt = acc[1] / k, acc[2] / k, acc[3] / k
        slope = (acc[5] / k - mt * mu) / jnp.maximum(acc[4] / k - mt * mt, 1e-6)        # least-squares trend of the speed
        return s, mu, mf, slope * acc[0] * step_s, acc[0]

    reset = jax.jit(jax.vmap(env.reset))
    run = jax.jit(jax.vmap(rollout))
    out = {k: [] for k in ("speed", "ice", "drift", "samples")}
    states = []
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        m = sl.stop - sl.start
        p = EnvParams(h=jnp.asarray(h[sl]), sigF=jnp.asarray(sigF[sl]), hvar=jnp.full(m, hvar), ridge_gain=jnp.zeros(m), edge_wave=jnp.full(m, 0.6))
        if state is None:
            _, s = reset(jax.random.split(jax.random.key(seed + a), m), p)
        else:
            s = jax.tree.map(lambda v: v[sl], state)
        z = jnp.zeros(m, jnp.float32)
        s = s._replace(x=z + x_start, y=z + cfg.H / 2, th=z, u=jnp.asarray(u0[sl]), v=z, r=z, rud=z, cover=z, t=z)
        t0 = time.time()
        s, u, f, d, k = run(s, p, jnp.asarray(thr[sl]), jnp.asarray(u0[sl]))
        for key, val in zip(("speed", "ice", "drift", "samples"), (u, f, d, k)):
            out[key].append(np.asarray(val))
        states.append(s)
        if log:
            log(f"  runs {sl.start}-{sl.stop - 1} of {n}: {time.time() - t0:.0f} s")
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["state"] = jax.tree.map(lambda *v: jnp.concatenate(v), *states)
    return res


# ------------------------------------------------------------------ up: hull scale -> route scale
H_GRID = (0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2)        # thicknesses measured, m
V_GRID = tuple(0.25 * k for k in range(2, 35))        # speeds at which the correction is tabulated, m/s
H_MIN = H_GRID[0]


def measure(cfg: Config | None = None, hs=H_GRID, thrs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), seeds=3, sigF=0.55e6, rounds=5, log=print) -> dict:
    """The hull-scale runs behind the table, in level ice of uniform thickness. Four sets:

      1. free running over thickness x throttle, `seeds` runs each;
      2. for each thickness, a bisection on the throttle between a run that stops and one that keeps
         moving (a throttle counts as moving only if every seed keeps moving for 300 s);
      3. a second ship down the channel of a first, at three throttles;
      4. a few runs at other flexural strengths, kept as a test of how far the table carries.
    """
    from . import ships

    cfg = cfg or run_config()
    ib = ships.ICEBREAKER_FORMULA
    TB, uow = cfg.TB, cfg.uow
    hs = np.asarray(hs, np.float32)

    def guess(h, thr, scale=0.85):                     # speed at which thrust balances scale x Lindqvist
        a, b = ships.lindqvist_terms(ib, jnp.asarray(h), sigF)
        a, b, kx = np.asarray(a) * scale, np.asarray(b) * scale, 0.7 * TB / uow ** 2
        lin, net = 0.3 * thr * TB / uow + b, thr * TB - a
        return np.where(net > 0, (-lin + np.sqrt(lin * lin + 4 * kx * np.maximum(net, 0))) / (2 * kx), 0.0)

    log("1. free running: %d thicknesses x %d throttles x %d seeds" % (len(hs), len(thrs), seeds))
    HH, TT, SS = [a.ravel() for a in np.meshgrid(hs, np.asarray(thrs, np.float32), np.arange(seeds), indexing="ij")]
    r = steady(cfg, HH, TT, sigF, u0=np.maximum(guess(HH, TT), 1.5), seed=1, seconds=300.0, settle=40.0, run_in=0.35, log=log)
    final = np.asarray(r["state"].u)
    moving = (final > 0.5) & (r["samples"] * cfg.outer * cfg.inner * cfg.dt > 20.0) & (np.abs(r["drift"]) < 0.5)

    lo = np.array([max([0.0] + [float(t) for t in thrs if (final[(HH == h) & (TT == np.float32(t))] <= 0.5).any()]) for h in hs], np.float32)
    hi = np.array([min([1.0] + [float(t) for t in thrs if t > l and (final[(HH == h) & (TT == np.float32(t))] > 0.5).all()]) for h, l in zip(hs, lo)], np.float32)
    log("2. stall: starting brackets " + " ".join(f"[{a:.2f},{b:.2f}]" for a, b in zip(lo, hi)))
    for k in range(rounds):
        mid = 0.5 * (lo + hi)
        rr = steady(cfg, np.repeat(hs, seeds), np.repeat(mid, seeds), sigF, u0=2.5, seed=100 + k, seconds=300.0, settle=200.0)
        keeps = (np.asarray(rr["state"].u).reshape(len(hs), seeds) > 0.5).all(axis=1)
        lo, hi = np.where(keeps, lo, mid), np.where(keeps, mid, hi)
        log(f"   round {k + 1}: " + " ".join(f"[{a:.3f},{b:.3f}]" for a, b in zip(lo, hi)))

    t3 = np.asarray([0.4, 0.7, 1.0], np.float32)
    H3, T3 = [a.ravel() for a in np.meshgrid(hs, t3, indexing="ij")]
    log("3. channel: %d runs behind a first ship" % H3.size)
    first = steady(cfg, H3, 1.0, sigF, u0=guess(H3, 1.0), seed=200, seconds=320.0)
    r3 = steady(cfg, H3, T3, sigF, u0=guess(H3, T3, 0.5), seconds=200.0, settle=30.0, run_in=0.3, state=first["state"])

    H4, S4 = [a.ravel() for a in np.meshgrid(np.asarray([0.3, 0.6, 1.0], np.float32), np.asarray([0.4e6, 0.7e6], np.float32), indexing="ij")]
    log("4. other flexural strengths: %d runs" % H4.size)
    r4 = steady(cfg, H4, 1.0, S4, u0=guess(H4, 1.0), seed=300, seconds=200.0, settle=40.0, run_in=0.35)

    f = lambda a, nd=3: [round(float(v), nd) for v in np.asarray(a)]
    return dict(
        config={k: getattr(cfg, k) for k in PHYSICS}, sigF=sigF, h=f(hs), stall_throttle=[f(lo), f(hi)],
        free_runs=dict(h=f(HH), thr=f(TT), speed=f(r["speed"]), ice_kN=f(r["ice"] / 1e3, 1), final_speed=f(final), used=[bool(v) for v in moving]),
        channel_runs=dict(h=f(H3), thr=f(T3), speed=f(r3["speed"]), ice_kN=f(r3["ice"] / 1e3, 1)),
        strength_runs=dict(h=f(H4), sigF=f(S4, 0), speed=f(r4["speed"]), ice_kN=f(r4["ice"] / 1e3, 1)),
    )


def reduce(raw: dict) -> dict:
    """Turn the runs of `measure` into what the route-scale model reads.

      c[h][v]   ice resistance of the moving ship as a multiple of Lindqvist's formula, on H_GRID x V_GRID.
                Each thickness and throttle at which every run kept moving gives one point, at the mean
                speed of its runs: the resistance there is thrust minus open-water drag at that speed, so
                that the route-scale force balance returns that speed. Between the points c is linear in
                speed; outside them it is held constant;
      stall[h]  the thrust below which the ship stops, as a multiple of Lindqvist's resistance at zero speed
                (the middle of the last bisection bracket);
      track[h]  resistance in the channel left by the same ship, as a share of unbroken ice at the same speed.
    """
    from . import ships

    k = raw["config"]
    TB, uow, sigF = k["TB"], k["uow"], raw["sigF"]
    kx, kx1 = 0.7 * TB / uow ** 2, 0.01 * TB / uow
    hs, vg = np.asarray(raw["h"]), np.asarray(V_GRID)
    ib = ships.ICEBREAKER_FORMULA

    def lindqvist(h, v):
        a, b = ships.lindqvist_terms(ib, jnp.asarray(h, jnp.float32), sigF)
        return np.asarray(a) + np.asarray(b) * np.asarray(v), np.asarray(a)

    resist = lambda thr, v: thr * TB * (1 - 0.3 * v / uow) - kx * v * v - kx1 * v          # thrust minus open-water drag
    fr = {key: np.asarray(v) for key, v in raw["free_runs"].items()}
    c = np.ones((len(hs), len(vg)))
    points = []
    for i, h in enumerate(hs):
        pv, pc = [], []
        for thr in sorted(set(fr["thr"])):
            m = np.isclose(fr["h"], h) & np.isclose(fr["thr"], thr)
            if (fr["final_speed"][m] > 0.5).all():                     # every seed kept moving
                v = fr["speed"][m].mean()
                pv.append(v), pc.append(resist(thr, v) / lindqvist(h, v)[0])
                points.append((float(h), float(thr), float(v), float(pc[-1]), float(fr["speed"][m].std(ddof=1)) if m.sum() > 1 else 0.0))
        order = np.argsort(pv)
        c[i] = np.interp(vg, np.asarray(pv)[order], np.asarray(pc)[order])
    lo, hi = np.asarray(raw["stall_throttle"])
    stall = 0.5 * (lo + hi) * TB / lindqvist(hs, 0.0)[1]

    cr = {key: np.asarray(v) for key, v in raw["channel_runs"].items()}
    c_at = np.array([np.interp(v, vg, c[int(np.argmin(np.abs(hs - h)))]) for h, v in zip(cr["h"], cr["speed"])])
    share = resist(cr["thr"], cr["speed"]) / (c_at * lindqvist(cr["h"], cr["speed"])[0])
    track = np.array([share[np.isclose(cr["h"], h)].mean() for h in hs])

    f = lambda a, nd=3: [round(float(v), nd) for v in np.asarray(a)]
    out = dict(what="Hull-scale runs of icebreaker_jax.py in level ice, and what the route-scale model reads from them. "
                    "Made by stlnav/hullscale.py; do not edit by hand.",
               config=raw["config"], sigF=sigF, h=f(hs), v=f(vg), c=[f(row) for row in c], stall=f(stall), track=f(track),
               stall_throttle=raw["stall_throttle"],
               points=dict(h=[q[0] for q in points], thr=[q[1] for q in points], speed=f([q[2] for q in points]), c=f([q[3] for q in points]), speed_sd=f([q[4] for q in points])),
               free_runs=raw["free_runs"], channel_runs=dict(raw["channel_runs"], share=f(share)), strength_runs=raw["strength_runs"])
    return out


# ------------------------------------------------------------------ down: route scale -> hull scale
def zoom(env, world, lon, lat, t, thr=1.0, sigF=0.55e6, seconds=120.0, cfg: Config | None = None, hvar=0.0, seed=0):
    """Run the hull-scale simulator in the ice found at places and times of a route-scale world.

    env, world: a stlnav environment and its world, whose class 0 must be the hull-scale ship. lon, lat,
    t: arrays [n] (t in seconds after the first field frame). The hull-scale ship is run at throttle thr
    through a sheet of the local thickness, starting at the speed the route-scale model predicts.

    Returns a dict of arrays [n]:
      h, conc      the route-scale ice at each point;
      usable       False where the hull-scale simulator does not represent that ice: concentration under
                   0.9 (it has no floes) or a thickness outside the measured range. The run is made
                   anyway, in level ice of that thickness, and should not be read as the local speed;
      route_speed  speed the route-scale model gives for this ship in level ice of that thickness, without
                   wind (so that the two numbers describe the same case);
      hull_speed, ice   settled speed (m/s) and mean ice force (N) at hull scale;
      state, cfg   the final hull-scale states (ice sheet with the channel cut) and their Config.
    """
    from . import ships
    from .fields import CONC, THICK

    cfg = cfg or run_config()
    own = jax.tree.map(lambda a: a[0], world.classes)
    if not (math.isclose(float(own.L), cfg.L) and math.isclose(float(own.B), cfg.B) and math.isclose(float(own.TB), cfg.TB) and math.isclose(float(own.uow), cfg.uow, rel_tol=1e-5)):
        raise ValueError("class 0 of this world is not the ship of the hull-scale Config")
    f = env.sample(world, jnp.atleast_1d(jnp.asarray(lon, jnp.float32)), jnp.atleast_1d(jnp.asarray(lat, jnp.float32)), jnp.atleast_1d(jnp.asarray(t, jnp.float32)))
    h, conc = np.asarray(f[..., THICK]), np.asarray(f[..., CONC])
    usable = (conc >= 0.9) & (h >= H_MIN) & (h <= H_GRID[-1])
    v_route = np.asarray(ships.attainable_speed(own, thr, h, 1.0, sigF))
    hh = np.clip(h, 0.05, cfg.h_max / (1.0 + hvar) - 0.01)
    r = steady(cfg, hh, thr, sigF, u0=np.maximum(v_route, 0.5), seed=seed, seconds=seconds, settle=seconds / 3, hvar=hvar)
    return dict(h=h, conc=conc, usable=usable, route_speed=v_route, hull_speed=r["speed"], ice=r["ice"], state=r["state"], cfg=cfg)


def picture(cfg: Config, state, i=0, x0=None, x1=None):
    """What a top view of hull-scale run i needs: the ice thickness map (0 where there is no ice) with its
    extent in metres, the broken-ice mask, and the hull outline in world coordinates."""
    sdf, brash = np.asarray(state.sdf[i]), np.asarray(state.brash[i])
    hmul = np.asarray(jax.image.resize(state.hmul[i], sdf.shape, "linear"))
    x, y, th = float(state.x[i]), float(state.y[i]), float(state.th[i])
    x0 = max(0.0, x - 330.0) if x0 is None else x0
    x1 = min(cfg.W, x + 110.0) if x1 is None else x1
    i0, i1 = int(x0 / cfg.cell), int(x1 / cfg.cell) + 1
    hl = cfg.hull
    px, py = np.append(hl["px"], hl["px"][0]), np.append(hl["py"], hl["py"][0])
    return dict(ice=np.where(sdf < 0, hmul, 0.0)[:, i0:i1], brash=brash[:, i0:i1], extent=(i0 * cfg.cell, (i1 - 1) * cfg.cell, 0.0, cfg.H),
                hull_x=x + np.cos(th) * px - np.sin(th) * py, hull_y=y + np.sin(th) * px + np.cos(th) * py)


def main():
    import argparse
    import datetime
    import subprocess

    ap = argparse.ArgumentParser(description="Measure the hull-scale icebreaker and write the table the route-scale model reads.")
    ap.add_argument("--out", default=str(TABLE))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reduce-only", action="store_true", help="redo the reduction from the runs already stored in the table (no new runs)")
    a = ap.parse_args()
    t0 = time.time()
    if a.reduce_only:
        old = json.loads(pathlib.Path(a.out).read_text())
        tab = dict(made=old["made"], commit=old["commit"], **reduce(old))
    else:
        raw = measure(seeds=a.seeds, log=lambda m: print(m, flush=True))
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=pathlib.Path(__file__).parent).stdout.strip()
        except OSError:
            commit = ""
        tab = dict(made=datetime.date.today().isoformat(), commit=commit, **reduce(raw))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(tab, indent=1))
    print("thickness (m)     " + "  ".join(f"{h:5.2f}" for h in tab["h"]))
    for v in (2.0, 4.0, 6.0):
        print(f"c at {v:.0f} m/s        " + "  ".join(f"{row[tab['v'].index(v)]:5.2f}" for row in tab["c"]))
    print("stall             " + "  ".join(f"{v:5.2f}" for v in tab["stall"]))
    print("track             " + "  ".join(f"{v:5.2f}" for v in tab["track"]))
    print(f"wrote {a.out} after {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
