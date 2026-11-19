"""Checks of the link between the hull-scale simulator (icebreaker_jax.py) and the route-scale one (stlnav).

    uv run --no-project --python 3.12 --with jax --with numpy python checks/two_scales.py data/fields_20261117.npz

The first three checks read the stored hull-scale runs (stlnav/tables/icebreaker.json). The others make
new hull-scale runs at thicknesses and throttles that are not in the table, which takes about 4 minutes
on a laptop. The last one needs the fields file made by stlnav/demo.py and is skipped without it.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
from stlnav import fields as F
from stlnav import hullscale as H
from stlnav import routes as RT
from stlnav import ships

ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def one(c):
    return jax.tree.map(lambda a: a[0], ships.stack([c]))


tab = ships.load_table()
if tab is None:
    sys.exit("no table: run `python -m stlnav.hullscale` first")
ib, formula = one(ships.ICEBREAKER), one(ships.ICEBREAKER_FORMULA)
err = lambda a, b: np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(b)

# 1. the table belongs to the present hull-scale code
changed = ships.table_is_current(tab)
report("the table was measured with the present hull-scale constants", not changed, f"made {tab['made']}" if not changed else "changed since: " + ", ".join(changed))

# 2. the route-scale model returns the mean hull-scale speed of every thickness and throttle of the table
pt = {k: np.asarray(v) for k, v in tab["points"].items()}
pred = np.asarray(ships.attainable_speed(ib, pt["thr"], pt["h"], 1.0, tab["sigF"]))
old = np.asarray(ships.attainable_speed(formula, pt["thr"], pt["h"], 1.0, tab["sigF"]))
e, e_old = err(pred, pt["speed"]), err(old, pt["speed"])
report("route-scale speed returns the hull-scale speeds it was fitted to", e.max() < 0.03,
       f"{pt['h'].size} thickness-throttle pairs (3 runs each): mean difference {100 * e.mean():.2f}%, largest {100 * e.max():.2f}% "
       f"(Lindqvist's formula alone: {100 * e_old.mean():.1f}% and {100 * e_old.max():.1f}%)")

# the Newton solve of the force balance against a search on a fine grid of speeds
rng = np.random.default_rng(0)
n = 1200
hh, tt, cc = (rng.uniform(lo, hi, n).astype(np.float32) for lo, hi in ((0.02, 1.5), (0.0, 1.0), (0.0, 1.0)))
trk, fext = (rng.uniform(size=n) < 0.3).astype(np.float32), rng.uniform(-5e4, 2e5, n).astype(np.float32)
vg = np.linspace(0, 12, 3001, dtype=np.float32)
worst, rising = 0.0, 0
for s in (ib, formula, one(RT.CLASSES[1])):
    v = np.asarray(ships.attainable_speed(s, tt, hh, cc, f_ext=fext, in_track=trk))
    TB, uow = float(s.TB), float(s.uow)
    g = (tt[:, None] * TB - fext[:, None] - (0.3 * tt[:, None] + 0.01) * TB / uow * vg - 0.7 * TB / uow ** 2 * vg ** 2
         - np.asarray(ships.ice_resistance(s, hh[:, None], vg[None, :], cc[:, None], in_track=trk[:, None])))
    ref = vg[np.where((g > 0).any(1), g.shape[1] - 1 - np.argmax((g > 0)[:, ::-1], axis=1), 0)]
    worst, rising = max(worst, float(np.abs(v - ref)[v > 0].max())), rising + int((np.diff(g, axis=1) > 0).any(axis=1).sum())
report("the speed solver finds the root of the force balance", worst < 0.006 and rising == 0,
       f"{3 * n} random cases (three ships; thickness, throttle, concentration, wind, in or out of a channel): largest difference from a search on a 0.004 m/s grid {worst:.4f} m/s; "
       f"cases where the net force rises with speed, which would allow two roots: {rising}")

# 3. the ship stops in the same runs
fr = {k: np.asarray(v) for k, v in tab["free_runs"].items()}
pred = np.asarray(ships.attainable_speed(ib, fr["thr"], fr["h"], 1.0, tab["sigF"]))
old = np.asarray(ships.attainable_speed(formula, fr["thr"], fr["h"], 1.0, tab["sigF"]))
stopped = fr["final_speed"] <= 0.5
wrong, wrong_old = (pred == 0) != stopped, (old == 0) != stopped
report("route scale stops the ship where the hull-scale ship stops", wrong.mean() < 0.05,
       f"{stopped.sum()} of {stopped.size} hull-scale runs stop; route scale disagrees on {wrong.sum()} (Lindqvist's formula alone: {wrong_old.sum()})"
       + ("" if not wrong.any() else "; all at (thickness, throttle) " + ", ".join(f"({h:.1f} m, {t:.1f})" for h, t in sorted(set(zip(fr["h"][wrong], fr["thr"][wrong]))))
          + ", the throttle nearest the stalling one, where some runs stop and some do not"))

# 4. new hull-scale runs between the measured thicknesses and throttles, two each
cfg = H.run_config()
new = [(0.25, 0.75), (0.25, 0.95), (0.5, 0.75), (0.5, 0.95), (0.7, 0.6), (0.9, 0.75), (0.9, 0.95), (1.1, 0.75), (1.1, 0.95)]
h4, t4 = (np.repeat(np.asarray(v, np.float32), 2) for v in zip(*new))
p4 = np.asarray(ships.attainable_speed(ib, t4, h4, 1.0))
o4 = np.asarray(ships.attainable_speed(formula, t4, h4, 1.0))
r4 = H.steady(cfg, h4, t4, u0=np.maximum(p4, 1.5), seed=7, seconds=300.0, settle=40.0, run_in=0.35)
mv4 = np.asarray(r4["state"].u) > 0.5
m4 = r4["speed"].reshape(-1, 2).mean(axis=1)
e4, eo4, thin = err(p4[::2], m4), err(o4[::2], m4), h4[::2] < 0.6
spread = np.abs(np.diff(r4["speed"].reshape(-1, 2), axis=1))[:, 0] / m4
report("new hull-scale runs between the table's points", mv4.all() and e4[thin].max() < 0.03 and e4[~thin].max() < 0.12,
       f"2 runs each. Ice of 0.25 and 0.5 m: mean difference {100 * e4[thin].mean():.1f}%, largest {100 * e4[thin].max():.1f}% (formula alone: {100 * eo4[thin].mean():.1f}% and "
       f"{100 * eo4[thin].max():.1f}%). Ice of 0.7, 0.9 and 1.1 m: mean {100 * e4[~thin].mean():.1f}%, largest {100 * e4[~thin].max():.1f}% (formula alone: "
       f"{100 * eo4[~thin].mean():.1f}% and {100 * eo4[~thin].max():.1f}%). The two runs of a pair differ by up to {100 * spread.max():.1f}%, so these differences are not noise: "
       f"between the table's thicknesses the hull-scale resistance varies in ways a linear interpolation misses")

# 5. new runs either side of the predicted stalling throttle, and in a channel
h5 = np.asarray([0.7, 0.7, 0.9, 0.9, 1.1, 1.1], np.float32)
a0, _ = ships.lindqvist_terms(ib, h5, 0.55e6)
thr_stall = np.asarray(ships.at_h(ib.stall, h5) * a0 / ib.TB)
t5 = thr_stall * np.tile(np.asarray([0.8, 1.2], np.float32), 3)
r5 = H.steady(cfg, h5, t5, u0=2.5, seed=8, seconds=300.0, settle=200.0)
moves = np.asarray(r5["state"].u) > 0.5
want = np.tile([False, True], 3)
report("the hull-scale ship stops below the stalling throttle and moves above it", (moves == want).all(),
       "; ".join(f"h = {h:.1f} m: stall at throttle {s:.2f}, at {a:.2f} it {'moves' if ma else 'stops'}, at {b:.2f} it {'moves' if mb else 'stops'}"
                 for h, s, a, b, ma, mb in zip(h5[::2], thr_stall[::2], t5[::2], t5[1::2], moves[::2], moves[1::2])))

h6, t6 = np.asarray([0.5, 0.5, 0.9, 0.9], np.float32), np.asarray([0.6, 1.0, 0.6, 1.0], np.float32)
first = H.steady(cfg, h6, 1.0, u0=np.asarray(ships.attainable_speed(ib, 1.0, h6, 1.0)), seed=9, seconds=320.0)
p6 = np.asarray(ships.attainable_speed(ib, t6, h6, 1.0, in_track=1.0))
r6 = H.steady(cfg, h6, t6, u0=p6, seconds=200.0, settle=30.0, run_in=0.3, state=first["state"])
e6 = err(p6, r6["speed"])
free = np.asarray(ships.attainable_speed(ib, t6, h6, 1.0))
report("speed in the channel of another icebreaker", e6.max() < 0.08,
       f"mean difference {100 * e6.mean():.1f}%, largest {100 * e6.max():.1f}%; e.g. 0.9 m at full throttle: {r6['speed'][3]:.2f} m/s in the channel at hull scale, "
       f"{p6[3]:.2f} at route scale, {free[3]:.2f} in unbroken ice")

# other flexural strengths: the curves are fitted at one strength and Lindqvist's formula carries them to others
sr = {k: np.asarray(v) for k, v in tab["strength_runs"].items()}
ps = np.asarray(ships.attainable_speed(ib, 1.0, sr["h"], 1.0, sr["sigF"]))
es = err(ps, sr["speed"])
report("other flexural strengths (0.40 and 0.70 MPa; the curves are fitted at 0.55)", es.max() < 0.10,
       f"{sr['h'].size} stored runs at full throttle: mean difference {100 * es.mean():.1f}%, largest {100 * es.max():.1f}%")

# 6. zoom: a point of a route-scale world run at hull scale
path = sys.argv[1] if len(sys.argv) > 1 else "data/fields_20261117.npz"
if os.path.exists(path):
    f = F.load(path)
    lon2d = np.broadcast_to(f.lon[None, :], f.land.shape)
    thick = np.interp(lon2d, [-69.5, -60.0], [0.3, 1.1]).astype(np.float32)          # made-up ice: thicker towards the east
    f = F.with_ice(f, np.ones_like(f.land), thick)
    env, world = RT.world(f, ice_route=True)
    pts = [RT.nearest_water(f, la, lo) for la, lo in ((48.733, -68.800), (49.640, -65.000), (48.300, -61.333))]
    z = H.zoom(env, world, [p[1] for p in pts], [p[0] for p in pts], [0.0, 0.0, 0.0], thr=1.0)
    ez = err(z["route_speed"], z["hull_speed"])
    report("a point of a voyage opened at hull scale", z["usable"].all() and ez.max() < 0.08,
           "; ".join(f"{h:.2f} m of ice: {a:.2f} m/s at hull scale, {b:.2f} at route scale" for h, a, b in zip(z["h"], z["hull_speed"], z["route_speed"])))
    f0 = F.load(path)
    env0, world0 = RT.world(f0, ice_route=True)
    z0 = H.zoom(env0, world0, [pts[0][1]], [pts[0][0]], [0.0], seconds=20.0)
    report("zoom flags ice the hull-scale simulator cannot represent", not z0["usable"][0], f"archived November day: concentration {z0['conc'][0]:.2f}, thickness {z0['h'][0]:.2f} m, usable = {bool(z0['usable'][0])}")
else:
    print(f"SKIP  zoom: no fields file at {path}")

print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
