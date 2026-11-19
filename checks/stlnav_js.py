"""The browser port of the route-scale model (stlnav-core.js) against the Python one, on the same data.

    uv run --no-project --python 3.12 --with jax --with numpy python checks/stlnav_js.py

Python is run on the fields exactly as the page sees them (stl-data.json, rebuilt by export_web.fields_from_web),
writes its results to a file, and node runs checks/stlnav_js.js, which repeats each case with stlnav-core.js
and prints the comparison. Random departures are switched off in the cases that are compared step by step,
because the two random generators differ; the generator itself is checked on its mean (in the .js file).
The last check has no Python side: it runs the page's own pilot (StlNav.pilot) over 60 voyages.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
from stlnav import env as E
from stlnav import export_web as X
from stlnav import fields as F
from stlnav import routes as RT
from stlnav import ships, traffic

root = pathlib.Path(__file__).resolve().parent.parent
d = json.loads((root / "stl-data.json").read_text())
THICK = 0.3
f = X.fields_from_web(d, THICK)
env, world = RT.world(f, ice_route=True)
rng = np.random.default_rng(0)
out = {"thick": THICK}
lst = lambda a: np.asarray(a, np.float64).tolist()

# 1. speeds of three classes
n = 400
h, thr, conc = (rng.uniform(lo, hi, n).astype(np.float32) for lo, hi in ((0.02, 1.5), (0.0, 1.0), (0.0, 1.0)))
trk, fext = (rng.uniform(size=n) < 0.3), rng.uniform(-5e4, 2e5, n).astype(np.float32)
out["speeds"] = dict(h=lst(h), thr=lst(thr), conc=lst(conc), track=trk.tolist(), fext=lst(fext),
                     v=[lst(ships.attainable_speed(jax.tree.map(lambda a: a[c], world.classes), thr, h, conc, f_ext=fext, in_track=trk)) for c in (0, 1, 2)])

# 2. fields at random places and times
lon, lat, t = rng.uniform(-70.5, -57.0, 300), rng.uniform(46.0, 51.5, 300), rng.uniform(0, 86400.0, 300)
s = np.asarray(env.sample(world, jnp.asarray(lon, jnp.float32), jnp.asarray(lat, jnp.float32), jnp.asarray(t, jnp.float32)))
out["fields"] = dict(lon=lst(np.float32(lon)), lat=lst(np.float32(lat)), t=lst(np.float32(t)), conc=lst(s[:, F.CONC]), thick=lst(s[:, F.THICK]), uc=lst(s[:, F.UC]), vc=lst(s[:, F.VC]),
                     uw=lst(s[:, F.UW]), vw=lst(s[:, F.VW]), land=lst(env.land(world, jnp.asarray(lon, jnp.float32), jnp.asarray(lat, jnp.float32))))

# 3. the own ship alone, and 4. with the traffic present at the start (no departures), under a fixed sequence of actions
quiet = world._replace(routes=world.routes._replace(rate=jnp.zeros_like(world.routes.rate), period=jnp.zeros_like(world.routes.period)))
p = RT.scenario("escoumins_to_sept_iles", f)
step = jax.jit(env.step)
act = lambda k: jnp.array([0.75 + 0.25 * np.sin(k / 50.0), 0.25 * np.sin(k / 23.0)], jnp.float32)


def state_of(st):
    return [float(st.lon), float(st.lat), float(st.th), float(st.v)]


def traffic_of(ts):
    return dict(active=np.asarray(ts.active).astype(int).tolist(), route=np.asarray(ts.route).tolist(), s=lst(ts.s), off=lst(ts.off), off0=lst(ts.off0), v=lst(ts.v), vdes=lst(ts.vdes),
                cls=np.asarray(ts.cls).tolist(), comply=np.asarray(ts.comply).astype(int).tolist())


_, st = env.reset(jax.random.key(0), p, quiet)
rec = []
for k in range(600):
    _, st, _, _, info = step(st, act(k), p, quiet)
    if k % 60 == 59:
        rec.append(state_of(st))
out["alone"] = dict(start=[float(p.lon), float(p.lat), float(p.th)], goal=[float(p.goal_lon), float(p.goal_lat)], states=rec)

_, st = env.reset(jax.random.key(4), p, world)
start_traffic, rec, near, closest, tracks = traffic_of(st.traffic), [], 0, [], 0
for k in range(360):
    _, st, _, _, info = step(st, act(k), p, quiet)
    near, tracks = near + int(info["near_misses"]), tracks + int(info["in_track"])
    closest.append(float(info["closest_km"]) if np.isfinite(float(info["closest_km"])) else -1.0)
    if k % 60 == 59:
        rec.append(dict(own=state_of(st), traffic=traffic_of(st.traffic)))
out["with_traffic"] = dict(traffic=start_traffic, records=rec, near=near, in_track=tracks, closest=closest)

# 5. the three encounters and the convoy of checks/stlnav_checks.py, on a test route in open water and in uniform ice
EAST = [dict(name="test", wp=[(47.5, -61.5), (47.5, -59.5)], per_day=0.0, mix={RT.CLASS_NAMES[1]: 1.0}, off=0.0, one_way=True)]
g2, w2 = E.make_world(f, EAST, RT.CLASSES, RT.CLASS_NAMES)
w2 = w2._replace(data=jnp.zeros_like(w2.data))
e2 = E.StlNav(g2)
ib = jax.tree.map(lambda a: a[0], world.classes)
step2 = jax.jit(e2.step)
enc = []
for kind in ("crossing", "head-on", "overtaking"):
    for comply in (0, 1):
        v_t = 6.0
        ts = traffic.TrafficState(active=jnp.arange(64) == 0, route=jnp.zeros(64, jnp.int32), s=jnp.full(64, 20.0), off=jnp.zeros(64), off0=jnp.zeros(64), v=jnp.full(64, v_t),
                                  vdes=jnp.full(64, v_t), cls=jnp.ones(64, jnp.int32), comply=jnp.full(64, bool(comply)))
        lon_t, lat_t, _ = traffic.place(w2.routes, ts.route, ts.s, ts.off)
        kx, T = traffic.KY * np.cos(np.deg2rad(47.5)), 1500.0
        mx, my = float(lon_t[0]) + v_t * T / 1e3 / kx, 47.5
        if kind == "crossing":
            lo, la, th, vo = mx, my - 6.0 * T / 1e3 / traffic.KY, np.pi / 2, 6.0
        elif kind == "head-on":
            lo, la, th, vo = mx + 6.0 * T / 1e3 / kx, my, np.pi, 6.0
        else:
            lo, la, th, vo = mx - 2.0 * T / 1e3 / kx, my, 0.0, 2.0
        pp = E.params(lo, la, lo + 3 * np.cos(th), la + 3 * np.sin(th), cls=0, th=th)
        s0 = E.State(jnp.float32(lo), jnp.float32(la), jnp.float32(th), jnp.float32(vo), jnp.float32(0.0), jax.random.key(0), ts)
        tq = float(jnp.interp(vo, ships.attainable_speed(ib, jnp.linspace(0, 1, 200), 0.0, 0.0), jnp.linspace(0, 1, 200)))
        best = np.inf
        for _ in range(60):
            _, s0, _, _, info = step2(s0, jnp.array([tq, 0.0]), pp, w2)
            best = min(best, float(info["closest_km"]))
        enc.append(dict(kind=kind, comply=comply, own=[float(np.float32(lo)), float(np.float32(la)), float(np.float32(th)), vo], goal=[float(pp.goal_lon), float(pp.goal_lat)], thr=tq, closest=best))
ice = w2._replace(data=jnp.zeros_like(w2.data).at[:, F.CONC].set(1.0).at[:, F.THICK].set(0.35))
conv = []
for leader in (0, 1):
    ts = traffic.TrafficState(active=(jnp.arange(64) == 0) | ((jnp.arange(64) == 1) & bool(leader)), route=jnp.zeros(64, jnp.int32), s=jnp.where(jnp.arange(64) == 1, 12.0, 10.0), off=jnp.zeros(64),
                              off0=jnp.zeros(64), v=jnp.full(64, 1.0), vdes=jnp.full(64, 7.0), cls=jnp.where(jnp.arange(64) == 1, 0, 1).astype(jnp.int32), comply=jnp.ones(64, bool))
    for i in range(60):
        ts = traffic.step(ts, jax.random.key(i), i * 60.0, 60.0, (jnp.float32(0.0),) * 4, lambda lo, la, t: e2.sample(ice, lo, la, t), w2.routes, w2.classes, env.cfg.traffic, 0.55e6)
    conv.append(dict(leader=leader, v=[float(ts.v[0]), float(ts.v[1])], s=[float(ts.s[0]), float(ts.s[1])]))
r2 = w2.routes
out["test_route"] = dict(lon=lst(r2.lon), lat=lst(r2.lat), cum=lst(r2.cum), length=lst(r2.length), rate=[0.0], period=[0.0], phase=[0.0], mix=lst(r2.mix), off=lst(r2.off))
out["encounters"], out["convoy"] = enc, conv

with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "vectors.json"
    path.write_text(json.dumps(out))
    sys.exit(subprocess.run(["node", str(root / "checks" / "stlnav_js.js"), str(path)], cwd=root).returncode)
