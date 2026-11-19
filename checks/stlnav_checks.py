"""Checks of the route-scale environment (stlnav). Each prints PASS or FAIL with the numbers behind it.

    uv run --no-project --python 3.12 --with jax --with numpy python checks/stlnav_checks.py data/fields_20261117.npz

The fields file is made by stlnav/demo.py (or fields.save). Small enough for a laptop.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")
from stlnav import env as E
from stlnav import fields as F
from stlnav import routes as RT
from stlnav import ships, traffic

f = F.load(sys.argv[1] if len(sys.argv) > 1 else "data/fields_20261117.npz")
ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def one(c):
    return jax.tree.map(lambda a: a[0], ships.stack([c]))


# 1. speed model
ib = one(ships.ICEBREAKER)
v0 = float(ships.attainable_speed(ib, 1.0, 0.0, 0.0))
hs = jnp.linspace(0.05, 1.5, 30)
vh = np.array(ships.attainable_speed(ib, 1.0, hs, 1.0))
vc = np.array(ships.attainable_speed(ib, 1.0, 0.8, jnp.linspace(0.05, 1.0, 20)))
x_ow = (-0.31 + (0.31 ** 2 + 4 * 0.7) ** 0.5) / 1.4        # 0.7 x^2 + (0.3 + 0.01) x = 1: thrust against the two open-water drag terms
report("open water gives the open-water speed", abs(v0 - x_ow * float(ib.uow)) < 1e-3,
       f"{v0:.3f} m/s, which is {x_ow:.3f} of the nominal {float(ib.uow):.1f} m/s because of the small linear drag shared with icebreaker_jax.py")
report("speed falls with thickness and with concentration", (np.diff(vh) <= 1e-6).all() and (np.diff(vc) <= 1e-6).all(), f"1.0 m level ice: {vh[np.argmin(abs(np.array(hs) - 1.0))]:.2f} m/s")
a, b = ships.lindqvist_terms(ib, 0.6, 0.55e6)
ref = 0.0
try:
    from icebreaker_jax import Config
    ref = Config().lindqvist(0.6, 3.0, 0.55e6)
    report("Lindqvist matches icebreaker_jax.Config.lindqvist", abs(float(a + 3.0 * b) - ref) / ref < 1e-4, f"{float(a + 3 * b) / 1e3:.1f} vs {ref / 1e3:.1f} kN at 0.6 m, 3 m/s")
except Exception as e:  # noqa: BLE001
    report("Lindqvist matches icebreaker_jax.Config.lindqvist", False, f"could not import: {e}")

# 2. routes lie in water
for ice_route in (True, False):
    env, world = RT.world(f, ice_route)
    rt = world.routes
    R = rt.length.shape[0]
    bad = []
    for r in range(R):
        s = jnp.linspace(0.0, rt.length[r], 400)
        lon, lat, _ = traffic.place(rt, jnp.full(400, r), s, jnp.full(400, rt.off[r]))
        land = np.array(env.land(world, lon, lat))
        if (land > 0.5).any():
            k = int(np.argmax(land > 0.5))
            bad.append(f"route {r} ({RT.route_name(r)}): {100 * (land > 0.5).mean():.0f}% on land, first at {float(lat[k]):.2f}, {float(lon[k]):.2f}")
    report(f"every route stays in water on the 2 km mask ({'Coast Guard ice route' if ice_route else 'open-water lanes'})", not bad,
           f"{R} routes, {float(rt.length.sum()):.0f} km" if not bad else "; ".join(bad))

# 3. traffic volume in open water matches rate x transit time
open_world = world._replace(data=world.data.at[:, :2].set(0.0).at[:, 2:4].set(0.0))
tc = env.cfg.traffic
samp = lambda lo, la, t: env.sample(open_world, lo, la, t)
far = (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))


def run_traffic(key, n_steps, dt=120.0):
    ts = traffic.reset(key, env.cfg.n_traffic, 0.0, rt, world.classes, tc)

    def body(ts, k):
        i, key = k
        ts = traffic.step(ts, key, i * dt, dt, far, samp, rt, world.classes, tc, 0.55e6)
        return ts, ts.active.sum()
    return jax.lax.scan(body, ts, (jnp.arange(n_steps), jax.random.split(key, n_steps)))


t0 = time.time()
_, counts = jax.jit(jax.vmap(lambda k: run_traffic(k, 1440)))(jax.random.split(jax.random.key(1), 16))     # 16 x 48 h
v_typ = jnp.sum(rt.mix * world.classes.uow[None, :], axis=1) * 0.5 * (tc.speed_lo + tc.speed_hi)
transit = rt.length * 1e3 / v_typ
expected = float(jnp.sum(rt.rate * transit) + jnp.sum(jnp.where(rt.period > 0, transit / jnp.maximum(rt.period, 1.0), 0.0)))
got = float(counts.mean())
report("mean number of ships at sea matches rate x transit time", abs(got - expected) / expected < 0.1 and int(counts.max()) < env.cfg.n_traffic,
       f"simulated {got:.1f}, expected {expected:.1f}, most at once {int(counts.max())} of {env.cfg.n_traffic} slots ({time.time() - t0:.0f} s)")

# 4. encounters: a ship that follows the rules passes further away than one that does not
EAST = [dict(name="test", wp=[(47.5, -61.5), (47.5, -59.5)], per_day=0.0, mix={RT.CLASS_NAMES[1]: 1.0}, off=0.0, one_way=True)]
g2, w2 = E.make_world(f, EAST, RT.CLASSES, RT.CLASS_NAMES)
w2 = w2._replace(data=jnp.zeros_like(w2.data))
e2 = E.StlNav(g2)


def encounter(kind, comply):
    v_t = 6.0
    ts = traffic.TrafficState(active=jnp.arange(64) == 0, route=jnp.zeros(64, jnp.int32), s=jnp.full(64, 20.0), off=jnp.zeros(64), off0=jnp.zeros(64),
                              v=jnp.full(64, v_t), vdes=jnp.full(64, v_t), cls=jnp.ones(64, jnp.int32), comply=jnp.full(64, comply))
    lon_t, lat_t, _ = traffic.place(w2.routes, ts.route, ts.s, ts.off)
    kx = traffic.KY * np.cos(np.deg2rad(47.5))
    T = 1500.0                                                   # both reach the meeting point after T seconds
    mx, my = float(lon_t[0]) + v_t * T / 1e3 / kx, 47.5
    if kind == "crossing from starboard":                        # own ship comes from the south, heading north
        lon, lat, th, vo = mx, my - 6.0 * T / 1e3 / traffic.KY, np.pi / 2, 6.0
    elif kind == "head-on":
        lon, lat, th, vo = mx + 6.0 * T / 1e3 / kx, my, np.pi, 6.0
    else:                                                        # the traffic ship overtakes a slow own ship
        lon, lat, th, vo = mx - 2.0 * T / 1e3 / kx, my, 0.0, 2.0
    p = E.params(lon, lat, lon + 3 * np.cos(th), lat + 3 * np.sin(th), cls=0, th=th)
    s = E.State(jnp.float32(lon), jnp.float32(lat), jnp.float32(th), jnp.float32(vo), jnp.float32(0.0), jax.random.key(0), ts)
    thr = float(jnp.interp(vo, ships.attainable_speed(ib, jnp.linspace(0, 1, 200), 0.0, 0.0), jnp.linspace(0, 1, 200)))
    step = jax.jit(e2.step)
    closest = np.inf
    for _ in range(60):
        _, s, _, _, info = step(s, jnp.array([thr, 0.0]), p, w2)
        closest = min(closest, float(info["closest_km"]))
    return closest


for kind in ("crossing from starboard", "head-on", "overtaking"):
    d_no, d_yes = encounter(kind, False), encounter(kind, True)
    report(f"{kind}: giving way opens the pass", d_no < 0.3 and d_yes > 0.5, f"closest approach {d_no:.2f} km without the rule, {d_yes:.2f} km with it")

# 5. convoy: in ice, a weak ship behind an icebreaker goes faster than alone
ice = w2._replace(data=jnp.zeros_like(w2.data).at[:, F.CONC].set(1.0).at[:, F.THICK].set(0.35))
samp_i = lambda lo, la, t: e2.sample(ice, lo, la, t)


def convoy(with_leader):
    ts = traffic.TrafficState(active=(jnp.arange(64) == 0) | ((jnp.arange(64) == 1) & with_leader), route=jnp.zeros(64, jnp.int32),
                              s=jnp.where(jnp.arange(64) == 1, 12.0, 10.0), off=jnp.zeros(64), off0=jnp.zeros(64), v=jnp.full(64, 1.0), vdes=jnp.full(64, 7.0),
                              cls=jnp.where(jnp.arange(64) == 1, 0, 1).astype(jnp.int32), comply=jnp.ones(64, bool))
    for i in range(60):
        ts = traffic.step(ts, jax.random.key(i), i * 60.0, 60.0, far, samp_i, w2.routes, w2.classes, tc, 0.55e6)
    return float(ts.v[0]), float(ts.v[1]), float(ts.s[1] - ts.s[0])


alone, _, _ = convoy(False)
esc, lead, gap = convoy(True)
report("a ship escorted through 0.35 m ice is faster than alone", esc > alone + 0.5 and gap > 0.5, f"{alone:.2f} m/s alone, {esc:.2f} m/s behind an icebreaker doing {lead:.2f} m/s, {gap:.1f} km behind")

# 6. current: a stopped ship drifts with the water
j, i = np.unravel_index(np.argmax(np.hypot(f.data[0, F.UC], f.data[0, F.VC]) * (f.land < 0.5)), f.land.shape)
lon, lat = float(f.lon[i]), float(f.lat[j])
p = E.params(lon, lat, lon + 1.0, lat)
_, s = env.reset(jax.random.key(0), p, world)
s = s._replace(v=jnp.float32(0.0))
_, s1, *_ = env.step(s, jnp.array([0.0, 0.0]), p, world)
drift = np.hypot((float(s1.lon) - lon) * traffic.KY * np.cos(np.deg2rad(lat)), (float(s1.lat) - lat) * traffic.KY) * 1e3 / 60.0
cur = float(np.hypot(f.data[0, F.UC, j, i], f.data[0, F.VC, j, i]))
report("a stopped ship drifts at the current's speed", abs(drift - cur) < 0.05 * cur + 0.02, f"{drift:.2f} m/s drift, {cur:.2f} m/s current at {lat:.2f}, {lon:.2f}")

# 7. batch and throughput
B = 256
ps = jax.tree.map(lambda a: jnp.broadcast_to(a, (B,) + a.shape), RT.scenario("escoumins_to_sept_iles", f))
_, st = jax.jit(jax.vmap(env.reset, in_axes=(0, 0, None)))(jax.random.split(jax.random.key(2), B), ps, world)


def rollout(st, p, key):
    def body(s, k):
        _, s, r, d, _ = env.step(s, jnp.stack([1.0, 0.2 * jax.random.normal(k)]), p, world)
        return s, (r, d)
    return jax.lax.scan(body, st, jax.random.split(key, 100))


run = jax.jit(jax.vmap(rollout, in_axes=(0, 0, 0)))
keys = jax.random.split(jax.random.key(3), B)
st2, (r, d) = run(st, ps, keys)
jax.block_until_ready(r)
t0 = time.time()
st2, (r, d) = run(st2, ps, keys)
jax.block_until_ready(r)
wall = time.time() - t0
report("batched rollout runs without NaN", bool(jnp.isfinite(r).all()), f"{B} environments x 100 steps: {B * 100 / wall:,.0f} steps/s on {jax.devices()[0]}")
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
