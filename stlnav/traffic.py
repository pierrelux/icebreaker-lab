"""Synthetic ship traffic: ships appear at the ends of fixed routes and sail along them.

What a traffic ship does
  * It is created at the start of a route, either at random (a Poisson process with the route's rate) or
    on a timetable (ferries), and removed at the end.
  * It keeps to the right of the route line by `off` kilometres, so opposite directions are separated.
  * Its speed is the lower of its service speed and the speed its class can hold in the ice where it is
    (ships.attainable_speed), plus the current along the route.
  * In ice it does not overtake: behind another ship on the same route it closes to `d_keep` and then
    matches speed, and within `d_channel` it gets the lower resistance of the leader's track. An
    icebreaker with a ship behind it slows down to keep that ship in its track. This is how convoys form. In open water ships on the same route pass freely.
  * Towards the learning ship ("own ship") it applies three collision rules when a close approach is
    predicted (closest approach under `d_risk` within `t_risk`): it slows down if the own ship is crossing
    from its starboard side, and moves further to starboard if they meet on nearly opposite courses or if
    it is overtaking.
    Each ship follows the rules with probability `p_comply`, drawn once when it is created.

What it does not do: traffic ships do not avoid one another across routes (a ferry crosses the lane
traffic without reacting), and they never leave their route, so they cannot go around ice.

Everything has a fixed shape: `n` ship slots, R routes of at most P waypoints.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import ships
from .fields import CONC, THICK, UC, VC

KY = 111.195                     # kilometres per degree of latitude


class Routes(NamedTuple):
    lon: jax.Array      # [R, P] waypoints, padded with the last one
    lat: jax.Array      # [R, P]
    cum: jax.Array      # [R, P] distance along the route, km (padding keeps it increasing)
    length: jax.Array   # [R] km
    rate: jax.Array     # [R] random departures per second
    period: jax.Array   # [R] seconds between timetabled departures; 0 = none
    phase: jax.Array    # [R] time of the first timetabled departure, s
    mix: jax.Array      # [R, C] probability of each ship class
    off: jax.Array      # [R] distance kept to the right of the line, km


class TrafficCfg(NamedTuple):
    off_spread: float = 0.3      # standard deviation of a ship's lateral position, km
    speed_lo: float = 0.85       # service speed as a share of the class's open-water speed: uniform in [lo, hi]
    speed_hi: float = 1.00
    tau: float = 180.0           # time constant of speed changes, s
    ice_conc: float = 0.3        # concentration above which ships queue instead of overtaking
    d_follow: float = 5.0        # km: a ship ahead closer than this is followed in ice
    d_keep: float = 1.0          # km: spacing held in a convoy
    d_channel: float = 3.0       # km: within this distance behind a leader the track is still open
    w_channel: float = 0.15      # km: half-width of that track (own ship)
    channel: float = 0.4         # PLACEHOLDER: ice resistance in a track, as a share of unbroken ice
    d_risk: float = 1.5          # km
    t_risk: float = 900.0        # s
    slow: float = 0.3            # speed kept while giving way to a crossing ship, share of target
    sidestep: float = 1.0        # km moved to starboard when meeting head-on or overtaking
    off_rate: float = 0.004      # km/s, how fast the lateral position changes
    p_comply: float = 0.9
    escort_cls: int = 0          # class that waits for the ship behind it in ice (the icebreaker)


class TrafficState(NamedTuple):
    active: jax.Array   # [n] bool
    route: jax.Array    # [n] int
    s: jax.Array        # [n] km along the route
    off: jax.Array      # [n] km to the right of the line
    off0: jax.Array     # [n] the ship's usual lateral position
    v: jax.Array        # [n] speed through the water, m/s
    vdes: jax.Array     # [n] service speed, m/s
    cls: jax.Array      # [n] ship class
    comply: jax.Array   # [n] bool


def build_routes(specs, class_names, both_ways=True) -> Routes:
    """specs: list of dicts with name, wp = [(lat, lon), ...], and per_day or period_h (+ phase_h),
    mix = {class name: weight}, optional off (km) and one_way. Each route is also created reversed
    unless one_way is set; a reversed timetable route leaves half a period later."""
    rows = []
    for sp in specs:
        ways = [False] if (sp.get("one_way") or not both_ways) else [False, True]
        for rev in ways:
            wp = sp["wp"][::-1] if rev else sp["wp"]
            mix = np.array([sp["mix"].get(c, 0.0) for c in class_names], np.float64)
            period = 3600.0 * sp.get("period_h", 0.0)
            rows.append(dict(wp=wp, rate=sp.get("per_day", 0.0) / 86400.0, period=period, mix=mix / mix.sum(),
                             phase=3600.0 * sp.get("phase_h", 0.0) + (0.5 * period if rev else 0.0), off=sp.get("off", 1.0)))
    P = max(len(r["wp"]) for r in rows)
    R = len(rows)
    lon, lat, cum = np.zeros((R, P)), np.zeros((R, P)), np.zeros((R, P))
    for k, r in enumerate(rows):
        la, lo = np.array([w[0] for w in r["wp"]]), np.array([w[1] for w in r["wp"]])
        seg = np.hypot(np.diff(lo) * KY * np.cos(np.deg2rad(0.5 * (la[1:] + la[:-1]))), np.diff(la) * KY)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        n = len(la)
        lon[k, :n], lat[k, :n], cum[k, :n] = lo, la, c
        lon[k, n:], lat[k, n:], cum[k, n:] = lo[-1], la[-1], c[-1] + 1e3 * np.arange(1, P - n + 1)
        r["length"] = c[-1]
    f = lambda a: jnp.asarray(a, jnp.float32)
    return Routes(f(lon), f(lat), f(cum), f([r["length"] for r in rows]), f([r["rate"] for r in rows]), f([r["period"] for r in rows]),
                  f([r["phase"] for r in rows]), f(np.stack([r["mix"] for r in rows])), f([r["off"] for r in rows]))


def place(rt: Routes, route, s, off):
    """Longitude, latitude and heading (radians from east) of ships at distance s along their routes."""
    def one(r, s, off):
        at = lambda q: (jnp.interp(q, rt.cum[r], rt.lon[r]), jnp.interp(q, rt.cum[r], rt.lat[r]))
        lo, la = at(s)
        lo1, la1 = at(jnp.maximum(s - 0.3, 0.0))
        lo2, la2 = at(jnp.minimum(s + 0.3, rt.length[r]))
        kx = KY * jnp.cos(jnp.deg2rad(la))
        hd = jnp.arctan2((la2 - la1) * KY, (lo2 - lo1) * kx)
        return lo + off * jnp.sin(hd) / kx, la - off * jnp.cos(hd) / KY, hd
    return jax.vmap(one)(route, s, off)


def _spawn(ts: TrafficState, key, r, s0, do, rt: Routes, classes: ships.ShipClass, cfg: TrafficCfg):
    """Put one new ship on route r at distance s0 in the first free slot, if `do` and a slot is free."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    slot = jnp.argmin(ts.active)
    put = (jnp.arange(ts.active.shape[0]) == slot) & do & ~ts.active[slot]
    cls = jax.random.categorical(k1, jnp.log(rt.mix[r] + 1e-9))
    vdes = classes.uow[cls] * jax.random.uniform(k2, (), jnp.float32, cfg.speed_lo, cfg.speed_hi)
    off = rt.off[r] + cfg.off_spread * jax.random.normal(k3, (), jnp.float32)
    w = lambda new, old: jnp.where(put, new, old)
    return TrafficState(active=ts.active | put, route=w(r, ts.route), s=w(s0, ts.s), off=w(off, ts.off), off0=w(off, ts.off0), v=w(vdes, ts.v),
                        vdes=w(vdes, ts.vdes), cls=w(cls, ts.cls), comply=w(jax.random.bernoulli(k4, cfg.p_comply), ts.comply))


def reset(key, n: int, t, rt: Routes, classes: ships.ShipClass, cfg: TrafficCfg) -> TrafficState:
    """Traffic as it would be at time t if the generator had always been running (open-water speeds)."""
    R = rt.length.shape[0]
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    v_typ = jnp.sum(rt.mix * classes.uow[None, :], axis=1) * 0.5 * (cfg.speed_lo + cfg.speed_hi)
    transit = rt.length * 1e3 / v_typ                              # s
    expect = rt.rate * transit                                     # mean number of ships on each route
    route = jax.random.categorical(k1, jnp.log(expect + 1e-12), shape=(n,))
    active = jax.random.bernoulli(k2, jnp.minimum(expect.sum() / n, 1.0), (n,)) & (expect.sum() > 0)
    cls = jax.random.categorical(k3, jnp.log(rt.mix[route] + 1e-9), axis=-1)
    vdes = classes.uow[cls] * jax.random.uniform(k4, (n,), jnp.float32, cfg.speed_lo, cfg.speed_hi)
    off = rt.off[route] + cfg.off_spread * jax.random.normal(k5, (n,), jnp.float32)
    ts = TrafficState(active=active, route=route, s=jax.random.uniform(k6, (n,), jnp.float32) * rt.length[route], off=off, off0=off, v=vdes, vdes=vdes,
                      cls=cls, comply=jax.random.bernoulli(k7, cfg.p_comply, (n,)))
    # a timetabled ship that left before t and has not arrived yet
    keys = jax.random.split(key, R)
    for r in range(R):
        since = jnp.mod(t - rt.phase[r], jnp.maximum(rt.period[r], 1.0))
        ts = _spawn(ts, keys[r], r, since * v_typ[r] / 1e3, (rt.period[r] > 0) & (since < transit[r]), rt, classes, cfg)
    return ts


def step(ts: TrafficState, key, t, dt, own, sample, rt: Routes, classes: ships.ShipClass, cfg: TrafficCfg, sigF):
    """Advance the traffic by dt seconds from time t.

    own = (lon, lat, east speed, north speed) of the own ship over the ground. sample(lon, lat, t) returns
    the field channels at points. Returns the new state.
    """
    n = ts.active.shape[0]
    lon, lat, hd = place(rt, ts.route, ts.s, ts.off)
    fld = sample(lon, lat, t)
    conc, thick = fld[..., CONC], fld[..., THICK]
    cs, sn = jnp.cos(hd), jnp.sin(hd)

    # ships ahead on the same route
    gap = jnp.where(ts.active[None, :] & (ts.route[None, :] == ts.route[:, None]) & (ts.s[None, :] > ts.s[:, None]), ts.s[None, :] - ts.s[:, None], jnp.inf)
    lead = jnp.argmin(gap, axis=1)
    g = jnp.min(gap, axis=1)
    in_ice = conc > cfg.ice_conc
    cls = jax.tree.map(lambda a: a[ts.cls], classes)
    v_ice = ships.attainable_speed(cls, 1.0, thick, conc, sigF, channel=jnp.where(in_ice & (g < cfg.d_channel), cfg.channel, 1.0))
    target = jnp.minimum(ts.vdes, v_ice)
    queue = in_ice & (g < cfg.d_follow)
    target = jnp.where(queue, jnp.minimum(target, jnp.maximum(ts.v[lead] + (g - cfg.d_keep) * 1e3 / 300.0, 0.0)), target)
    # an icebreaker with a ship behind it holds back so that the ship stays in its track
    gb = jnp.min(gap, axis=0)                                      # distance to the nearest ship behind
    escort = in_ice & (ts.cls == cfg.escort_cls) & (gb < cfg.d_follow)
    target = jnp.where(escort, jnp.minimum(target, jnp.maximum(ts.v[jnp.argmin(gap, axis=0)] + (0.5 * cfg.d_channel - gb) * 1e3 / 300.0, 0.0)), target)

    # collision rules towards the own ship
    olon, olat, ove, ovn = own
    kx = KY * jnp.cos(jnp.deg2rad(lat))
    dx, dy = (olon - lon) * kx, (olat - lat) * KY                  # own ship seen from each traffic ship, km
    ve, vn = ts.v * cs + fld[..., UC], ts.v * sn + fld[..., VC]
    rvx, rvy = (ove - ve) * 1e-3, (ovn - vn) * 1e-3                # km/s
    rv2 = rvx * rvx + rvy * rvy + 1e-12
    tcpa = -(dx * rvx + dy * rvy) / rv2
    dcpa = jnp.hypot(dx + rvx * tcpa, dy + rvy * tcpa)
    risk = ts.active & ts.comply & (tcpa > 0) & (tcpa < cfg.t_risk) & (dcpa < cfg.d_risk)
    brg = jnp.rad2deg(jnp.arctan2(dx * sn - dy * cs, dx * cs + dy * sn))           # own ship's bearing off the bow, + to starboard
    ospd = jnp.hypot(ove, ovn) + 1e-6
    same = (ove * cs + ovn * sn) / ospd                                              # cosine of the angle between the courses
    obrg = jnp.rad2deg(jnp.arctan2(-(dx * ovn - dy * ove) / ospd, -(dx * ove + dy * ovn) / ospd))  # this ship's bearing off the own ship's bow
    head_on = (jnp.abs(brg) < 90.0) & (same < -0.94)                               # nearly opposite courses, own ship ahead
    overtaking = (jnp.abs(obrg) > 112.5) & (ts.v > ospd) & (same > 0.0)
    crossing = (brg >= 10.0) & (brg <= 112.5) & ~head_on & ~overtaking
    target = jnp.where(risk & crossing, cfg.slow * target, target)
    off_goal = jnp.where(risk & (head_on | overtaking), ts.off0 + cfg.sidestep, ts.off0)

    v = ts.v + (target - ts.v) * (1.0 - jnp.exp(-dt / cfg.tau))
    off = ts.off + jnp.clip(off_goal - ts.off, -cfg.off_rate * dt, cfg.off_rate * dt)
    along = jnp.maximum(v + fld[..., UC] * cs + fld[..., VC] * sn, 0.0)
    s = ts.s + along * dt * 1e-3
    ts = ts._replace(s=s, v=v, off=off, active=ts.active & (s < rt.length[ts.route]))

    # departures
    R = rt.length.shape[0]
    keys = jax.random.split(key, 2 * R)
    for r in range(R):
        per = jnp.maximum(rt.period[r], 1.0)
        timetable = (rt.period[r] > 0) & (jnp.floor((t + dt - rt.phase[r]) / per) > jnp.floor((t - rt.phase[r]) / per))
        do = timetable | jax.random.bernoulli(keys[2 * r], jnp.minimum(rt.rate[r] * dt, 1.0))
        ts = _spawn(ts, keys[2 * r + 1], r, 0.0, do, rt, classes, cfg)
    return ts


def kinematics(ts: TrafficState, t, sample, rt: Routes):
    """Longitude, latitude, heading and velocity over the ground (east, north; m/s) of every slot."""
    lon, lat, hd = place(rt, ts.route, ts.s, ts.off)
    fld = sample(lon, lat, t)
    return lon, lat, hd, ts.v * jnp.cos(hd) + fld[..., UC], ts.v * jnp.sin(hd) + fld[..., VC]
