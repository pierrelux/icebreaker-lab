"""Route-scale navigation in the St. Lawrence estuary and Gulf, as a batched JAX environment.

One ship (the "own ship") sails from a start to a goal through real fields of ice, current, wind and
waves (fields.py), among generated traffic (traffic.py). A step is `dt` seconds (default 60 s). The ship
is a point with a heading: its speed through the water relaxes to the speed its class can hold in the
local ice and wind at the commanded throttle (ships.py), and the current carries it.

This is a different scale from icebreaker_jax.py, which resolves the hull breaking ice at 0.5 m and
1/360 s over a few hundred metres. Here ice is a concentration and a thickness on a 2 km grid. The two
are linked by hullscale.py: the speed the icebreaker (class 0) holds here in level ice is the one the
hull-scale simulator gives, and any place and time of an episode can be run at hull scale (`zoom`).

Usage
    grid, world = make_world(fields, route_specs, classes, class_names)
    env = StlNav(grid)
    obs, state = env.reset(key, params, world)
    obs, state, reward, done, info = env.step(state, action, params, world)   # action = [throttle, turn]
    # batched: jax.vmap(env.step, in_axes=(0, 0, 0, None))

The throttle means what it means at hull scale: 1 is full ahead and 0 is no thrust. A negative
throttle counts as 0, because there is no going astern at this scale. The turn command is in [-1, 1].

Positions are longitude and latitude in degrees. Headings are radians from east, counterclockwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import ships, traffic
from .fields import CONC, HS, THICK, UC, UW, VC, VW, Fields
from .traffic import KY


class Grid(NamedTuple):
    """Static description of the field grid."""
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    dt: float       # seconds between field frames
    nt: int
    ny: int
    nx: int


class World(NamedTuple):
    """Arrays shared by every environment in a batch."""
    data: jax.Array                 # [T, 7, ny, nx]
    land: jax.Array                 # [ny, nx]
    routes: traffic.Routes
    classes: ships.ShipClass        # fields are [n_classes]


@dataclass(frozen=True)
class Config:
    dt: float = 60.0
    t_max: float = 48 * 3600.0
    n_traffic: int = 64
    k_obs: int = 8                  # nearest ships reported in the observation
    patch: int = 15                 # the observation holds patch x patch points of each field, north up
    patch_km: float = 4.0           # spacing of those points
    tau_v: float = 120.0            # time constant of the own ship's speed, s
    goal_km: float = 3.0            # the episode ends within this distance of the goal
    near_km: float = 0.926          # a pass closer than this (0.5 nautical mile) is a near miss
    hit_margin: float = 50.0        # collision when closer than half the two lengths plus this, m
    w_near: float = 5.0             # reward terms, in kilometres of progress
    w_hit: float = 200.0
    w_goal: float = 50.0
    traffic: traffic.TrafficCfg = traffic.TrafficCfg()


class EnvParams(NamedTuple):
    lon: jax.Array          # start
    lat: jax.Array
    th: jax.Array           # initial heading, radians from east
    goal_lon: jax.Array
    goal_lat: jax.Array
    cls: jax.Array          # own ship's class id
    sigF: jax.Array         # flexural strength of the ice, Pa
    t0: jax.Array           # start time, seconds after the first field frame


class State(NamedTuple):
    lon: jax.Array
    lat: jax.Array
    th: jax.Array
    v: jax.Array            # speed through the water, m/s
    t: jax.Array            # seconds after the first field frame
    key: jax.Array
    traffic: traffic.TrafficState


def make_world(f: Fields, route_specs, classes, class_names):
    grid = Grid(f.lon0, f.lat0, f.dlon, f.dlat, f.dt, *f.data.shape[0:1], *f.data.shape[2:])
    world = World(jnp.asarray(f.data), jnp.asarray(f.land), traffic.build_routes(route_specs, class_names), ships.stack(classes))
    return grid, world


def params(lon, lat, goal_lon, goal_lat, cls=0, sigF=0.55e6, t0=0.0, th=None) -> EnvParams:
    """Start and goal; by default the ship starts pointing at the goal."""
    if th is None:
        th = math.atan2((goal_lat - lat) * KY, (goal_lon - lon) * KY * math.cos(math.radians(lat)))
    f = lambda v: jnp.asarray(v, jnp.float32)
    return EnvParams(f(lon), f(lat), f(th), f(goal_lon), f(goal_lat), jnp.asarray(cls, jnp.int32), f(sigF), f(t0))


class StlNav:
    def __init__(self, grid: Grid, cfg: Config = Config()):
        self.g, self.cfg = grid, cfg
        o = (jnp.arange(cfg.patch, dtype=jnp.float32) - (cfg.patch - 1) / 2) * cfg.patch_km
        self._px, self._py = jnp.meshgrid(o, o)                    # km east and north of the ship

    # ------------------------------------------------------------------ fields
    def sample(self, world: World, lon, lat, t):
        """Field channels at points and time t, linear in time and space. Shape [..., 7]."""
        g = self.g
        fx = jnp.clip((lon - g.lon0) / g.dlon, 0.0, g.nx - 1.0)
        fy = jnp.clip((lat - g.lat0) / g.dlat, 0.0, g.ny - 1.0)
        ft = jnp.clip(t / g.dt, 0.0, g.nt - 1.0)
        i, j, k = jnp.minimum(fx.astype(jnp.int32), g.nx - 2), jnp.minimum(fy.astype(jnp.int32), g.ny - 2), jnp.minimum(ft.astype(jnp.int32), max(g.nt - 2, 0))
        ax, ay, at = (fx - i)[..., None], (fy - j)[..., None], jnp.asarray(ft - k)[..., None]
        k1 = jnp.minimum(k + 1, g.nt - 1)
        d = world.data

        def frame(kk):
            return ((d[kk, :, j, i] * (1 - ax) + d[kk, :, j, i + 1] * ax) * (1 - ay) + (d[kk, :, j + 1, i] * (1 - ax) + d[kk, :, j + 1, i + 1] * ax) * ay)
        return frame(k) * (1 - at) + frame(k1) * at

    def land(self, world: World, lon, lat):
        """Share of land around a point (bilinear); 1 outside the grid."""
        g = self.g
        fx, fy = (lon - g.lon0) / g.dlon, (lat - g.lat0) / g.dlat
        out = (fx < 0) | (fy < 0) | (fx > g.nx - 1) | (fy > g.ny - 1)
        fx, fy = jnp.clip(fx, 0.0, g.nx - 1.0), jnp.clip(fy, 0.0, g.ny - 1.0)
        i, j = jnp.minimum(fx.astype(jnp.int32), g.nx - 2), jnp.minimum(fy.astype(jnp.int32), g.ny - 2)
        ax, ay = fx - i, fy - j
        m = world.land
        v = (m[j, i] * (1 - ax) + m[j, i + 1] * ax) * (1 - ay) + (m[j + 1, i] * (1 - ax) + m[j + 1, i + 1] * ax) * ay
        return jnp.where(out, 1.0, v)

    # ------------------------------------------------------------------ reset / observation
    def reset(self, key, p: EnvParams, world: World):
        k1, k2 = jax.random.split(key)
        ts = traffic.reset(k1, self.cfg.n_traffic, p.t0, world.routes, world.classes, self.cfg.traffic)
        own = jax.tree.map(lambda a: a[p.cls], world.classes)
        state = State(lon=p.lon, lat=p.lat, th=p.th, v=0.8 * own.uow, t=p.t0, key=k2, traffic=ts)
        return self.observe(state, p, world), state

    def _goal(self, lon, lat, p: EnvParams):
        dx, dy = (p.goal_lon - lon) * KY * jnp.cos(jnp.deg2rad(lat)), (p.goal_lat - lat) * KY
        return jnp.hypot(dx, dy), jnp.arctan2(dy, dx)

    def observe(self, s: State, p: EnvParams, world: World):
        """ship (9,), fields (8, patch, patch) north up, traffic (k_obs, 7) nearest first."""
        c = self.cfg
        kx = KY * jnp.cos(jnp.deg2rad(s.lat))
        plon, plat = s.lon + self._px / kx, s.lat + self._py / KY
        f = self.sample(world, plon, plat, s.t)
        fields = jnp.concatenate([self.land(world, plon, plat)[None], jnp.moveaxis(f, -1, 0) / jnp.asarray([1, 1, 1, 1, 10, 10, 3], jnp.float32)[:, None, None]])
        dist, brg = self._goal(s.lon, s.lat, p)
        here = self.sample(world, s.lon, s.lat, s.t)
        ship = jnp.stack([s.v / 8.0, jnp.cos(s.th), jnp.sin(s.th), jnp.tanh(dist / 100.0), jnp.cos(brg - s.th), jnp.sin(brg - s.th),
                          here[CONC], here[THICK], here[HS] / 3.0])
        tlon, tlat, _, tve, tvn = traffic.kinematics(s.traffic, s.t, lambda lo, la, t: self.sample(world, lo, la, t), world.routes)
        dx, dy = (tlon - s.lon) * kx, (tlat - s.lat) * KY
        d = jnp.where(s.traffic.active, jnp.hypot(dx, dy), jnp.inf)
        _, idx = jax.lax.top_k(-d, c.k_obs)
        on = jnp.isfinite(d[idx])
        near = jnp.stack([dx[idx] / 10.0, dy[idx] / 10.0, tve[idx] / 8.0, tvn[idx] / 8.0, world.classes.L[s.traffic.cls[idx]] / 200.0,
                          jnp.tanh(d[idx] / 20.0), jnp.ones_like(dx[idx])], axis=1)
        return {"ship": ship, "fields": fields, "traffic": jnp.where(on[:, None], near, 0.0)}

    # ------------------------------------------------------------------ step
    def step(self, s: State, action, p: EnvParams, world: World):
        c, tc = self.cfg, self.cfg.traffic
        thr, turn = jnp.clip(action[0], 0.0, 1.0), jnp.clip(action[1], -1.0, 1.0)
        key, kt = jax.random.split(s.key)
        samp = lambda lo, la, t: self.sample(world, lo, la, t)
        own = jax.tree.map(lambda a: a[p.cls], world.classes)
        here = samp(s.lon, s.lat, s.t)
        cs, sn = jnp.cos(s.th), jnp.sin(s.th)
        tlon, tlat, thd, tve, tvn = traffic.kinematics(s.traffic, s.t, samp, world.routes)
        kx = KY * jnp.cos(jnp.deg2rad(s.lat))
        rx0, ry0 = (tlon - s.lon) * kx, (tlat - s.lat) * KY                       # traffic seen from the own ship, km

        # in the track of a ship ahead: behind it along its heading, and close to its line
        back = -(-rx0 * jnp.cos(thd) - ry0 * jnp.sin(thd))                        # distance of the own ship behind each ship
        side = jnp.abs(-rx0 * jnp.sin(thd) + ry0 * jnp.cos(thd))
        in_track = jnp.any(s.traffic.active & (back > 0) & (back < tc.d_channel) & (side < tc.w_channel))

        f_wind = ships.wind_force(own, s.v, s.th, here[UW], here[VW])
        v_goal = ships.attainable_speed(own, thr, here[THICK], here[CONC], p.sigF, f_ext=f_wind, in_track=in_track)
        v = s.v + (v_goal - s.v) * (1.0 - math.exp(-c.dt / c.tau_v))
        th = s.th + turn * own.turn * jnp.clip(v / own.uow, 0.2, 1.0) * c.dt
        ve, vn = v * cs + here[UC], v * sn + here[VC]
        lon = s.lon + ve * c.dt * 1e-3 / kx
        lat = s.lat + vn * c.dt * 1e-3 / KY

        ts = traffic.step(s.traffic, kt, s.t, c.dt, (s.lon, s.lat, ve, vn), samp, world.routes, world.classes, tc, p.sigF)
        t = s.t + c.dt
        new = State(lon=lon, lat=lat, th=th, v=v, t=t, key=key, traffic=ts)

        # closest approach to each ship during the step (straight-line motion of both)
        nlon, nlat, _ = traffic.place(world.routes, ts.route, ts.s, ts.off)
        rx1, ry1 = (nlon - lon) * kx, (nlat - lat) * KY
        ex, ey = rx1 - rx0, ry1 - ry0
        q = jnp.clip(-(rx0 * ex + ry0 * ey) / (ex * ex + ey * ey + 1e-12), 0.0, 1.0)
        there = s.traffic.active & ts.active & (s.traffic.route == ts.route) & (ts.s >= s.traffic.s)
        dmin = jnp.where(there, jnp.hypot(rx0 + q * ex, ry0 + q * ey), jnp.inf)
        closest = jnp.min(dmin)
        hit = jnp.any(dmin * 1e3 < 0.5 * (own.L + world.classes.L[ts.cls]) + c.hit_margin)
        near = jnp.sum(dmin < c.near_km)

        d0, _ = self._goal(s.lon, s.lat, p)
        d1, _ = self._goal(lon, lat, p)
        aground = self.land(world, lon, lat) > 0.5
        arrived = d1 < c.goal_km
        reward = (d0 - d1) - c.w_near * near - c.w_hit * (hit | aground) + c.w_goal * arrived
        done = arrived | hit | aground | (t - p.t0 >= c.t_max) | (t >= (self.g.nt - 1) * self.g.dt)
        info = {"arrived": arrived, "collision": hit, "aground": aground, "near_misses": near, "closest_km": closest, "in_track": in_track,
                "speed": v, "goal_km": d1}
        return self.observe(new, p, world), new, reward, done, info
