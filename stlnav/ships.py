"""Ship classes and the speed a ship can hold in given ice, at the scale of a route (kilometres, minutes).

The speed is the one at which thrust equals resistance:

    thr * TB * (1 - 0.3 v / uow)  =  kx v^2 + kx1 v  +  conc^n * track * c(h, v) * R_lindqvist(h, v)  +  F_wind

  * Thrust and open-water resistance are the ones in icebreaker_jax.py: bollard pull TB falling by 30%
    at the open-water speed uow, kx = 0.7 TB / uow^2 and a small linear drag kx1 = 0.01 TB / uow. Full
    throttle in open water therefore gives 0.994 uow, at both scales.
  * R_lindqvist is Lindqvist's (1989) level-ice resistance, which is linear in v. The balance is solved
    for v by a few Newton steps.
  * c(h, v), stall(h) and track(h) tie this model to the hull-scale simulator. They are measured by
    hullscale.py and stored in tables/icebreaker.json:
      - c(h, v): the ice resistance of the moving hull-scale ship, as a multiple of Lindqvist's;
      - stall(h): the ship is stopped when its thrust is below stall(h) times Lindqvist's resistance at
        zero speed. The hull-scale ship stops at a higher thrust than the mean resistance suggests,
        because a ship moving slowly is brought to rest by one load peak;
      - track(h): resistance in the channel left by another ship, as a share of unbroken ice.
    c and stall are measured for the 76 m icebreaker only, the one hull the hull-scale simulator has.
    The other classes keep c = stall = 1, that is, Lindqvist's formula as it is. track is measured for
    the icebreaker in an icebreaker's channel and APPLIED TO EVERY CLASS; the merchant ships here are
    wider than that channel, so it is optimistic for them.
  * conc^n scales level-ice resistance (and the stall threshold) down in pack ice of concentration conc.
    n = 2 is a PLACEHOLDER, not taken from a source: the hull-scale JAX simulator has no floes, so it
    cannot supply it.
  * F_wind is the drag of the head wind on the frontal area: 0.5 rho_air Cd A w |w|.

Classes other than the icebreaker have illustrative particulars (see routes.py). For a ship whose bollard
pull is not known it is derived from installed power P: TB = eta P / (0.7 uow), with eta = 0.65, because
in this thrust model the propulsive force at uow is 0.7 TB.
"""
from __future__ import annotations

import json
import math
import pathlib
import warnings
from typing import NamedTuple

import jax.numpy as jnp

from icebreaker_jax import Config as HullConfig

_HULL = HullConfig()                                 # one source for the ice constants and the icebreaker
G, RHO_W, RHO_I, RHO_A = _HULL.g, _HULL.rhoW, _HULL.rhoI, 1.25
E_ICE, NU, MU = _HULL.E, _HULL.nu, _HULL.mu
KN = 0.5144
TABLE = pathlib.Path(__file__).parent / "tables" / "icebreaker.json"


def load_table(path=TABLE):
    """The hull-scale measurements (see hullscale.py), or None if they have not been made."""
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def table_is_current(table, cfg: HullConfig = _HULL) -> list:
    """Names of the hull-scale constants that have changed since the table was made (empty if none)."""
    return [k for k, v in table["config"].items() if not math.isclose(getattr(cfg, k), v, rel_tol=1e-6)]


_T = load_table()
if _T and not all(k in _T for k in ("h", "v", "c", "stall", "track")):      # a table from an older version of hullscale.py
    _T = None
if _T and table_is_current(_T):
    warnings.warn(f"stlnav/tables/icebreaker.json was measured with other hull-scale constants ({', '.join(table_is_current(_T))}); "
                  "rerun `python -m stlnav.hullscale`", stacklevel=2)
H_GRID = tuple(_T["h"]) if _T else (0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2)      # thicknesses of the curves, m
V_GRID = tuple(_T["v"]) if _T else (1.0, 8.0)                                # speeds at which c is tabulated, m/s
_ONES = (1.0,) * len(H_GRID)
_ONES2 = (((1.0,) * len(V_GRID)),) * len(H_GRID)
TRACK = tuple(_T["track"]) if _T else (0.4,) * len(H_GRID)                   # without a table: the old placeholder


class ShipClass(NamedTuple):
    L: float          # length at the waterline, m
    B: float          # beam, m
    T: float          # draught, m
    TB: float         # bollard pull, N
    uow: float        # open-water speed at full throttle, m/s
    stem: float       # stem angle from the horizontal, rad (small = icebreaking bow)
    entrance: float   # waterline entrance half-angle, rad
    area: float       # frontal windage area, m^2
    turn: float       # largest rate of turn at full speed, rad/s
    c: tuple = _ONES2     # [n_h][n_v] on H_GRID x V_GRID: ice resistance as a multiple of Lindqvist's
    stall: tuple = _ONES  # [n_h]: thrust below which the ship stops, as a multiple of Lindqvist's resistance at rest
    track: tuple = TRACK  # [n_h]: resistance in another ship's channel, as a share of unbroken ice


def from_power(L, B, T, power_kw, speed_kn, stem_deg, entrance_deg, area, eta=0.65):
    uow = speed_kn * KN
    return ShipClass(L, B, T, eta * power_kw * 1e3 / (0.7 * uow), uow, math.radians(stem_deg), math.radians(entrance_deg), area, uow / (2.5 * L))


# The 76 m icebreaker of icebreaker_jax.py: particulars from its Config (stem 28 degrees from the horizontal).
# ICEBREAKER_FORMULA is the same ship on Lindqvist's formula alone; ICEBREAKER carries the hull-scale curves.
# The other classes are filled in by routes.py from named ships; see the notes there on what is sourced.
ICEBREAKER_FORMULA = ShipClass(_HULL.L, _HULL.B, _HULL.T, _HULL.TB, _HULL.uow, math.pi / 2 - _HULL.phi_stem, math.atan(_HULL.B / (_HULL.bow_frac * _HULL.L)),
                               350.0, _HULL.uow / (2.5 * _HULL.L), track=(0.4,) * len(H_GRID))
ICEBREAKER = ICEBREAKER_FORMULA._replace(c=tuple(tuple(r) for r in _T["c"]), stall=tuple(_T["stall"]), track=TRACK) if _T else ICEBREAKER_FORMULA


def stack(classes):
    """A ShipClass whose fields are arrays [n_classes] (curves: [n_classes, n_h], c: [n_classes, n_h, n_v]),
    to be indexed by class id."""
    return ShipClass(*[jnp.asarray([getattr(c, k) for c in classes], jnp.float32) for k in ShipClass._fields])


def _cell(grid, x):
    """Index of the grid interval that holds x and the weight of its upper end (clamped outside the grid)."""
    grid = jnp.asarray(grid, jnp.float32)
    i = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)
    return i, jnp.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)


def _lead(shape):
    """Index arrays for the leading axes of a table, so that table[(*_lead, i, j)] picks, for each element
    of the index arrays i and j, the entry of the table that broadcasting would pair it with."""
    r = len(shape)
    return tuple(jnp.zeros((), jnp.int32) if n == 1 else jnp.arange(n).reshape((n,) + (1,) * (r - 1 - k)) for k, n in enumerate(shape))


def at_h(curve, h):
    """A curve over H_GRID (shape [..., n_h]) at thickness h, linear between the measured thicknesses and
    held constant outside them. The leading shape of `curve` broadcasts with h."""
    curve, h = jnp.asarray(curve, jnp.float32), jnp.asarray(h, jnp.float32)
    i, w = _cell(H_GRID, h)
    lead = _lead(curve.shape[:-1])
    return curve[(*lead, i)] * (1.0 - w) + curve[(*lead, i + 1)] * w


def at_hv(table, h, v, slope=False):
    """A table over H_GRID x V_GRID (shape [..., n_h, n_v]) at thickness h and speed v, bilinear and held
    constant outside the grids. The leading shape of `table` broadcasts with h and v. With `slope`, also
    its derivative with respect to v."""
    table, h, v = jnp.asarray(table, jnp.float32), jnp.asarray(h, jnp.float32), jnp.asarray(v, jnp.float32)
    (i, wh), (j, wv) = _cell(H_GRID, h), _cell(V_GRID, v)
    lead = _lead(table.shape[:-2])
    at = lambda di, dj: table[(*lead, i + di, j + dj)]
    c0, c1 = at(0, 0) * (1 - wh) + at(1, 0) * wh, at(0, 1) * (1 - wh) + at(1, 1) * wh      # at the two speeds of the cell
    c = c0 * (1 - wv) + c1 * wv
    if not slope:
        return c
    grid = jnp.asarray(V_GRID, jnp.float32)
    return c, jnp.where((v >= grid[0]) & (v <= grid[-1]), (c1 - c0) / (grid[j + 1] - grid[j]), 0.0)


def lindqvist_terms(s: ShipClass, h, sigF):
    """Level-ice resistance as a + b v (N and N s/m), after Lindqvist (1989)."""
    h = jnp.maximum(h, 1e-3)
    phi, alpha = s.stem, s.entrance
    psi = jnp.arctan(jnp.tan(phi) / jnp.sin(alpha))
    rc = 0.5 * sigF * h ** 2 * (jnp.tan(phi) + MU * jnp.cos(phi) / jnp.cos(psi)) / (1 - MU * jnp.sin(phi) / jnp.cos(psi))
    rb = (27 / 64 * sigF * s.B * h ** 1.5 / math.sqrt(E_ICE / (12 * (1 - NU ** 2) * G * RHO_W))
          * (jnp.tan(psi) + MU * jnp.cos(phi) / (jnp.cos(psi) * jnp.sin(alpha))) * (1 + 1 / jnp.cos(psi)))
    k = MU * (0.7 * s.L - s.T / jnp.tan(phi) - s.B / (4 * jnp.tan(alpha))
              + s.T * jnp.cos(phi) * jnp.cos(psi) * jnp.sqrt(1 / jnp.sin(phi) ** 2 + 1 / jnp.tan(alpha) ** 2))
    rs = (RHO_W - RHO_I) * G * h * s.B * (s.T * (s.B + s.T) / (s.B + 2 * s.T) + k)
    a = rc + rb + rs
    b = (rc + rb) * 1.4 / jnp.sqrt(G * h) + rs * 9.4 / jnp.sqrt(G * s.L)
    return a, b


def ice_scale(s: ShipClass, h, conc, in_track=0.0, conc_exp=2.0):
    """Share of level-ice resistance met in ice of concentration conc, in or out of another ship's channel."""
    track = 1.0 - jnp.asarray(in_track, jnp.float32) * (1.0 - at_h(s.track, h))
    return jnp.where((conc > 0.01) & (h > 0.01), jnp.clip(conc, 0.0, 1.0) ** conc_exp * track, 0.0)


def ice_resistance(s: ShipClass, h, v, conc=1.0, sigF=0.55e6, in_track=0.0, conc_exp=2.0):
    """Ice resistance (N) of a ship moving at v through the water."""
    a, b = lindqvist_terms(s, h, sigF)
    return ice_scale(s, h, conc, in_track, conc_exp) * at_hv(s.c, h, v) * (a + b * v)


def attainable_speed(s: ShipClass, thr, h, conc, sigF=0.55e6, f_ext=0.0, in_track=0.0, conc_exp=2.0, iters=8):
    """Steady speed through the water (m/s) at throttle thr in [0, 1]; 0 if the ship is stopped by the ice.

    h: mean thickness of the ice present, m. conc: concentration in [0, 1]. f_ext: any other resisting
    force, N (wind). in_track: 1 (or True) in the channel left by another ship. All arguments broadcast.

    The balance is solved by Newton's method from the speed the ship would have without ice, which is
    above the root; the net force falls with speed, so the iterates come down onto the root.
    """
    a, b = lindqvist_terms(s, h, sigF)
    scale = ice_scale(s, h, conc, in_track, conc_exp)
    kx, lin = 0.7 * s.TB / s.uow ** 2, (0.3 * thr + 0.01) * s.TB / s.uow
    push = thr * s.TB - f_ext
    top = (-lin + jnp.sqrt(lin * lin + 4 * kx * jnp.maximum(push, 0.0))) / (2 * kx)       # speed without ice
    v = top
    for _ in range(iters):
        c, dc = at_hv(s.c, h, v, slope=True)
        f = push - lin * v - kx * v * v - scale * c * (a + b * v)
        df = -lin - 2 * kx * v - scale * (dc * (a + b * v) + c * b)
        v = jnp.clip(v - f / jnp.minimum(df, -1.0), 0.0, top)
    moving = (push > a * at_h(s.stall, h) * scale) & (push > a * at_hv(s.c, h, 0.0) * scale)
    return jnp.where(moving, v, 0.0)


def wind_force(s: ShipClass, v, heading, uw, vw, cd=0.8):
    """Resisting force (N) from the wind component along the ship's heading; negative with a following wind.
    heading in radians from east, counterclockwise. v is the ship's speed, m/s."""
    w = v - (uw * jnp.cos(heading) + vw * jnp.sin(heading))      # air speed met head-on
    return 0.5 * RHO_A * cd * s.area * w * jnp.abs(w)
