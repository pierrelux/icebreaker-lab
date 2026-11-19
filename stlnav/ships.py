"""Ship classes and the speed a ship can hold in given ice, at the scale of a route (kilometres, minutes).

The speed is the one at which thrust equals resistance:

    thr * TB * (1 - 0.3 v / uow)  =  kx v^2  +  conc^n * R_lindqvist(h, v) * channel  +  F_wind

  * Thrust and open-water resistance are the ones in icebreaker_jax.py: bollard pull TB falling by 30%
    at the open-water speed uow, and kx = 0.7 TB / uow^2, so that full throttle gives uow in open water.
  * R_lindqvist is Lindqvist's (1989) level-ice resistance, which is linear in v, so the balance is a
    quadratic with a closed-form root. If thrust cannot overcome the ice at zero speed the ship is stopped.
  * conc^n scales level-ice resistance down in pack ice of concentration conc. n = 2 is a PLACEHOLDER,
    not taken from a source. It should be replaced by a fit to the floe simulator or to model tests.
  * channel < 1 lowers ice resistance for a ship following in another ship's track. Also a placeholder.
  * F_wind is the drag of the head wind on the frontal area: 0.5 rho_air Cd A w |w|.

Classes other than the icebreaker have illustrative particulars (see CLASSES). For a ship whose bollard
pull is not known it is derived from installed power P: TB = eta P / (0.7 uow), with eta = 0.65, because
in this thrust model the propulsive force at uow is 0.7 TB.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

G, RHO_W, RHO_I, RHO_A = 9.81, 1025.0, 880.0, 1.25
E_ICE, NU, MU = 5.4e9, 0.33, 0.15
KN = 0.5144


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


def from_power(L, B, T, power_kw, speed_kn, stem_deg, entrance_deg, area, eta=0.65):
    uow = speed_kn * KN
    return ShipClass(L, B, T, eta * power_kw * 1e3 / (0.7 * uow), uow, math.radians(stem_deg), math.radians(entrance_deg), area, uow / (2.5 * L))


# Index 0 is the 76 m icebreaker of icebreaker_jax.py (stem 28 degrees, entrance atan(B / (0.36 L))).
# The others are filled in by routes.py from named ships; see the notes there on what is sourced.
ICEBREAKER = ShipClass(76.0, 18.0, 6.5, 2.0e6, 8.4, math.radians(28.0), math.atan(18.0 / (0.36 * 76.0)), 350.0, 8.4 / (2.5 * 76.0))


def stack(classes):
    """A ShipClass whose fields are arrays [n_classes], to be indexed by class id."""
    return ShipClass(*[jnp.asarray([getattr(c, k) for c in classes], jnp.float32) for k in ShipClass._fields])


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


def attainable_speed(s: ShipClass, thr, h, conc, sigF=0.55e6, f_ext=0.0, channel=1.0, conc_exp=2.0):
    """Steady speed through the water (m/s) at throttle thr in [0, 1]; 0 if the ship cannot move.

    h: mean thickness of the ice present, m. conc: concentration in [0, 1]. f_ext: any other resisting
    force, N (wind). All arguments broadcast.
    """
    a, b = lindqvist_terms(s, h, sigF)
    scale = jnp.where((conc > 0.01) & (h > 0.01), jnp.clip(conc, 0.0, 1.0) ** conc_exp * channel, 0.0)
    a, b = a * scale, b * scale
    kx = 0.7 * s.TB / s.uow ** 2
    lin = 0.3 * thr * s.TB / s.uow + b
    net = thr * s.TB - a - f_ext
    v = (-lin + jnp.sqrt(lin * lin + 4 * kx * jnp.maximum(net, 0.0))) / (2 * kx)
    return jnp.where(net > 0, v, 0.0)


def wind_force(s: ShipClass, v, heading, uw, vw, cd=0.8):
    """Resisting force (N) from the wind component along the ship's heading; negative with a following wind.
    heading in radians from east, counterclockwise. v is the ship's speed, m/s."""
    w = v - (uw * jnp.cos(heading) + vw * jnp.sin(heading))      # air speed met head-on
    return 0.5 * RHO_A * cd * s.area * w * jnp.abs(w)
