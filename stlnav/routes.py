"""Routes, traffic volumes, ship classes and test scenarios for the estuary and Gulf of St. Lawrence.

HOW FAR TO TRUST THE NUMBERS. Each is marked:
  [V] read on a primary source on 2026-11-18 by a research subagent (I did not re-read the sources);
  [S] secondary source or search snippet;
  [A] an approximation or an allocation made here.
Sources are listed in docs/st-lawrence-plan.md.

Lines
  * No traffic separation scheme is published as coordinates (they are "shown on the charts"). The
    open-water trunk line follows the centre of Transport Canada's dynamic right-whale zones, which are
    laid on the shipping lanes south of Anticosti [derived from V]. Other waypoints are [A].
  * ICE_ROUTE is a real recommended ice route of the Canadian Coast Guard (12 February 2026, with the
    Cabot Strait leg of 11 March 2026) [V]. The Coast Guard changes these daily; this is one sample.
  * Port positions are [A]; `world()` moves an end point that falls on land on the 2 km grid to the
    nearest water cell.

Volumes (departures per day, each direction)
  * Laurentian Pilotage Authority, 2024: 6,889 voyages between Montréal and Les Escoumins including the
    Saguenay [V], so at most 9.4 a day each way past Les Escoumins. Cabot Strait: about 6,400 commercial
    transits a year, a 2001 figure [V], 8.8 a day each way.
  * Calls per year: Sept-Îles 575 (2025) [S], Port-Cartier 409 (2021) [S], Baie-Comeau 263 (2004) [S].
  * The split of these totals among the routes below is [A]. They are annual means: the first quarter is
    the quietest [V] and no winter figure was found, so winter traffic is overstated by an unknown factor.
  * Type mix of the 2024 pilotage voyages [V]: bulk 40.5%, tanker 25.1%, general cargo 13.0%,
    container 11.1%, passenger 4.7%, ro-ro 2.8%. General cargo is counted with bulk here, ro-ro with
    container; passenger ships are left out.

Ferries, January to March
  * Matane - Baie-Comeau: 5 round trips a week; Matane - Godbout: 3 a week (STQ timetable, 4 January to
    31 March 2027) [V]. North Sydney - Port aux Basques: about 2 departures a day each way [S].
    Souris - Cap-aux-Meules: about 3 a week [S, unconfirmed]. Each is spread evenly here (one departure
    every 168/n hours), not put on its real days and hours.
  * Left out because they are smaller than a grid cell or outside the area: Tadoussac - Baie-Sainte-
    Catherine, L'Isle-aux-Coudres, Québec - Lévis, St. Barbe - Blanc-Sablon. Rivière-du-Loup - Saint-Siméon
    and the Bella Desgagnés do not run in winter [S].

Ship classes: lengths, beams, draughts, powers and speeds marked [V] or [S] are those of the named ship.
Stem angle, entrance angle and windage area are [A] for every class except the icebreaker, and they decide
the ice resistance, so the speed of the merchant classes in ice is indicative only.
"""
from __future__ import annotations

import numpy as np

from . import env, ships
from .fields import Fields

# name -> (lat, lon), all [A] except where noted
PLACES = {
    "escoumins": (48.3467, -69.39),        # point defining the compulsory pilotage line [V]
    "baie_comeau": (49.23, -68.13),
    "sept_iles": (50.167, -66.40),         # pilot embarkation, Chenal du Milieu [V]
    "port_cartier": (50.03, -66.78),
    "matane": (48.84, -67.58),
    "godbout": (49.32, -67.59),
    "cabot_strait": (47.33, -59.85),       # midpoint of the voluntary slow-down zone across the strait [derived from V]
    "souris": (46.35, -62.25),
    "cap_aux_meules": (47.38, -61.86),
    "north_sydney": (46.21, -60.25),
    "port_aux_basques": (47.57, -59.14),
}

CLASS_NAMES = ["icebreaker", "bulk", "tanker", "container", "ferry", "ropax"]
CLASSES = [
    ships.ICEBREAKER,                                                        # the 76 m ship of icebreaker_jax.py
    #                 L      B     T     kW      kn   stem entrance windage
    ships.from_power(199.9, 23.2, 9.0, 8000.0, 14.5, 60.0, 30.0, 600.0),     # Federal Montreal class laker, ice class 1C: L, B [S]; rest [A]
    ships.from_power(135.0, 23.5, 7.9, 5450.0, 13.5, 50.0, 30.0, 450.0),     # Mia Desgagnés, Polar Class 7: L, B, T, kW, speed [V]
    ships.from_power(210.0, 29.6, 8.0, 15000.0, 20.0, 40.0, 25.0, 1100.0),   # Oceanex Connaigra, ICE-1A: L, B, speed [S]; rest [A]
    ships.from_power(133.3, 22.0, 5.5, 14000.0, 19.0, 35.0, 30.0, 900.0),    # F.-A.-Gauthier: L [S]; rest [A]
    ships.from_power(199.5, 26.7, 6.2, 21600.0, 22.0, 40.0, 28.0, 1500.0),   # Blue Puttees, ICE-1A: L, B, T, kW, speed [V]
]

_CARGO = {"bulk": 0.535, "tanker": 0.251, "container": 0.139}
_E, _CS = PLACES["escoumins"], PLACES["cabot_strait"]
# open water: Les Escoumins to Pointe des Monts [A], then the lane south of Anticosti [derived from V]
_UP = [_E, (48.75, -68.60), (49.20, -67.30)]
_LANE = [(49.51, -65.00), (49.275, -64.00), (48.90, -63.00), (48.49, -62.00), (48.01, -61.09), _CS]
# Coast Guard recommended ice route of 12 February 2026, then the Cabot Strait leg of 11 March 2026 [V]
_UP_ICE = [_E, (48.483, -69.000), (48.733, -68.800), (49.250, -67.300)]
_LANE_ICE = [(49.533, -66.550), (49.643, -66.000), (49.640, -65.000), (49.450, -63.883), (48.883, -63.000), (48.300, -61.333), (48.050, -60.283), (47.750, -60.000), _CS]
_J = (49.20, -67.30)


def specs(ice_route: bool = False):
    """Route list. With ice_route the trunk follows the Coast Guard's 12 February 2026 ice route."""
    up, lane = (_UP_ICE, _LANE_ICE) if ice_route else (_UP, _LANE)
    off = 0.3 if ice_route else 1.0                     # in ice both directions use nearly the same track
    j = up[-1]
    return [
        dict(name="Les Escoumins - Cabot Strait", per_day=6.0, mix=_CARGO, off=off, wp=up + lane),
        dict(name="Sept-Iles - Cabot Strait", per_day=1.6, mix={"bulk": 1.0}, off=off, wp=[PLACES["sept_iles"], (49.85, -66.20)] + lane),
        dict(name="Port-Cartier - Cabot Strait", per_day=1.1, mix={"bulk": 1.0}, off=off, wp=[PLACES["port_cartier"], (49.75, -66.55)] + lane),
        dict(name="Baie-Comeau - Cabot Strait", per_day=0.4, mix=_CARGO, off=off, wp=[PLACES["baie_comeau"], (49.12, -67.90), j] + lane),
        dict(name="Baie-Comeau - Les Escoumins", per_day=0.3, mix=_CARGO, off=off, wp=[PLACES["baie_comeau"], (49.10, -68.00), (48.95, -68.25)] + up[-2::-1]),
        dict(name="icebreaker, estuary and central Gulf", per_day=0.5, mix={"icebreaker": 1.0}, off=0.0, wp=up + lane[:5]),   # rate [A]
        dict(name="ferry Matane - Baie-Comeau", period_h=168.0 / 5, phase_h=8.0, mix={"ferry": 1.0}, off=0.3, wp=[PLACES["matane"], (49.05, -67.88), PLACES["baie_comeau"]]),
        dict(name="ferry Matane - Godbout", period_h=168.0 / 3, phase_h=14.0, mix={"ferry": 1.0}, off=0.3, wp=[PLACES["matane"], PLACES["godbout"]]),
        dict(name="ferry Souris - Cap-aux-Meules", period_h=168.0 / 3, phase_h=8.0, mix={"ferry": 1.0}, off=0.3, wp=[PLACES["souris"], (46.33, -62.05), (46.47, -61.82), (47.10, -61.72), PLACES["cap_aux_meules"]]),
        dict(name="ferry North Sydney - Port aux Basques", period_h=12.0, phase_h=11.0, mix={"ropax": 1.0}, off=0.3, wp=[PLACES["north_sydney"], (46.40, -60.10), PLACES["port_aux_basques"]]),
    ]


SPECS = specs(False)


def route_name(r: int, sp=None) -> str:
    """Name of route index r as built by traffic.build_routes (each spec gives an outbound and a return route)."""
    names = []
    for s in sp or SPECS:
        names += [s["name"]] if s.get("one_way") else [s["name"], s["name"] + " (return)"]
    return names[r]


def nearest_water(f: Fields, lat, lon, max_km=15.0):
    """(lat, lon) itself if it is in water on the grid, else the centre of the nearest water cell."""
    LON, LAT = np.meshgrid(f.lon, f.lat)
    d = np.hypot((LON - lon) * 111.195 * np.cos(np.deg2rad(lat)), (LAT - lat) * 111.195)
    if f.land.ravel()[np.argmin(d)] < 0.5 and d.min() < 3.0:
        j, i = np.unravel_index(np.argmin(d), d.shape)
        near = f.land[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]
        if near.max() < 0.5:
            return (lat, lon)
    # require a cell whose neighbours are water too, so that bilinear lookups around it stay below 0.5
    pad = np.pad(f.land, 1, constant_values=1.0)
    open_ = sum(pad[1 + a:pad.shape[0] - 1 + a, 1 + b:pad.shape[1] - 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)) < 0.5
    d = np.where(open_, d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    if d[j, i] > max_km:
        raise ValueError(f"no open water within {max_km} km of {lat}, {lon}")
    return (float(f.lat[j]), float(f.lon[i]))


def snapped(f: Fields, sp):
    """Copy of the route list with every waypoint that is a named place moved to open water on the grid."""
    places = {v: nearest_water(f, *v) for v in PLACES.values()}
    return [dict(s, wp=[places.get(w, w) for w in s["wp"]]) for s in sp]


def world(f: Fields, ice_route: bool = False, cfg: env.Config = env.Config()):
    """(environment, world) for these fields with the routes and classes of this module."""
    grid, w = env.make_world(f, snapped(f, specs(ice_route)), CLASSES, CLASS_NAMES)
    return env.StlNav(grid, cfg), w


SCENARIOS = {
    "escoumins_to_sept_iles": ("escoumins", "sept_iles"),
    "cabot_strait_to_escoumins": ("cabot_strait", "escoumins"),
    "matane_to_baie_comeau": ("matane", "baie_comeau"),
    "sept_iles_to_cabot_strait": ("sept_iles", "cabot_strait"),
}


def scenario(name: str, f: Fields, cls: int = 0, t0: float = 0.0) -> env.EnvParams:
    a, b = (nearest_water(f, *PLACES[k]) for k in SCENARIOS[name])
    return env.params(a[1], a[0], b[1], b[0], cls=cls, t0=t0)
