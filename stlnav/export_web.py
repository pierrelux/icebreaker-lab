"""Write stl-data.json, what the web page needs to run the route-scale model in the browser.

    uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy \
        python -m stlnav.export_web --cache data/fields_20261117.npz --ice data/ice_hist/sic_2026-02-15.nc

The file holds one day of fields at reduced resolution, the routes, the ship classes with the hull-scale
table, and the settings of traffic.py and env.py, so that stlnav-core.js copies no number by hand.

  ice      concentration of the chart day on the 2 km grid, one byte per cell (255 = land, else c * 250);
  current  east and north components every third cell (6 km), 9 frames 3 h apart, one byte each (0.02 m/s);
  wind     every eighth cell (16 km), same frames, one byte each (0.25 m/s).

`fields_from_web` rebuilds a Fields from the file, so that Python can be run on exactly the data the
page uses (checks/stlnav_js.py does that).
"""
from __future__ import annotations

import argparse
import base64
import json

import numpy as np

from . import demo, fields as F, routes as RT, ships, traffic
from . import env as E

CONC_SCALE, CUR_STEP, CUR_SCALE, WIND_STEP, WIND_SCALE = 250, 3, 0.02, 8, 0.25
VOYAGES = [("Les Escoumins to Sept-Îles", "demo"), ("Les Escoumins to Cabot Strait", 0), ("Cabot Strait to Les Escoumins", 1), ("Sept-Îles to Cabot Strait", 2)]


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export(f: F.Fields, chart_date: str) -> dict:
    """f: fields with the chart's concentration laid over them (any thickness; the page sets its own)."""
    ny, nx = f.land.shape
    ice = np.where(f.land > 0.5, 255, np.round(f.data[0, F.CONC] * CONC_SCALE)).astype(np.uint8)

    def pack(ch, step, scale):
        a = f.data[:, ch, ::step, ::step]
        return np.clip(np.round(a / scale), -127, 127).astype(np.int8)

    cur = [pack(c, CUR_STEP, CUR_SCALE) for c in (F.UC, F.VC)]
    wind = [pack(c, WIND_STEP, WIND_SCALE) for c in (F.UW, F.VW)]
    specs = RT.snapped(f, RT.specs(True))
    rt = traffic.build_routes(specs, RT.CLASS_NAMES)
    names = [RT.route_name(r, specs) for r in range(rt.length.shape[0])]
    lst = lambda a: np.asarray(a, np.float64).round(6).tolist()
    voyages = []
    for name, src in VOYAGES:
        if src == "demo":
            path = [RT.nearest_water(f, *demo.PATH[0])] + list(demo.PATH[1:-1]) + [RT.nearest_water(f, *demo.PATH[-1])]
        else:
            n = int(np.sum(np.diff(np.asarray(rt.cum[src])) < 999.0)) + 1          # waypoints before the padding
            path = [(float(rt.lat[src, k]), float(rt.lon[src, k])) for k in range(n)]
        voyages.append(dict(name=name, path=[[round(float(a), 5), round(float(b), 5)] for a, b in path]))
    cfg, tc = E.Config(), traffic.TrafficCfg()
    return dict(
        about=dict(
            ice=f"Sea ice concentration of {chart_date}: Fisheries and Oceans Canada, Daily Ocean Variables from Satellite, Estuary and Gulf of St. Lawrence "
                "(St. Lawrence Global Observatory, catalogue.ogsl.ca). Open Government Licence - Canada.",
            fields=f"Currents (CIOPS-East) and wind (HRDPS) of {f.t0[:10]}: Environment and Climate Change Canada, MSC Datamart.",
            note="Reduced in resolution and rounded for the web page by stlnav/export_web.py. The ice thickness is not in the data: the page assumes one."),
        chart_date=chart_date, fields_date=f.t0[:10],
        grid=dict(lon0=f.lon0, lat0=f.lat0, dlon=f.dlon, dlat=f.dlat, nx=nx, ny=ny, dt=f.dt, nt=int(f.data.shape[0])),
        ice=dict(scale=CONC_SCALE, data=_b64(ice)),
        cur=dict(step=CUR_STEP, scale=CUR_SCALE, ny=int(cur[0].shape[1]), nx=int(cur[0].shape[2]), u=_b64(cur[0]), v=_b64(cur[1])),
        wind=dict(step=WIND_STEP, scale=WIND_SCALE, ny=int(wind[0].shape[1]), nx=int(wind[0].shape[2]), u=_b64(wind[0]), v=_b64(wind[1])),
        consts=dict(G=ships.G, RHO_W=ships.RHO_W, RHO_I=ships.RHO_I, RHO_A=ships.RHO_A, E=ships.E_ICE, NU=ships.NU, MU=ships.MU),
        h_grid=list(ships.H_GRID), v_grid=list(ships.V_GRID),
        classes=[dict(name=n, **{k: (getattr(c, k) if not isinstance(getattr(c, k), tuple) else [list(r) if isinstance(r, tuple) else r for r in getattr(c, k)])
                                for k in ships.ShipClass._fields}) for n, c in zip(RT.CLASS_NAMES, RT.CLASSES)],
        routes=dict(name=names, lon=lst(rt.lon), lat=lst(rt.lat), cum=lst(rt.cum), length=lst(rt.length), rate=np.asarray(rt.rate, np.float64).tolist(),
                    period=lst(rt.period), phase=lst(rt.phase), mix=lst(rt.mix), off=lst(rt.off)),
        traffic=tc._asdict(),
        env={k: getattr(cfg, k) for k in ("dt", "t_max", "n_traffic", "tau_v", "goal_km", "near_km", "hit_margin")},
        places={k: [v[0], v[1]] for k, v in RT.PLACES.items()},
        voyages=voyages,
    )


def fields_from_web(d: dict, thick: float) -> F.Fields:
    """The fields exactly as the page sees them, on the full grid (the coarse ones are spread bilinearly)."""
    g = d["grid"]
    ny, nx, nt = g["ny"], g["nx"], g["nt"]
    raw = np.frombuffer(base64.b64decode(d["ice"]["data"]), np.uint8).reshape(ny, nx)
    land = (raw == 255).astype(np.float32)
    conc = np.where(raw == 255, 0.0, raw / d["ice"]["scale"]).astype(np.float32)

    def spread(spec, key):
        a = np.frombuffer(base64.b64decode(spec[key]), np.int8).reshape(nt, spec["ny"], spec["nx"]).astype(np.float64) * spec["scale"]
        y = np.clip(np.arange(ny) / spec["step"], 0, spec["ny"] - 1)
        x = np.clip(np.arange(nx) / spec["step"], 0, spec["nx"] - 1)
        j, i = np.minimum(y.astype(int), spec["ny"] - 2), np.minimum(x.astype(int), spec["nx"] - 2)
        wy, wx = (y - j)[None, :, None], (x - i)[None, None, :]
        top = a[:, j][:, :, i] * (1 - wx) + a[:, j][:, :, i + 1] * wx
        bot = a[:, j + 1][:, :, i] * (1 - wx) + a[:, j + 1][:, :, i + 1] * wx
        return (top * (1 - wy) + bot * wy).astype(np.float32)

    data = np.zeros((nt, 7, ny, nx), np.float32)
    data[:, F.UC], data[:, F.VC] = spread(d["cur"], "u"), spread(d["cur"], "v")
    data[:, F.UW], data[:, F.VW] = spread(d["wind"], "u"), spread(d["wind"], "v")
    f = F.Fields(data, land, g["lon0"], g["lat0"], g["dlon"], g["dlat"], g["dt"], d["fields_date"])
    return F.with_ice(f, conc, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/fields_20261117.npz")
    ap.add_argument("--ice", default="data/ice_hist/sic_2026-02-15.nc")
    ap.add_argument("--out", default="stl-data.json")
    a = ap.parse_args()
    f = F.load(a.cache)
    f = F.with_ice(f, F.read_chart_concentration(a.ice, f), 0.3)
    d = export(f, a.ice.split("sic_")[-1][:10])
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    import gzip
    import os
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e3:.0f} kB, {len(gzip.compress(open(a.out, 'rb').read())) / 1e3:.0f} kB gzipped; "
          f"{len(d['routes']['name'])} routes, {len(d['classes'])} classes, {len(d['voyages'])} voyages")


if __name__ == "__main__":
    main()
