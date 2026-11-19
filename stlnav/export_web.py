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

It also writes stl-map.png, which the page only draws: the same chart on its own grid of about 1 km, twice
to three times finer than the model's, so that coasts and ice edges are not drawn as 2 km squares. One
byte per cell: 255 = land, else concentration * 250. Land is where the chart has no data, which follows
the coast at 1 km. The exception is an inlet that the chart leaves out and the model has as water (at
least 3 x 3 model cells): it is drawn as open water, which is what the model takes it for.

The edge of that land also goes into stl-data.json as closed lines (`picture.coast`), which the page fills
as a shape: a picture of 1 km cells shows as steps on a screen, a line does not.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

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


def chart_picture(path, f: F.Fields):
    """The chart on its own grid, cut to the model's area: (bytes [ny, nx] with north at the top, description)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        a = np.squeeze(ds["sic"].values).astype(np.float32) / 100.0
        lat, lon = ds["latitude"].values, ds["longitude"].values
    if lat[0] < lat[-1]:
        a, lat = a[::-1], lat[::-1]
    kj = (lat >= f.lat[0] - 0.5 * f.dlat) & (lat <= f.lat[-1] + 0.5 * f.dlat)
    ki = (lon >= f.lon[0] - 0.5 * f.dlon) & (lon <= f.lon[-1] + 0.5 * f.dlon)
    a, lat, lon = a[kj][:, ki], lat[kj], lon[ki]
    j = np.clip(np.round((lat - f.lat0) / f.dlat).astype(int), 0, f.land.shape[0] - 1)
    i = np.clip(np.round((lon - f.lon0) / f.dlon).astype(int), 0, f.land.shape[1] - 1)
    from scipy import ndimage

    # water of the model that the chart has no data for: a fringe one cell wide along the coasts, where the chart's coast is
    # the better one, and a few inlets, which stay water. An opening by 3 x 3 cells keeps the inlets only.
    gap = np.isnan(a) & ~(f.land[np.ix_(j, i)] > 0.5)
    lat_m, lon_m = np.abs(lat[None, :] - f.lat[:, None]).argmin(1), np.abs(lon[None, :] - f.lon[:, None]).argmin(1)
    gap_m = np.isnan(a[np.ix_(lat_m, lon_m)]) & ~(f.land > 0.5)
    inlet = ndimage.binary_opening(gap_m, structure=np.ones((3, 3), bool))[np.ix_(j, i)]
    land = np.isnan(a) & ~(gap & inlet)
    img = np.where(land, 255, np.round(np.clip(np.nan_to_num(a), 0, 1) * CONC_SCALE)).astype(np.uint8)
    return img, dict(file="stl-map.png", scale=CONC_SCALE, nx=int(img.shape[1]), ny=int(img.shape[0]), lon0=float(lon[0]), lat0=float(lat[0]),
                     dlon=float(lon[1] - lon[0]), dlat=float(lat[0] - lat[1]))


def coast_rings(land, q=8):
    """The coast of a land mask as closed lines, so that the page can draw land as a shape with a sharp edge at any zoom.

    land: [ny, nx] bool, north at the top. Returns a list of rings, each a flat list [x0, y0, x1, y1, ...] in cells of the mask
    times q (integers; x along columns, y along rows from the top). Filling all of them by the even-odd rule gives the land.
    The rings follow the cells' edges at half-way points, are then rounded by a little averaging along the line (small islands
    are left as they are, or they would shrink away), and are thinned to within an eighth of a cell.
    """
    from contourpy import FillType, contour_generator

    pad = 2
    z = np.pad(land.astype(np.float64), pad, mode="edge")
    polys, codes = contour_generator(z=z, fill_type=FillType.OuterCode).filled(0.5, 2.0)

    def thin(p, tol):                           # Douglas-Peucker on an open line
        keep = np.zeros(len(p), bool); keep[[0, -1]] = True; stack = [(0, len(p) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            d, seg = p[b] - p[a], p[a + 1:b] - p[a]
            n = np.hypot(*d)
            dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n if n > 0 else np.hypot(seg[:, 0], seg[:, 1])
            k = int(dist.argmax())
            if dist[k] > tol:
                keep[a + 1 + k] = True; stack += [(a, a + 1 + k), (a + 1 + k, b)]
        return p[keep]

    rings = []
    for pts, cds in zip(polys, codes):
        starts = list(np.flatnonzero(cds == 1)) + [len(pts)]
        for a, b in zip(starts[:-1], starts[1:]):
            r = pts[a:b] - pad
            if len(r) > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            if len(r) < 3:
                continue
            for _ in range(min(3, len(r) // 8)):
                r = 0.25 * np.roll(r, 1, axis=0) + 0.5 * r + 0.25 * np.roll(r, -1, axis=0)
            if len(r) > 8:                      # thin the two halves of the ring between its first point and the one farthest from it
                k = int(np.hypot(*(r - r[0]).T).argmax())
                r = np.concatenate([thin(r[:k + 1], 0.125)[:-1], thin(np.concatenate([r[k:], r[:1]]), 0.125)[:-1]])
            r = np.round(r * q).astype(int)
            r = r[np.any(r != np.roll(r, 1, axis=0), axis=1)]
            if len(r) >= 3:
                rings.append(r.ravel().tolist())
    return rings


def write_png(path, img):
    """An 8-bit greyscale PNG, written with the standard library."""
    import struct
    import zlib

    h, w = img.shape
    chunk = lambda kind, body: struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + np.ascontiguousarray(img[r]).tobytes() for r in range(h))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


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
    img, d["picture"] = chart_picture(a.ice, f)
    d["picture"]["coast"] = dict(q=8, rings=coast_rings(img == 255, 8))
    png = os.path.join(os.path.dirname(a.out), d["picture"]["file"])
    write_png(png, img)
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    import gzip
    print(f"wrote {png}: {img.shape[1]} x {img.shape[0]} cells of {d['picture']['dlon'] * 111.195 * np.cos(np.deg2rad(48.75)):.2f} x {d['picture']['dlat'] * 111.195:.2f} km, "
          f"{os.path.getsize(png) / 1e3:.0f} kB; land share {np.mean(img == 255):.2f}; coast: {len(d['picture']['coast']['rings'])} closed lines, "
          f"{sum(len(r) for r in d['picture']['coast']['rings']) // 2} points")
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e3:.0f} kB, {len(gzip.compress(open(a.out, 'rb').read())) / 1e3:.0f} kB gzipped; "
          f"{len(d['routes']['name'])} routes, {len(d['classes'])} classes, {len(d['voyages'])} voyages")


if __name__ == "__main__":
    main()
