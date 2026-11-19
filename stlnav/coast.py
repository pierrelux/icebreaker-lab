"""Write stl-coast.json, what the web page draws of the land: the coastline of GSHHG cut to the map, with lakes,
rivers and the names of towns, so that places can be found.

    uv run --no-project --python 3.12 --with numpy python -I stlnav/coast.py data/gshhg data/osm/places.json stl-data.json stl-coast.json

GSHHG (Wessel and Smith 1996, version 2.3.7) is a world coastline with points 50 to 500 m apart in this region,
five to ten times finer than the ice chart whose land mask the page drew before. The file read here is the full
resolution of the binary release, gshhs_f.b, from gshhg-bin-2.3.7.zip (119 MB):
https://www.soest.hawaii.edu/pwessel/gshhg/ . It is distributed under the GNU Lesser General Public License.

The land against the sea (level 1) is cut to the model's grid, then thinned twice with the Douglas-Peucker rule:
`fine` for close views, `coarse` for the whole Gulf. Islands smaller than a threshold are left out of each.
Lakes (level 2 of the same file) and rivers (wdb_rivers_f.b of the same release, the larger classes) are only
there to be recognised: they are not water of the model.

Towns come from OpenStreetMap (place = city, town or village), fetched from the Overpass API with

    [out:json][timeout:120];(node["place"~"^(city|town|village)$"](45.4,-71.6,52.1,-55.9););out body;

(c) OpenStreetMap contributors, Open Database License. Cities and towns are all kept; villages only within a few
kilometres of the sea, where they help to place a ship.

Points are whole numbers of `q` degrees east of `lon0` and south of `lat0`. A ring is a flat list
[x0, y0, dx1, dy1, dx2, dy2, ...]: the first point, then steps from one point to the next.

The script reads nothing but its two input files and uses numpy only, so that it can be run with `python -I`.
It also reports whether the routes and voyages of stl-data.json touch this land, because the model's own land
mask has cells of 2 km and the two do not agree everywhere.
"""
import json
import struct
import sys

import numpy as np

Q = 2e-5                     # degrees per unit: 2.2 m north-south
FINE = dict(tol=0.035, min_area=0.004)      # km, km^2
COARSE = dict(tol=0.45, min_area=1.5)
LAKE = dict(tol=0.12, min_area=4.0)
RIVER = dict(tol=0.15, min_km=12.0, levels=(0, 1, 2, 3, 4))
VILLAGE_KM = 5.0             # villages are kept this close to the sea
KY = 111.195


def read(path, levels, west, east, south, north, closed=True):
    """Polygons or lines of the given levels in a GSHHG binary file whose box meets the given one: list of (level, [n, 2] array of lon, lat)."""
    buf = open(path, "rb").read()
    out, pos = [], 0
    while pos < len(buf):
        _, n, flag, w, e, s, nn, _, _, _, _ = struct.unpack(">11i", buf[pos:pos + 44])
        pos += 44
        if (flag & 255) in levels:
            w, e = w * 1e-6, e * 1e-6
            if w > 180:
                w -= 360
            if e > 180:
                e -= 360
            hit = s * 1e-6 <= north and nn * 1e-6 >= south and (w <= east and e >= west if w <= e else True)
            if hit:
                p = np.frombuffer(buf, ">i4", 2 * n, pos).reshape(n, 2).astype(np.float64) * 1e-6
                p[:, 0] = np.where(p[:, 0] > 180, p[:, 0] - 360, p[:, 0])
                if closed and len(p) > 1 and np.all(p[0] == p[-1]):
                    p = p[:-1]
                out.append((flag & 255, p))
        pos += 8 * n
    return out


def clip(p, axis, bound, keep_greater):
    """Sutherland-Hodgman: the part of a closed ring on one side of a line parallel to an axis."""
    v = p[:, axis]
    inside = v >= bound if keep_greater else v <= bound
    if inside.all():
        return p
    if not inside.any():
        return p[:0]
    q, inq = np.roll(p, -1, axis=0), np.roll(inside, -1)
    cross = inside != inq
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, (bound - v) / (q[:, axis] - v), 0.0)
    x = p + (q - p) * t[:, None]
    x[:, axis] = bound
    out, mask = np.empty((2 * len(p), 2)), np.zeros(2 * len(p), bool)
    out[0::2], mask[0::2], out[1::2], mask[1::2] = p, inside, x, cross
    return out[mask]


def thin(p, tol):
    """Douglas-Peucker on an open line [n, 2]: the points to keep."""
    keep = np.zeros(len(p), bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d, seg = p[b] - p[a], p[a + 1:b] - p[a]
        n = np.hypot(*d)
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n if n > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(dist.argmax())
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return p[keep]


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def inside(rings, pts):
    """Even-odd test of points [m, 2] against rings (each [n, 2]); a bool per point."""
    odd = np.zeros(len(pts), bool)
    for r in rings:
        sel = np.flatnonzero((pts[:, 0] >= r[:, 0].min()) & (pts[:, 0] <= r[:, 0].max()) & (pts[:, 1] >= r[:, 1].min()) & (pts[:, 1] <= r[:, 1].max()))
        a, b = r, np.roll(r, -1, axis=0)
        for k0 in range(0, len(sel), 256):
            k = sel[k0:k0 + 256]
            x, y = pts[k, 0][:, None], pts[k, 1][:, None]
            c = (a[None, :, 1] > y) != (b[None, :, 1] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = a[None, :, 0] + (y - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (b[None, :, 1] - a[None, :, 1])
            odd[k] ^= (np.count_nonzero(c & (xi > x), axis=1) & 1).astype(bool)
    return odd


def main():
    gdir, places, data, out = sys.argv[1:5]
    d = json.load(open(data))
    g = d["grid"]
    west, south = g["lon0"] - 0.5 * g["dlon"], g["lat0"] - 0.5 * g["dlat"]
    east, north = west + g["nx"] * g["dlon"], south + g["ny"] * g["dlat"]
    kx = KY * np.cos(np.deg2rad(0.5 * (south + north)))
    km = lambda p: np.column_stack([(p[:, 0] - west) * kx, (north - p[:, 1]) * KY])          # kilometres east and south of the north-west corner
    units = lambda x: np.column_stack([np.round(x[:, 0] / kx / Q), np.round(x[:, 1] / KY / Q)]).astype(np.int64)
    box = (west, east, south, north)

    def cut(polys):
        res = []
        for _, p in polys:
            for axis, bound, greater in ((0, west, True), (0, east, False), (1, south, True), (1, north, False)):
                p = clip(p, axis, bound, greater)
                if len(p) < 3:
                    break
            if len(p) >= 3:
                res.append(p)
        return res

    def rings_of(polys, tol, min_area):
        rings = []
        for p in polys:
            x = km(p)
            if area(x) < min_area:
                continue
            if len(x) > 8:                      # thin the two halves of the ring between its first point and the one farthest from it
                k = int(np.hypot(*(x - x[0]).T).argmax())
                x = np.concatenate([thin(x[:k + 1], tol)[:-1], thin(np.concatenate([x[k:], x[:1]]), tol)[:-1]])
            u = units(x)
            u = u[np.any(u != np.roll(u, 1, axis=0), axis=1)]
            if len(u) >= 3:
                rings.append(u)
        return rings

    land = cut(read(f"{gdir}/gshhs_f.b", (1,), *box))
    print(f"land polygons that meet the map: {len(land)}, {sum(len(p) for p in land)} points at full resolution")
    fine, coarse = rings_of(land, **FINE), rings_of(land, **COARSE)
    lakes = rings_of(cut(read(f"{gdir}/gshhs_f.b", (2,), *box)), **LAKE)

    rivers, by_level = [], {}
    for lev, p in read(f"{gdir}/wdb_rivers_f.b", RIVER["levels"], *box, closed=False):
        ok = (p[:, 0] >= west) & (p[:, 0] <= east) & (p[:, 1] >= south) & (p[:, 1] <= north)
        for run in np.split(np.arange(len(p)), np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1):          # the runs of points inside the map
            if not ok[run[0]] or len(run) < 2:
                continue
            x = km(p[run])
            if np.hypot(*np.diff(x, axis=0).T).sum() < RIVER["min_km"]:
                continue
            u = units(thin(x, RIVER["tol"]))
            u = u[np.r_[True, np.any(np.diff(u, axis=0) != 0, axis=1)]]
            if len(u) >= 2:
                rivers.append(u)
                by_level[lev] = by_level.get(lev, 0) + 1

    # towns: cities and towns, and the villages near the sea
    coast_km = np.concatenate([r * Q * np.array([kx, KY]) for r in fine])
    towns = []
    for e in json.load(open(places))["elements"]:
        t = e.get("tags", {})
        name, kind = str(t.get("name", "")).strip()[:40], t.get("place")
        if not name or kind not in ("city", "town", "village") or not (west < e["lon"] < east and south < e["lat"] < north):
            continue
        if kind == "village":
            x = np.array([(e["lon"] - west) * kx, (north - e["lat"]) * KY])
            near = coast_km[(np.abs(coast_km[:, 0] - x[0]) < VILLAGE_KM) & (np.abs(coast_km[:, 1] - x[1]) < VILLAGE_KM)]
            if not len(near) or np.hypot(*(near - x).T).min() > VILLAGE_KM:
                continue
        try:
            pop = int(str(t.get("population", "0")).replace(" ", "").replace(",", ""))
        except ValueError:
            pop = 0
        towns.append([name, round(e["lon"], 4), round(e["lat"], 4), {"city": 0, "town": 1, "village": 2}[kind], pop])
    towns.sort(key=lambda r: (r[3], -r[4], r[0]))

    enc = lambda rings: [np.concatenate([r[:1], np.diff(r, axis=0)]).ravel().tolist() for r in rings]
    doc = dict(
        about="Coastline, lakes and rivers: GSHHG 2.3.7, full resolution (Wessel and Smith 1996), https://www.soest.hawaii.edu/pwessel/gshhg/ , GNU Lesser General Public License. "
              "Names of towns: (c) OpenStreetMap contributors, Open Database License. Cut to the map and thinned by stlnav/coast.py.",
        q=Q, lon0=west, lat0=north,
        fine=dict(tol_km=FINE["tol"], rings=enc(fine)), coarse=dict(tol_km=COARSE["tol"], rings=enc(coarse)),
        lakes=enc(lakes), rivers=enc(rivers), towns=towns)
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"), ensure_ascii=False)
    import gzip
    import os
    for name, rings in (("fine", fine), ("coarse", coarse)):
        n = sum(len(r) for r in rings)
        seg = np.concatenate([np.hypot(*(np.diff(np.vstack([r, r[:1]]), axis=0) * Q * np.array([kx, KY])).T) for r in rings])
        print(f"{name}: {len(rings)} rings, {n} points; segments: median {1e3 * np.median(seg):.0f} m, 90% under {1e3 * np.quantile(seg, 0.9):.0f} m; largest ring {max(len(r) for r in rings)} points")
    print(f"lakes: {len(lakes)} rings, {sum(len(r) for r in lakes)} points; rivers: {len(rivers)} lines, {sum(len(r) for r in rivers)} points, by class {dict(sorted(by_level.items()))}")
    print(f"towns: {len(towns)} ({sum(r[3] == 0 for r in towns)} cities, {sum(r[3] == 1 for r in towns)} towns, {sum(r[3] == 2 for r in towns)} villages within {VILLAGE_KM:.0f} km of the sea)")
    print(f"wrote {out}: {os.path.getsize(out) / 1e3:.0f} kB, {len(gzip.compress(open(out, 'rb').read())) / 1e3:.0f} kB gzipped")

    # do the routes and voyages of the page keep off this land?
    rings_ll = [np.column_stack([west + r[:, 0] * Q, north - r[:, 1] * Q]) for r in fine]
    rt = d["routes"]
    for r, name in enumerate(rt["name"]):
        lon, lat, cum = np.array(rt["lon"][r]), np.array(rt["lat"][r]), np.array(rt["cum"][r])
        n = int(np.sum(np.diff(cum) < 999.0)) + 1                     # waypoints before the padding
        s = np.arange(0.0, cum[n - 1], 0.25)
        hit = inside(rings_ll, np.column_stack([np.interp(s, cum[:n], lon[:n]), np.interp(s, cum[:n], lat[:n])]))
        print(f"route {name}: " + ("clear of land" if not hit.any() else f"{hit.sum() * 0.25:.1f} km of {cum[n - 1]:.0f} km on land, between km {s[hit].min():.1f} and {s[hit].max():.1f}"))
    for v in d["voyages"]:
        hit = inside(rings_ll, np.array(v["path"])[:, ::-1])
        print(f"voyage {v['name']}: " + ("clear of land" if not hit.any() else f"points on land: {np.flatnonzero(hit).tolist()} of {len(hit)}"))


if __name__ == "__main__":
    main()
