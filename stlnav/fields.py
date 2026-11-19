"""Environment fields for the St. Lawrence estuary and Gulf, on one regular longitude/latitude grid.

`load_window` reads the archive written by archiver/stl_archive.py and returns a `Fields` tuple of
fixed-shape arrays for a time window:

    data[t, c, j, i]   c = CONC, THICK, UC, VC, UW, VW, HS
    land[j, i]         1 where the ocean model has no water

Sources and what is done to them
  * CIOPS-East (already on a 0.03 x 0.02 degree grid, about 2.2 km): ice concentration, ice volume per
    unit area and the 0.5 m current. Mean thickness of the ice present = volume / concentration. Its
    grid is the common grid, and its missing values define the land mask.
  * HRDPS 10 m wind (rotated grid, 2.5 km): the components are relative to the rotated grid
    (GRIB uvRelativeToGrid = 1), so they are turned to east/north with the local grid direction, then
    copied to the common grid by nearest neighbour.
  * RDWPS significant wave height (same rotated grid): nearest neighbour; 0 where the model has none.

For each valid time the file with the shortest lead time is used, so a window is made of the first
hours of successive runs: close to an analysis, not what a ship would have had as a forecast.

Assumption not yet checked: the CIOPS "x" and "y" velocities on the lat/lon product are east and north.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

CONC, THICK, UC, VC, UW, VW, HS = range(7)
CHANNELS = ("conc", "thick", "uc", "vc", "uw", "vw", "hs")
R_EARTH = 6.371e6
_NAME = re.compile(r"^(\d{8}T\d{2})Z_MSC_[^_]+_(.+)_R?LatLon[^_]*_PT(\d+)H\.nc$")
_VARS = {
    "ciops": {"conc": "SeaIceAreaFraction_Sfc", "vol": "SeaIceVol_Sfc", "uc": "SeaWaterVelocityX_DBS-0.5m", "vc": "SeaWaterVelocityY_DBS-0.5m"},
    "hrdps": {"uw": "UGRD_AGL-10m", "vw": "VGRD_AGL-10m"},
    "rdwps": {"hs": "HTSGW_Sfc"},
}


class Fields(NamedTuple):
    data: np.ndarray     # [T, 7, ny, nx] float32, no NaN
    land: np.ndarray     # [ny, nx] float32, 1 = land or outside the ocean model
    lon0: float          # longitude of column 0, degrees east (negative west)
    lat0: float          # latitude of row 0
    dlon: float
    dlat: float
    dt: float            # seconds between frames
    t0: str              # ISO time of frame 0 (UTC); a label, not used in the computation

    @property
    def lon(self):
        return self.lon0 + self.dlon * np.arange(self.data.shape[-1])

    @property
    def lat(self):
        return self.lat0 + self.dlat * np.arange(self.data.shape[-2])


def _index(root: Path, model: str):
    """{variable: {valid time: (lead hours, path)}} keeping the shortest lead for each valid time."""
    out: dict = {}
    for p in (root / model).glob("*/*/*/*.nc"):
        m = _NAME.match(p.name)
        if not m:
            continue
        run = datetime.strptime(m.group(1), "%Y%m%dT%H").replace(tzinfo=timezone.utc)
        lead = int(m.group(3))
        d = out.setdefault(m.group(2), {})
        valid = run + timedelta(hours=lead)
        if valid not in d or lead < d[valid][0]:
            d[valid] = (lead, p)
    return out


def _read(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        v = next(k for k in ds.data_vars if ds[k].ndim >= 2)
        return np.squeeze(ds[v].values).astype(np.float32)


def _pick(table: dict, when: datetime, tol_h: float):
    """File for `when`, or the nearest valid time within tol_h hours; None if there is none."""
    if when in table:
        return table[when][1]
    if not table:
        return None
    near = min(table, key=lambda t: abs((t - when).total_seconds()))
    return table[near][1] if abs((near - when).total_seconds()) <= tol_h * 3600 else None


def _rotated_grid(root: Path, model: str, lon_t, lat_t):
    """Nearest-neighbour map from a rotated grid to the target grid, and the angle of its x axis from east."""
    import xarray as xr
    from scipy.spatial import cKDTree

    with xr.open_dataset(root / model / "grid_latlon.nc") as g:
        lat, lon = g["latitude"].values.astype(np.float64), g["longitude"].values.astype(np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    k = np.cos(np.deg2rad(48.5))
    tree = cKDTree(np.column_stack([(lon * k).ravel(), lat.ravel()]))
    LON, LAT = np.meshgrid(lon_t, lat_t)
    dist, idx = tree.query(np.column_stack([(LON * k).ravel(), LAT.ravel()]))
    far = (dist > 0.05).reshape(LON.shape)                      # more than ~5 km from any source point
    ang = np.arctan2(np.gradient(lat, axis=1), np.gradient(lon, axis=1) * np.cos(np.deg2rad(lat)))
    return idx.reshape(LON.shape), far, ang


def load_window(root, start: str, hours: int = 24, step_h: int = 1, bbox=(-71.5, -56.0, 45.5, 52.0), tol_h: float = 3.0, verbose: bool = True) -> Fields:
    """Fields for `hours` hours from `start` ("2026-11-17T00"), one frame every `step_h` hours.

    bbox = (west, east, south, north) in degrees. A frame whose file is missing takes the nearest valid
    time within `tol_h` hours; if there is none the field is 0 for that frame and the gap is reported.
    """
    import xarray as xr

    root = Path(root)
    t_start = datetime.strptime(start, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    times = [t_start + timedelta(hours=h) for h in range(0, hours + 1, step_h)]
    idx = {m: _index(root, m) for m in _VARS}
    first = next(iter(idx["ciops"][_VARS["ciops"]["conc"]].values()))[1]
    with xr.open_dataset(first) as ds:
        lat, lon = ds["latitude"].values, ds["longitude"].values
    lon = np.where(lon > 180, lon - 360, lon)
    jj = np.where((lat >= bbox[2] - 1e-9) & (lat <= bbox[3] + 1e-9))[0]
    ii = np.where((lon >= bbox[0] - 1e-9) & (lon <= bbox[1] + 1e-9))[0]
    sl = (slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1))
    lat_t, lon_t = lat[sl[0]], lon[sl[1]]
    maps = {m: _rotated_grid(root, m, lon_t, lat_t) for m in ("hrdps", "rdwps")}

    data = np.zeros((len(times), 7, len(lat_t), len(lon_t)), np.float32)
    land, missing = None, []
    for n, when in enumerate(times):
        got = {}
        for model, names in _VARS.items():
            for key, var in names.items():
                p = _pick(idx[model].get(var, {}), when, tol_h)
                if p is None:
                    missing.append(f"{when:%m-%dT%H} {key}")
                got[key] = None if p is None else _read(p)
        if got["conc"] is not None:
            c = got["conc"][sl]
            if land is None:
                land = np.isnan(c).astype(np.float32)
            c = np.nan_to_num(c)
            data[n, CONC] = c
            if got["vol"] is not None:
                data[n, THICK] = np.where(c > 0.01, np.nan_to_num(got["vol"][sl]) / np.maximum(c, 0.01), 0.0)
        for key, ch in (("uc", UC), ("vc", VC)):
            if got[key] is not None:
                data[n, ch] = np.nan_to_num(got[key][sl])
        if got["uw"] is not None and got["vw"] is not None:
            ix, far, ang = maps["hrdps"]
            u, v = got["uw"], got["vw"]
            east, north = u * np.cos(ang) - v * np.sin(ang), u * np.sin(ang) + v * np.cos(ang)
            data[n, UW], data[n, VW] = np.where(far, 0, east.ravel()[ix]), np.where(far, 0, north.ravel()[ix])
        if got["hs"] is not None:
            ix, far, _ = maps["rdwps"]
            data[n, HS] = np.where(far, 0, np.nan_to_num(got["hs"].ravel()[ix]))
    if land is None:
        raise FileNotFoundError(f"no CIOPS ice concentration file within {tol_h} h of the window starting {start} under {root}")
    if verbose and missing:
        print(f"load_window: {len(missing)} missing field-times, e.g. {missing[:6]}")
    return Fields(data=data, land=land, lon0=float(lon_t[0]), lat0=float(lat_t[0]), dlon=float(lon_t[1] - lon_t[0]), dlat=float(lat_t[1] - lat_t[0]),
                  dt=3600.0 * step_h, t0=t_start.isoformat())


def with_ice(f: Fields, conc, thick) -> Fields:
    """Replace the ice in every frame. conc and thick are [ny, nx] (or a scalar thickness) on the fields' grid."""
    data = f.data.copy()
    conc = np.where(f.land > 0.5, 0.0, np.clip(np.nan_to_num(conc), 0.0, 1.0)).astype(np.float32)
    data[:, CONC] = conc
    data[:, THICK] = np.where(conc > 0.01, np.broadcast_to(np.asarray(thick, np.float32), conc.shape), 0.0)
    return f._replace(data=data)


def read_chart_concentration(path, f: Fields):
    """Daily ice concentration from the DFO/OGSL product built on Canadian Ice Service charts
    (ERDDAP mpoDailySatelliteDataSic, percent, about 1 km), brought to the fields' grid by nearest neighbour."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        a = np.squeeze(ds["sic"].values).astype(np.float32) / 100.0
        lat, lon = ds["latitude"].values, ds["longitude"].values
    j = np.abs(lat[None, :] - f.lat[:, None]).argmin(1)
    i = np.abs(lon[None, :] - f.lon[:, None]).argmin(1)
    out = a[np.ix_(j, i)]
    outside = (f.lat < lat.min() - 0.01) | (f.lat > lat.max() + 0.01)
    out[outside, :] = np.nan
    out[:, (f.lon < lon.min() - 0.01) | (f.lon > lon.max() + 0.01)] = np.nan
    return out


def save(f: Fields, path):
    np.savez_compressed(path, **f._asdict())


def load(path) -> Fields:
    z = np.load(path)
    return Fields(data=z["data"], land=z["land"], lon0=float(z["lon0"]), lat0=float(z["lat0"]), dlon=float(z["dlon"]), dlat=float(z["dlat"]), dt=float(z["dt"]), t0=str(z["t0"]))
