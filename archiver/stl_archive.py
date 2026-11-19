# /// script
# requires-python = ">=3.11"
# dependencies = ["xarray>=2024.6", "netCDF4>=1.7,<1.7.5", "cfgrib>=0.9.14", "eccodes>=2.38", "numpy"]
# ///
"""Daily archive of St. Lawrence forecast and ice data, before the public servers drop it.

The MSC Datamart (dd.weather.gc.ca) keeps about 30 days of model output, the CIS SIGRID-3 ice charts stay
about two weeks, and the copies behind Nautilo keep 48 h. This script saves, for one UTC day:

  ciops   CIOPS-East (2 km): sea ice (concentration, volume, drift, pressure, strength, ...), sea level, surface currents
  hrdps   HRDPS (2.5 km): 10 m wind, 2 m air temperature, sea-level pressure
  rdwps   RDWPS (2.5 km): significant wave height, peak period and direction, ice concentration seen by the wave model
  ohps    OHPS-SLFE (100 m river model, Montréal to Saint-Joseph-de-la-Rive): depth-averaged currents, water level
  cis     every CIS SIGRID-3 ice chart (egg codes) currently published
  iwls    CHS water levels for stations in the domain: observations, predictions, forecasts, SPINE forecasts

Gridded fields are cropped to BBOX and written as compressed NetCDF, one file per variable, run and lead time.
Every run's analysis and first 5 h are kept, plus the 00Z run out to 48 h every 3 h (forecast archive).
Files already present are skipped, so the script can be rerun and can backfill:

    uv run stl_archive.py --root /path/to/archive                 # yesterday (UTC)
    uv run stl_archive.py --root /path/to/archive --backfill 30   # the last 30 days the Datamart still has
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

DD = "https://dd.weather.gc.ca"
BBOX = dict(lat=(45.0, 52.5), lon=(-74.5, -55.5))       # Montréal to Cabot Strait
IO_LOCK = threading.Lock()
UA = {"User-Agent": "stl-archive/1.0 (research data archiving)"}

SHORT = range(0, 6)                                      # leads kept from every run
LONG = range(6, 49, 3)                                   # extra leads kept from the 00Z run


def leads(run):
    return list(SHORT) + (list(LONG) if run == "00" else [])


MODELS = {
    "ciops": dict(
        dirs=lambda d, r, h: f"{d}/WXO-DD/model_ciops/east/2km/{r}/{h:03d}/",
        runs=["00", "06", "12", "18"], leads=leads,
        vars=r"_(SeaIce[A-Za-z]+_Sfc|SeaSfcHeight_Sfc|SeaWaterVelocity[XY]_DBS-0\.5m)_"),
    "hrdps": dict(
        dirs=lambda d, r, h: f"{d}/WXO-DD/model_hrdps/continental/2.5km/{r}/{h:03d}/",
        runs=["00", "06", "12", "18"], leads=leads,
        vars=r"_(UGRD_AGL-10m|VGRD_AGL-10m|TMP_AGL-2m|PRMSL_MSL)_"),
    "rdwps": dict(
        dirs=lambda d, r, h: f"{d}/WXO-DD/model_rdwps/national/2.5km/{r}/",
        runs=["00", "06", "12"], leads=leads,
        vars=r"_(HTSGW_Sfc|PWPER_Sfc|PWAVEDIR_Sfc|ICEC_Sfc)_",
        lead_in_name=True),
    "ohps": dict(
        dirs=lambda d, r, h: f"{d}/WXO-DD/model_ohps/slfe/100m/{r}/{h:03d}/",
        runs=["00", "12"], leads=lambda r: list(range(0, 12, 3)),
        vars=r"_(RiverVelocity[XY]_DBS-Avg|WaterLvlRiver_Sfc)_"),
}


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except Exception as e:  # noqa: BLE001 - network errors of every kind get retried
            err = e
        time.sleep(5 * (k + 1))
    raise err


def listing(url):
    try:
        html = get(url).decode()
    except urllib.error.HTTPError:
        return []
    return sorted(set(re.findall(r'href="([^"?/][^"/]*\.(?:nc|grib2|tar))"', html)))


def crop(src, dst, gridfile=None):
    """Crop a NetCDF or GRIB2 file to BBOX and write compressed NetCDF."""
    import numpy as np
    import xarray as xr
    if src.suffix == ".grib2":
        ds = xr.open_dataset(src, engine="cfgrib", backend_kwargs={"indexpath": ""})
    else:
        ds = xr.open_dataset(src)
    names = {n.lower(): n for n in list(ds.coords) + list(ds.variables)}
    lat = ds[names.get("latitude") or names["lat"]].values
    lon = ds[names.get("longitude") or names["lon"]].values
    lon = np.where(lon > 180, lon - 360, lon)
    if lat.ndim == 1 and lon.ndim == 1:
        latn, lonn = names.get("latitude") or names["lat"], names.get("longitude") or names["lon"]
        iy = np.where((lat >= BBOX["lat"][0]) & (lat <= BBOX["lat"][1]))[0]
        ix = np.where((lon >= BBOX["lon"][0]) & (lon <= BBOX["lon"][1]))[0]
        ds = ds.isel({ds[latn].dims[0]: slice(iy.min(), iy.max() + 1), ds[lonn].dims[0]: slice(ix.min(), ix.max() + 1)})
    else:
        m = (lat >= BBOX["lat"][0]) & (lat <= BBOX["lat"][1]) & (lon >= BBOX["lon"][0]) & (lon <= BBOX["lon"][1])
        rows, cols = np.where(m)
        latn, lonn = names.get("latitude") or names["lat"], names.get("longitude") or names["lon"]
        ydim, xdim = ds[latn].dims
        ds = ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})
        # The 2D lat/lon of a rotated grid is the same in every file and larger than the field: store it once.
        if gridfile is not None and not gridfile.exists():
            g = ds[[latn, lonn]].reset_coords()[[latn, lonn]].astype("float32").load()
            g.to_netcdf(gridfile.with_suffix(".part"), encoding={k: {"zlib": True} for k in g.data_vars})
            gridfile.with_suffix(".part").rename(gridfile)
        ds = ds.drop_vars([latn, lonn])
    # Compress coordinates too (the rotated grids carry 2D float64 lat/lon), and keep 3 significant decimals:
    # the float noise below that is what defeats zlib, and no field here is known to better than 0.001.
    enc = {}
    for v in list(ds.data_vars) + [c for c in ds.coords if ds[c].ndim >= 1]:
        e = {"zlib": True, "complevel": 4, "shuffle": True}
        if np.issubdtype(ds[v].dtype, np.floating):
            if v in ds.coords and ds[v].ndim == 2:
                ds[v] = ds[v].astype("float32")
            elif v in ds.data_vars:
                e["least_significant_digit"] = 3
        enc[v] = e
    tmp = dst.with_suffix(".part")
    ds.load().to_netcdf(tmp, encoding=enc)
    ds.close()
    tmp.rename(dst)


def fetch_model(name, day, root, pool, log):
    m = MODELS[name]
    D = day.strftime("%Y%m%d")
    out = root / name / day.strftime("%Y/%m/%d")
    out.mkdir(parents=True, exist_ok=True)
    pat = re.compile(m["vars"])
    jobs = []
    for r in m["runs"]:
        lds = m["leads"](r)
        if m.get("lead_in_name"):
            files = listing(f"{DD}/{m['dirs'](D, r, 0)}")
            want = {f"PT{h:03d}H" for h in lds}
            for f in files:
                if pat.search(f) and re.search(r"PT\d{3}H", f).group(0) in want:
                    jobs.append((f"{DD}/{m['dirs'](D, r, 0)}{f}", f))
        else:
            for h in lds:
                base = f"{DD}/{m['dirs'](D, r, h)}"
                for f in listing(base):
                    if pat.search(f):
                        jobs.append((base + f, f))

    def one(job):
        url, f = job
        dst = out / (Path(f).stem + ".nc")
        if dst.exists():
            return "skip", 0
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as td:
            src = Path(td) / f
            src.write_bytes(get(url))
            if m.get("crop", True):
                with IO_LOCK:          # netCDF4/HDF5 and eccodes are not thread-safe: download in parallel, crop serially
                    crop(src, dst, root / name / "grid_latlon.nc")
            else:
                shutil.copyfile(src, dst.with_suffix(".part"))
                dst.with_suffix(".part").rename(dst)
        return "ok", dst.stat().st_size

    stats = dict(listed=len(jobs), ok=0, skip=0, failed=0, bytes=0)
    for fut, job in [(pool.submit(one, j), j) for j in jobs]:
        try:
            st, n = fut.result()
            stats[st] += 1
            stats["bytes"] += n
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            log(f"  {name} FAILED {job[1]}: {e!r}")
    return stats


def fetch_cis(root, log):
    out = root / "cis"
    out.mkdir(parents=True, exist_ok=True)
    base = "https://ice-glaces.ec.gc.ca/prods/sigrids/"
    stats = dict(listed=0, ok=0, skip=0, failed=0, bytes=0)
    for f in listing(base):
        stats["listed"] += 1
        m = re.search(r"_(\d{4})(\d{2})\d{2}T", f)
        d = out / (f"{m.group(1)}/{m.group(2)}" if m else "undated")
        d.mkdir(parents=True, exist_ok=True)
        if (d / f).exists():
            stats["skip"] += 1
            continue
        try:
            data = get(base + f)
            (d / (f + ".part")).write_bytes(data)
            (d / (f + ".part")).rename(d / f)
            stats["ok"] += 1
            stats["bytes"] += len(data)
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            log(f"  cis FAILED {f}: {e!r}")
    return stats


IWLS = "https://api-iwls.dfo-mpo.gc.ca/api/v1"
IWLS_CODES = {"wlo", "wlp", "wlf", "wlf-spine", "wcs1-derv", "wcd1-derv", "wcso", "wcdo"}


def fetch_iwls(day, root, log):
    out = root / "iwls" / day.strftime("%Y/%m")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"iwls_{day:%Y%m%d}.json"
    if dst.exists():
        return dict(listed=0, ok=0, skip=1, failed=0, bytes=0)
    stations = json.loads(get(f"{IWLS}/stations"))
    stations = [s for s in stations
                if BBOX["lat"][0] <= s.get("latitude", 0) <= BBOX["lat"][1]
                and BBOX["lon"][0] <= s.get("longitude", 0) <= BBOX["lon"][1]]
    t0 = f"{day:%Y-%m-%d}T00:00:00Z"
    t1 = f"{day + dt.timedelta(days=1):%Y-%m-%d}T00:00:00Z"
    data, stats = {}, dict(listed=0, ok=0, skip=0, failed=0, bytes=0)
    for s in stations:
        codes = {ts.get("code") for ts in s.get("timeSeries", [])} & IWLS_CODES
        for c in sorted(codes):
            stats["listed"] += 1
            try:
                data.setdefault(s["code"], {"name": s.get("officialName"), "lat": s["latitude"],
                                            "lon": s["longitude"]})[c] = json.loads(
                    get(f"{IWLS}/stations/{s['id']}/data?time-series-code={c}&from={t0}&to={t1}"))
                stats["ok"] += 1
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                log(f"  iwls FAILED {s.get('code')} {c}: {e!r}")
            time.sleep(0.2)
    blob = json.dumps(data).encode()
    (dst.with_suffix(".part")).write_bytes(blob)
    dst.with_suffix(".part").rename(dst)
    stats["bytes"] = len(blob)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", help="UTC day YYYYMMDD (default: yesterday)")
    ap.add_argument("--backfill", type=int, default=0, help="also do this many days before --date")
    ap.add_argument("--only", nargs="*", default=None, help="subset of: ciops hrdps rdwps ohps cis iwls")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    root = a.root
    (root / ".tmp").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    end = dt.datetime.strptime(a.date, "%Y%m%d").date() if a.date else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    days = [end - dt.timedelta(days=k) for k in range(a.backfill, -1, -1)]
    only = set(a.only or ["ciops", "hrdps", "rdwps", "ohps", "cis", "iwls"])
    logf = open(root / "logs" / f"archive_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.log", "a")

    def log(msg):
        line = f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    summary = {}
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for day in days:
            for name in ["ciops", "hrdps", "rdwps", "ohps"]:
                if name in only:
                    try:
                        st = fetch_model(name, day, root, pool, log)
                    except Exception:  # noqa: BLE001
                        st = dict(error=traceback.format_exc(limit=2))
                    summary[f"{day}/{name}"] = st
                    log(f"{day} {name}: {st}")
            if "iwls" in only:
                try:
                    st = fetch_iwls(day, root, log)
                except Exception:  # noqa: BLE001
                    st = dict(error=traceback.format_exc(limit=2))
                summary[f"{day}/iwls"] = st
                log(f"{day} iwls: {st}")
    if "cis" in only:
        st = fetch_cis(root, log)
        summary["cis"] = st
        log(f"cis: {st}")

    bad = [k for k, v in summary.items() if "error" in v or v.get("failed") or v.get("listed") == 0]
    status = dict(finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  days=[str(d) for d in days], problems=bad, summary=summary)
    (root / "STATUS.json").write_text(json.dumps(status, indent=1))
    log(f"done; problems: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    os.umask(0o002)
    main()
