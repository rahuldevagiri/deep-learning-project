"""Sentinel-1 monthly composites from Google Earth Engine - IDENTICAL to training (notebooks 01/04/10).

Authentication (never commit keys):
  * hosted: set EE_SERVICE_ACCOUNT_JSON to the service-account key JSON *content* (platform secret)
  * local : run `earthengine authenticate` once; then EE_PROJECT (default deep-learning-27) is used.
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import CACHE_DIR, EE_PROJECT, NODATA, RES

_EE_READY = False
FROM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]          # ESA WorldCover codes
TO = [0, 1, 1, 2, 3, 4, 255, 5, 6, 0, 1]                      # -> 7 project classes (snow/ice -> 255)


def init_ee():
    global _EE_READY
    if _EE_READY:
        return
    import ee
    key = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if key:
        info = json.loads(key)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key)
        ee.Initialize(creds, project=info.get("project_id", EE_PROJECT))
    else:
        ee.Initialize(project=EE_PROJECT)
    _EE_READY = True


def validate(lat, lon, start, size):
    if lat is None or lon is None:
        raise ValueError("Enter coordinates or pick a place first.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"lat/lon out of range: {lat}, {lon}")
    if size not in (256, 512, 768):
        raise ValueError("Size must be 256, 512 or 768 px.")
    first = pd.Timestamp(start + "-01") - pd.DateOffset(months=1)
    last = pd.Timestamp(start + "-01") + pd.DateOffset(months=13)
    if first < pd.Timestamp("2014-10-01"):
        raise ValueError("Sentinel-1 IW data start in Oct 2014: choose a start month from 2014-11 on.")
    if last > pd.Timestamp.today().normalize().replace(day=1):
        raise ValueError(f"Start month too late: the 12 months + 1 neighbour need complete data through "
                         f"{last - pd.DateOffset(months=1):%Y-%m}.")


def utm_epsg(lat, lon):
    z = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + z if lat >= 0 else 32700 + z}"


def grid(lat, lon, size):
    """UTM grid of the analysis square + its lat/lon bounds [[south, west], [north, east]]."""
    crs = utm_epsg(lat, lon)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    x0, y0 = round(x / RES) * RES - size * RES // 2, round(y / RES) * RES + size * RES // 2
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = [x0, x0 + size * RES]
    ys = [y0 - size * RES, y0]
    pts = [inv.transform(a, b) for a in xs for b in ys]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    return crs, x0, y0, [[min(lats), min(lons)], [max(lats), max(lons)]]


def _s1_collection(ee, region, start, end):
    def mask_edges(img):
        a = img.select("angle")
        return img.updateMask(a.gt(30.6).And(a.lt(45.2)))
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .map(mask_edges).select(["VV", "VH"]))


def _monthly_composite(ee, region, start, i):
    s = ee.Date(start).advance(i, "month")
    col = _s1_collection(ee, region, s, s.advance(1, "month"))
    empty = ee.Image.constant([0, 0]).rename(["VV", "VH"]).updateMask(0)
    img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), empty))
    return img.multiply(100).round().unmask(NODATA).toInt16().rename([f"VV_{i:02d}", f"VH_{i:02d}"])


def cache_path(lat, lon, start, size):
    return CACHE_DIR / f"s1_{lat:.4f}_{lon:.4f}_{start}_{size}.npz"


def fetch(lat, lon, start, size, use_cache=True):
    """-> dict(s1 (14,2,H,W) int16 dB*100 for months start-1..start+12, label (H,W) uint8, counts list[14],
    bounds [[s,w],[n,e]], from_cache bool). Results are cached on disk (presets work without Earth Engine)."""
    p = cache_path(lat, lon, start, size)
    if use_cache and p.exists():
        z = np.load(p)
        return dict(s1=z["s1"], label=z["label"], counts=z["counts"].tolist(),
                    bounds=z["bounds"].tolist(), from_cache=True)
    init_ee()
    import ee
    s0 = (pd.Timestamp(start + "-01") - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    crs, x0, y0, bounds = grid(lat, lon, size)
    region = ee.Geometry.Point([lon, lat]).buffer(size * RES * 0.8).bounds()
    label = (ee.ImageCollection("ESA/WorldCover/v200").first()
             .remap(FROM, TO, 255).unmask(255).toInt16().rename("label"))
    stack = ee.Image.cat([_monthly_composite(ee, region, s0, i) for i in range(14)] + [label])
    arr = ee.data.computePixels({
        "expression": stack, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": size, "height": size}, "crsCode": crs,
                 "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": x0,
                                     "shearY": 0, "scaleY": -RES, "translateY": y0}}})
    s1 = np.stack([arr[f"{b}_{i:02d}"] for i in range(14) for b in ("VV", "VH")]).reshape(14, 2, size, size)
    s = ee.Date(s0)
    counts = ee.List.sequence(0, 13).map(lambda i: _s1_collection(
        ee, region, s.advance(i, "month"), s.advance(ee.Number(i).add(1), "month")).size()).getInfo()
    lab = arr["label"].astype(np.uint8)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, s1=s1.astype(np.int16), label=lab, counts=np.array(counts), bounds=np.array(bounds))
    return dict(s1=s1.astype(np.int16), label=lab, counts=list(counts), bounds=bounds, from_cache=False)
