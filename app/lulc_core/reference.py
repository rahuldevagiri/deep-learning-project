"""Independent reference: Google Dynamic World (Sentinel-2 based, never used in training).

Monthly reference map = per-pixel mode of the Dynamic World 'label' band over all Sentinel-2 images of that month.
Dynamic World classes -> project classes:
  water->Water, trees->Tree, grass->Shrub/Grass, flooded_vegetation->Wetland, crops->Cropland,
  shrub_and_scrub->Shrub/Grass, built->Built-up, bare->Bare, snow_and_ice->excluded (255).
"""
import numpy as np
import pandas as pd

from .config import CACHE_DIR, CLASSES, NC, RES
from .s1_fetch import grid, init_ee

DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_TO = [5, 0, 1, 6, 2, 1, 3, 4, 255]


def fetch_dw(lat, lon, start, size, use_cache=True):
    """-> dict(dw (12, H, W) uint8 with 255 = no reference, counts list[12] of Sentinel-2 scenes per month)."""
    p = CACHE_DIR / f"dw_{lat:.4f}_{lon:.4f}_{start}_{size}.npz"
    if use_cache and p.exists():
        z = np.load(p)
        return dict(dw=z["dw"], counts=z["counts"].tolist())
    init_ee()
    import ee
    crs, x0, y0, _ = grid(lat, lon, size)
    region = ee.Geometry.Point([lon, lat]).buffer(size * RES * 0.8).bounds()
    s0 = ee.Date(start + "-01")
    col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
    imgs, sizes = [], []
    for i in range(12):
        c = col.filterDate(s0.advance(i, "month"), s0.advance(i + 1, "month")).select("label")
        empty = ee.Image.constant(0).rename("label").updateMask(0)
        m = ee.Image(ee.Algorithms.If(c.size().gt(0), c.mode(), empty))
        imgs.append(m.remap(DW_FROM, DW_TO, 255).unmask(255).toInt16().rename(f"dw_{i:02d}"))
        sizes.append(c.size())
    arr = ee.data.computePixels({
        "expression": ee.Image.cat(imgs), "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": size, "height": size}, "crsCode": crs,
                 "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": x0,
                                     "shearY": 0, "scaleY": -RES, "translateY": y0}}})
    dw = np.stack([arr[f"dw_{i:02d}"] for i in range(12)]).astype(np.uint8)
    counts = ee.List(sizes).getInfo()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, dw=dw, counts=np.array(counts))
    return dict(dw=dw, counts=list(counts))


def compare(pred, dw, labels):
    """Model vs Dynamic World. -> per-month table, pooled confusion matrix (rows = Dynamic World, cols = model),
    and mean class shares of both maps (pixels valid in both)."""
    rows, cm = [], np.zeros((NC, NC), np.int64)
    for m, lab in enumerate(labels):
        v = (pred[m] != 255) & (dw[m] != 255)
        n = int(v.sum())
        row = {"month": lab, "reference_%": round(100 * n / v.size, 1),
               "agree_DW_%": round(100 * float((pred[m][v] == dw[m][v]).mean()), 1) if n else np.nan}
        if n:
            cm += np.bincount(dw[m][v].astype(np.int64) * NC + pred[m][v], minlength=NC * NC).reshape(NC, NC)
        rows.append(row)
    per_month = pd.DataFrame(rows).set_index("month")
    tot = cm.sum()
    shares = pd.DataFrame({"model %": (100 * cm.sum(0) / tot).round(2) if tot else np.nan,
                           "Dynamic World %": (100 * cm.sum(1) / tot).round(2) if tot else np.nan},
                          index=CLASSES)
    return per_month, cm, shares
