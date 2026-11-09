"""Place-name search on OpenStreetMap data: Nominatim first, Photon (komoot) as a fallback.

Usage policies: Nominatim max 1 request/second with an identifying User-Agent and cached results
(https://operations.osmfoundation.org/policies/nominatim/); Photon fair use (https://photon.komoot.io).
The query text is sent to these services. No bulk geocoding.
"""
import threading
import time
from functools import lru_cache

import requests

_UA = "lulc-s1-demo/1.0 (university course project; Sentinel-1 land cover demo)"
_lock = threading.Lock()
_last = [0.0]


def _get_nominatim(path, params):
    with _lock:                                   # global 1 request / second
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        r = requests.get(f"https://nominatim.openstreetmap.org/{path}", params=params,
                         headers={"User-Agent": _UA, "Accept-Language": "en"}, timeout=15)
        _last[0] = time.time()
    r.raise_for_status()
    return r


def _nominatim(q, limit):
    r = _get_nominatim("search", {"q": q, "format": "jsonv2", "limit": limit})
    return [(f"{d['display_name']} [{d.get('type', '')}]", float(d["lat"]), float(d["lon"])) for d in r.json()]


def _photon(q, limit):
    r = requests.get("https://photon.komoot.io/api/", params={"q": q, "limit": limit, "lang": "en"},
                     headers={"User-Agent": _UA}, timeout=15)
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        parts = [p.get(k) for k in ("name", "city", "state", "country") if p.get(k)]
        out.append((", ".join(dict.fromkeys(parts)) + f" [{p.get('osm_value', '')}]", lat, lon))
    return out


@lru_cache(maxsize=512)
def search(query, limit=6):
    """-> list of (label, lat, lon), best first. Empty list if nothing found."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    hits, errors = [], []
    for fn in (_nominatim, _photon):
        try:
            hits += fn(q, limit)
        except Exception as e:               # one provider down -> use the other
            errors.append(f"{fn.__name__[1:]}: {e}")
        if len(hits) >= 3:
            break
    if not hits and errors:
        raise RuntimeError("; ".join(errors))
    seen, out = set(), []
    for name, la, lo in hits:                # drop near-duplicates (< ~1 km apart)
        k = (round(la, 2), round(lo, 2))
        if k not in seen:
            seen.add(k)
            out.append((name, round(la, 5), round(lo, 5)))
    return out[:limit]


@lru_cache(maxsize=512)
def reverse(lat, lon):
    """Coordinates -> place name (town/city level), or '' if nothing is there (e.g. open ocean)."""
    lat, lon = round(float(lat), 4), round(float(lon), 4)
    try:
        d = _get_nominatim("reverse", {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10}).json()
        if "display_name" in d:
            return d["display_name"]
    except Exception:
        pass
    try:                                     # fallback: Photon
        r = requests.get("https://photon.komoot.io/reverse", params={"lat": lat, "lon": lon, "lang": "en"},
                         headers={"User-Agent": _UA}, timeout=15)
        r.raise_for_status()
        f = r.json().get("features", [])
        if f:
            p = f[0]["properties"]
            return ", ".join(dict.fromkeys(p.get(k) for k in ("name", "city", "state", "country") if p.get(k)))
    except Exception:
        pass
    return ""
