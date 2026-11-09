"""Pre-fetch demo locations from Earth Engine into cache/ so they run later without Earth Engine.

Usage (after `earthengine authenticate`):  python build_cache.py
"""
from lulc_core.s1_fetch import fetch

DEMO = {   # all in TEST cells; start Jan 2021 (training year)
    "Lagos": (6.52, 3.38), "Lima": (-12.05, -77.04), "Amazon near Iquitos": (-3.50, -72.50),
    "Lake Superior": (47.50, -88.00), "Taklamakan": (38.50, 79.50), "Huascaran": (-9.12, -77.60),
}

for name, (lat, lon) in DEMO.items():
    d = fetch(lat, lon, "2021-01", 512)
    print(f"{name}: images/month {d['counts']} | cached={d['from_cache']}")
