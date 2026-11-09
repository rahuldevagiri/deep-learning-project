"""P(c,m) = 100 * A(c,m) / A(valid,m); dP in percentage points; flags from the M5 thresholds."""
import numpy as np
import pandas as pd

from .config import AREA_ERR, CLASSES, CONF_MIN, NC, THR_FL, THR_M2M


def stats_tables(pred, conf, lab, months, calibrated=True):
    """Returns P (per month), dP (month-to-month), flags, fl (first-to-last), wc (WorldCover shares).
    For uncalibrated models the confidence check is not applied (reliable = 'n/a') and flags are indicative:
    the thresholds were measured for the FINAL model only."""
    rows = []
    for m, mo in enumerate(months):
        v = pred[m] != 255
        nv = int(v.sum())
        mc = round(float(np.nanmean(conf[m])), 3) if nv else np.nan
        if not nv:
            rel = "LOW"
        elif not calibrated:
            rel = "n/a"
        else:
            rel = "yes" if mc >= CONF_MIN else "LOW"
        row = {"month": mo, "valid_%": round(100 * nv / v.size, 1), "mean_conf": mc, "reliable": rel}
        cnt = np.bincount(pred[m][v], minlength=NC)
        for c, name in enumerate(CLASSES):
            row[name] = round(100 * cnt[c] / nv, 2) if nv else np.nan
        both = v & (lab != 255)
        row["agree_WC_%"] = round(100 * float((pred[m][both] == lab[both]).mean()), 1) if both.any() else np.nan
        rows.append(row)
    P = pd.DataFrame(rows).set_index("month")

    dP = P[CLASSES].diff().iloc[1:].round(2)
    big = dP.abs().gt(pd.Series(THR_M2M))
    rel = P["reliable"].ne("LOW")
    pair_ok = (rel & rel.shift(1, fill_value=False)).iloc[1:]
    flags = pd.DataFrame("", index=dP.index, columns=CLASSES)
    tag = "CHANGE?" if calibrated else "CHANGE? (indicative)"
    flags[big & pair_ok.values[:, None]] = tag
    flags[big & ~pair_ok.values[:, None]] = "(low conf)"

    ok = P[P[CLASSES].notna().all(axis=1)]
    fl = pd.DataFrame()
    if len(ok) >= 2:
        fl = pd.DataFrame({"first_%": ok[CLASSES].iloc[0], "last_%": ok[CLASSES].iloc[-1]})
        fl["dP_pp"] = (fl["last_%"] - fl["first_%"]).round(2)
        fl["threshold_pp"] = pd.Series(THR_FL)
        ends_ok = ok["reliable"].iloc[0] != "LOW" and ok["reliable"].iloc[-1] != "LOW"
        yes = "yes" if calibrated else "yes (indicative)"
        fl["reliable_change"] = np.where(fl.dP_pp.abs() <= fl.threshold_pp, "no (within seasonal/noise range)",
                                         yes if ends_ok else "unknown (first or last month low confidence)")
        k = min(3, len(ok) // 2)
        fl[f"dP_{k}mo_mean_pp"] = (ok[CLASSES].iloc[-k:].mean() - ok[CLASSES].iloc[:k].mean()).round(2)
        fl["typical_area_error_pp"] = pd.Series(AREA_ERR)
    v = lab != 255
    wc = pd.Series([round(100 * float((lab[v] == c).mean()), 2) if v.any() else np.nan for c in range(NC)],
                   index=CLASSES, name="WorldCover2021_%")
    return P, dP, flags, fl, wc


def month_label(iso):
    """'2024-03' -> 'March 2024'."""
    return pd.Timestamp(iso + "-01").strftime("%B %Y")


def change_summary(pred, P, fl, size_px):
    """Summary metrics + first-to-last transition matrix + changed-pixel mask."""
    ok = [m for m in range(pred.shape[0]) if P[CLASSES].iloc[m].notna().all()]
    a, b = pred[ok[0]], pred[ok[-1]]
    v = (a != 255) & (b != 255)
    tm = np.bincount(a[v].astype(np.int64) * NC + b[v], minlength=NC * NC).reshape(NC, NC)
    trans = pd.DataFrame(tm, index=[f"{c} (first)" for c in CLASSES], columns=[f"{c} (last)" for c in CLASSES])
    changed = v & (a != b)
    d = fl["dP_pp"]
    return dict(
        area_km2=(size_px * 10 / 1000) ** 2,
        changed_pct=round(100 * float(changed.sum()) / max(1, int(v.sum())), 2),
        gain=(d.idxmax(), float(d.max())), loss=(d.idxmin(), float(d.min())),
        first=ok[0], last=ok[-1], changed=changed, transition=trans)


def warnings_list(P, counts, subst, pred, lab, months, all_months, split):
    """Plain-language reliability warnings for the Trust panel.
    months: display labels of the 12 output months; all_months: ISO 'YYYY-MM' of the 14 fetched months."""
    w = []
    if "train" in split or "val" in split:
        w.append("This 5-degree cell was used for training/validation: agreement with WorldCover is optimistic.")
    if subst:
        w.append("Neighbour month without data: centre month repeated (as in training) for "
                 + ", ".join(months[i] for i in subst) + ".")
    few = [month_label(m) for m, n in zip(all_months, counts) if 0 < n < 3]
    if few:
        w.append("Fewer than 3 Sentinel-1 images in " + ", ".join(few) + ": noisier monthly composite.")
    low = P[P["valid_%"] < 50].index.tolist()
    if low:
        w.append("Less than 50% valid pixels in " + ", ".join(low) + ": percentages there are unreliable.")
    lowc = P[P["reliable"] == "LOW"].index.tolist()
    if lowc:
        w.append(f"Mean confidence below {CONF_MIN} in " + ", ".join(lowc) + ": maps and changes there are not trusted.")
    s1_ok = pred.min(0) != 255
    out_cls = 100 * float((lab[s1_ok] == 255).mean()) if s1_ok.any() else 0.0
    if out_cls > 20:
        w.append(f"{out_cls:.0f}% of the area is outside the 7 classes in WorldCover (e.g. snow/ice) or unmapped. "
                 "The model still has to pick one of 7 classes there.")
    if any(m[:4] != "2021" for m in all_months[1:13]):
        w.append("The model was trained on 2021 only. Other years are not validated (Sentinel-1B failed in Dec 2021, "
                 "so 2022-2024 composites use fewer images).")
    return w, out_cls
