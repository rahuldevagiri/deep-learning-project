"""Sentinel-1 Land Cover Monitor - Gradio prototype.

Input : preset city, place name OR latitude/longitude (each fills the other), start month, area size, model.
Output (tabs): monthly maps, class shares, change, independent sanity check (Dynamic World), downloads,
               and a static model-evaluation tab (test-set results, training curves, model comparison).

Run:  python app.py   ->  http://127.0.0.1:7860
All science is in lulc_core/ (same code path as notebook 10); this file is UI only.
"""
import json
import math
import os
import tempfile
import time
import zipfile

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from lulc_core import evaluation, geocode, viz
from lulc_core.config import CLASSES, MODELS, ROOT, THR_M2M, available_models
from lulc_core.inference import predict_12
from lulc_core.reference import compare, fetch_dw
from lulc_core.s1_fetch import fetch, grid, validate
from lulc_core.stats import change_summary, month_label, stats_tables, warnings_list

CELL_SPLIT = json.loads((ROOT / "config" / "cell_split.json").read_text())
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
YEARS = list(range(2015, pd.Timestamp.today().year + 1))
MODEL_KEYS = available_models()
MODEL_CHOICES = [(f"{MODELS[k]['label']}  (val mIoU {MODELS[k]['val_mIoU']:.3f})", k) for k in MODEL_KEYS]
SIZES = [("2.56 km x 2.56 km", 256), ("5.12 km x 5.12 km", 512), ("7.68 km x 7.68 km", 768)]
# Cities in 5-degree cells of the TEST split (never used for training or validation) -> honest results.
PRESETS = {
    "Lagos, Nigeria": (6.52, 3.38), "Lima, Peru": (-12.05, -77.04), "Beijing, China": (39.90, 116.40),
    "Tianjin, China": (39.13, 117.20), "Sydney, Australia": (-33.87, 151.21), "Seattle, USA": (47.61, -122.33),
    "Dallas, USA": (32.78, -96.80), "Warsaw, Poland": (52.23, 21.01), "Marrakech, Morocco": (31.63, -8.00),
    "Valencia, Spain": (39.47, -0.38),
}
EVAL = evaluation.build()
SEASON_NOTE = ("**Read change with care.** These numbers are differences between model predictions. Part of the "
               "month-to-month variation is seasonal radar response (soil moisture, crop growth, flooding, snow) "
               "and classification noise, not real land-use change. Only bars beyond the dashed lines (95% noise level "
               "measured on the test set) are flagged as possible real change.")
SANITY_NOTE = ("**Sanity check against an independent map.** Google **Dynamic World** is made from Sentinel-2 "
               "*optical* images with a different method and was **not used anywhere in training**. For each month "
               "its reference map is the most frequent Dynamic World class per pixel (grass + shrub = Shrub/Grass, "
               "flooded vegetation = Wetland, snow/ice excluded). Agreement is not accuracy: both maps make errors, "
               "class definitions differ slightly, and cloudy months have few optical images. ESA WorldCover 2021 is "
               "shown for comparison only - it is the source of the training labels, so it is *not* independent.")


def clean(label):
    """'Lagos, Nigeria [city]' -> 'Lagos, Nigeria'."""
    return label.rsplit(" [", 1)[0] if label.endswith("]") else label


# ----------------------------------------------------------------------------- input callbacks
def preview_map(la, lo, size):
    if la is None or lo is None:
        return viz.location_map(None, None)
    try:
        bounds = grid(float(la), float(lo), int(size))[3]
    except Exception:
        bounds = None
    return viz.location_map(float(la), float(lo), bounds, size=int(size))


def pick_preset(name, size):
    if not name or name not in PRESETS:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    la, lo = PRESETS[name]
    return name, la, lo, gr.update(visible=False), preview_map(la, lo, size)


def find_place(name, size):
    """Place name -> coordinates (best match) + list of other matches."""
    try:
        hits = geocode.search(name)
    except Exception as e:
        raise gr.Error(f"Place search failed: {e}")
    if not hits:
        raise gr.Error(f"No place found for '{name}'. Check the spelling or enter coordinates.")
    label, la, lo = hits[0]
    alts = gr.update(choices=[(clean(n), f"{a},{b}") for n, a, b in hits], value=f"{la},{lo}",
                     visible=len(hits) > 1)
    return clean(label), la, lo, alts, preview_map(la, lo, size), gr.update(value=None)


def pick_alternative(v, size):
    if not v:
        return gr.update(), gr.update(), gr.update(), gr.update()
    la, lo = (float(x) for x in v.split(","))
    return geocode.reverse(la, lo) or gr.update(), la, lo, preview_map(la, lo, size)


def coords_entered(la, lo, size):
    """Coordinates typed by the user -> place name + map."""
    if la is None or lo is None:
        return gr.update(), gr.update(visible=False), preview_map(None, None, size), gr.update()
    name = geocode.reverse(la, lo) or "(no named place here, e.g. open sea)"
    return name, gr.update(visible=False), preview_map(la, lo, size), gr.update(value=None)


def inspect(state, mi):
    """Monthly-maps tab: model map, Dynamic World map and radar composite for one month."""
    if not state:
        return None, None, None
    m = int(mi) - 1
    model = viz.upscale(viz.colorize(state["pred"][m]))
    dw = viz.upscale(viz.colorize(state["dw"][m])) if state["dw"] is not None else None
    radar = viz.upscale(viz.radar_image(state["s1"][m + 1]))
    return model, dw, radar


# ----------------------------------------------------------------------------- run
def run(name, la, lo, year, month, size, key, progress=gr.Progress()):
    if la is None or lo is None or (float(la) == 0 and float(lo) == 0 and not name):
        raise gr.Error("Choose a preset, enter a place name (and press Find) or enter coordinates first.")
    try:
        la, lo, size = float(la), float(lo), int(size)
        start = f"{int(year)}-{int(month):02d}"
        validate(la, lo, start, size)
        if key not in MODELS:
            raise ValueError("Choose a model.")
    except (TypeError, ValueError) as e:
        raise gr.Error(str(e))

    t0 = time.time()
    progress(0.02, desc="Fetching 14 monthly Sentinel-1 composites from Earth Engine...")
    try:
        d = fetch(la, lo, start, size)
    except Exception as e:
        raise gr.Error(f"Sentinel-1 fetch failed: {e}. Check Earth Engine authentication (README).")
    iso = pd.date_range(start + "-01", periods=12, freq="MS").strftime("%Y-%m").tolist()
    labels = [month_label(m) for m in iso]
    all_iso = (pd.date_range(start + "-01", periods=14, freq="MS") - pd.DateOffset(months=1)).strftime("%Y-%m").tolist()
    if sum(d["counts"][1:13]) == 0:
        msg = (f"### {name or ''} ({la:.4f}, {lo:.4f})\n**No Sentinel-1 data** for this location and period "
               "(typical for open ocean). No maps can be produced.")
        return (None, msg, "") + (None,) * (len(OUTPUTS) - 3)

    spec = MODELS[key]
    pred, conf, subst = predict_12(
        d["s1"], key, progress=lambda i, n: progress(0.1 + 0.7 * i / n, desc=f"Predicting {labels[i]} ({i + 1}/12)"))
    P, dP, flags, fl, wc = stats_tables(pred, conf, d["label"], labels, spec["calibrated"])
    split = CELL_SPLIT.get(f"{math.floor(la / 5)}_{math.floor(lo / 5)}", "not in dataset")
    warns, _ = warnings_list(P, d["counts"], subst, pred, d["label"], labels, all_iso, split)
    if not spec["calibrated"]:
        warns.append("This model is not calibrated: its confidence is too high, and the change thresholds were "
                     "measured for the final model only (flags are indicative).")

    progress(0.82, desc="Fetching the independent Dynamic World reference...")
    dw = dw_counts = per_month = dw_cm = dw_shares = None
    try:
        r = fetch_dw(la, lo, start, size)
        dw, dw_counts = r["dw"], r["counts"]
        per_month, dw_cm, dw_shares = compare(pred, dw, labels)
        per_month.insert(0, "S2 scenes", dw_counts)
    except Exception as e:
        warns.append(f"Dynamic World reference could not be loaded ({e}); sanity-check tab is empty.")

    cs = change_summary(pred, P, fl, size)
    km = size * 10 / 1000
    agree_dw = f"{per_month['agree_DW_%'].mean():.1f}%" if per_month is not None else "n/a"
    summary = "  \n".join(
        [f"### {name or 'Selected location'}",
         f"{la:.4f}, {lo:.4f} | {labels[0]} to {labels[-1]} | area {km:.2f} km x {km:.2f} km | "
         f"region in dataset: **{split}** | run time {time.time() - t0:.0f} s",
         f"**Model:** {spec['label']}"]
        + (["", "**Reliability notes**"] + [f"- {w}" for w in warns] if warns else ["", "No reliability warnings."]))
    cards = viz.metric_cards([
        ("Area", f"{cs['area_km2']:.2f} km2", f"{size} x {size} pixels at 10 m"),
        ("Pixels changed", f"{cs['changed_pct']:.1f}%", f"{labels[cs['first']]} vs {labels[cs['last']]}"),
        ("Largest net gain", cs["gain"][0], f"{cs['gain'][1]:+.2f} pp"),
        ("Largest net loss", cs["loss"][0], f"{cs['loss'][1]:+.2f} pp"),
        ("Agreement with Dynamic World", agree_dw, "independent map, mean of months"),
    ])

    progress(0.92, desc="Drawing maps and charts...")
    grid_img = viz.month_grid(pred, conf, d["label"], P, labels, f"{name or ''}  ({la:.3f}, {lo:.3f})")
    out_dir = tempfile.mkdtemp(prefix="lulc_")
    gif_path = viz.make_gif(pred, labels, os.path.join(out_dir, "lulc_12_months.gif"))
    first = viz.upscale(viz.colorize(pred[cs["first"]]))
    last = viz.upscale(viz.colorize(pred[cs["last"]]))
    valid = (pred[cs["first"]] != 255) & (pred[cs["last"]] != 255)
    changed = viz.upscale(viz.changed_image(cs["changed"], valid))

    shares_tbl = P.drop(columns=["reliable"]).reset_index()
    change_tbl = dP.astype(str).where(flags == "", dP.astype(str) + " " + flags).reset_index()
    fl_tbl = fl.reset_index().rename(columns={"index": "class"})
    trans_tbl = cs["transition"].reset_index(names="first month \\ last month")

    if per_month is not None:
        dw_bar = viz.shares_compare_chart(dw_shares, wc)
        dw_line = viz.agreement_chart(per_month, P)
        dw_cm_fig = viz.confusion_heatmap(dw_cm, "Model vs Dynamic World, all months (row-normalised)",
                                          "Dynamic World (independent)", "model prediction")
        dw_tbl = per_month.reset_index()
    else:
        dw_bar = dw_line = dw_cm_fig = dw_tbl = None

    # downloads: CSV tables + PNG figure + GIF in one zip
    tag = (name or f"{la:.3f}_{lo:.3f}").split(",")[0].replace(" ", "_") + f"_{start}"
    files = {f"{tag}_class_shares.csv": shares_tbl, f"{tag}_change_month_to_month_pp.csv": change_tbl,
             f"{tag}_change_first_to_last_pp.csv": fl_tbl, f"{tag}_transition_first_last.csv": trans_tbl}
    if dw_tbl is not None:
        files[f"{tag}_sanity_dynamic_world.csv"] = dw_tbl
        files[f"{tag}_sanity_class_shares.csv"] = dw_shares.assign(**{"WorldCover 2021 %": wc.values}).reset_index(names="class")
    paths = []
    for fn, df in files.items():
        p = os.path.join(out_dir, fn)
        df.to_csv(p, index=False)
        paths.append(p)
    png = os.path.join(out_dir, f"{tag}_lulc_12_months.png")
    Image.fromarray(grid_img).save(png)
    paths += [png, gif_path]
    zp = os.path.join(out_dir, f"{tag}_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.basename(p))

    state = dict(pred=pred, s1=d["s1"], dw=dw, labels=labels)
    m0 = inspect(state, 1)
    progress(1.0, desc="Done")
    return (state, summary, cards,
            grid_img, viz.gif_html(gif_path), gr.update(value=1, maximum=12), *m0,
            viz.lines_chart(P, "Share of each class per month"), shares_tbl,
            viz.change_bars(dP, flags, THR_M2M, "Month-to-month change per class (pp)", P), change_tbl,
            viz.first_last_bars(fl, f"Change from {labels[cs['first']]} to {labels[cs['last']]} (pp)"), fl_tbl,
            first, last, changed, trans_tbl,
            dw_bar, dw_line, dw_cm_fig, dw_tbl,
            [zp] + paths)


# ----------------------------------------------------------------------------- layout
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter")] + [gr.themes.Font(f) for f in ("Segoe UI", "Arial", "sans-serif")],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono")] + [gr.themes.Font(f) for f in ("Consolas", "monospace")],
).set(
    body_background_fill="#F5F7FA", block_background_fill="white", block_border_color="#E5E7EB",
    block_radius="12px", block_shadow="0 1px 2px rgba(16,24,40,0.05)",
    block_label_background_fill="white", block_label_text_color="#374151", block_label_border_color="#E5E7EB",
    block_title_text_color="#374151", body_text_color="#1F2937", body_text_color_subdued="#6B7280",
    button_primary_background_fill="#1D4ED8", button_primary_background_fill_hover="#1E40AF",
    button_primary_text_color="white", button_secondary_background_fill="#EEF2FF",
    button_secondary_text_color="#1E3A8A",
    # dark-mode counterparts
    body_background_fill_dark="#0B1220", block_background_fill_dark="#111827", block_border_color_dark="#1F2937",
    block_label_background_fill_dark="#111827", block_label_text_color_dark="#D1D5DB",
    block_label_border_color_dark="#1F2937", block_title_text_color_dark="#D1D5DB",
    body_text_color_dark="#E5E7EB", body_text_color_subdued_dark="#9CA3AF",
    button_primary_background_fill_dark="#2563EB", button_primary_background_fill_hover_dark="#1D4ED8",
    button_secondary_background_fill_dark="#1E293B", button_secondary_text_color_dark="#BFDBFE",
)
CSS = """
.gradio-container {max-width: 1500px !important; margin: auto}
#lulc-header {background: linear-gradient(90deg, #0F2A4A 0%, #1D4ED8 100%); color: white; border-radius: 14px;
  padding: 18px 24px; margin-bottom: 6px}
#lulc-header h1 {color: white; margin: 0 0 4px 0; font-size: 26px; font-weight: 700; letter-spacing: -0.01em}
#lulc-header p {color: #DBEAFE; margin: 0; font-size: 15px}
#lulc-header b {color: white}
.lulc-cards {display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 8px 0}
.lulc-card {flex: 1; min-width: 160px; background: white; border: 1px solid #E5E7EB; border-left: 4px solid #1D4ED8;
  border-radius: 10px; padding: 10px 14px; box-shadow: 0 1px 2px rgba(16,24,40,0.05)}
.lulc-card-t {font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: .04em}
.lulc-card-v {font-size: 22px; font-weight: 700; color: #111827; margin-top: 2px}
.lulc-card-s {font-size: 12px; color: #6B7280}
.tab-nav button, button[role="tab"] {font-weight: 600; font-size: 15px}
.lulc-note {background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 10px 14px}
.lulc-info {background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 10px 14px}
footer {display: none !important}
#lulc-top {align-items: stretch}
#lulc-theme-btn {min-width: 150px; max-width: 170px; align-self: center}
/* dark mode */
.dark #lulc-header {background: linear-gradient(90deg, #0B1B33 0%, #1E3A8A 100%)}
.dark .lulc-card {background: #111827; border-color: #1F2937; border-left-color: #3B82F6}
.dark .lulc-card-v {color: #F9FAFB}
.dark .lulc-card-t, .dark .lulc-card-s {color: #9CA3AF}
.dark .lulc-note {background: #2A2108; border-color: #6B5513; color: #FDE68A}
.dark .lulc-info {background: #0C1E3A; border-color: #1E3A8A; color: #BFDBFE}
.dark .lulc-legend > span {color: #E5E7EB !important}
"""
# Theme: light by default; the toggle button switches instantly (no reload, results stay) and the choice is
# remembered in the browser. "?__theme=dark" or "?__theme=light" in the URL overrides it.
APPLY_THEME = """() => { let t = 'light';
  try { t = localStorage.getItem('lulc-theme') || 'light'; } catch (e) {}
  const u = new URLSearchParams(window.location.search).get('__theme'); if (u) t = u;
  document.body.classList.toggle('dark', t === 'dark'); }"""
TOGGLE_THEME = """() => { const d = document.body.classList.toggle('dark');
  try { localStorage.setItem('lulc-theme', d ? 'dark' : 'light'); } catch (e) {} }"""

with gr.Blocks(title="Sentinel-1 Land Cover Monitor") as demo:
    state = gr.State(None)
    with gr.Row(elem_id="lulc-top"):
        gr.HTML("<div id='lulc-header'><h1>Sentinel-1 Land Cover Monitor</h1><p>Monthly land use / land cover from "
                "Sentinel-1 radar only &middot; 7 classes &middot; 10 m pixels &middot; pick a preset city, type a "
                "place name or enter coordinates, choose a start month and press <b>Run</b>.</p></div>", scale=12)
        btn_theme = gr.Button("Light / Dark theme", elem_id="lulc-theme-btn", size="sm", variant="secondary",
                              scale=1)
    with gr.Row():
        # ------------------------------------------------------------------ inputs
        with gr.Column(scale=1, min_width=330):
            preset = gr.Dropdown(label="Preset city (test regions, not used in training)", choices=list(PRESETS),
                                 value=None)
            with gr.Row():
                place = gr.Textbox(label="Place name", placeholder="e.g. Lagos", scale=4)
                btn_find = gr.Button("Find", scale=1, min_width=60)
            alts = gr.Dropdown(label="Other matches", choices=[], visible=False)
            with gr.Row():
                lat = gr.Number(label="Latitude", precision=5)
                lon = gr.Number(label="Longitude", precision=5)
            with gr.Row():
                year = gr.Dropdown(label="Start year", choices=YEARS, value=2021)
                month = gr.Dropdown(label="Start month", choices=[(m, i + 1) for i, m in enumerate(MONTHS)], value=1)
            size = gr.Radio(label="Area", choices=SIZES, value=512)
            model = gr.Dropdown(label="Model", choices=MODEL_CHOICES,
                                value="final" if "final" in MODEL_KEYS else (MODEL_KEYS[0] if MODEL_KEYS else None))
            btn_run = gr.Button("Run", variant="primary")
            gr.Markdown("A run takes about 1-3 minutes (Sentinel-1 download + 12 model passes + reference map). "
                        "2021 is the training year; other years are not validated.")
            loc_map = gr.HTML(viz.location_map(None, None))
        # ------------------------------------------------------------------ outputs
        with gr.Column(scale=2):
            summary = gr.Markdown("Results appear here after **Run**.")
            cards = gr.HTML()
            with gr.Tab("Monthly maps"):
                grid_img = gr.Image(label="LULC map over 12 months", show_label=False, interactive=False)
                gr.HTML(viz.legend_html())
                mslider = gr.Slider(1, 12, value=1, step=1, label="Inspect a month (1 = start month)")
                with gr.Row():
                    im_model = gr.Image(label="Model prediction", interactive=False, height=300)
                    im_dw = gr.Image(label="Dynamic World (independent reference)", interactive=False, height=300)
                    im_radar = gr.Image(label="Sentinel-1 radar (R=VV, G=VH, B=VV-VH)", interactive=False, height=300)
                gr.Markdown("**12-month animation**")
                gif = gr.HTML()
            with gr.Tab("Class shares"):
                fig_shares = gr.Plot()
                tbl_shares = gr.Dataframe(label="% of valid area per class, per month (+ valid pixels, mean "
                                                "confidence, agreement with WorldCover)", wrap=True)
            with gr.Tab("Change"):
                gr.Markdown(SEASON_NOTE, elem_classes=["lulc-note"])
                fig_m2m = gr.Plot()
                tbl_m2m = gr.Dataframe(label="Month-to-month change (percentage points). CHANGE? = above the 95% "
                                             "noise level", wrap=True)
                fig_fl = gr.Plot()
                tbl_fl = gr.Dataframe(label="First-to-last change (percentage points) with noise threshold", wrap=True)
                with gr.Row():
                    im_first = gr.Image(label="First month", interactive=False, height=280)
                    im_last = gr.Image(label="Last month", interactive=False, height=280)
                    im_changed = gr.Image(label="Changed pixels (red)", interactive=False, height=280)
                tbl_trans = gr.Dataframe(label="Transition matrix, first to last month (pixels)", wrap=True)
            with gr.Tab("Sanity check"):
                gr.Markdown(SANITY_NOTE, elem_classes=["lulc-info"])
                fig_dw_bar = gr.Plot()
                fig_dw_line = gr.Plot()
                fig_dw_cm = gr.Plot()
                tbl_dw = gr.Dataframe(label="Per month: Sentinel-2 scenes, share of pixels with a reference, agreement",
                                      wrap=True)
            with gr.Tab("Downloads"):
                files = gr.File(label="All results: zip, CSV tables, PNG figure, GIF animation", file_count="multiple")
            with gr.Tab("Model evaluation"):
                gr.Markdown(EVAL["md"])
                gr.Plot(EVAL["per_class"])
                gr.Dataframe(EVAL["per_class_tbl"], label="Test IoU and F1 per class")
                gr.Plot(EVAL["cm"])
                gr.Markdown("### Model variants")
                gr.Plot(EVAL["models"])
                gr.Plot(EVAL["ensemble"])
                gr.Dataframe(EVAL["bootstrap"], label="Paired cell-bootstrap comparisons (mIoU difference, 95% CI)")
                gr.Markdown("### Training curves")
                run_pick = gr.Dropdown(label="Run", choices=evaluation.history_runs(), value="segformer_b2_t3_s42")
                fig_hist = gr.Plot(evaluation.training_curves("segformer_b2_t3_s42"))
                gr.Markdown("### Calibration (temperature scaling, fitted on validation half A, checked on half B)")
                gr.Dataframe(EVAL["calibration"], label="Temperature and calibration error before/after")
                gr.Image(EVAL["reliability_png"], label="Reliability diagram", interactive=False)
                gr.Markdown("### Where the model is weaker (test set)")
                gr.Plot(EVAL["terrain"])
                gr.Plot(EVAL["lat"])
                gr.Plot(EVAL["biome"])

    OUTPUTS = [state, summary, cards, grid_img, gif, mslider, im_model, im_dw, im_radar,
               fig_shares, tbl_shares, fig_m2m, tbl_m2m, fig_fl, tbl_fl, im_first, im_last, im_changed, tbl_trans,
               fig_dw_bar, fig_dw_line, fig_dw_cm, tbl_dw, files]

    preset.input(pick_preset, [preset, size], [place, lat, lon, alts, loc_map])
    btn_find.click(find_place, [place, size], [place, lat, lon, alts, loc_map, preset])
    place.submit(find_place, [place, size], [place, lat, lon, alts, loc_map, preset])
    alts.input(pick_alternative, [alts, size], [place, lat, lon, loc_map])
    for c in (lat, lon):
        c.submit(coords_entered, [lat, lon, size], [place, alts, loc_map, preset])
        c.blur(coords_entered, [lat, lon, size], [place, alts, loc_map, preset])
    size.change(preview_map, [lat, lon, size], loc_map)
    btn_run.click(run, [place, lat, lon, year, month, size, model], OUTPUTS)
    mslider.change(inspect, [state, mslider], [im_model, im_dw, im_radar])
    run_pick.change(evaluation.training_curves, run_pick, fig_hist)
    btn_theme.click(fn=None, js=TOGGLE_THEME)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=THEME, css=CSS, js=APPLY_THEME)
