"""Images, maps and charts for the UI."""
from html import escape

import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import colormaps

from .config import CLASSES, COLORS, NODATA, PRE, RES

matplotlib.use("Agg")
_RGB = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in COLORS], np.uint8)
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"

# one clean, light chart style for every Plotly figure
_t = go.layout.Template(pio.templates["plotly_white"])
_t.layout.font = dict(family=FONT, size=13, color=INK)
_t.layout.title = dict(font=dict(size=16, color=INK), x=0.01, xanchor="left")
_t.layout.paper_bgcolor = "white"
_t.layout.plot_bgcolor = "white"
_t.layout.xaxis = dict(gridcolor=GRID, linecolor="#D1D5DB", zerolinecolor="#9CA3AF")
_t.layout.yaxis = dict(gridcolor=GRID, linecolor="#D1D5DB", zerolinecolor="#9CA3AF")
_t.layout.hoverlabel = dict(font=dict(family=FONT))
pio.templates["lulc"] = _t
pio.templates.default = "lulc"
matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
                            "text.color": INK, "axes.titlecolor": INK})


def colorize(pred, alpha=False):
    """Class map (H, W) -> RGB(A) uint8; nodata (255) white / transparent."""
    out = np.full(pred.shape + (4 if alpha else 3,), 255, np.uint8)
    v = pred != 255
    out[v, :3] = _RGB[pred[v]]
    if alpha:
        out[~v, 3] = 0
    return out


def radar_image(x2):
    """Sentinel-1 composite (2, H, W) int16 dB*100 -> false colour RGB: R=VV, G=VH, B=VV-VH."""
    nod = (x2[0] == NODATA) | (x2[1] == NODATA)
    vv = np.clip(x2[0] / PRE["scale"], -25, 0)
    vh = np.clip(x2[1] / PRE["scale"], -30, -5)
    r = (vv + 25) / 25
    g = (vh + 30) / 25
    b = np.clip((vv - vh) / 15, 0, 1)
    img = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    img[nod] = 0
    return img


def conf_image(conf):
    c = np.nan_to_num((conf - 0.3) / 0.7, nan=0.0).clip(0, 1)
    img = (colormaps["viridis"](c)[..., :3] * 255).astype(np.uint8)
    img[np.isnan(conf)] = 255
    return img


def disagreement_image(pa, pb):
    """Grey where two models agree, magenta where they disagree, white = nodata."""
    out = np.full(pa.shape + (3,), 255, np.uint8)
    v = (pa != 255) & (pb != 255)
    out[v & (pa == pb)] = (200, 200, 200)
    out[v & (pa != pb)] = (220, 0, 160)
    return out


def legend_html():
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:3px 16px 3px 0;color:{INK}">'
        f'<span style="width:16px;height:16px;border-radius:4px;background:{c};border:1px solid #9CA3AF;'
        f'margin-right:7px"></span>{n}</span>'
        for n, c in zip(CLASSES, COLORS))
    return f'<div class="lulc-legend" style="font-size:14px;line-height:1.9">{items}</div>'


def location_map(lat, lon, bounds=None, overlay=None, opacity=0.65, zoom=None, size=512):
    """Folium map (OSM + satellite basemap) with the analysis square; optional LULC overlay (RGBA array).
    Returned as HTML (iframe) for gr.HTML."""
    import folium
    if lat is None or lon is None:
        lat, lon, zoom = 20.0, 0.0, 2
    zoom = zoom or {256: 14, 512: 13, 768: 12}.get(int(size), 13)
    m = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Map (OpenStreetMap)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics", name="Satellite (Esri)",
        show=False).add_to(m)
    if overlay is not None and bounds is not None:
        folium.raster_layers.ImageOverlay(image=overlay, bounds=bounds, opacity=opacity,
                                          name="Predicted land cover", interactive=False).add_to(m)
    if bounds is not None:
        folium.Rectangle(bounds, color="#ff3333", weight=2, fill=False, tooltip="analysis area").add_to(m)
    if lat is not None and zoom != 2:
        folium.Marker([lat, lon], tooltip=f"{lat:.4f}, {lon:.4f}").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    html = escape(m.get_root().render(), quote=True)          # full page inside the iframe's srcdoc
    return (f'<iframe srcdoc="{html}" style="width:100%;height:420px;border:1px solid #ccc;border-radius:8px" '
            f'loading="lazy"></iframe>')


def shares_chart(P, title):
    fig = go.Figure()
    for c, col in zip(CLASSES, COLORS):
        fig.add_trace(go.Scatter(x=P.index, y=P[c], name=c, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=col), fillcolor=col,
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + c + "</extra>"))
    low = P.index[P["reliable"] == "LOW"].tolist()
    for mo in low:
        fig.add_vrect(x0=mo, x1=mo, line=dict(color="black", dash="dot", width=1))
    fig.update_layout(title=title + ("  (dotted = low-confidence month)" if low else ""),
                      yaxis=dict(title="% of valid area", range=[0, 100]), height=380,
                      margin=dict(l=40, r=10, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    return fig


def change_heatmap(dP, flags, title):
    z = dP.T.values
    text = [[f"{z[i][j]:+.1f}" + ("*" if flags.T.values[i][j].startswith("CHANGE") else
                                  " ~" if flags.T.values[i][j] == "(low conf)" else "")
             for j in range(z.shape[1])] for i in range(z.shape[0])]
    lim = max(5.0, float(np.nanmax(np.abs(z))) if z.size else 5.0)
    fig = go.Figure(go.Heatmap(z=z, x=dP.index, y=CLASSES, text=text, texttemplate="%{text}",
                               colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=dict(title="pp")))
    fig.update_layout(title=title + "<br><sup>* above M5 95% pseudo-change level | ~ large but low confidence</sup>",
                      height=380, margin=dict(l=90, r=10, t=70, b=40), yaxis=dict(autorange="reversed"))
    return fig


def compare_chart(Pa, Pb, la, lb):
    fig = go.Figure()
    fig.add_bar(x=CLASSES, y=Pa[CLASSES].mean(), name=la, marker_color="#1f77b4")
    fig.add_bar(x=CLASSES, y=Pb[CLASSES].mean(), name=lb, marker_color="#ff7f0e")
    fig.update_layout(barmode="group", title="Mean class share over the 12 months", height=360,
                      yaxis_title="% of valid area", margin=dict(l=40, r=10, t=50, b=40),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def month_grid(pred, conf, lab, P, months, title):
    """One figure: 12 monthly maps + WorldCover reference + mean confidence + legend -> RGB array."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(3, 5, figsize=(17, 11.2), facecolor="white")

    def frame(a):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#D1D5DB")

    for m in range(12):
        a = ax.ravel()[m]
        a.imshow(np.ma.masked_equal(pred[m], 255), cmap=cmap, vmin=0, vmax=len(CLASSES) - 1,
                 interpolation="nearest")
        low = P["reliable"].iloc[m] == "LOW"
        valid = P["valid_%"].iloc[m]
        sub = ("low confidence" if low else "") + (f"  valid {valid:.0f}%" if valid < 99.5 else "")
        a.set_title(months[m] + (f"\n{sub.strip()}" if sub.strip() else ""), fontsize=13,
                    color="#B91C1C" if low else INK, fontweight="semibold", pad=6)
        frame(a)
    a = ax.ravel()[12]
    a.imshow(np.ma.masked_equal(lab, 255), cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, interpolation="nearest")
    a.set_title("ESA WorldCover 2021\n(training-label source)", fontsize=12, color=MUTED, pad=6)
    frame(a)
    a = ax.ravel()[13]
    im = a.imshow(np.nanmean(conf, 0), cmap="viridis", vmin=0.3, vmax=1)
    a.set_title("Mean confidence\n(yellow = high)", fontsize=12, color=MUTED, pad=6)
    frame(a)
    fig.colorbar(im, ax=a, fraction=0.046)
    a = ax.ravel()[14]
    a.axis("off")
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#9CA3AF") for c in COLORS],
             labels=CLASSES, loc="center", fontsize=13, frameon=False, handlelength=1.6, handleheight=1.2,
             labelspacing=0.7)
    fig.suptitle(title, fontsize=17, fontweight="semibold", color=INK)
    fig.tight_layout(h_pad=1.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def lines_chart(P, title):
    fig = go.Figure()
    for c, col in zip(CLASSES, COLORS):
        fig.add_trace(go.Scatter(x=P.index, y=P[c], name=c, mode="lines+markers",
                                 line=dict(color=col, width=3), marker=dict(size=7, line=dict(color="white", width=1)),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + c + "</extra>"))
    fig.update_layout(title=title, yaxis=dict(title="% of valid area", range=[0, 100]), height=420,
                      margin=dict(l=50, r=10, t=50, b=40), legend=dict(orientation="h", y=-0.22))
    return fig


def _short(label):
    """'February 2021' -> 'Feb'."""
    return label[:3]


def upscale(img, target=384):
    """Nearest-neighbour upscale of an (H, W, 3) image so small maps stay crisp in the UI."""
    k = max(1, target // img.shape[0])
    return np.repeat(np.repeat(img, k, 0), k, 1)


def change_bars(dP, flags, thr, title, P=None):
    """Small multiples, one panel per class present in the area: month-to-month change (pp) as bars in the class
    colour. Dashed lines = +/- 95% noise level from the test set (shown when it fits the scale). Bars above the
    noise level are drawn solid with a dark outline; bars inside it are pale."""
    from plotly.subplots import make_subplots
    present = [c for c in CLASSES if P is None or P[c].max() >= 0.5] or CLASSES
    cols = 4 if len(present) > 3 else len(present)
    rows = int(np.ceil(len(present) / cols))
    x = [_short(v) for v in dP.index]
    titles = [f"<b>{c}</b>  <span style='color:{MUTED}'>noise +/-{thr[c]:.1f} pp</span>" for c in present]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.06,
                        vertical_spacing=0.22 if rows > 1 else 0.1)
    for i, c in enumerate(present):
        r, k = i // cols + 1, i % cols + 1
        col = COLORS[CLASSES.index(c)]
        big = flags[c].str.startswith("CHANGE").values
        y = dP[c].values
        lim = max(1.3 * float(np.nanmax(np.abs(y))) if len(y) else 0, 0.5)
        if thr[c] <= 2.5 * lim:                        # noise level fits the scale -> draw it
            lim = max(lim, 1.15 * thr[c])
            for s in (1, -1):
                fig.add_trace(go.Scatter(x=x, y=[s * thr[c]] * len(x), mode="lines", hoverinfo="skip",
                                         line=dict(color="#6B7280", dash="dash", width=1.5), showlegend=False), r, k)
        fig.add_trace(go.Bar(x=x, y=y, showlegend=False,
                             marker=dict(color=col, opacity=[1.0 if b else 0.55 for b in big],
                                         line=dict(color=INK, width=[2 if b else 0 for b in big])),
                             customdata=list(dP.index),
                             hovertemplate="%{customdata}: %{y:+.2f} pp<extra>" + c + "</extra>"), r, k)
        fig.update_yaxes(range=[-lim, lim], row=r, col=k)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=11))
    fig.update_layout(title=title + f"<br><sup style='color:{MUTED}'>Bars = change in class share vs previous month. "
                                    "Dashed lines = 95% noise level (seasonality + model noise). Solid outlined bar = "
                                    "larger than noise (possible real change).</sup>",
                      height=260 * rows + 110, margin=dict(l=50, r=10, t=110, b=30), bargap=0.25)
    fig.update_annotations(font_size=13)
    return fig


def first_last_bars(fl, title):
    """Horizontal bars: first-to-last change per class (pp) with the class noise level marked."""
    d, t = fl["dP_pp"].values, fl["threshold_pp"].values
    lim = max(1.4 * float(np.abs(d).max()), 0.5)
    show_t = t <= 2.5 * lim
    lim = max(lim, 1.15 * float(t[show_t].max()) if show_t.any() else lim)
    ylab = [f"<b>{c}</b>  <span style='color:{MUTED}'>(noise +/-{v:.1f})</span>" for c, v in zip(CLASSES, t)]
    big = np.abs(d) > t
    fig = go.Figure(go.Bar(y=ylab, x=d, orientation="h",
                           marker=dict(color=COLORS, opacity=[1.0 if b else 0.6 for b in big],
                                       line=dict(color=INK, width=[2 if b else 0 for b in big])),
                           text=[f"{v:+.2f} pp" for v in d], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x:+.2f} pp<extra></extra>"))
    for i, (v, s) in enumerate(zip(t, show_t)):
        if s:
            for sign in (1, -1):
                fig.add_shape(type="line", x0=sign * v, x1=sign * v, y0=i - 0.4, y1=i + 0.4,
                              line=dict(color="#6B7280", dash="dash", width=1.5))
    fig.update_layout(title=title + f"<br><sup style='color:{MUTED}'>Outlined bar = larger than the 95% noise "
                                    "level (possible real change); dashed marks = noise level when it fits the scale"
                                    "</sup>",
                      xaxis=dict(title="change in class share (percentage points)", range=[-lim, lim],
                                 zeroline=True, zerolinewidth=2),
                      yaxis=dict(autorange="reversed"), height=420, margin=dict(l=210, r=40, t=90, b=50))
    return fig


def changed_image(changed, valid):
    out = np.full(changed.shape + (3,), 255, np.uint8)
    out[valid] = (229, 231, 235)
    out[changed] = (220, 38, 38)
    return out


REF_COLORS = {"model": "#2563EB", "dw": "#059669", "wc": "#9CA3AF"}


def shares_compare_chart(shares, wc):
    fig = go.Figure()
    fig.add_bar(x=CLASSES, y=shares["model %"], name="Model (12-month mean)", marker_color=REF_COLORS["model"],
                text=shares["model %"].round(1), textposition="outside")
    fig.add_bar(x=CLASSES, y=shares["Dynamic World %"], name="Dynamic World (independent)",
                marker_color=REF_COLORS["dw"], text=shares["Dynamic World %"].round(1), textposition="outside")
    fig.add_bar(x=CLASSES, y=wc.values, name="WorldCover 2021 (label source)", marker_color=REF_COLORS["wc"])
    fig.update_layout(barmode="group", title="Class shares: model vs independent reference", height=420,
                      yaxis=dict(title="% of area", range=[0, 105]), margin=dict(l=50, r=10, t=50, b=40),
                      legend=dict(orientation="h", y=-0.18), bargap=0.25)
    return fig


def agreement_chart(per_month, P):
    fig = go.Figure()
    fig.add_scatter(x=per_month.index, y=per_month["agree_DW_%"], mode="lines+markers",
                    name="vs Dynamic World (independent)", line=dict(color=REF_COLORS["dw"], width=3),
                    marker=dict(size=8), connectgaps=False)
    fig.add_scatter(x=P.index, y=P["agree_WC_%"], mode="lines+markers", name="vs WorldCover 2021 (label source)",
                    line=dict(color=REF_COLORS["wc"], dash="dash", width=2), marker=dict(size=6))
    fig.update_layout(title="Pixel agreement per month", yaxis=dict(title="%", range=[0, 100]), height=360,
                      margin=dict(l=40, r=10, t=50, b=40), legend=dict(orientation="h", y=-0.25))
    return fig


def confusion_heatmap(cm, title, ylab, xlab):
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=rown * 100, x=CLASSES, y=CLASSES, colorscale="Blues", zmin=0, zmax=100,
                               text=np.round(rown * 100, 1), texttemplate="%{text}", colorbar=dict(title="% of row")))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis=dict(title=ylab, autorange="reversed"), height=460,
                      margin=dict(l=100, r=10, t=50, b=60))
    return fig


def make_gif(pred, labels, path, size=384):
    """12-month animation of the predicted maps (one frame per month, month name on top)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    frames = []
    for m, lab in enumerate(labels):
        img = Image.fromarray(colorize(pred[m])).resize((size, size), Image.NEAREST)
        canvas = Image.new("RGB", (size, size + 34), "white")
        canvas.paste(img, (0, 34))
        ImageDraw.Draw(canvas).text((8, 6), lab, fill="black", font=font)
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=900, loop=0)
    return path


def gif_html(path):
    import base64
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/gif;base64,{b}" style="max-width:100%;border-radius:6px"/>'


def metric_cards(items):
    """items: list of (title, value, sub) -> HTML row of cards."""
    cards = "".join(
        f'<div class="lulc-card"><div class="lulc-card-t">{t}</div><div class="lulc-card-v">{v}</div>'
        f'<div class="lulc-card-s">{s}</div></div>' for t, v, s in items)
    return f'<div class="lulc-cards">{cards}</div>'


def area_km2(size):
    return (size * RES / 1000) ** 2


def fmt_table(df):
    return df.reset_index() if isinstance(df, pd.DataFrame) else df
