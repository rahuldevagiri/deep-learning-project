"""Static model-evaluation charts from the saved result files (eval_data/). Nothing is re-computed on test data:
the test confusion matrix and tables were produced once by notebook 09 (M5)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import CLASSES, COLORS, ROOT
from .viz import confusion_heatmap

D = ROOT / "eval_data"
CONFIGS = [("segformer_b2_t3", "SegFormer-B2 temporal"), ("deeplabv3p_r34_t3", "DeepLabV3+ R34 temporal"),
           ("segformer_b2", "SegFormer-B2 single month"), ("deeplabv3p_r34", "DeepLabV3+ R34 single month"),
           ("deeplabv3p_r34_scratch", "DeepLabV3+ R34 no pre-training")]


def test_metrics():
    cm = np.load(D / "M5_confusion_all.npy").astype(float)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou, f1 = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
    res = pd.read_csv(D / "M5_results.csv")
    ens = res[(res.subset == "all_test") & (res.model == "ensemble_T")].iloc[0]
    per_class = pd.DataFrame({"IoU": iou.round(3), "F1": f1.round(3)}, index=CLASSES)
    return dict(cm=cm, OA=tp.sum() / cm.sum(), mIoU=iou.mean(), F1=f1.mean(),
                ci=(ens.CI_low, ens.CI_high), per_class=per_class, results=res)


def per_class_chart(pc):
    fig = go.Figure()
    fig.add_bar(x=CLASSES, y=pc["IoU"], name="IoU", marker_color="#2563EB", text=pc["IoU"], textposition="outside")
    fig.add_bar(x=CLASSES, y=pc["F1"], name="F1", marker_color="#F59E0B", text=pc["F1"], textposition="outside")
    fig.update_layout(barmode="group", title="Test set: IoU and F1 per class (final model)", yaxis=dict(range=[0, 1]),
                      height=380, margin=dict(l=40, r=10, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    return fig


def model_comparison_chart():
    names, means, sds, ns = [], [], [], []
    for key, label in CONFIGS:
        s = pd.read_csv(D / f"summary_{key}.csv")
        names.append(label)
        means.append(s.val_mIoU.mean())
        sds.append(s.val_mIoU.std() if len(s) > 1 else 0)
        ns.append(len(s))
    fig = go.Figure(go.Bar(x=names, y=means, error_y=dict(type="data", array=sds), marker_color="#2563EB",
                           text=[f"{m:.3f} (n={n})" for m, n in zip(means, ns)], textposition="outside"))
    fig.update_layout(title="Validation mIoU of the model variants (mean +/- sd over seeds)",
                      yaxis=dict(title="val mIoU", range=[0.4, 0.6]), height=400, margin=dict(l=40, r=10, t=50, b=80))
    return fig


def ensemble_chart(res):
    r = res[(res.subset == "all_test") & res.model.isin(["seed42", "seed43", "seed44", "ensemble_T"])]
    fig = go.Figure(go.Bar(x=r.model, y=r.mIoU, marker_color=["#93C5FD"] * 3 + ["#1D4ED8"],
                           error_y=dict(type="data", symmetric=False, array=r.CI_high - r.mIoU,
                                        arrayminus=r.mIoU - r.CI_low),
                           text=r.mIoU.round(3), textposition="outside"))
    fig.update_layout(title="Test mIoU: single seeds vs calibrated 3-seed ensemble (95% CI, cell bootstrap)",
                      yaxis=dict(range=[0.4, 0.7]), height=380, margin=dict(l=40, r=10, t=50, b=40))
    return fig


def history_runs():
    return sorted(p.stem for p in (D / "histories").glob("*.csv"))


def training_curves(run):
    h = pd.read_csv(D / "histories" / f"{run}.csv")
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "mIoU"))
    fig.add_scatter(x=h.epoch, y=h.train_loss, name="train loss", line=dict(color="#2563EB"), row=1, col=1)
    fig.add_scatter(x=h.epoch, y=h.val_loss, name="val loss", line=dict(color="#F59E0B"), row=1, col=1)
    fig.add_scatter(x=h.epoch, y=h.train_mIoU, name="train mIoU", line=dict(color="#2563EB", dash="dot"), row=1, col=2)
    fig.add_scatter(x=h.epoch, y=h.val_mIoU, name="val mIoU", line=dict(color="#F59E0B", dash="dot"), row=1, col=2)
    best = h.loc[h.val_mIoU.idxmax()]
    fig.add_vline(x=best.epoch, line=dict(color="grey", dash="dash"), row=1, col=2)
    fig.update_layout(title=f"Training curves - {run} (best epoch {int(best.epoch)}, val mIoU {best.val_mIoU:.3f})",
                      height=380, margin=dict(l=40, r=10, t=60, b=40), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="epoch")
    return fig


def group_chart(file, col, title, horizontal=False):
    g = pd.read_csv(D / file)
    lab = g[col] + " (n=" + g.n_locations.astype(str) + ")"
    if horizontal:
        g = g.assign(lab=lab).sort_values("mIoU_ensemble")
        fig = go.Figure(go.Bar(y=g.lab, x=g.mIoU_ensemble, orientation="h", marker_color="#2563EB",
                               text=g.mIoU_ensemble, textposition="outside"))
        fig.update_layout(xaxis=dict(range=[0, 0.75], title="test mIoU"), height=520, margin=dict(l=330, r=10, t=50, b=40))
    else:
        fig = go.Figure(go.Bar(x=lab, y=g.mIoU_ensemble, marker_color="#2563EB", text=g.mIoU_ensemble,
                               textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 0.75], title="test mIoU"), height=340, margin=dict(l=40, r=10, t=50, b=40))
    fig.update_layout(title=title)
    return fig


def build():
    """All static evaluation outputs (computed once at app start)."""
    t = test_metrics()
    lo, hi = t["ci"]
    md = (f"### Final model on the held-out test set\n"
          f"Overall accuracy **{t['OA']:.3f}** | mIoU **{t['mIoU']:.3f}** (95% CI {lo:.3f}-{hi:.3f}) | "
          f"mean F1 **{t['F1']:.3f}**  \n"
          "5,854 samples at 489 locations in 89 five-degree cells never used for training or validation "
          "(geographic split). Test set evaluated once. Reference labels: ESA WorldCover 2021 (~77% accurate itself). "
          "Trivial baseline (always the majority class): mIoU 0.037.")
    cal = pd.read_csv(D / "M45_calibration.csv").round(4)
    boot = pd.concat([pd.read_csv(D / "M45_bootstrap.csv"), pd.read_csv(D / "M4_scratch_vs_imagenet_bootstrap.csv")])
    return dict(
        md=md, per_class=per_class_chart(t["per_class"]), per_class_tbl=t["per_class"].reset_index(names="class"),
        cm=confusion_heatmap(t["cm"], "Test confusion matrix (row-normalised)", "WorldCover 2021 (reference)",
                             "model prediction"),
        models=model_comparison_chart(), ensemble=ensemble_chart(t["results"]),
        terrain=group_chart("M5_by_terrain.csv", "terrain", "Test mIoU by terrain slope (degrees)"),
        lat=group_chart("M5_by_lat_zone.csv", "lat_zone", "Test mIoU by latitude zone"),
        biome=group_chart("M5_by_biome_name.csv", "biome_name", "Test mIoU by biome", horizontal=True),
        calibration=cal, bootstrap=boot.round(4), reliability_png=str(D / "M45_reliability.png"))
