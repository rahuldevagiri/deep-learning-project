"""Preprocessing + model inference - IDENTICAL preprocessing to training (notebook 06, K2.1)."""
from functools import lru_cache

import numpy as np
import torch

from .config import MODELS, NC, NODATA, PRE, TILE, WEIGHTS_DIR

torch.set_num_threads(max(1, torch.get_num_threads()))


def preprocess(x2):
    """x2: (2, H, W) int16 dB*100 -> (3, H, W) float32 normalised [VV, VH, VV-VH], nodata mask (H, W)."""
    nod = (x2[0] == NODATA) | (x2[1] == NODATA)
    vv = np.clip(x2[0] / PRE["scale"], *PRE["clip"]["VV"])
    vh = np.clip(x2[1] / PRE["scale"], *PRE["clip"]["VH"])
    img = np.stack([vv, vh, vv - vh]).astype(np.float32)
    img = (img - np.array(PRE["mean"], np.float32)[:, None, None]) / np.array(PRE["std"], np.float32)[:, None, None]
    img[:, nod] = 0.0
    return img, nod


@lru_cache(maxsize=8)
def load_member(run, arch, in_ch):
    import segmentation_models_pytorch as smp
    if arch == "segformer":
        m = smp.Segformer("mit_b2", encoder_weights=None, in_channels=in_ch, classes=NC)
    else:
        m = smp.DeepLabV3Plus("resnet34", encoder_weights=None, in_channels=in_ch, classes=NC)
    sd = torch.load(WEIGHTS_DIR / run / "best.pth", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    m.load_state_dict(sd)                 # strict: fails loudly if architecture differs
    return m.eval()


def window_probs(x, members, temperature, stride=TILE // 2, batch=8):
    """x: (C, H, W) -> probabilities (NC, H, W). Overlapping 256-px windows (stride 128) blended with a
    tapered weight (no tile seams); temperature applied AFTER blending the member-mean probabilities.
    A 256-px input is exactly one window = the M5 test setting."""
    H, W = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - TILE + 1, stride)) + [n - TILE]))
    tiles = [(r, c) for r in starts(H) for c in starts(W)]
    h = 0.1 + 0.9 * np.sin(np.pi * (np.arange(TILE) + 0.5) / TILE)
    w = np.outer(h, h).astype(np.float32)
    acc = np.zeros((NC, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            part = tiles[i:i + batch]
            xb = torch.from_numpy(np.stack([x[:, r:r + TILE, c:c + TILE] for r, c in part]))
            pbar = (sum(m(xb).softmax(1) for m in members) / len(members)).numpy()
            for k, (r, c) in enumerate(part):
                acc[:, r:r + TILE, c:c + TILE] += pbar[k] * w
                wsum[r:r + TILE, c:c + TILE] += w
    p = torch.from_numpy(acc / wsum)
    if temperature != 1.0:
        p = torch.softmax(torch.log(p.clamp_min(1e-8)) / temperature, 0)
    return p.numpy()


def predict_12(s1, model_key, progress=None):
    """s1: (14, 2, H, W) -> pred (12, H, W) uint8 (255 = nodata), conf (12, H, W) float32, substituted months.
    Temporal models use months m-1, m, m+1; a neighbour month with NO data repeats the centre month (as in
    training). Single-month models use month m only."""
    spec = MODELS[model_key]
    in_ch = 9 if spec["temporal"] else 3
    members = [load_member(r, spec["arch"], in_ch) for r in spec["members"]]
    pre = [preprocess(s1[i]) for i in range(14)]
    empty = [bool(p[1].all()) for p in pre]
    preds, confs, subst = [], [], []
    for m in range(1, 13):
        if progress:
            progress(m - 1, 12)
        if spec["temporal"]:
            prev = pre[m][0] if empty[m - 1] else pre[m - 1][0]
            nxt = pre[m][0] if empty[m + 1] else pre[m + 1][0]
            if empty[m - 1] or empty[m + 1]:
                subst.append(m - 1)
            x = np.concatenate([prev, pre[m][0], nxt])
        else:
            x = pre[m][0]
        p = window_probs(x, members, spec["temperature"])
        pred, conf = p.argmax(0).astype(np.uint8), p.max(0).astype(np.float32)
        nod = pre[m][1]
        pred[nod], conf[nod] = 255, np.nan
        preds.append(pred)
        confs.append(conf)
    return np.stack(preds), np.stack(confs), subst
