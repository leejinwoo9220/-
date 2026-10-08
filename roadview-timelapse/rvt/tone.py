"""Second pass: gentle tone normalisation toward the reference capture.

Only brightness/contrast is pulled toward the reference by default, and only
partially, so each year keeps its own camera colour (part of the era look).
"""
from __future__ import annotations

import cv2
import numpy as np


def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_vals, s_idx, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_cdf = np.cumsum(s_cnt) / src.size
    r_cdf = np.cumsum(r_cnt) / ref.size
    mapped = np.interp(s_cdf, r_cdf, r_vals)
    return mapped[s_idx].reshape(src.shape)


def normalize(img: np.ndarray, ref: np.ndarray, luma: float = 0.35, color: float = 0.0) -> np.ndarray:
    if luma <= 0 and color <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    rlab = cv2.cvtColor(cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2LAB).astype(np.float32)
    if luma > 0:
        L = lab[..., 0]
        lab[..., 0] = (1 - luma) * L + luma * _match_channel(L.astype(np.uint8), rlab[..., 0].astype(np.uint8))
    if color > 0:
        for c in (1, 2):
            ch = lab[..., c]
            m, s = ch.mean(), ch.std() + 1e-6
            rm, rs = rlab[..., c].mean(), rlab[..., c].std() + 1e-6
            lab[..., c] = (1 - color) * ch + color * ((ch - m) * (rs / s) + rm)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
