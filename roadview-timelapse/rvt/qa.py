"""Review sheets for the stills (labels live only on QA sheets, never in the video)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .util import imwrite


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 18 + 15 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(images: list[np.ndarray], labels: list[str], path: str | Path, cell_w: int = 640, cols: int = 2) -> None:
    cells = []
    for img, lab in zip(images, labels):
        h, w = img.shape[:2]
        c = cv2.resize(img, (cell_w, round(h * cell_w / w)), interpolation=cv2.INTER_AREA)
        cells.append(_label(c, lab))
    ch = max(c.shape[0] for c in cells)
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        row += [np.zeros((ch, cell_w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack([np.pad(c, ((0, ch - c.shape[0]), (0, 0), (0, 0))) for c in row]))
    imwrite(path, np.vstack(rows), quality=90)


def grid_overlay(img: np.ndarray, path: str | Path, step: float = 0.1) -> None:
    """Normalised-coordinate grid, to read off sign boxes for project.json."""
    out = img.copy()
    h, w = out.shape[:2]
    for k in range(1, int(1 / step)):
        x, y = int(k * step * w), int(k * step * h)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(out, f"{k * step:.1f}", (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(out, f"{k * step:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    imwrite(path, out, quality=90)


def edge_overlay(img: np.ndarray, ref: np.ndarray, path: str | Path) -> None:
    """Reference edges (magenta) over an aligned capture: misalignment shows as doubled lines."""
    g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 60, 160) > 0
    out = (img.astype(np.float32) * 0.75).astype(np.uint8)
    out[e] = (255, 0, 255)
    imwrite(path, out, quality=90)


def residual_flow(img: np.ndarray, ref: np.ndarray, width: int = 960) -> dict:
    """Dense optical flow between an aligned still and the reference.

    On unchanged structure the flow should be ~0; the median over textured
    pixels is the remaining misalignment (in output pixels of ``img``)."""
    s = width / img.shape[1]
    a = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(ref, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1]) / s
    grad = cv2.Sobel(b, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(b, cv2.CV_32F, 0, 1) ** 2
    textured = grad > np.percentile(grad, 70)
    m = mag[textured]
    return {"median_px": round(float(np.median(m)), 3), "p90_px": round(float(np.percentile(m, 90)), 3)}


def blink_gif(images: list[np.ndarray], labels: list[str], path: str | Path, width: int = 960, ms: int = 700) -> None:
    frames = []
    for img, lab in zip(images, labels):
        h, w = img.shape[:2]
        c = _label(cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA), lab)
        frames.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
