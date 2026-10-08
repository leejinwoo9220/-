"""Pixel locking: keep real capture pixels wherever nothing moves.

Video models redraw every pixel of every frame, which is why sign lettering
wobbles, melts or turns into pseudo-text.  The camera never moves in this
film, so every pixel that is not a passing person/car/shadow can be taken
straight from the aligned capture instead - at full capture resolution and
with the exact original lettering.

HOLD clips   moving-mask composite: generated pixels only where something
             moves (people, cars, their shadows), aligned capture elsewhere.
TRANSITIONS  sign boxes "pop-swap": a sign shows the exact old capture until
             the model's sign has turned into the new one, then the exact new
             capture (one abrupt jump, like a real time-lapse; no melting
             pseudo-text in between), with passing occluders let through.
             The first/last frames are blended into exact locks so every
             seam lands on the real stills.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BLUR_SIGMA = 1.2


def blurred(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)


def absdiff_max(a_blur: np.ndarray, b_blur: np.ndarray) -> np.ndarray:
    """Per-pixel max channel difference (uint8)."""
    d = cv2.absdiff(a_blur, b_blur)
    return cv2.max(cv2.max(d[..., 0], d[..., 1]), d[..., 2])


def box_mask(boxes: list[list[float]], w: int, h: int) -> np.ndarray:
    """Normalised [x, y, w, h] boxes -> uint8 mask."""
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        if not b:
            continue
        x, y, bw, bh = b
        m[int(y * h):int(np.ceil((y + bh) * h)), int(x * w):int(np.ceil((x + bw) * w))] = 255
    return m


def color_fit(gen: np.ndarray, plate: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Per-channel gain/offset that maps gen onto the plate on static pixels."""
    idx = np.flatnonzero(static.ravel())
    if idx.size < 500:
        return gen
    if idx.size > 60000:
        idx = idx[:: idx.size // 60000]
    out = np.empty_like(gen)
    g2, p2 = gen.reshape(-1, 3), plate.reshape(-1, 3)
    for c in range(3):
        x = g2[idx, c].astype(np.float64)
        y = p2[idx, c].astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        (gain, off), *_ = np.linalg.lstsq(A, y, rcond=None)
        gain = float(np.clip(gain, 0.8, 1.25))
        off = float(np.clip(off, -40, 40))
        out[..., c] = np.clip(gen[..., c].astype(np.float32) * gain + off, 0, 255).astype(np.uint8)
    return out


def stabilize(gen: np.ndarray, plate_gray: np.ndarray, max_shift: float = 8.0) -> tuple[np.ndarray, tuple[float, float]]:
    """Undo small whole-frame drift of the generated camera (phase correlation)."""
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(plate_gray, g, win)
    if np.hypot(dx, dy) < 0.25 or np.hypot(dx, dy) > max_shift:
        return gen, (0.0, 0.0)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(gen, M, (gen.shape[1], gen.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (float(dx), float(dy))


def clean_mask(raw: np.ndarray, min_area: int, dilate: int) -> np.ndarray:
    m = raw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab].astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    return m


def drop_inside(mask01: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Remove blobs that lie entirely inside ``inside`` (sign boxes).

    A change confined to a sign box is the model redrawing the lettering, not
    something passing in front of it; real occluders (people, buses,
    scaffolding) always continue outside the box."""
    if not inside.any() or not mask01.any():
        return mask01
    n, lab = cv2.connectedComponents(mask01.astype(np.uint8), connectivity=8)
    outside = np.bincount(lab[inside == 0].ravel(), minlength=n) > 0
    outside[0] = False
    return outside[lab].astype(np.uint8)


def moving_mask(gen: np.ndarray, plate_blur: np.ndarray, sign: np.ndarray, cfg: dict) -> np.ndarray:
    """Binary mask (uint8 0/1) of pixels where gen visibly differs from the plate."""
    d = absdiff_max(blurred(gen), plate_blur)
    thr = np.where(sign > 0, cfg["sign_threshold"], cfg["lock_threshold"]).astype(np.float32)
    h, w = d.shape
    raw = drop_inside((d > thr).astype(np.uint8), sign)
    return clean_mask(raw, int(cfg["min_blob_frac"] * w * h), int(cfg["mask_dilate"]))


def temporal_max(masks: list[np.ndarray], radius: int = 2) -> list[np.ndarray]:
    out = []
    for t in range(len(masks)):
        lo, hi = max(0, t - radius), min(len(masks), t + radius + 1)
        out.append(np.maximum.reduce(masks[lo:hi]))
    return out


def feather(mask01: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(mask01.astype(np.float32), (0, 0), sigma), 0, 1)


def composite(gen_up: np.ndarray, plate_up: np.ndarray, alpha_up: np.ndarray) -> np.ndarray:
    """alpha * gen + (1 - alpha) * plate."""
    a = alpha_up[..., None]
    return (gen_up.astype(np.float32) * a + plate_up.astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)


def composite_into(base: np.ndarray, top: np.ndarray, alpha: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    """In place: base = alpha * top + (1 - alpha) * base inside rect (x, y, w, h)."""
    x, y, w, h = rect
    sl = (slice(y, y + h), slice(x, x + w))
    a = alpha[sl][..., None]
    base[sl] = (top[sl].astype(np.float32) * a + base[sl].astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)


@dataclass
class HoldLock:
    """Result of locking one HOLD clip at canvas resolution."""
    frames: list[np.ndarray]  # colour-corrected, stabilised generated frames (canvas)
    alphas: list[np.ndarray]  # feathered moving masks (canvas, float 0..1)
    masks: list[np.ndarray]  # binary moving masks after temporal max
    plate: np.ndarray  # aligned capture (canvas)
    shifts: list[tuple[float, float]]

    def locked_canvas(self, t: int) -> np.ndarray:
        return composite(self.frames[t], self.plate, self.alphas[t])

    def motion_fraction(self) -> float:
        return float(np.mean([m.mean() for m in self.masks]))

    def background(self) -> np.ndarray:
        """Temporal median of the locked frames: the street without the passers-by."""
        stack = np.stack([self.locked_canvas(t) for t in range(0, len(self.frames), 2)])
        return np.median(stack, axis=0).astype(np.uint8)

    def subjects(self, t0: int, t1: int) -> np.ndarray:
        """Union of moving masks over [t0, t1): where people/cars are at that moment."""
        t0, t1 = max(0, t0), min(len(self.masks), t1)
        return np.maximum.reduce(self.masks[t0:t1]) if t1 > t0 else np.zeros_like(self.masks[0])


def lock_hold(frames: list[np.ndarray], plate: np.ndarray, sign: np.ndarray, cfg: dict) -> HoldLock:
    plate_blur = blurred(plate)
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fixed, shifts, raw_masks = [], [], []
    for f in frames:
        if f.shape[:2] != plate.shape[:2]:
            f = cv2.resize(f, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_AREA)
        if cfg.get("stabilize", True):
            f, sh = stabilize(f, plate_gray)
        else:
            sh = (0.0, 0.0)
        static0 = absdiff_max(blurred(f), plate_blur) < cfg["lock_threshold"] / 2
        f = color_fit(f, plate, static0)
        raw_masks.append(moving_mask(f, plate_blur, sign, cfg))
        fixed.append(f)
        shifts.append(sh)
    masks = temporal_max(raw_masks, 2)
    alphas = [feather(m, 2.0) for m in masks]
    return HoldLock(fixed, alphas, masks, plate, shifts)


# ------------------------------------------------------------- transitions

def sign_swap_frames(dist, boxes: list[list[float]]) -> list[dict]:
    """For each sign box, the frame where the model's sign has become the new one.

    d_old(t), d_new(t): mean blurred distance of the box to the start/end
    still.  The swap is the first frame where the box is closer to the new
    still and stays closer (avoids flicker).  Unchanged signs never swap."""
    h, w = dist.sb.shape[:2]
    out = []
    for i, b in enumerate(boxes):
        if not b:
            continue
        m = box_mask([b], w, h) > 0
        changed = float(dist.start_end[m].mean())
        d_old = np.array([float(d[m].mean()) for d in dist.ds])
        d_new = np.array([float(d[m].mean()) for d in dist.de])
        n = len(dist.ds)
        if changed < 6.0:
            t_swap = None  # same sign in both captures: lock it all the way
        else:
            closer = d_new < d_old
            t_swap = n - 1
            for t in range(n):
                if closer[t:].mean() > 0.8:
                    t_swap = t
                    break
            t_swap = int(np.clip(t_swap, 1, n - 1))
        out.append({"index": i, "box": b, "changed": round(changed, 2), "swap_frame": t_swap})
    return out


class Dist:
    """Blurred transition frames and their distance maps to the start/end stills (computed once)."""

    def __init__(self, frames: list[np.ndarray], start: np.ndarray, end: np.ndarray, sign: np.ndarray | None = None):
        self.sign = sign if sign is not None else np.zeros(start.shape[:2], np.uint8)
        self.sb, self.eb = blurred(start), blurred(end)
        self.start_end = absdiff_max(self.sb, self.eb)
        self.fb = [blurred(f) for f in frames]
        self.ds = [absdiff_max(f, self.sb) for f in self.fb]
        self.de = [absdiff_max(f, self.eb) for f in self.fb]

    def occluders(self, j: int, thr: float, motion_thr: float = 10.0) -> np.ndarray:
        """Pixels of frame j that (a) match neither still, (b) changed since the
        neighbouring frames and (c) continue outside the sign boxes: something
        passing in front.  Lettering that slowly morphs fails (b) and (c)."""
        n = len(self.fb)
        motion = np.zeros_like(self.ds[j])
        for k in (j - 1, j + 1):
            if 0 <= k < n:
                motion = cv2.max(motion, absdiff_max(self.fb[j], self.fb[k]))
        occ = (cv2.min(self.ds[j], self.de[j]) > thr) & (motion > motion_thr)
        occ = cv2.dilate(occ.astype(np.uint8), np.ones((5, 5), np.uint8))
        return drop_inside(occ, self.sign)
