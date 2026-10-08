"""Detect (and optionally fix) people/cars that freeze inside a time-lapse transition.

The classic failure: buildings and signs evolve while the pedestrians and cars
of the anchor frames stand still for seconds.  We know exactly who those
anchor subjects are, because the HOLD clips show them moving:

  start subjects = moving-mask of the start hold at its last shown frames
  end subjects   = moving-mask of the next hold at its first frames

A start subject is frozen when its pixels still match the start still after a
short grace period; an end subject is frozen when it already matches the end
still long before the end (it walked in early and stood there).  Fixing
replaces the frozen subject with the street background from the hold clips
(temporal median), faded in and out, so it reads as having walked off.
"""
from __future__ import annotations

import cv2
import numpy as np

from .lock import absdiff_max, blurred


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for t, f in enumerate(flags):
        if f and start is None:
            start = t
        if not f and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def detect(dist, start_subj: np.ndarray, end_subj: np.ndarray, grace: int, thr: float = 14.0,
           min_frames: int = 6, stuck_frac: float = 0.6) -> list[dict]:
    """``dist`` is a lock.Dist of the transition (distances to start/end stills)."""
    n = len(dist.ds)
    h, w = dist.sb.shape[:2]
    differs = dist.start_end > thr  # only judge where start and end stills disagree
    min_area = int(0.0004 * w * h)
    events = []
    for kind, subj, dmaps, window in (
        ("start_subject_frozen", start_subj, dist.ds, range(grace, n)),
        ("end_subject_frozen", end_subj, dist.de, range(0, max(0, n - grace))),
    ):
        for comp in _components((subj > 0) & differs, min_area):
            ys, xs = np.nonzero(comp)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = comp[y0:y1, x0:x1]
            flags = np.zeros(n, bool)
            for t in window:
                flags[t] = (dmaps[t][y0:y1, x0:x1][sub] < thr).mean() > stuck_frac
            for a, b in _runs(flags):
                if b - a >= min_frames:
                    events.append({
                        "kind": kind, "from_frame": a, "to_frame": b,
                        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                        "mask": comp,
                    })
    return events


def stalled_transients(dist, thr: float = 18.0, run: int = 18) -> list[dict]:
    """Warn-only: person/car sized blobs that match neither still and do not move for `run` frames."""
    n = len(dist.ds)
    h, w = dist.sb.shape[:2]
    if n <= run:
        return []
    count = np.zeros((h, w), np.int32)
    best = np.zeros((h, w), np.int32)
    for t in range(1, n):
        new = cv2.min(dist.ds[t], dist.de[t]) > thr
        still = absdiff_max(dist.fb[t], dist.fb[t - 1]) < 4
        count = np.where(new & still, count + 1, 0)
        np.maximum(best, count, out=best)
    out = []
    for comp in _components(best >= run, int(0.0005 * w * h)):
        if comp.mean() > 0.03:  # facade-sized: construction, not a person
            continue
        ys, xs = np.nonzero(comp)
        out.append({"kind": "stalled_transient_warning", "frames": int(best[comp].max()),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]})
    return out


def fix(frames: list[np.ndarray], events: list[dict], bg_start: np.ndarray, bg_end: np.ndarray, start: np.ndarray,
        end: np.ndarray, fade: int = 3) -> list[np.ndarray]:
    out = [f.copy() for f in frames]
    for ev in events:
        comp = ev["mask"].astype(np.uint8)
        comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if ev["kind"] == "start_subject_frozen":
            bg, still = bg_start, start
        else:
            bg, still = bg_end, end
        # background unknown (subject never moved in the hold) -> inpaint the still
        if absdiff_max(blurred(bg), blurred(still))[comp > 0].mean() < 8:
            bg = cv2.inpaint(still, comp * 255, 7, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 2.0)[..., None]
        a, b = ev["from_frame"], ev["to_frame"]
        for t in range(max(0, a - fade), min(len(out), b + fade)):
            ramp = 1.0
            if t < a:
                ramp = (t - (a - fade) + 1) / (fade + 1)
            elif t >= b:
                ramp = (b + fade - t) / (fade + 1)
            k = alpha * ramp
            out[t] = (out[t].astype(np.float32) * (1 - k) + bg.astype(np.float32) * k + 0.5).astype(np.uint8)
    return out
