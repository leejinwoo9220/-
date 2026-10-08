"""Offline end-to-end demo: synthetic captures + fake H3 clips -> final video.

The fake transitions are plain cross-fades with the anchor people left in
place - exactly the "buildings change, people stand still" failure - so the
demo shows the frozen-subject detector and fixer at work.  No network or
Sogni account is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import finish, prep, sogni, synth
from . import project as P
from .util import imread, log, read_json, write_json
from .video import write_frames


def _sign_boxes_from_scene(p: dict) -> list[list[float]]:
    """Map the synthetic scene's sign rectangles into normalised aligned-frame boxes."""
    prepj = read_json(P.work(p, "prep.json"))
    ref = p["reference"]
    era = [str(e["year"]) for e in synth.ERAS].index(ref)
    _, Hv = synth.make_capture(era, ui=False)
    g = prepj["geometry"][ref]
    vw, vh = synth.UI_CROP[2], synth.UI_CROP[3]
    a, out_w = g["half_w"], g["out_w"]
    x0, y0, cw, ch = prepj["crop_ref_px"]
    boxes = []
    for x, y, w, h in synth.SIGN_BOXES:
        pts = np.array([[x, y, 1], [x + w, y + h, 1]], float).T
        q = Hv @ pts
        q = q[:2] / q[2]
        u = (q[0] - (vw - 1) / 2) * out_w / (2 * a) + (out_w - 1) / 2
        v = (q[1] - (vh - 1) / 2) * out_w / (2 * a) + (g["out_h"] - 1) / 2
        bx0, by0 = (u[0] - x0) / cw, (v[0] - y0) / ch
        bx1, by1 = (u[1] - x0) / cw, (v[1] - y0) / ch
        pad = 0.01
        boxes.append([round(bx0 - pad, 4), round(by0 - pad, 4), round(bx1 - bx0 + 2 * pad, 4), round(by1 - by0 + 2 * pad, 4)])
    return boxes


def run_demo(out_dir: str) -> dict:
    root = Path(out_dir)
    pj = synth.write_demo_project(root)
    d = read_json(pj)
    d["signs"] = []
    d["transitions"] = {"2009>2014": {"changes": "The photo lab on the middle building closes and a phone shop opens."}}
    write_json(pj, d)
    p = P.load(pj)
    prep.run(p)

    d["signs"] = [{"id": f"sign{k}", "box": b} for k, b in enumerate(_sign_boxes_from_scene(p))]
    write_json(pj, d)
    p = P.load(pj)
    plan = sogni.build_plan(p)
    log(f"plan: {len(plan['segments'])} segments, {plan['estimate']['generated_seconds']}s of H3")

    h, w = P.canvas(p)[1], P.canvas(p)[0]
    boxes_px = [[int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)] for b in (s["box"] for s in d["signs"])]
    for k, s in enumerate(x for x in plan["segments"] if x["kind"] == "hold"):
        still = imread(p["_root"] / s["first"])
        frames = synth.fake_clip(still, s["frames"], seed=k, drift=0.04, garble_boxes=boxes_px)
        write_frames(p["_root"] / s["out"], frames)
        finish.make_seam(p, plan, s["id"])
    for k, s in enumerate(x for x in plan["segments"] if x["kind"] == "transition"):
        start = imread(p["_root"] / s["first"])
        end = imread(p["_root"] / s["last"])
        frames = synth.fake_clip(start, s["frames"], seed=100 + k, walkers=10, end_still=end, garble_boxes=boxes_px)
        # fake model behaviour: the opening frame's people stay put while the street cross-fades
        frames = [_keep_anchor_people(f, start, end, t / (len(frames) - 1)) for t, f in enumerate(frames)]
        write_frames(p["_root"] / s["out"], frames)
    report = finish.run(p)
    log(json.dumps({s["segment"]: {k: v for k, v in s.items() if k in ("frozen_subjects", "sign_swaps", "peak_speed", "motion_area_fraction")}
                    for s in report["segments"]}, default=str)[:3000])
    return report


def _keep_anchor_people(frame: np.ndarray, start: np.ndarray, end: np.ndarray, a: float) -> np.ndarray:
    """Paste the start frame's moving people back for the first 70% (frozen)."""
    if a > 0.7:
        return frame
    diff = np.abs(start.astype(np.int16) - end.astype(np.int16)).max(axis=2)
    people = (diff > 60) & (np.arange(start.shape[0])[:, None] > 0.6 * start.shape[0])
    out = frame.copy()
    out[people] = start[people]
    return out
