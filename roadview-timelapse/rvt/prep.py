"""Stills stage: fisheye removal -> common framing -> tone -> QA.

Outputs (under work/):
  undistorted/<id>.jpg       ordinary rectilinear photo of each capture
  undistorted/<id>_grid.jpg  same with a pixel grid, for manual anchors
  aligned/master/<id>.png    hi-res aligned still (used to lock signs in the video)
  aligned/canvas/<id>.png    exact H3 canvas size (Sogni input frame)
  qa/stills_sheet.jpg, qa/blink.gif, qa/edges_<id>.jpg, qa/sign_grid.jpg
  prep.json                  geometry, homographies, crop and metrics
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import project as P
from . import qa, tone
from .align import Item, footprint, intersect_polys, largest_rect, prepare, register, render
from .undistort import Geometry, solve_capture
from .util import imread, imwrite, log, write_json


def _pixel_grid(img: np.ndarray, step: int = 100) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(out, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


def run(p: dict) -> dict:
    caps = p["captures"]
    ref_idx = [c["id"] for c in caps].index(p["reference"])
    items: list[Item] = []
    geoms: dict[str, dict] = {}
    for c in caps:
        raw = imread(P.file(p, c["file"]))
        log(f"{c['id']}: raw {raw.shape[1]}x{raw.shape[0]}")
        geom = solve_capture(raw, c)
        it = Item(id=c["id"], raw=raw, geom=geom, anchors=c.get("anchors", []))
        prepare(it, c.get("ignore"))
        imwrite(P.work(p, "undistorted", f"{c['id']}.jpg"), it.und)
        imwrite(P.work(p, "undistorted", f"{c['id']}_grid.jpg"), _pixel_grid(it.und), quality=88)
        items.append(it)
        geoms[c["id"]] = geom.to_dict()

    Hs, reg_report = register(items, ref_idx)

    ref = items[ref_idx]
    rh, rw = ref.und.shape[:2]
    polys = [footprint(H, it.und.shape[1], it.und.shape[0]) for H, it in zip(Hs, items)]
    inter = intersect_polys(polys)
    cw, ch = P.canvas(p)
    aspect = cw / ch
    cx, cy, half = largest_rect(inter, aspect, prefer=(rw / 2, rh / 2), diag=math.hypot(rw, rh))
    zoom = float(p.get("crop_zoom", 1.0))  # < 1 tightens the crop
    half *= zoom
    crop = (cx - half, cy - half / aspect, 2 * half, 2 * half / aspect)
    coverage = (crop[2] * crop[3]) / (rw * rh)
    log(f"common crop {crop[2]:.0f}x{crop[3]:.0f} px of the reference ({coverage:.0%} of its area)")

    mw = int(min(max(round(crop[2]), cw), p["master_max_width"]))
    mw -= mw % 2
    mh = int(round(mw / aspect))
    mh -= mh % 2

    masters, canvases = [], []
    for H, it in zip(Hs, items):
        masters.append(render(it, H, crop, mw, mh))
    ref_master = masters[ref_idx]
    t = p["tone"]
    for i, (c, m) in enumerate(zip(caps, masters)):
        if i != ref_idx:
            m = tone.normalize(m, ref_master, luma=float(t.get("luma", 0)), color=float(t.get("color", 0)))
            masters[i] = m
        cv_img = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA)
        canvases.append(cv_img)
        imwrite(P.work(p, "aligned", "master", f"{c['id']}.png"), m)
        imwrite(P.work(p, "aligned", "canvas", f"{c['id']}.png"), cv_img)

    labels = [f"{c['id']}" for c in caps]
    qa.contact_sheet(masters, labels, P.work(p, "qa", "stills_sheet.jpg"))
    qa.blink_gif(masters, labels, P.work(p, "qa", "blink.gif"))
    for c, m in zip(caps, masters):
        qa.edge_overlay(m, ref_master, P.work(p, "qa", f"edges_{c['id']}.jpg"))
    qa.grid_overlay(ref_master, P.work(p, "qa", "sign_grid.jpg"))
    for rep, m in zip(reg_report, masters):
        rep["residual_flow_vs_reference"] = qa.residual_flow(m, ref_master)

    out = {
        "reference": p["reference"],
        "captures": [c["id"] for c in caps],
        "geometry": geoms,
        "homographies": {it.id: H.tolist() for it, H in zip(items, Hs)},
        "registration": reg_report,
        "crop_ref_px": [round(v, 2) for v in crop],
        "crop_coverage_of_reference": round(coverage, 4),
        "master_size": [mw, mh],
        "canvas_size": [cw, ch],
    }
    write_json(P.work(p, "prep.json"), out)
    return out
