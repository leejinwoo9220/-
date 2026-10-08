"""Synthetic street scenes used by the tests and the offline demo.

Generates the same street in several "eras" (different shop signs and
facades), viewed from slightly different camera positions and through a
barrel-distorting road-view lens, plus fake H3 clips with walking people.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SCENE_W, SCENE_H = 2400, 1350

ERAS = [
    {"year": 2009, "signs": ["KIMBAP 24", "PHOTO LAB", "BOOKS"], "colors": [(40, 60, 170), (60, 140, 60), (150, 90, 40)], "tower": False},
    {"year": 2014, "signs": ["KIMBAP 24", "MOBILE", "CAFE 7"], "colors": [(40, 60, 170), (170, 120, 30), (60, 60, 60)], "tower": False},
    {"year": 2019, "signs": ["BURGER", "MOBILE", "CAFE 7"], "colors": [(30, 30, 200), (170, 120, 30), (60, 60, 60)], "tower": True},
    {"year": 2026, "signs": ["BURGER", "PHARMACY", "BAKERY"], "colors": [(30, 30, 200), (60, 160, 60), (40, 120, 200)], "tower": True},
]

# sign boxes in scene coordinates [x, y, w, h]
SIGN_BOXES = [[260, 610, 520, 120], [930, 610, 520, 120], [1600, 610, 520, 120]]


def draw_scene(era: dict, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((SCENE_H, SCENE_W, 3), (215, 200, 185), np.uint8)  # sky-ish
    cv2.rectangle(img, (0, 0), (SCENE_W, 300), (235, 215, 190), -1)
    # distant tower (appears in later eras)
    if era.get("tower"):
        cv2.rectangle(img, (1050, 40), (1350, 560), (150, 150, 140), -1)
        for y in range(70, 540, 40):
            for x in range(1075, 1330, 50):
                cv2.rectangle(img, (x, y), (x + 30, y + 22), (110, 90, 70), -1)
    # three low-rise buildings
    bx = [200, 870, 1540]
    for i, x0 in enumerate(bx):
        col = (170 + 15 * i, 165, 150 - 10 * i)
        cv2.rectangle(img, (x0, 300), (x0 + 640, 1000), col, -1)
        cv2.rectangle(img, (x0, 300), (x0 + 640, 1000), (60, 60, 60), 4)
        for y in range(330, 590, 85):
            for x in range(x0 + 30, x0 + 620, 110):
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (90, 80, 70), -1)
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (230, 230, 230), 3)
        # shop window
        cv2.rectangle(img, (x0 + 40, 760), (x0 + 600, 990), (70, 60, 50), -1)
    # signs
    for (x, y, w, h), text, color in zip(SIGN_BOXES, era["signs"], era["colors"]):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 250), 4)
        cv2.putText(img, text, (x + 25, y + 85), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 5, cv2.LINE_AA)
    # sidewalk, curb and road with lane marks
    cv2.rectangle(img, (0, 1000), (SCENE_W, 1090), (150, 150, 155), -1)
    cv2.line(img, (0, 1090), (SCENE_W, 1090), (90, 90, 90), 6)
    cv2.rectangle(img, (0, 1093), (SCENE_W, SCENE_H), (80, 80, 85), -1)
    for x in range(0, SCENE_W, 240):
        cv2.rectangle(img, (x, 1210), (x + 130, 1225), (240, 240, 240), -1)
    # poles and wires
    for x in (130, 1210, 2290):
        cv2.rectangle(img, (x, 180), (x + 18, 1090), (50, 50, 50), -1)
    cv2.line(img, (0, 220), (SCENE_W, 235), (30, 30, 30), 3)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def camera_homography(dx: float, dy: float, zoom: float, rot_deg: float, persp: float = 0.0) -> np.ndarray:
    """Scene -> image homography for a slightly different camera pose."""
    cx, cy = SCENE_W / 2, SCENE_H / 2
    a = math.radians(rot_deg)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    R = np.array([[zoom * math.cos(a), -zoom * math.sin(a), 0], [zoom * math.sin(a), zoom * math.cos(a), 0], [0, 0, 1]], float)
    P = np.array([[1, 0, 0], [0, 1, 0], [persp, 0, 1]], float)
    T2 = np.array([[1, 0, cx + dx], [0, 1, cy + dy], [0, 0, 1]], float)
    return T2 @ P @ R @ T1


def barrel_distort(img: np.ndarray, lam: float, out_w: int, out_h: int) -> np.ndarray:
    """Render what a road-view viewer would show: each output (distorted) pixel
    samples the rectilinear image at r_u = r_d / (1 + lam r_d^2)."""
    h, w = img.shape[:2]
    R = math.hypot(out_w / 2, out_h / 2)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    x = xx - (out_w - 1) / 2
    y = yy - (out_h - 1) / 2
    rd = np.hypot(x, y) / R
    k = 1.0 / (1.0 + lam * rd * rd)
    mx = (w - 1) / 2 + x * k
    my = (h - 1) / 2 + y * k
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


POSES = [
    dict(dx=-30, dy=12, zoom=1.04, rot_deg=0.8, persp=0.00002),
    dict(dx=20, dy=-8, zoom=0.98, rot_deg=-0.6, persp=-0.00001),
    dict(dx=45, dy=5, zoom=1.07, rot_deg=0.3, persp=0.00001),
    dict(dx=0, dy=0, zoom=1.0, rot_deg=0.0, persp=0.0),
]


def make_capture(era_index: int, lam: float = -0.22, size=(1600, 900), ui: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw capture with fake viewer UI, ground-truth scene->viewport H)."""
    scene = draw_scene(ERAS[era_index], seed=era_index)
    H = camera_homography(**POSES[era_index])
    # viewport: the camera image is the scene warped then cropped to `size`
    vw, vh = size
    S = np.array([[vw / SCENE_W, 0, 0], [0, vw / SCENE_W, (vh - SCENE_H * vw / SCENE_W) / 2], [0, 0, 1]])
    Hv = S @ H
    rect = cv2.warpPerspective(scene, Hv, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    viewport = barrel_distort(rect, lam, vw, vh)
    if not ui:
        return viewport, Hv
    # wrap into a fake browser/app window with a toolbar and a date chip
    raw = np.full((vh + 140, vw + 80, 3), 245, np.uint8)
    raw[100:100 + vh, 40:40 + vw] = viewport
    cv2.putText(raw, f"roadview {ERAS[era_index]['year']}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
    return raw, Hv


UI_CROP = [40, 100, 1600, 900]


def write_demo_project(root, lam: float = 0.0) -> str:
    """Write 4 synthetic road-view captures and a project.json; returns its path."""
    from pathlib import Path

    from .util import imwrite, write_json

    root = Path(root)
    caps = []
    for i, era in enumerate(ERAS):
        raw, _ = make_capture(i, lam=lam)
        f = f"captures/{era['year']}.png"
        imwrite(root / f, raw)
        caps.append({"id": str(era["year"]), "file": f, "date": f"{era['year']}-06", "crop": UI_CROP})
    proj = {
        "name": "synthetic-demo",
        "scene": "a two-lane commercial street lined with three low-rise shop buildings",
        "orientation": "landscape",
        "captures": caps,
        "signs": [{"id": f"sign{k}", "box": None} for k in range(3)],
    }
    write_json(root / "project.json", proj)
    return str(root / "project.json")


def fake_clip(still: np.ndarray, n_frames: int, seed: int = 0, walkers: int = 6, frozen: list[tuple[int, int]] | None = None,
              end_still: np.ndarray | None = None, drift: float = 0.0, garble_boxes: list[list[int]] | None = None) -> list[np.ndarray]:
    """Fake an H3 clip: moving people/cars over the still (optionally blending
    to end_still), with slight exposure drift and garbled sign texture."""
    rng = np.random.default_rng(seed)
    h, w = still.shape[:2]
    peds = []
    for _ in range(walkers):
        y = rng.uniform(0.72, 0.86) * h
        x = rng.uniform(0, w)
        v = rng.choice([-1, 1]) * rng.uniform(3, 7) * w / 1344
        color = tuple(int(c) for c in rng.integers(20, 230, 3))
        peds.append([x, y, v, color])
    frames = []
    for t in range(n_frames):
        a = t / max(1, n_frames - 1)
        base = still.astype(np.float32)
        if end_still is not None:
            base = (1 - a) * base + a * end_still.astype(np.float32)
        base = base * (1.0 + drift * a)
        f = np.clip(base, 0, 255).astype(np.uint8)
        if garble_boxes:
            for x, y, bw, bh in garble_boxes:
                patch = f[y:y + bh, x:x + bw].astype(np.int16)
                patch += rng.integers(-35, 35, patch.shape, dtype=np.int16)
                f[y:y + bh, x:x + bw] = np.clip(patch, 0, 255).astype(np.uint8)
        for p in peds:
            p[0] = (p[0] + p[2]) % w
            pw, ph = int(0.025 * w), int(0.11 * h)
            cv2.rectangle(f, (int(p[0]), int(p[1] - ph)), (int(p[0]) + pw, int(p[1])), p[3], -1)
        for fx, fy in frozen or []:
            pw, ph = int(0.03 * w), int(0.13 * h)
            cv2.rectangle(f, (fx, fy - ph), (fx + pw, fy), (20, 20, 160), -1)
        frames.append(f)
    return frames
