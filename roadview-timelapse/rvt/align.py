"""Bring every capture into the reference capture's framing.

1. Undistorted previews are matched pairwise with RootSIFT + MAGSAC
   homographies (buildings and road are far enough that a homography holds
   for the facade plane; moving people/cars are rejected as outliers).
2. Old and new eras can share few features (2009 vs 2026), so each capture
   reaches the reference through the *widest path* in the match graph
   (e.g. 2009 -> 2014 -> 2026) instead of a weak direct match.
3. A guided pass then re-matches each warped capture directly against the
   reference with a tight spatial window and refines the homography.
4. Manual anchors (>= 4 point pairs) override everything for hard cases.
5. All warped footprints are intersected and the largest rectangle with the
   target aspect is cut, so no frame needs invented border pixels.
6. Each output is rendered in ONE resampling step straight from the raw
   capture (lens undistortion + homography + crop composed into one map).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .undistort import Geometry, remap_maps, undistort
from .util import log, resize_long, to_gray

WORK_LONG = 1600


@dataclass
class Item:
    id: str
    raw: np.ndarray
    geom: Geometry
    und: np.ndarray = None  # undistorted full-res preview
    mask: np.ndarray = None  # feature mask (undistorted coords), 255 = usable
    anchors: list = field(default_factory=list)  # [[x, y, x_ref, y_ref], ...] in undistorted px
    work: np.ndarray = None
    scale: float = 1.0
    kp: np.ndarray = None
    desc: np.ndarray = None


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    return _SIFT


def _features(gray: np.ndarray, mask: np.ndarray | None):
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    kps, desc = _sift().detectAndCompute(g, mask)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    desc = desc / (np.sum(desc, axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc).astype(np.float32)  # RootSIFT
    pts = np.array([k.pt for k in kps], np.float32)
    return pts, desc


def _match(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.82) -> np.ndarray:
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(d1, d2, k=2)
    bwd = bf.knnMatch(d2, d1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in bwd if m}
    out = []
    for m in fwd:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx:
            out.append((m[0].queryIdx, m[0].trainIdx))
    return np.array(out, int).reshape(-1, 2)


def _homography(p1: np.ndarray, p2: np.ndarray, thresh: float):
    if len(p1) < 8:
        return None, np.zeros(len(p1), bool)
    H, inl = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, thresh, maxIters=20000, confidence=0.9999)
    if H is None:
        return None, np.zeros(len(p1), bool)
    return H, inl.ravel().astype(bool)


def sane(H: np.ndarray, w: int, h: int) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    area = cv2.contourArea(q.astype(np.float32), oriented=True)
    if area <= 0:  # flipped
        return False
    ratio = abs(area) / (w * h)
    if not (0.35 < ratio < 2.8):
        return False
    return cv2.isContourConvex(q.astype(np.float32))


def _scale_h(H: np.ndarray, s_src: float, s_dst: float) -> np.ndarray:
    """Homography estimated on scaled images -> full-res homography."""
    S1 = np.diag([s_src, s_src, 1.0])
    S2i = np.diag([1 / s_dst, 1 / s_dst, 1.0])
    return S2i @ H @ S1


def prepare(item: Item, ignore_raw: list[list[int]] | None = None) -> None:
    item.und = undistort(item.raw, item.geom)
    h, w = item.und.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    if ignore_raw:
        raw_mask = np.full(item.raw.shape[:2], 255, np.uint8)
        for x, y, rw, rh in ignore_raw:
            raw_mask[y:y + rh, x:x + rw] = 0
        mx, my = remap_maps(item.geom, raw_offset=True)
        mask = cv2.remap(raw_mask, mx, my, cv2.INTER_NEAREST, borderValue=0)
    item.mask = mask
    item.work, item.scale = resize_long(to_gray(item.und), WORK_LONG)
    wm = cv2.resize(mask, (item.work.shape[1], item.work.shape[0]), interpolation=cv2.INTER_NEAREST)
    item.kp, item.desc = _features(item.work, wm)


def pair_homography(a: Item, b: Item) -> tuple[np.ndarray | None, int, float]:
    """Full-res homography a -> b, inlier count, median residual (full-res px of b)."""
    m = _match(a.desc, b.desc)
    if len(m) < 8:
        return None, 0, float("inf")
    p1, p2 = a.kp[m[:, 0]], b.kp[m[:, 1]]
    H, inl = _homography(p1, p2, thresh=3.0)
    if H is None:
        return None, 0, float("inf")
    Hf = _scale_h(H, a.scale, b.scale)
    if not sane(Hf, a.und.shape[1], a.und.shape[0]):
        return None, 0, float("inf")
    proj = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    res = float(np.median(np.hypot(*(proj - p2[inl]).T))) / b.scale if inl.any() else float("inf")
    return Hf, int(inl.sum()), res


def widest_paths(n: int, weight: dict[tuple[int, int], int], ref: int) -> dict[int, list[int]]:
    """Maximum spanning tree (Prim) rooted at ref: path with the best weakest link."""
    in_tree = {ref}
    parent = {ref: None}
    best = {i: (-1, None) for i in range(n) if i != ref}
    for j in best:
        best[j] = (weight.get((ref, j), weight.get((j, ref), 0)), ref)
    while best:
        j, (wgt, p) = max(best.items(), key=lambda kv: kv[1][0])
        if wgt <= 0:
            break
        in_tree.add(j)
        parent[j] = p
        del best[j]
        for k in best:
            wk = weight.get((j, k), weight.get((k, j), 0))
            if wk > best[k][0]:
                best[k] = (wk, j)
    paths = {}
    for i in range(n):
        if i not in parent:
            continue
        path = [i]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        paths[i] = path
    return paths


def guided_refine(item: Item, ref: Item, H: np.ndarray, window: float = 30.0):
    """Warp item into the reference, re-match with a spatial window, refine H.

    Returns (H_refined, inliers, median residual in ref full-res px)."""
    rh, rw = ref.work.shape[:2]
    S_ref = np.diag([ref.scale, ref.scale, 1.0])
    S_it_inv = np.diag([1 / item.scale, 1 / item.scale, 1.0])
    Hw = S_ref @ H @ S_it_inv  # item.work -> ref.work
    warped = cv2.warpPerspective(item.work, Hw, (rw, rh), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(np.full(item.work.shape[:2], 255, np.uint8), Hw, (rw, rh), flags=cv2.INTER_NEAREST)
    wmask = cv2.erode(wmask, np.ones((9, 9), np.uint8))
    refmask = cv2.resize(ref.mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    kp1, d1 = _features(warped, wmask)
    kp2, d2 = _features(ref.work, refmask)
    m = _match(d1, d2, ratio=0.9)
    if len(m) < 8:
        return H, 0, float("inf")
    p1, p2 = kp1[m[:, 0]], kp2[m[:, 1]]
    near = np.hypot(*(p1 - p2).T) < window
    p1, p2 = p1[near], p2[near]
    Hc, inl = _homography(p1, p2, thresh=2.0)
    if Hc is None or inl.sum() < 12:
        return H, int(inl.sum()), float("inf")
    proj = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), Hc).reshape(-1, 2)
    res = float(np.median(np.hypot(*(proj - p2[inl]).T))) / ref.scale
    Hc_full = np.diag([1 / ref.scale, 1 / ref.scale, 1.0]) @ Hc @ np.diag([ref.scale, ref.scale, 1.0])
    Hn = Hc_full @ H
    if not sane(Hn, item.und.shape[1], item.und.shape[0]):
        return H, 0, float("inf")
    return Hn, int(inl.sum()), res


def register(items: list[Item], ref_index: int) -> tuple[list[np.ndarray], list[dict]]:
    """Homographies (undistorted item px -> undistorted reference px) + report."""
    n = len(items)
    pairs: dict[tuple[int, int], tuple] = {}
    weight: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            H, inl, res = pair_homography(items[i], items[j])
            pairs[(i, j)] = (H, inl, res)
            weight[(i, j)] = inl if H is not None else 0
            log(f"match {items[i].id} <-> {items[j].id}: {inl} inliers, median residual {res:.2f}px")
    paths = widest_paths(n, weight, ref_index)
    Hs: list[np.ndarray] = []
    report: list[dict] = []
    for i, it in enumerate(items):
        rep: dict = {"id": it.id}
        if i == ref_index:
            Hs.append(np.eye(3))
            rep.update(method="reference")
            report.append(rep)
            continue
        H = None
        if len(it.anchors) >= 4:
            a = np.array(it.anchors, np.float64)
            H, _ = cv2.findHomography(a[:, :2], a[:, 2:4], cv2.RANSAC, 4.0)
            rep.update(method="manual_anchors", anchors=len(a))
        elif i in paths:
            path = paths[i]
            H = np.eye(3)
            links = []
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in pairs:
                    Huv = pairs[(u, v)][0]
                    links.append(pairs[(u, v)][1])
                else:
                    Huv = np.linalg.inv(pairs[(v, u)][0])
                    links.append(pairs[(v, u)][1])
                H = Huv @ H
            rep.update(method="features", path=[items[k].id for k in path], weakest_link_inliers=min(links))
        if H is None:
            raise RuntimeError(
                f"capture {it.id}: could not be matched to the reference. Add >= 4 manual anchors "
                f"([x, y, x_ref, y_ref] in undistorted pixels, see work/undistorted/) to project.json")
        H2, ginl, gres = guided_refine(it, items[ref_index], H)
        rep.update(guided_inliers=ginl, guided_median_residual_px=round(gres, 3) if math.isfinite(gres) else None)
        Hs.append(H2 / H2[2, 2])
        report.append(rep)
    return Hs, report


# ------------------------------------------------------------------ crop

def footprint(H: np.ndarray, w: int, h: int, inset: float = 2.0) -> np.ndarray:
    c = np.array([[inset, inset], [w - inset, inset], [w - inset, h - inset], [inset, h - inset]], np.float32)
    return cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2)


def intersect_polys(polys: list[np.ndarray]) -> np.ndarray:
    cur = polys[0].astype(np.float32)
    for p in polys[1:]:
        area, inter = cv2.intersectConvexConvex(cur, p.astype(np.float32))
        if area <= 0 or inter is None:
            raise RuntimeError("captures do not overlap after alignment")
        cur = inter.reshape(-1, 2)
    return cur


def largest_rect(poly: np.ndarray, aspect: float, prefer: tuple[float, float], diag: float, bias: float = 0.15):
    """Largest axis-aligned rect (w/h = aspect) inside a convex polygon,
    mildly preferring centres close to ``prefer``. Returns (cx, cy, half_w)."""
    pts = poly.astype(np.float64)
    if cv2.contourArea(pts.astype(np.float32), oriented=True) < 0:
        pts = pts[::-1]
    # half-planes n.x <= c for a counter-clockwise polygon in image coords (y down)
    e = np.roll(pts, -1, axis=0) - pts
    normals = np.stack([e[:, 1], -e[:, 0]], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    offs = np.sum(normals * pts, axis=1)
    # make sure the polygon centroid satisfies n.x <= c, else flip
    cen = pts.mean(0)
    if np.mean(normals @ cen - offs > 0) > 0.5:
        normals, offs = -normals, -offs

    corners = np.array([[-1, -1 / aspect], [1, -1 / aspect], [1, 1 / aspect], [-1, 1 / aspect]])

    def max_half(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        # for each corner k and edge j: n.(c + s*corner_k) <= o  ->  s <= (o - n.c) / (n.corner_k)
        nc = cx[:, None] * normals[None, :, 0] + cy[:, None] * normals[None, :, 1]  # (M, E)
        slack = offs[None, :] - nc
        s = np.full(len(cx), np.inf)
        for k in corners:
            nk = normals @ k  # (E,)
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.where(nk[None, :] > 1e-12, slack / nk[None, :], np.inf)
            s = np.minimum(s, lim.min(axis=1))
        s[np.any(slack < 0, axis=1)] = 0
        return np.maximum(s, 0)

    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    best = None
    for it in range(3):
        if best is None:
            gx, gy = np.meshgrid(np.linspace(x0, x1, 41), np.linspace(y0, y1, 41))
        else:
            span = (x1 - x0) / (41 * (2 ** it))
            gx, gy = np.meshgrid(np.linspace(best[0] - 4 * span, best[0] + 4 * span, 21),
                                 np.linspace(best[1] - 4 * span, best[1] + 4 * span, 21))
        cx, cy = gx.ravel(), gy.ravel()
        s = max_half(cx, cy)
        score = s * (1 - bias * np.hypot(cx - prefer[0], cy - prefer[1]) / diag)
        k = int(np.argmax(score))
        best = (float(cx[k]), float(cy[k]), float(s[k]))
    return best


# ------------------------------------------------------------------ render

def render(item: Item, H: np.ndarray, crop: tuple[float, float, float, float], out_w: int, out_h: int,
           interp: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """crop = (x0, y0, w, h) in reference-undistorted px. One resampling from raw."""
    x0, y0, cw, ch = crop
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    rx = x0 + (xx + 0.5) * (cw / out_w) - 0.5
    ry = y0 + (yy + 0.5) * (ch / out_h) - 0.5
    Hi = np.linalg.inv(H)
    d = Hi[2, 0] * rx + Hi[2, 1] * ry + Hi[2, 2]
    ux = (Hi[0, 0] * rx + Hi[0, 1] * ry + Hi[0, 2]) / d
    uy = (Hi[1, 0] * rx + Hi[1, 1] * ry + Hi[1, 2]) / d
    if item.geom.lens.model == "none":
        X = ux + (item.geom.crop[0] if item.geom.crop else 0)
        Y = uy + (item.geom.crop[1] if item.geom.crop else 0)
    else:
        X, Y, _ = item.geom.undist_to_raw(ux, uy)
    return cv2.remap(item.raw, X.astype(np.float32), Y.astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)
