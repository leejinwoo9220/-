"""Fisheye / wide-angle removal for road-view screenshots.

Road-view viewers render a wide field of view, so straight building edges bow
outward (barrel / "fisheye" look).  We model the capture as a radially
symmetric projection around the viewport centre and remap it to an ordinary
rectilinear photo, where straight lines in the world are straight in the image.

Two lens families are supported:

* ``division`` - one-parameter division model ``r_u = r_d / (1 + lam * r_d^2)``
  (radii normalised by the half diagonal).  Covers barrel (lam < 0) and
  pincushion (lam > 0) and is the default because it fits most viewers.
* ``equidistant`` / ``stereographic`` / ``equisolid`` / ``orthographic`` -
  true fisheye projections parameterised by the horizontal field of view.

``auto`` picks the parameter with the plumb-line criterion: short line pieces
are detected (LSD) in the distorted capture, undistorted with each candidate,
and voted into a Hough histogram.  Pieces of one bowed building edge only fall
on a common line when the candidate is right, so the most concentrated
histogram wins.  On synthetic captures this recovers lambda within ~0.01.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .util import log, resize_long, to_gray

FISHEYE_MODELS = ("equidistant", "stereographic", "equisolid", "orthographic")


@dataclass
class Lens:
    model: str = "none"  # none | division | equidistant | stereographic | equisolid | orthographic
    param: float = 0.0  # division: lambda ; fisheye: horizontal FOV in degrees
    width: int = 0
    height: int = 0

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def half_diag(self) -> float:
        return math.hypot(self.width / 2.0, self.height / 2.0)

    def _fisheye_g(self, theta: np.ndarray) -> np.ndarray:
        m = self.model
        if m == "equidistant":
            return theta
        if m == "stereographic":
            return 2.0 * np.tan(theta / 2.0)
        if m == "equisolid":
            return 2.0 * np.sin(theta / 2.0)
        if m == "orthographic":
            return np.sin(theta)
        raise ValueError(m)

    def src_radius(self, ru: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map undistorted radius (centre-pixel units) to source radius.

        Returns (rd, valid)."""
        ru = np.asarray(ru, dtype=np.float64)
        if self.model == "none" or (self.model == "division" and abs(self.param) < 1e-9):
            return ru.copy(), np.ones_like(ru, dtype=bool)
        if self.model == "division":
            R = self.half_diag
            lam = self.param
            run = ru / R
            disc = 1.0 - 4.0 * lam * run * run
            valid = disc >= 0
            disc = np.clip(disc, 0, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                rdn = np.where(run > 1e-12, (1.0 - np.sqrt(disc)) / (2.0 * lam * run), run)
            return rdn * R, valid & np.isfinite(rdn) & (rdn >= 0)
        if self.model in FISHEYE_MODELS:
            half = math.radians(self.param) / 2.0
            fs = (self.width / 2.0) / float(self._fisheye_g(np.array(half)))
            theta = np.arctan(ru / fs)  # rectilinear output keeps the centre scale
            rd = fs * self._fisheye_g(theta)
            return rd, theta < math.radians(89.0)
        raise ValueError(f"unknown lens model {self.model}")

    def undist_radius(self, rd: np.ndarray) -> np.ndarray:
        """Source radius -> undistorted radius (inverse of src_radius)."""
        rd = np.asarray(rd, dtype=np.float64)
        if self.model == "none" or (self.model == "division" and abs(self.param) < 1e-9):
            return rd.copy()
        if self.model == "division":
            R = self.half_diag
            rdn = rd / R
            with np.errstate(divide="ignore", invalid="ignore"):
                return R * rdn / (1.0 + self.param * rdn * rdn)
        half = math.radians(self.param) / 2.0
        fs = (self.width / 2.0) / float(self._fisheye_g(np.array(half)))
        q = rd / fs
        m = self.model
        with np.errstate(invalid="ignore"):
            if m == "equidistant":
                theta = q
            elif m == "stereographic":
                theta = 2 * np.arctan(q / 2)
            elif m == "equisolid":
                theta = 2 * np.arcsin(np.clip(q / 2, -1, 1))
            else:
                theta = np.arcsin(np.clip(q, -1, 1))
        return fs * np.tan(np.clip(theta, 0, math.radians(89.0)))

    def undist_points(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Source pixel coords -> undistorted coords (origin at principal point)."""
        xd, yd = x - self.cx, y - self.cy
        rd = np.hypot(xd, yd)
        ru = self.undist_radius(rd)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(rd > 1e-9, ru / rd, 1.0)
        return xd * k, yd * k

    def src_coords(self, xu: np.ndarray, yu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undistorted coords (origin = principal point) -> source pixel coords."""
        ru = np.hypot(xu, yu)
        rd, valid = self.src_radius(ru)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(ru > 1e-9, rd / ru, 1.0)
        return self.cx + xu * k, self.cy + yu * k, valid


@dataclass
class Geometry:
    """Everything needed to go from a raw capture to its undistorted photo."""
    crop: list[int] = field(default_factory=list)  # [x, y, w, h] of the viewport in the raw capture
    lens: Lens = field(default_factory=Lens)
    half_w: float = 0.0  # half width of the inscribed rectangle, undistorted units
    half_h: float = 0.0
    out_w: int = 0
    out_h: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Geometry":
        g = Geometry(**{k: v for k, v in d.items() if k != "lens"})
        g.lens = Lens(**d["lens"])
        return g

    def undist_to_raw(self, xu: np.ndarray, yu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undistorted output pixel coords -> raw capture pixel coords."""
        sx = (2 * self.half_w) / self.out_w
        sy = (2 * self.half_h) / self.out_h
        x = (xu - (self.out_w - 1) / 2.0) * sx
        y = (yu - (self.out_h - 1) / 2.0) * sy
        X, Y, valid = self.lens.src_coords(x, y)
        cx0, cy0 = (self.crop[0], self.crop[1]) if self.crop else (0, 0)
        return X + cx0, Y + cy0, valid


def _inscribed_half_width(lens: Lens, aspect: float, samples: int = 64) -> float:
    """Largest centred rectangle (width/height = aspect) whose pixels all come
    from inside the source frame."""
    w, h = lens.width, lens.height

    def ok(a: float) -> bool:
        b = a / aspect
        t = np.linspace(-1, 1, samples)
        xs = np.concatenate([t * a, t * a, np.full(samples, -a), np.full(samples, a)])
        ys = np.concatenate([np.full(samples, -b), np.full(samples, b), t * b, t * b])
        X, Y, valid = lens.src_coords(xs, ys)
        return bool(np.all(valid) and X.min() >= -0.5 and Y.min() >= -0.5
                    and X.max() <= w - 0.5 and Y.max() <= h - 0.5)

    lo, hi = 4.0, 4.0 * max(w, h)
    if not ok(lo):
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def build_geometry(lens: Lens, crop: list[int] | None = None, max_side: int = 6000) -> Geometry:
    aspect = lens.width / lens.height
    a = _inscribed_half_width(lens, aspect)
    if a <= 0:
        raise ValueError(f"lens {lens} leaves no valid area")
    b = a / aspect
    out_w, out_h = int(round(2 * a)), int(round(2 * b))
    s = min(1.0, max_side / max(out_w, out_h))
    out_w, out_h = max(16, int(round(out_w * s))), max(16, int(round(out_h * s)))
    return Geometry(crop=list(crop) if crop else [], lens=lens, half_w=a, half_h=b, out_w=out_w, out_h=out_h)


def remap_maps(geom: Geometry, out_w: int | None = None, out_h: int | None = None,
               raw_offset: bool = False) -> tuple[np.ndarray, np.ndarray]:
    ow = out_w or geom.out_w
    oh = out_h or geom.out_h
    g = geom
    if (ow, oh) != (geom.out_w, geom.out_h):
        g = Geometry(crop=geom.crop, lens=geom.lens, half_w=geom.half_w, half_h=geom.half_h, out_w=ow, out_h=oh)
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    X, Y, valid = g.undist_to_raw(xx, yy)
    if not raw_offset and g.crop:
        X -= g.crop[0]
        Y -= g.crop[1]
    X[~valid] = -1e4
    Y[~valid] = -1e4
    return X.astype(np.float32), Y.astype(np.float32)


def apply_crop(img: np.ndarray, crop: list[int] | None) -> np.ndarray:
    if not crop:
        return img
    x, y, w, h = crop
    return img[y:y + h, x:x + w]


def undistort(img_raw: np.ndarray, geom: Geometry, interp: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    src = apply_crop(img_raw, geom.crop)
    if geom.lens.model == "none":
        return src.copy()
    mx, my = remap_maps(geom)
    return cv2.remap(src, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- estimation

_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


def _segment_pieces(gray: np.ndarray, piece: float = 18.0) -> np.ndarray:
    """LSD on the *distorted* image, long segments cut into short pieces.

    Returns an (N, 4) array of piece endpoints.  Each piece is short enough to
    be straight even on a bowed edge; after correct undistortion all pieces of
    one world line become collinear."""
    lines = _lsd().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    out = []
    diag = math.hypot(*gray.shape[:2])
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(np.float64):
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 0.012 * diag:
            continue
        n = max(1, int(L // piece))
        t = np.linspace(0, 1, n + 1)
        xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        out.append(np.stack([xs[:-1], ys[:-1], xs[1:], ys[1:]], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def _collinearity(pieces: np.ndarray, lens: Lens, rho_bin: float, theta_bins: int = 360) -> float:
    """Length-weighted Hough concentration of undistorted pieces.

    Weights use the *distorted* length, so stretching the periphery does not
    earn votes; only pieces lining up on common lines raise the score."""
    if len(pieces) == 0:
        return 0.0
    x1, y1 = lens.undist_points(pieces[:, 0], pieces[:, 1])
    x2, y2 = lens.undist_points(pieces[:, 2], pieces[:, 3])
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y1) & np.isfinite(y2)
    x1, y1, x2, y2 = x1[ok], y1[ok], x2[ok], y2[ok]
    wgt = np.hypot(pieces[ok, 2] - pieces[ok, 0], pieces[ok, 3] - pieces[ok, 1])
    theta = np.arctan2(x2 - x1, -(y2 - y1))  # normal direction of the piece
    theta = np.mod(theta, np.pi)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    rho = mx * np.cos(theta) + my * np.sin(theta)
    rmax = float(np.max(np.abs(rho))) + rho_bin
    hist, _, _ = np.histogram2d(theta, rho, bins=[theta_bins, max(8, int(2 * rmax / rho_bin))],
                                range=[[0, np.pi], [-rmax, rmax]], weights=wgt)
    hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), 0.8)
    return float(np.sum(hist.astype(np.float64) ** 2) / (np.sum(wgt) ** 2 + 1e-9))


def estimate_lens(img: np.ndarray, model: str = "division", ignore: list[list[int]] | None = None) -> tuple[Lens, list[tuple[float, float]]]:
    """Search the lens parameter that makes lines straightest (plumb-line).

    ``img`` must already be cropped to the viewport.  Returns the best lens and
    the (param, score) curve for the report."""
    h, w = img.shape[:2]
    small, s = resize_long(to_gray(img), 1400)
    small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    if ignore:
        for x, y, rw, rh in ignore:
            small[int(y * s):int((y + rh) * s), int(x * s):int((x + rw) * s)] = int(np.median(small))
    pieces = _segment_pieces(small)
    sh, sw = small.shape[:2]
    rho_bin = max(1.0, 0.0012 * math.hypot(sw, sh))

    if model == "division":
        coarse = np.round(np.arange(-0.70, 0.2501, 0.01), 4)
        step = 0.001
    elif model in FISHEYE_MODELS:
        coarse = np.arange(50.0, 178.01, 2.0)
        step = 0.25
    else:
        raise ValueError(model)

    def score(p: float) -> float:
        return _collinearity(pieces, Lens(model, p, sw, sh), rho_bin)

    curve: list[tuple[float, float]] = [(float(p), score(float(p))) for p in coarse]
    best = max(curve, key=lambda t: t[1])[0]
    span = coarse[1] - coarse[0]
    for p in np.arange(best - span, best + span + 1e-9, step):
        if model in FISHEYE_MODELS and not (30 <= p <= 179):
            continue
        curve.append((float(round(p, 4)), score(float(p))))
    curve.sort()
    best_p, best_s = max(curve, key=lambda t: t[1])
    log(f"lens {model}: best param {best_p:.4f} score {best_s:.5f} (no correction {score(0.0) if model == 'division' else float('nan'):.5f}, {len(pieces)} pieces)")
    return Lens(model=model, param=best_p, width=w, height=h), curve


def solve_capture(img_raw: np.ndarray, spec: dict) -> Geometry:
    """Build the undistortion geometry for one capture from its project spec.

    spec keys (all optional): crop [x,y,w,h], ignore [[x,y,w,h]...] (raw coords),
    undistort: {model: auto|division|none|stereographic|..., param: number|null}
    """
    crop = spec.get("crop") or []
    vp = apply_crop(img_raw, crop)
    h, w = vp.shape[:2]
    ud = spec.get("undistort") or {}
    model = ud.get("model", "auto")
    param = ud.get("param")
    ignore = []
    for x, y, rw, rh in spec.get("ignore", []) or []:
        ignore.append([x - (crop[0] if crop else 0), y - (crop[1] if crop else 0), rw, rh])

    if model == "none":
        lens = Lens("none", 0.0, w, h)
        score = 0.0
    else:
        m = "division" if model == "auto" else model
        if param is None:
            lens, curve = estimate_lens(vp, m, ignore)
            score = max(s for _, s in curve)
        else:
            lens = Lens(m, float(param), w, h)
            score = 0.0
    geom = build_geometry(lens, crop)
    geom.score = score
    return geom
