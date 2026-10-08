"""project.json loading and defaults."""
from __future__ import annotations

import copy
from pathlib import Path

from .util import read_json

CANVAS = {"landscape": (1344, 768), "portrait": (768, 1344)}
DELIVERY = {"landscape": (1920, 1080), "portrait": (1080, 1920)}

VIDEO_DEFAULTS = {
    "tier": "standard",          # standard | balanced | turbo | fasth3  (MiniMax H3 on Sogni)
    "hold_frames": 124,          # generated length of each normal-speed hold (124 = 5.17 s, the H3 minimum)
    "hold_seconds": 3.0,         # how much of each hold is shown at 1x
    "first_hold_seconds": 3.0,
    "last_hold_seconds": 4.0,
    "transition_frames": "auto", # auto = by year gap (124 / 141 / 192) or an explicit grid value
    "transition_seconds": 2.6,   # on-screen length of each fast-forward after retiming
    "shutter": 1.0,              # 1.0 = 360-degree shutter: streaks proportional to speed
    "audio": "keep",             # keep | none
    "loras": [],                 # e.g. [{"id": "h3-realism-people", "strength": 0.6}]
    "billing_mode": "subscription",
    "fps": 24,
}

FINISH_DEFAULTS = {
    "lock_threshold": 14.0,       # blurred abs-diff (0..255) above which a pixel counts as moving
    "sign_threshold": 26.0,       # stricter threshold inside sign boxes
    "min_blob_frac": 0.00008,     # ignore moving specks smaller than this fraction of the frame
    "mask_dilate": 5,
    "landing_frames": 8,          # transition frames blended into the exact stills at each end
    "fix_frozen": True,           # replace frozen people/cars in transitions with background
    "frozen_grace_seconds": 0.75, # model-time grace before a start/end subject counts as frozen
    "stabilize": True,
    "crf": 16,
}


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    p = read_json(path)
    p["_root"] = path.parent
    p["_path"] = path
    p.setdefault("name", path.parent.name)
    p.setdefault("scene", "a city street lined with shop buildings")
    p.setdefault("orientation", "landscape")
    if p["orientation"] not in CANVAS:
        raise ValueError("orientation must be landscape or portrait")
    p.setdefault("tone", {"luma": 0.35, "color": 0.0})
    p.setdefault("master_max_width", 3840)
    p.setdefault("signs", [])
    p.setdefault("transitions", {})
    p["video"] = {**VIDEO_DEFAULTS, **p.get("video", {})}
    p["finish"] = {**FINISH_DEFAULTS, **p.get("finish", {})}
    caps = p.get("captures") or []
    if len(caps) < 2:
        raise ValueError("need at least two captures")
    ids = [c["id"] for c in caps]
    if len(set(ids)) != len(ids):
        raise ValueError("capture ids must be unique")
    caps.sort(key=lambda c: str(c.get("date", c["id"])))
    p["captures"] = caps
    p.setdefault("reference", caps[-1]["id"])
    if p["reference"] not in ids:
        raise ValueError(f"reference {p['reference']} is not a capture id")
    return p


def work(p: dict, *parts: str) -> Path:
    d = Path(p.get("work_dir", "work"))
    if not d.is_absolute():
        d = p["_root"] / d
    out = d.joinpath(*parts)
    return out


def file(p: dict, rel: str) -> Path:
    f = Path(rel)
    return f if f.is_absolute() else p["_root"] / f


def year(c: dict) -> int:
    return int(str(c.get("date", c["id"]))[:4])


def canvas(p: dict) -> tuple[int, int]:
    return CANVAS[p["orientation"]]


def delivery(p: dict) -> tuple[int, int]:
    d = p.get("delivery")
    return tuple(d) if d else DELIVERY[p["orientation"]]


def transition_key(a: dict, b: dict) -> str:
    return f"{a['id']}>{b['id']}"


def clone(p: dict) -> dict:
    return copy.deepcopy(p)


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scaffold(captures_dir: str | Path, out: str | Path, scene: str | None = None) -> Path:
    """Write a starter project.json from a folder of captures named by date
    (e.g. 2009-08.png, roadview_2014_05.jpg)."""
    import re

    from .util import write_json

    cdir = Path(captures_dir).resolve()
    out = Path(out).resolve()
    files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if len(files) < 2:
        raise ValueError(f"need at least two capture images in {cdir}")
    caps, seen = [], set()
    for f in files:
        m = re.search(r"((?:19|20)\d{2})(?:[-_. ]?(0[1-9]|1[0-2]))?", f.stem)
        date = f"{m.group(1)}-{m.group(2)}" if m and m.group(2) else (m.group(1) if m else None)
        cid = date if date and date not in seen else f.stem
        seen.add(cid)
        try:
            rel = f.relative_to(out.parent).as_posix()
        except ValueError:
            rel = str(f)
        caps.append({"id": cid, "file": rel, "date": date or cid})
    proj = {
        "name": out.parent.name or "roadview",
        "scene": scene or "a city street lined with shop buildings",
        "orientation": "landscape",
        "captures": caps,
        "transitions": {},
        "signs": [],
    }
    write_json(out, proj)
    return out
