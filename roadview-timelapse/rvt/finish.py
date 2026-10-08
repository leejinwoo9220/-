"""Post stage: lock real pixels, fix frozen subjects, ramp speed, assemble.

Reads work/plan.json and the generated clips in work/clips/, writes
work/final/<name>.mp4 and work/qa/finish_report.json (+ review sheets).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import frozen, lock, retime
from . import project as P
from .qa import contact_sheet
from .util import imread, imwrite, log, read_json, write_json
from .video import Writer, mux_audio, read_frames

SUBJECT_WINDOW = 12  # frames of a hold that define "who is in the anchor frame"


def work_size(p: dict) -> tuple[int, int]:
    """Canvas aspect scaled to cover the delivery size (then centre-cropped)."""
    cw, ch = P.canvas(p)
    dw, dh = P.delivery(p)
    s = max(dw / cw, dh / ch)
    w, h = int(round(cw * s)), int(round(ch * s))
    return w + (w % 2), h + (h % 2)


def crop_delivery(img: np.ndarray, dw: int, dh: int) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = (w - dw) // 2, (h - dh) // 2
    return img[y:y + dh, x:x + dw]


def sign_boxes(p: dict) -> list[list[float]]:
    return [s["box"] for s in p.get("signs", []) if s.get("box")]


def _plate(p: dict, cap_id: str, kind: str) -> np.ndarray:
    return imread(P.work(p, "aligned", kind, f"{cap_id}.png"))


def _seg(plan: dict, sid: str) -> dict:
    return next(s for s in plan["segments"] if s["id"] == sid)


def hold_lock(p: dict, plan: dict, sid: str) -> tuple[lock.HoldLock, dict]:
    seg = _seg(plan, sid)
    plate = _plate(p, seg["capture"], "canvas")
    frames = read_frames(p["_root"] / seg["out"], limit=max(seg["show_frames"], SUBJECT_WINDOW))
    h, w = plate.shape[:2]
    sign = lock.box_mask(sign_boxes(p), w, h)
    return lock.lock_hold(frames, plate, sign, p["finish"]), seg


def make_seam(p: dict, plan: dict, hold_id: str) -> Path:
    hl, seg = hold_lock(p, plan, hold_id)
    n = min(seg["show_frames"], len(hl.frames))
    out = P.work(p, "seams", f"{seg['capture']}_end.png")
    imwrite(out, hl.locked_canvas(n - 1))
    log(f"seam {seg['capture']}: locked frame {n - 1} -> {out.name}")
    return out


def _up(img: np.ndarray, W: int, H: int, interp=cv2.INTER_LANCZOS4) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == (W, H):
        return img
    return cv2.resize(img, (W, H), interpolation=interp)


def _lock_single(gen: np.ndarray, plate: np.ndarray, plate_blur: np.ndarray, plate_work: np.ndarray,
                 sign: np.ndarray, cfg: dict, W: int, H: int) -> np.ndarray:
    static0 = lock.absdiff_max(lock.blurred(gen), plate_blur) < cfg["lock_threshold"] / 2
    g = lock.color_fit(gen, plate, static0)
    a = lock.feather(lock.moving_mask(g, plate_blur, sign, cfg), 2.0)
    return lock.composite(_up(g, W, H), plate_work, _up(a, W, H, cv2.INTER_LINEAR))


class TransitionSource:
    """Lazily builds full-resolution, locked source frames of one transition."""

    def __init__(self, p, frames, dist, start_c, end_c, start_w, end_w, swaps, W, H):
        self.p, self.cfg, self.frames, self.dist = p, p["finish"], frames, dist
        self.start_c, self.end_c, self.start_w, self.end_w = start_c, end_c, start_w, end_w
        self.swaps, self.W, self.H = swaps, W, H
        h, w = start_c.shape[:2]
        self.sign = lock.box_mask(sign_boxes(p), w, h)
        # per sign: feathered box alpha (work res) and its padded bounding rect
        self.boxes = []
        for sw in swaps:
            bm = lock.box_mask([sw["box"]], W, H)
            ys, xs = np.nonzero(bm)
            pad = 6
            rect = (max(0, xs.min() - pad), max(0, ys.min() - pad), 0, 0)
            rect = (rect[0], rect[1], min(W, xs.max() + pad + 1) - rect[0], min(H, ys.max() + pad + 1) - rect[1])
            self.boxes.append((cv2.GaussianBlur(bm.astype(np.float32) / 255, (0, 0), 2.0), rect))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __call__(self, j: int) -> np.ndarray:
        j = int(np.clip(j, 0, len(self.frames) - 1))
        if j in self.cache:
            self.cache.move_to_end(j)
            return self.cache[j]
        f = self._build(j)
        self.cache[j] = f
        if len(self.cache) > 40:
            self.cache.popitem(last=False)
        return f

    def _build(self, j: int) -> np.ndarray:
        cfg, W, H = self.cfg, self.W, self.H
        gen = self.frames[j]
        out = _up(gen, W, H)
        n = len(self.frames)
        # sign pop-swap: exact old sign until the swap frame, exact new sign after,
        # passing occluders (buses, scaffolding) let through
        if self.swaps:
            occ = _up(lock.feather(self.dist.occluders(j, cfg["sign_threshold"]), 1.5), W, H, cv2.INTER_LINEAR)
            for sw, (alpha, rect) in zip(self.swaps, self.boxes):
                use_old = sw["swap_frame"] is not None and j < sw["swap_frame"]
                plate = self.start_w if use_old else self.end_w
                lock.composite_into(out, plate, alpha * (1 - occ), rect)
        L = int(cfg["landing_frames"])
        if L > 0 and j < L:
            k = 1 - j / L
            locked = _lock_single(gen, self.start_c, self.dist.sb, self.start_w, self.sign, cfg, W, H)
            out = cv2.addWeighted(out, 1 - k, locked, k, 0)
        if L > 0 and j >= n - L:
            k = (j - (n - L) + 1) / L
            locked = _lock_single(gen, self.end_c, self.dist.eb, self.end_w, self.sign, cfg, W, H)
            out = cv2.addWeighted(out, 1 - k, locked, k, 0)
        return out


def _strip_masks(events: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "mask"} for e in events]


def run(p: dict) -> dict:
    plan = read_json(P.work(p, "plan.json"))
    missing = [s["out"] for s in plan["segments"] if not (p["_root"] / s["out"]).exists()]
    if missing:
        raise RuntimeError(f"clips not generated yet: {missing}")
    cfg = p["finish"]
    fps = p["video"]["fps"]
    W, H = work_size(p)
    dw, dh = P.delivery(p)
    caps = p["captures"]
    name = p["name"]
    video_tmp = P.work(p, "final", f"{name}.video.mp4")
    writer = Writer(video_tmp, dw, dh, fps, cfg["crf"])
    report: dict = {"work_size": [W, H], "delivery": [dw, dh], "segments": []}
    audio_pieces: list[dict] = []
    review: list[np.ndarray] = []
    review_labels: list[str] = []

    hold_cache: dict[str, tuple] = {}

    def get_hold(cap_id: str):
        # keep at most the current and the next hold in memory
        if cap_id not in hold_cache:
            if len(hold_cache) >= 2:
                hold_cache.pop(next(iter(hold_cache)))
            hold_cache[cap_id] = hold_lock(p, plan, f"hold_{cap_id}")
        return hold_cache[cap_id]

    last_written_work = None
    for i, c in enumerate(caps):
        hl, seg = get_hold(c["id"])
        n_show = min(seg["show_frames"], len(hl.frames))
        plate_w = _up(_plate(p, c["id"], "master"), W, H, cv2.INTER_AREA)
        for t in range(n_show):
            fw = lock.composite(_up(hl.frames[t], W, H), plate_w, _up(hl.alphas[t], W, H, cv2.INTER_LINEAR))
            writer.write(crop_delivery(fw, dw, dh))
            last_written_work = fw
            if t in (0, n_show - 1):
                review.append(crop_delivery(fw, dw, dh))
                review_labels.append(f"hold {c['id']} f{t}")
        motion = hl.motion_fraction()
        hrep = {"segment": seg["id"], "frames_shown": n_show, "motion_area_fraction": round(motion, 4),
                "max_camera_shift_px": round(max(np.hypot(*s) for s in hl.shifts), 3)}
        if motion < 0.003:
            hrep["warning"] = "almost nothing moves in this hold: people/cars may be frozen -> regenerate the hold"
        report["segments"].append(hrep)
        audio_pieces.append({"src": p["_root"] / seg["out"], "start": 0.0, "dur_src": n_show / fps, "dur_out": n_show / fps})
        log(f"hold {c['id']}: {n_show} frames, motion {motion:.2%}")

        if i == len(caps) - 1:
            break
        b = caps[i + 1]
        tseg = _seg(plan, f"trans_{c['id']}_{b['id']}")
        frames = read_frames(p["_root"] / tseg["out"])
        start_c = imread(p["_root"] / tseg["first"])
        end_c = _plate(p, b["id"], "canvas")
        frames = [f if f.shape[:2] == end_c.shape[:2] else cv2.resize(f, (end_c.shape[1], end_c.shape[0]), interpolation=cv2.INTER_AREA)
                  for f in frames]
        hl_b, _ = get_hold(b["id"])
        start_subj = hl.subjects(n_show - SUBJECT_WINDOW, n_show)
        end_subj = hl_b.subjects(0, SUBJECT_WINDOW)
        grace = int(round(cfg["frozen_grace_seconds"] * fps))
        sign_c = lock.box_mask(sign_boxes(p), end_c.shape[1], end_c.shape[0])
        dist = lock.Dist(frames, start_c, end_c, sign_c)
        events = frozen.detect(dist, start_subj, end_subj, grace, cfg["lock_threshold"])
        stalls = frozen.stalled_transients(dist)
        if cfg["fix_frozen"] and events:
            frames = frozen.fix(frames, events, hl.background(), hl_b.background(), start_c, end_c)
            dist = lock.Dist(frames, start_c, end_c, sign_c)
        swaps = lock.sign_swap_frames(dist, sign_boxes(p))
        end_w = _up(_plate(p, b["id"], "master"), W, H, cv2.INTER_AREA)
        src = TransitionSource(p, frames, dist, start_c, end_c, last_written_work, end_w, swaps, W, H)
        n_src = len(frames)
        n_out = tseg["show_frames"] + 1
        tau, speed, peak = retime.curve(n_src, n_out)
        for k in range(1, n_out):
            fw = retime.blend(src, retime.sample_times(tau[k], speed[k], p["video"]["shutter"], n_src))
            writer.write(crop_delivery(fw, dw, dh))
            if k == n_out // 2:
                review.append(crop_delivery(fw, dw, dh))
                review_labels.append(f"{c['id']}->{b['id']} mid")
        if events or stalls:
            dbg = frames[n_src // 2].copy()
            for e in events + stalls:
                x, y, bw, bh = e["bbox"]
                cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 0, 255) if "frozen" in e["kind"] else (0, 200, 255), 2)
            imwrite(P.work(p, "qa", f"frozen_{tseg['id']}.jpg"), dbg)
        report["segments"].append({
            "segment": tseg["id"], "source_frames": n_src, "frames_shown": n_out - 1, "peak_speed": round(peak, 2),
            "frozen_subjects": _strip_masks(events), "frozen_fixed": bool(cfg["fix_frozen"] and events),
            "stalled_transient_warnings": stalls, "sign_swaps": swaps,
        })
        audio_pieces.append({"src": p["_root"] / tseg["out"], "start": 1 / fps, "dur_src": (n_src - 1) / fps,
                             "dur_out": (n_out - 1) / fps})
        log(f"transition {c['id']}->{b['id']}: {n_src} -> {n_out - 1} frames (peak {peak:.1f}x), "
            f"{len(events)} frozen subject(s){' fixed' if cfg['fix_frozen'] and events else ''}, "
            f"{len(stalls)} stall warning(s)")

    writer.close()
    total = writer.count / fps
    final = P.work(p, "final", f"{name}.mp4")
    if p["video"]["audio"] == "none":
        video_tmp.replace(final)
    else:
        mux_audio(video_tmp, final, audio_pieces, total)
        video_tmp.unlink(missing_ok=True)
    contact_sheet(review, review_labels, P.work(p, "qa", "final_frames.jpg"), cell_w=480, cols=3)
    report.update(final=str(final), frames=writer.count, seconds=round(total, 3))
    write_json(P.work(p, "qa", "finish_report.json"), report)
    log(f"final: {final} ({total:.2f}s, {writer.count} frames)")
    return report
