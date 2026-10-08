"""Generation plan and runner for MiniMax H3 on Sogni (sogni-agent CLI).

`plan` only writes prompts + commands (no cost).  `generate --execute` runs
them one at a time: all HOLD clips first, then each TRANSITION, whose first
frame is the locked last frame of the previous hold (exact seam).

Set SOGNI_AGENT to override the executable, e.g.
  SOGNI_AGENT="npx -y @sogni-ai/sogni-creative-agent-skill@3.55.0"
The API key comes from SOGNI_API_KEY or ~/.config/sogni/credentials and is
never written to the plan or logs.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from . import h3
from . import project as P
from .util import log, read_json, write_json


def hold_show_frames(p: dict, i: int) -> int:
    v = p["video"]
    n = len(p["captures"])
    sec = v["first_hold_seconds"] if i == 0 else v["last_hold_seconds"] if i == n - 1 else v["hold_seconds"]
    return max(2, min(int(round(sec * v["fps"])), h3.snap_frames(v["hold_frames"])))


def _rel(p: dict, path: Path) -> str:
    try:
        return str(path.relative_to(p["_root"]))
    except ValueError:
        return str(path)


def build_plan(p: dict) -> dict:
    v = p["video"]
    tier = v["tier"]
    if tier not in h3.TIERS:
        raise ValueError(f"video.tier must be one of {list(h3.TIERS)}")
    sel_i2v, sel_flf, spark_s = h3.TIERS[tier]
    cw, ch = P.canvas(p)
    caps = p["captures"]
    loras = v.get("loras") or []
    segs = []
    hold_frames = h3.snap_frames(v["hold_frames"])
    for i, c in enumerate(caps):
        sid = f"hold_{c['id']}"
        prompt = h3.hold_prompt(p["scene"], hold_frames, c.get("crowd", "moderate"), loras)
        segs.append({
            "id": sid, "kind": "hold", "capture": c["id"], "selector": sel_i2v, "frames": hold_frames,
            "show_frames": hold_show_frames(p, i),
            "first": _rel(p, P.work(p, "aligned", "canvas", f"{c['id']}.png")),
            "out": _rel(p, P.work(p, "clips", f"{sid}.mp4")),
            "prompt_file": _rel(p, P.work(p, "prompts", f"{sid}.txt")), "prompt": prompt,
        })
    for a, b in zip(caps[:-1], caps[1:]):
        sid = f"trans_{a['id']}_{b['id']}"
        gap = P.year(b) - P.year(a)
        frames = h3.transition_frames(v["transition_frames"], gap)
        changes = (p["transitions"].get(P.transition_key(a, b)) or {}).get("changes", "")
        prompt = h3.transition_prompt(p["scene"], P.year(a), P.year(b), frames, changes, loras)
        segs.append({
            "id": sid, "kind": "transition", "from": a["id"], "to": b["id"], "selector": sel_flf, "frames": frames,
            "show_frames": int(round(v["transition_seconds"] * v["fps"])),
            "first": _rel(p, P.work(p, "seams", f"{a['id']}_end.png")),
            "last": _rel(p, P.work(p, "aligned", "canvas", f"{b['id']}.png")),
            "depends_on": f"hold_{a['id']}",
            "out": _rel(p, P.work(p, "clips", f"{sid}.mp4")),
            "prompt_file": _rel(p, P.work(p, "prompts", f"{sid}.txt")), "prompt": prompt,
        })
    gen_seconds = sum(s["frames"] / h3.FPS for s in segs)
    plan = {
        "project": p["name"], "tier": tier, "canvas": [cw, ch], "fps": h3.FPS,
        "segments": segs,
        "estimate": {
            "generated_seconds": round(gen_seconds, 2),
            "spark_if_not_covered": round(gen_seconds * spark_s),
            "usd_if_not_covered": round(gen_seconds * spark_s * 0.005, 2),
            "note": "Standard H3 runs one job at a time on Sogni Unlimited; with billing_mode=subscription "
                    "a job that Unlimited cannot cover is refused (4078/4080) instead of charging Spark.",
        },
    }
    for s in segs:
        pf = p["_root"] / s["prompt_file"]
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(s["prompt"], encoding="utf-8")
        s["command"] = command(p, s)
    write_json(P.work(p, "plan.json"), plan)
    return plan


def _exe() -> list[str]:
    env = os.environ.get("SOGNI_AGENT")
    if env:
        return shlex.split(env, posix=os.name != "nt")
    if os.name == "nt":
        # npm's sogni-agent.cmd goes through cmd.exe, which breaks the multi-line H3 prompt:
        # run the CLI script with node directly instead.
        try:
            root = subprocess.run("npm root -g", shell=True, capture_output=True, text=True).stdout.strip()
            mjs = Path(root) / "@sogni-ai" / "sogni-creative-agent-skill" / "sogni-agent.mjs"
            if mjs.exists():
                return ["node", str(mjs)]
        except OSError:
            pass
    return ["sogni-agent"]


def command(p: dict, s: dict) -> list[str]:
    v = p["video"]
    cw, ch = P.canvas(p)
    cmd = _exe() + ["--json", "-q", "--video", "-m", s["selector"], "--ref", s["first"]]
    if s.get("last"):
        cmd += ["--ref-end", s["last"]]
    cmd += ["--frames", str(s["frames"]), "-w", str(cw), "-h", str(ch), "--no-expand-prompt",
            "-t", "5400", "-o", s["out"]]
    if v.get("billing_mode"):
        cmd += ["--billing-mode", v["billing_mode"]]
    for l in v.get("loras") or []:
        cmd += ["--lora", l["id"]]
        if "strength" in l:
            cmd += ["--lora-strength", str(l["strength"])]
    cmd.append("@PROMPT_FILE")  # replaced with the prompt text at run time
    return cmd


def run(p: dict, execute: bool = False, only: list[str] | None = None) -> None:
    from . import finish  # seams need the lock pipeline

    plan = read_json(P.work(p, "plan.json"))
    log_path = P.work(p, "clips", "generation_log.json")
    history = read_json(log_path) if log_path.exists() else []
    ordered = [s for s in plan["segments"] if s["kind"] == "hold"] + [s for s in plan["segments"] if s["kind"] == "transition"]
    if execute and not shutil.which(_exe()[0]):
        raise RuntimeError("sogni-agent not found: npm i -g @sogni-ai/sogni-creative-agent-skill@3.55.0 "
                           "(or set SOGNI_AGENT), and set SOGNI_API_KEY")
    for s in ordered:
        if only and s["id"] not in only:
            continue
        out = p["_root"] / s["out"]
        if s["kind"] == "transition":
            seam = p["_root"] / s["first"]
            hold_clip = p["_root"] / next(x["out"] for x in plan["segments"] if x["id"] == s["depends_on"])
            if hold_clip.exists() and not seam.exists():
                finish.make_seam(p, plan, s["depends_on"])
            if not seam.exists():
                log(f"{s['id']}: waiting for {s['depends_on']} (seam frame not available yet)")
                continue
        if out.exists():
            log(f"{s['id']}: already generated -> {s['out']}")
            continue
        prompt = (p["_root"] / s["prompt_file"]).read_text(encoding="utf-8")
        cmd = [prompt if a == "@PROMPT_FILE" else a for a in s["command"]]
        if not execute:
            shown = " ".join(shlex.quote(a) for a in s["command"][:-1]) + f' "$(cat {s["prompt_file"]})"'
            log(f"[dry-run] {s['id']}: {shown}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"{s['id']}: submitting {s['selector']} {s['frames']} frames")
        t0 = time.time()
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        proc = subprocess.run(cmd, cwd=p["_root"], capture_output=True, text=True)
        rec = {"segment": s["id"], "selector": s["selector"], "frames": s["frames"], "started": started,
               "elapsed_s": round(time.time() - t0, 3), "exit_code": proc.returncode}
        try:
            res = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except (json.JSONDecodeError, IndexError):
            res = {"raw_stdout_tail": proc.stdout[-2000:]}
        rec["result"] = {k: res.get(k) for k in ("success", "error", "errorCode", "localPath", "projectId", "model",
                                                  "width", "height", "paymentModel") if k in res}
        if proc.returncode != 0:
            rec["stderr_tail"] = proc.stderr[-2000:]
        history.append(rec)
        write_json(log_path, history)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"{s['id']} failed: {rec['result'] or rec.get('stderr_tail')}. "
                               "Do not resubmit blindly: check `sogni-agent --recent --json` first.")
        log(f"{s['id']}: done in {rec['elapsed_s']:.0f}s -> {s['out']}")
        if s["kind"] == "hold":
            finish.make_seam(p, plan, s["id"])
