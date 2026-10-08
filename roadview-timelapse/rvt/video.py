"""Video IO through ffmpeg/ffprobe (frames as BGR uint8 numpy arrays)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(path: str | Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def has_audio(path: str | Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def read_frames(path: str | Path, limit: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while limit is None or len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def write_frames(path: str | Path, frames: list[np.ndarray], fps: int = 24, crf: int = 16) -> None:
    w = Writer(path, frames[0].shape[1], frames[0].shape[0], fps, crf)
    for f in frames:
        w.write(f)
    w.close()


class Writer:
    def __init__(self, path: str | Path, w: int, h: int, fps: int = 24, crf: int = 16, preset: str = "slow"):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.size = (w, h)
        self.count = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame {frame.shape[1]}x{frame.shape[0]} != writer {self.size}")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def atempo_chain(factor: float) -> str:
    """ffmpeg atempo accepts 0.5..2.0 per stage."""
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.6f}")
    return ",".join(parts)


def mux_audio(video: Path, out: Path, pieces: list[dict], total: float) -> None:
    """pieces: [{src, start, dur_src, dur_out}] in order. Missing audio -> silence."""
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    filters, labels = [], []
    k = 1
    for i, pc in enumerate(pieces):
        lab = f"a{i}"
        if pc.get("src") and has_audio(pc["src"]):
            args += ["-i", str(pc["src"])]
            tempo = pc["dur_src"] / pc["dur_out"]
            chain = f"[{k}:a]atrim=start={pc['start']:.4f}:duration={pc['dur_src']:.4f},asetpts=PTS-STARTPTS,"
            chain += "aresample=48000,aformat=channel_layouts=stereo,"
            if abs(tempo - 1) > 1e-3:
                chain += atempo_chain(tempo) + ","
            fo = max(0.0, pc["dur_out"] - 0.02)
            chain += f"apad,atrim=duration={pc['dur_out']:.4f},afade=t=in:d=0.02,afade=t=out:st={fo:.4f}:d=0.02[{lab}]"
            k += 1
        else:
            chain = f"anullsrc=r=48000:cl=stereo,atrim=duration={pc['dur_out']:.4f}[{lab}]"
        filters.append(chain)
        labels.append(f"[{lab}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")
    args += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.4f}", "-movflags", "+faststart", str(out)]
    subprocess.run(args, check=True)
