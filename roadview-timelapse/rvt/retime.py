"""Deterministic speed ramp with a simulated shutter.

The transition clip is played 1x at both ends (so it joins the real-time
holds without a jump) and accelerates in the middle.  Each output frame
averages the source frames its "shutter" spans, so faster passages turn
passers-by into streaks - the real time-lapse look - instead of the model
having to fake speed.
"""
from __future__ import annotations

import math

import numpy as np


def curve(n_src: int, n_out: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Source time tau(k) and speed dtau/dk for output frames k = 0..n_out.

    speed(x) = 1 + (P - 1) * sin^2(pi x), x = k / n_out, chosen so that
    tau(0) = 0 and tau(n_out) = n_src - 1.  Returns (tau, speed, peak P)."""
    span = n_src - 1
    if n_out >= span:
        k = np.arange(n_out + 1)
        tau = k * (span / n_out)
        return tau, np.full(n_out + 1, span / n_out), span / n_out
    P = 1 + 2 * (span - n_out) / n_out
    x = np.arange(n_out + 1) / n_out
    tau = n_out * x + (P - 1) * n_out * (x / 2 - np.sin(2 * math.pi * x) / (4 * math.pi))
    speed = 1 + (P - 1) * np.sin(math.pi * x) ** 2
    tau[-1] = span
    return tau, speed, P


def sample_times(tau: float, speed: float, shutter: float, n_src: int) -> np.ndarray:
    width = shutter * max(speed - 1.0, 0.0)
    if width < 1e-3:
        return np.array([tau])
    m = int(min(24, math.ceil(width) + 1))
    ts = np.linspace(tau - width / 2, tau + width / 2, m)
    return np.clip(ts, 0, n_src - 1)


def blend(get_frame, times: np.ndarray) -> np.ndarray:
    """Average of time-interpolated source frames."""
    acc = None
    for t in times:
        i0 = int(math.floor(t))
        f = t - i0
        a = get_frame(i0).astype(np.float32)
        if f > 1e-3:
            a = a * (1 - f) + get_frame(i0 + 1).astype(np.float32) * f
        acc = a if acc is None else acc + a
    return (acc / len(times) + 0.5).astype(np.uint8)
