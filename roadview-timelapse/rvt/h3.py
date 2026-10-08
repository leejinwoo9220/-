"""MiniMax H3 prompt authoring (official ordered-field contract, as used on Sogni).

Structure of the film (chained so every seam is an exact frame match):

    HOLD 2009 (I2V, real-time)  -> last shown frame E_2009
    TRANS 2009->2014 (FLF2V: E_2009 -> K_2014, fixed-camera time-lapse)
    HOLD 2014 (I2V from K_2014, real-time) -> E_2014
    ...
    HOLD 2026 (I2V from K_2026, real-time)

Why "people/cars freeze while buildings change" happens and what the prompts
do about it:
  * In a first/last-frame clip the people in both anchor frames are treated as
    persistent subjects, so the model keeps them parked while it morphs the
    facades.  The transition prompt therefore makes the anchor people leave in
    the first second and arrive only in the last second, and describes the
    middle as a rush of short-lived, motion-blurred passers-by.
  * Real-time behaviour is never asked of the transition clip; real-time
    footage comes only from the HOLD clips, and speed is applied afterwards in
    post (frame-blended retime), so the speed curve is deterministic.
  * Post-processing (rvt.finish) detects anchor people that still freeze and
    replaces them with background, see rvt.frozen.

H3 rules followed here (Sogni CLI 3.55 reference, MiniMax h3-prompt-writing):
  exact alignment preamble line, three ordered fields, "[Shot 1]" with no
  timestamp, every sentence a visible/audible depiction, exclusions phrased as
  positive facts, static shot when both pictures share one camera position,
  no negative prompt, non_diegetic_music N/A for no score.
"""
from __future__ import annotations

import math

TIERS = {
    # tier: (i2v selector, flf2v selector, Spark per output second at 768p class)
    "standard": ("minimax-h3-i2v", "minimax-h3-flf2v", 16),
    "balanced": ("minimax-h3-i2v-balanced", "minimax-h3-flf2v-balanced", 10),
    "turbo": ("minimax-h3-i2v-turbo", "minimax-h3-flf2v-turbo", 6),
    "fasth3": ("minimax-h3-fasth3-i2v-turbo", "minimax-h3-fasth3-flf2v-turbo", 4),
}

FPS = 24
GRID = [124 + 17 * n for n in range(15)]  # 124 .. 362


def snap_frames(frames: int) -> int:
    return min(GRID, key=lambda g: abs(g - frames))


def seconds_label(frames: int) -> str:
    """H3 alignment-line seconds: frames/24 rounded UP to 2 decimals (243 -> 10.13)."""
    return f"{math.ceil(frames / FPS * 100 - 1e-9) / 100:.2f}"


def transition_frames(setting, year_gap: int) -> int:
    if setting != "auto":
        return snap_frames(int(setting))
    if year_gap <= 3:
        return 124
    if year_gap <= 6:
        return 141
    return 192


CROWD = {
    "light": "A few pedestrians",
    "moderate": "Pedestrians",
    "busy": "A steady crowd of pedestrians",
}


def _trigger(loras: list[dict]) -> str:
    return "r34l1sm, " if any(l.get("id") == "h3-realism-people" for l in loras) else ""


def hold_prompt(scene: str, frames: int, crowd: str = "moderate", loras: list[dict] | None = None) -> str:
    who = CROWD.get(crowd, CROWD["moderate"])
    desc = (
        f"[Shot 1] {_trigger(loras or [])}Live-action, documentary street footage, a Static Shot from a camera locked on a "
        f"tripod at roof height of a car begins exactly from <Picture 1>, framing {scene}. "
        "The picture has the flat perspective of an ordinary lens, with building edges and utility poles standing "
        "straight and upright. "
        "The buildings, storefronts, signboards, utility poles, overhead wires, trees and road markings stay fixed in "
        "place and unchanged for the whole shot, and the daylight and shadows hold steady. "
        f"{who} on both sidewalks walk at an ordinary real-time pace from the first frame to the last, stepping "
        "forward with swinging arms, and the people standing in <Picture 1> start walking away early in the shot. "
        "Cars, taxis, buses and scooters roll along the lanes at normal city speed with turning wheels, entering at "
        "one edge of the frame and leaving at the other, while the parked vehicles stay at the curb. "
        "Early in the shot a car glides through the near lane and a pedestrian near the curb walks along the "
        "sidewalk. Midway, two people pass each other on the sidewalk and a scooter weaves past the parked cars. "
        "Toward the end a van slides through the far lane and new pedestrians step in from the edges of the frame, "
        "still walking as the shot ends."
    )
    sound = (
        "Continuous real-time street ambience: steady footsteps on the pavement, car engines and tyres passing from "
        "side to side, a scooter buzzing by, a distant horn and an indistinct murmur of voices."
    )
    return (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        f"integrated_multimodal_description: {desc}\n\n"
        f"overall_soundscape: {sound}\n\n"
        "non_diegetic_music: N/A\n"
    )


def transition_prompt(scene: str, year_a: int, year_b: int, frames: int, changes: str = "",
                      loras: list[dict] | None = None) -> str:
    end = seconds_label(frames)
    span = f"from {year_a} to {year_b}" if year_b > year_a else "over the following years"
    changes = changes.strip()
    if changes and not changes.endswith("."):
        changes += "."
    if not changes:
        changes = ("Through the middle of the shot the storefronts, facades and signboards gradually take on the "
                   "appearance they have in Picture 2.")
    desc = (
        f"[Shot 1] {_trigger(loras or [])}Live-action, fixed-camera time-lapse footage, a Static Shot from the same tripod "
        f"position begins in the position and framing established by Picture 1, showing {scene} with the flat "
        "perspective of an ordinary lens, building edges and utility poles standing straight and upright. "
        "In the first second the pedestrians and vehicles of the opening moment hurry out of frame and the footage "
        f"accelerates into a time-lapse spanning the years {span}: days and nights flicker past, shadows sweep across "
        "the facades, and the sidewalks and lanes carry a constant rush of translucent, motion-blurred pedestrians "
        "and streaking cars, each one crossing the frame in a fraction of a second, so every person and vehicle is "
        "always in motion. "
        f"{changes} "
        "Each signboard switches to its next design in a single abrupt jump, the new sign appearing whole and sharp. "
        "The colour and angle of the daylight shift with the passing seasons. "
        "Toward the end the flicker slows, the rush of traffic eases back to an ordinary real-time pace, the people "
        "of Picture 2 walk into their places, and the street settles into the buildings, signboards, light, "
        "pedestrians, spacing and composition established by Picture 2 at the end of the shot, with the camera "
        "unmoved throughout."
    )
    sound = (
        "A fast-forwarded rush of city sound: a compressed hiss of traffic, rapid overlapping footsteps and brief "
        "flickers of voices and engines, slowing into ordinary street ambience with passing cars and "
        "footsteps in the final second."
    )
    return (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
        f"mark of the target video; Picture 2 (from Shot 1) aligns with the {end}-second mark of the target video.\n\n"
        f"integrated_multimodal_description: {desc}\n\n"
        f"overall_soundscape: {sound}\n\n"
        "non_diegetic_music: N/A\n"
    )
