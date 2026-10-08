import numpy as np

from rvt import frozen, h3, lock, retime
from rvt.project import FINISH_DEFAULTS


def test_h3_frames_and_labels():
    assert h3.snap_frames(120) == 124
    assert h3.snap_frames(144) == 141
    assert h3.snap_frames(192) == 192
    assert h3.seconds_label(124) == "5.17"
    assert h3.seconds_label(192) == "8.00"
    assert h3.seconds_label(243) == "10.13"
    assert h3.transition_frames("auto", 2) == 124
    assert h3.transition_frames("auto", 5) == 141
    assert h3.transition_frames("auto", 9) == 192


def test_h3_prompt_contract():
    i2v = h3.hold_prompt("a street", 124)
    assert i2v.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n")
    flf = h3.transition_prompt("a street", 2009, 2014, 141, "the shop closes")
    assert flf.startswith("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
                          "the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.88-second mark")
    for text in (i2v, flf):
        a = text.index("integrated_multimodal_description: [Shot 1]")
        b = text.index("\n\noverall_soundscape: ")
        c = text.index("\n\nnon_diegetic_music: N/A")
        assert a < b < c
        assert "[Shot 2]" not in text  # one continuous static shot
    assert "the shop closes." in flf
    assert h3.hold_prompt("x", 124, loras=[{"id": "h3-realism-people"}]).count("r34l1sm") == 1


def test_retime_curve():
    tau, speed, peak = retime.curve(141, 63)
    assert tau[0] == 0 and abs(tau[-1] - 140) < 1e-9
    assert np.all(np.diff(tau) > 0)
    assert abs(speed[0] - 1) < 1e-9 and abs(speed[-1] - 1) < 1e-9
    assert peak > 2
    assert len(retime.sample_times(tau[0], speed[0], 1.0, 141)) == 1  # 1x: no blur at the seams
    assert len(retime.sample_times(tau[31], speed[31], 1.0, 141)) > 2  # fast: streaks


def _street(h=192, w=336, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 180, np.uint8)
    img[40:70, 60:160] = (40, 60, 170)  # sign
    img[120:, :] = 90  # road
    return np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def test_lock_hold_restores_sign_and_keeps_walkers():
    plate = _street()
    cfg = dict(FINISH_DEFAULTS, min_blob_frac=0.0005)
    sign = lock.box_mask([[60 / 336, 40 / 192, 100 / 336, 30 / 192]], 336, 192)
    frames = []
    rng = np.random.default_rng(1)
    for t in range(20):
        f = plate.copy()
        f[40:70, 60:160] = np.clip(f[40:70, 60:160].astype(int) + rng.integers(-30, 30, (30, 100, 3)), 0, 255)  # garbled sign
        x = 10 + 8 * t
        f[100:150, x:x + 12] = (20, 200, 20)  # walker
        frames.append(f)
    hl = lock.lock_hold(frames, plate, sign, cfg)
    out = hl.locked_canvas(10)
    assert np.abs(out[42:68, 62:158].astype(int) - plate[42:68, 62:158]).mean() < 2  # sign back to the capture
    assert hl.masks[10][120, 10 + 80 + 6] == 1  # walker kept from the generated frame
    assert out[125, 96, 1] > 150


def test_frozen_subject_detected_and_fixed():
    start = _street()
    end = _street(seed=1)
    end[40:70, 60:160] = (30, 150, 40)  # sign changed
    start[100:150, 200:215] = (20, 20, 200)  # person in the start frame
    frames = []
    for t in range(60):
        a = t / 59
        f = ((1 - a) * _street(seed=2).astype(float) + a * end.astype(float)).astype(np.uint8)
        f[100:150, 200:215] = (20, 20, 200)  # ...who never leaves during the time-lapse
        frames.append(f)
    subj = np.zeros(start.shape[:2], np.uint8)
    subj[100:150, 200:215] = 1
    dist = lock.Dist(frames, start, end)
    events = frozen.detect(dist, subj, np.zeros_like(subj), grace=8)
    assert any(e["kind"] == "start_subject_frozen" and e["to_frame"] - e["from_frame"] > 30 for e in events)
    bg = _street(seed=2)
    fixed = frozen.fix(frames, events, bg, bg, start, end)
    assert np.abs(fixed[30][110:140, 203:212].astype(int) - bg[110:140, 203:212]).mean() < 10


def test_sign_swap_is_single_jump():
    start, end = _street(), _street(seed=1)
    end[40:70, 60:160] = (30, 150, 40)
    frames = [((1 - t / 39) * start.astype(float) + t / 39 * end.astype(float)).astype(np.uint8) for t in range(40)]
    dist = lock.Dist(frames, start, end)
    (sw,) = lock.sign_swap_frames(dist, [[60 / 336, 40 / 192, 100 / 336, 30 / 192]])
    assert 15 <= sw["swap_frame"] <= 25
