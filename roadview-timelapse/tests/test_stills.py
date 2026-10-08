import numpy as np
import pytest

from rvt import prep, synth, undistort
from rvt import project as P


@pytest.mark.parametrize("lam", [-0.35, -0.2, -0.1, 0.0, 0.08])
def test_lens_recovered(lam):
    raw, _ = synth.make_capture(3, lam=lam)
    geom = undistort.solve_capture(raw, {"crop": synth.UI_CROP, "undistort": {"model": "auto"}})
    assert abs(geom.lens.param - lam) < 0.03


def test_undistort_roundtrip():
    lens = undistort.Lens("division", -0.25, 1600, 900)
    r = np.linspace(0, 800, 50)
    rd, ok = lens.src_radius(r)
    assert ok.all()
    np.testing.assert_allclose(lens.undist_radius(rd), r, atol=1e-6)
    fish = undistort.Lens("stereographic", 120.0, 1600, 900)
    rd, ok = fish.src_radius(r)
    np.testing.assert_allclose(fish.undist_radius(rd), r, rtol=1e-6, atol=1e-6)


def test_default_applies_no_lens_warp():
    raw, _ = synth.make_capture(3, lam=0.0)
    geom = undistort.solve_capture(raw, {"crop": synth.UI_CROP})
    assert geom.lens.model == "none"
    out = undistort.undistort(raw, geom)
    np.testing.assert_array_equal(out, undistort.apply_crop(raw, synth.UI_CROP))


def test_fisheye_model_estimate_runs():
    raw, _ = synth.make_capture(3, lam=-0.25)
    geom = undistort.solve_capture(raw, {"crop": synth.UI_CROP, "undistort": {"model": "stereographic"}})
    assert 60 <= geom.lens.param <= 178
    assert geom.out_w > 0 and geom.out_h > 0


def test_prep_aligns_all_eras(tmp_path):
    pj = synth.write_demo_project(tmp_path)
    p = P.load(pj)
    out = prep.run(p)
    assert out["canvas_size"] == [1344, 768]
    assert out["crop_coverage_of_reference"] > 0.6
    for rep in out["registration"]:
        assert rep["residual_flow_vs_reference"]["median_px"] < 1.2, rep
    for c in p["captures"]:
        assert P.work(p, "aligned", "canvas", f"{c['id']}.png").exists()
        assert P.work(p, "undistorted", f"{c['id']}.jpg").exists()
    assert P.work(p, "qa", "blink.gif").exists()
