from pathlib import Path

import pytest

from visionguard.inference import decide
from visionguard.validation import sanitize_filename

ROOT = Path(__file__).resolve().parents[1]


def test_decide_on_the_threshold_is_a_coin_flip():
    label, conf = decide(0.75, 0.75)
    assert label == "ai" and conf == pytest.approx(0.5)


def test_decide_extremes():
    assert decide(1.0, 0.75) == ("ai", pytest.approx(1.0))
    assert decide(0.0, 0.75) == ("real", pytest.approx(1.0))


def test_decide_between_half_and_threshold_is_real():
    label, conf = decide(0.6, 0.75)
    assert label == "real" and 0.5 < conf < 1.0


def test_decide_rejects_bad_threshold():
    with pytest.raises(ValueError):
        decide(0.5, 1.0)


@pytest.mark.parametrize("raw, expected", [
    ("holiday photo.JPG", "holiday_photo.JPG"),
    ("../../etc/passwd.png", "etc_passwd.png"),
    ("..\\..\\windows\\evil.jpg", "windows_evil.jpg"),
    ("C:\\Users\\me\\pic.jpg", "C_Users_me_pic.jpg"),
    ("<script>alert(1)</script>.jpg", "scriptalert1_script.jpg"),
    ("", "upload"),
    (None, "upload"),
    ("日本語", "upload"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.skipif(not (ROOT / "models" / "champion.keras").exists(), reason="champion model not downloaded")
def test_champion_model_on_known_samples():
    """Integration check with the real model (skipped in CI, where the model file is absent)."""
    from visionguard.inference import LocalPredictor
    predictor = LocalPredictor(ROOT / "models" / "champion.keras")
    ai = predictor.predict_bytes((ROOT / "samples" / "known_ai.jpg").read_bytes())
    real = predictor.predict_bytes((ROOT / "samples" / "known_real.jpg").read_bytes())
    assert ai["label"] == "ai" and real["label"] == "real"
