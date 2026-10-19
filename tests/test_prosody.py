import numpy as np
import pytest

pytest.importorskip("librosa")

from omnivoice_dpo.rewards import prosody  # noqa: E402


def test_inner_pause_and_rate():
    st = np.zeros(300, np.float32)
    db = np.full(300, -10.0, np.float32)
    db[:20] = db[-20:] = -80.0  # leading / trailing silence: not pauses
    db[100:140] = -80.0  # 400 ms inner pause
    db[200:205] = -80.0  # 50 ms dip: too short
    s = prosody.clip_stats(st, db, n_words=10)
    assert s["pauses"] == [[1.0, 0.4]]
    assert s["rate"] == pytest.approx(10 / (2.6 - 0.4), rel=1e-2)


def test_dtw_identical_is_zero_and_shift_is_not():
    t = np.linspace(0, 2 * np.pi, 400)
    st = (3 * np.sin(t)).astype(np.float32)
    db = np.full(400, -10.0, np.float32)
    x = prosody.dtw_features(st, db)
    assert prosody.dtw_distance(x, x) == pytest.approx(0.0, abs=1e-6)
    assert prosody.dtw_distance(x, prosody.dtw_features(st + 2, db)) > 1.0


def test_content_error():
    asr = {"words": [["Anh", 0, 0.2], ["đi", 0.2, 0.4], ["đâu", 0.4, 0.6]]}
    assert prosody.content_error("Anh đi đâu thế?", asr) == pytest.approx(0.25)
