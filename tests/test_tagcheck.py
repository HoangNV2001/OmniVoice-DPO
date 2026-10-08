import os
import random

import numpy as np

from omnivoice_dpo.pairs import select_pair
from omnivoice_dpo.rewards.tagcheck import check_candidate
from omnivoice_dpo.tags import TagInventory

INV = TagInventory.from_yaml(os.path.join(os.path.dirname(__file__), "..", "configs", "tags_vi.yaml"))
CLASSES = ["Speech", "Sigh", "Laughter", "Cough"]


def windows(events, dur=2.0):
    """AST-like array: one row per 0.25 s window centre, sigmoid probs per class (events: {centre: {class: p}})."""
    cs = np.arange(0.5, dur, 0.25)
    w = np.zeros((len(cs), 1 + len(CLASSES)), np.float32)
    w[:, 0] = cs
    for c, probs in events.items():
        i = int(np.argmin(abs(cs - c)))
        for k, p in probs.items():
            w[i, 1 + CLASSES.index(k)] = p
    return w


def asr(*words):
    return {"text": " ".join(w for w, _, _ in words), "words": [list(w) for w in words]}


def test_nv_and_filler_realized():
    text = "[sighs] Thôi được [ừm] để em xem nhé."
    a = asr(("Thôi", 0.6, 0.8), ("được", 0.8, 1.0), ("ừm", 1.1, 1.3), ("để", 1.4, 1.5), ("em", 1.5, 1.6),
            ("xem", 1.6, 1.8), ("nhé.", 1.8, 2.0))
    c = check_candidate(INV, text, a, windows({0.5: {"Sigh": 0.8}}), CLASSES, 2.0)
    assert c["err"] == 0 and not c["leak"]
    sighs, um = c["tags"]
    assert sighs["ok"] and sighs["ast_pos"] > 0.7 and not sighs["chen"]
    assert um["pos"] == 2 and um["ins"] == ["um"] and um["ok"]


def test_nv_missing_and_wrong_class():
    text = "[sighs] Thôi được rồi."
    a = asr(("Thôi", 0.2, 0.4), ("được", 0.4, 0.6), ("rồi.", 0.6, 0.8))
    c = check_candidate(INV, text, a, windows({0.5: {"Laughter": 0.9}}), CLASSES, 1.0)
    assert c["tags"][0]["ok"] is False


def test_leak_and_content_error():
    c = check_candidate(INV, "[laughs] Chào anh nhé", asr(("laughs", 0.0, 0.3), ("chào", 0.4, 0.6), ("anh", 0.6, 0.8)),
                        windows({}), CLASSES, 1.0)
    assert c["leak"] and c["tags"][0]["leak"]
    assert abs(c["err"] - 1 / 3) < 1e-3  # "nhé" missing


def test_select_pair_prefers_different_realization():
    def cand(k, ok, err=0.0, leak=False):
        return {"k": k, "err": err, "leak": leak, "tags": [{"cat": "NV", "ok": ok, "ast_pos": 0.9 if ok else 0.0}]}
    cands = [cand(0, True), cand(1, True), cand(2, False), cand(3, False, leak=True)]
    (a, b, diff), n_valid = select_pair(cands, random.Random(0))
    assert n_valid == 3 and diff == 1 and {a["k"], b["k"]} & {2} and 3 not in (a["k"], b["k"])
    assert select_pair([cand(0, True), cand(1, True, err=0.5)], random.Random(0))[0] is None
