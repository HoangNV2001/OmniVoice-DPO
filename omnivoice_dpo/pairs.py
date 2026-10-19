"""Pick one pair of candidates per prompt for a listener to compare.

The machine never decides which candidate is better. It only:

1. drops broken candidates: content error above ``max_err``, a tag read out as text, or other words inserted where an
   NV sound should be;
2. among the rest, picks the pair that differs most in which tags were realized (NV event found, filler heard), so a
   listening turn is not spent on two near-identical clips. Ties go to the larger difference in NV event strength, then
   to chance.

With ``--prosody <feat>`` (prompts without tags, prosody preference) step 1 only drops candidates with a content error
above ``max_err`` and more than one missed word (on a five-word sentence one ASR slip is already 20 %), and step 2 picks
the valid pair with the largest prosodic distance (``rewards.prosody``).

    python -m omnivoice_dpo.pairs --checks run/feat/checks.jsonl --out run/pairs.jsonl
    python -m omnivoice_dpo.pairs --prosody run/feat --out run/pairs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random


def realized(c) -> list[bool]:
    return [bool(t["ok"]) for t in c["tags"] if t["ok"] is not None]


def strength(c) -> float:
    return sum(t.get("ast_pos", 0.0) for t in c["tags"] if t["cat"] == "NV")


def select_pair(cands: list[dict], rng: random.Random, max_err: float = 0.2):
    valid = [c for c in cands if c["err"] <= max_err and not c["leak"] and not any(t.get("chen") for t in c["tags"])]
    if len(valid) < 2:
        return None, len(valid)

    def key(pair):
        a, b = pair
        va, vb = realized(a), realized(b)
        return (sum(x != y for x, y in zip(va, vb)), abs(strength(a) - strength(b)), rng.random())

    a, b = max(itertools.combinations(valid, 2), key=key)
    return (a, b, key((a, b))[0]), len(valid)


def select_pair_prosody(cands: list[dict], dist: dict, max_err: float = 0.2):
    valid = [c for c in cands if c["err"] <= max_err or round(c["err"] * c.get("n_words", 0)) <= 1]
    if len(valid) < 2:
        return None, len(valid)
    d = lambda a, b: dist["dist"][dist["ks"].index(a["k"])][dist["ks"].index(b["k"])]
    a, b = max(itertools.combinations(valid, 2), key=lambda p: (d(*p) if d(*p) == d(*p) else -1.0))  # NaN last
    return (a, b, d(a, b)), len(valid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks", help="checks.jsonl from rewards.tagcheck (tag prompts)")
    ap.add_argument("--prosody", help="feat dir with prosody.jsonl and dist.jsonl from rewards.prosody")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-err", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if not (a.checks or a.prosody):
        ap.error("give --checks or --prosody")
    groups = collections.defaultdict(list)
    src = os.path.join(a.prosody, "prosody.jsonl") if a.prosody else a.checks
    dists = {d["id"]: d for d in map(json.loads, open(os.path.join(a.prosody, "dist.jsonl")))} if a.prosody else {}
    for l in open(src, encoding="utf-8"):
        c = json.loads(l)
        groups[c["id"]].append(c)
    n_valid = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for pid in sorted(groups):
            cands = sorted(groups[pid], key=lambda c: c["k"])
            if a.prosody:
                got, nv = select_pair_prosody(cands, dists[pid], a.max_err)
            else:
                got, nv = select_pair(cands, rng, a.max_err)
            n_valid[nv] += 1
            if got is None:
                continue
            x, y, diff = got
            key = "prosody_dist" if a.prosody else "tag_diff"
            f.write(json.dumps({"id": pid, "ks": [x["k"], y["k"]], key: diff, "n_valid": nv}) + "\n")
    print("valid candidates per prompt:", dict(sorted(n_valid.items())))


if __name__ == "__main__":
    main()
