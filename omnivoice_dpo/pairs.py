"""Pick one pair of candidates per prompt for a listener to compare.

The machine never decides which candidate is better. It only:

1. drops broken candidates: content error above ``max_err``, a tag read out as text, or other words inserted where an
   NV sound should be;
2. among the rest, picks the pair that differs most in which tags were realized (NV event found, filler heard), so a
   listening turn is not spent on two near-identical clips. Ties go to the larger difference in NV event strength, then
   to chance.

    python -m omnivoice_dpo.pairs --checks run/feat/checks.jsonl --out run/pairs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-err", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    groups = collections.defaultdict(list)
    for l in open(a.checks, encoding="utf-8"):
        c = json.loads(l)
        groups[c["id"]].append(c)
    n_valid = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for pid in sorted(groups):
            got, nv = select_pair(sorted(groups[pid], key=lambda c: c["k"]), rng, a.max_err)
            n_valid[nv] += 1
            if got is None:
                continue
            x, y, diff = got
            f.write(json.dumps({"id": pid, "ks": [x["k"], y["k"]], "tag_diff": diff, "n_valid": nv}) + "\n")
    print("valid candidates per prompt:", dict(sorted(n_valid.items())))


if __name__ == "__main__":
    main()
