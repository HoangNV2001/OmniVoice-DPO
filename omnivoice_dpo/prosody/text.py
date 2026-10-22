"""Syllables of a label text and their mapping to text-token positions.

A syllable is a maximal run of letters outside ``[tags]`` and ``/ipa/`` spans (Vietnamese syllables are separated by spaces
or hyphens). Its character span refers to the label text exactly as stored, so token offsets can be matched against it.
"""

from __future__ import annotations

import re
import unicodedata

_SKIP = re.compile(r"\[[^\[\]]*\]|/[^/\s][^/]*/")
_LETTERS = re.compile(r"[^\W\d_]+")


def syllables(text: str) -> list[tuple[int, int]]:
    """Character spans ``(start, end)`` of the syllables of ``text``."""
    skip = [m.span() for m in _SKIP.finditer(text)]
    out = []
    for m in _LETTERS.finditer(text):
        if not any(a <= m.start() < b for a, b in skip):
            out.append(m.span())
    return out


def has_unalignable(text: str) -> bool:
    """IPA spans or digits: the utterance cannot be aligned with a letter-level CTC model."""
    return bool(re.search(r"/[^/\s][^/]*/|\d", text))


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def token_to_syllable(offsets: list[tuple[int, int]], spans: list[tuple[int, int]], shift: int) -> list[int]:
    """For each token (character ``offsets`` in the tokenized string), the index of the syllable it overlaps, else -1.

    ``shift`` is where the label text starts inside the tokenized string (e.g. ``len("<|text_start|>")``). A token that
    overlaps several syllables takes the first one.
    """
    out = []
    j = 0
    for a, b in offsets:
        a, b = a - shift, b - shift
        k = -1
        if b > a:
            while j < len(spans) and spans[j][1] <= a:
                j += 1
            if j < len(spans) and spans[j][0] < b:
                k = j
        out.append(k)
    return out
