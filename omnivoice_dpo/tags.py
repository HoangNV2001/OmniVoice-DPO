"""Inline tag inventory and parsing.

A tag is a bracketed control written inside the text to synthesize, e.g. ``[laughs]`` or ``[sad]``. Three categories:

- ``NV``: a non-verbal event that takes time (laugh, cough). Has a duration prior and the AudioSet labels a detector
  may report for it.
- ``FIL``: a filler that is spoken (``[ừm]``). ``spoken`` is the word expected in an ASR transcript.
- ``EMO``: an emotion or delivery style for the following speech. Takes no time of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import yaml

CATEGORIES = ("NV", "FIL", "EMO")
_ANY_TAG = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class Tag:
    name: str
    category: str
    duration: float = 0.0
    spoken: str | None = None
    audioset: tuple[str, ...] = field(default_factory=tuple)

    @property
    def token(self) -> str:
        return f"[{self.name}]"


@dataclass(frozen=True)
class TagSpan:
    tag: Tag
    start: int  # character offsets in the original text
    end: int
    word_index: int  # number of words before the tag in the tag-free text


class UnknownTagError(ValueError):
    pass


class TagInventory:
    def __init__(self, tags: Iterable[Tag]):
        self.tags: dict[str, Tag] = {}
        for t in tags:
            if t.category not in CATEGORIES:
                raise ValueError(f"{t.token}: unknown category {t.category!r}")
            if t.name in self.tags:
                raise ValueError(f"duplicate tag {t.token}")
            self.tags[t.name] = t
        names = sorted(self.tags, key=len, reverse=True)
        self.pattern = re.compile(r"\[(" + "|".join(re.escape(n) for n in names) + r")\]")

    @classmethod
    def from_yaml(cls, path: str) -> "TagInventory":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tags = []
        for cat in CATEGORIES:
            defaults = cfg.get(cat, {}).get("defaults", {})
            for item in cfg.get(cat, {}).get("tags", []):
                item = {"name": item} if isinstance(item, str) else dict(item)
                kw = {**defaults, **item}
                if cat == "FIL":
                    kw.setdefault("spoken", kw["name"])
                kw["audioset"] = tuple(kw.get("audioset", ()))
                tags.append(Tag(category=cat, **kw))
        return cls(tags)

    def __len__(self) -> int:
        return len(self.tags)

    def by_category(self, category: str) -> list[Tag]:
        return [t for t in self.tags.values() if t.category == category]

    def parse(self, text: str, strict: bool = True) -> list[TagSpan]:
        """Tags in order of appearance. ``strict`` raises on bracketed text that is not in the inventory."""
        if strict:
            unknown = [m.group(0) for m in _ANY_TAG.finditer(text) if m.group(1) not in self.tags]
            if unknown:
                raise UnknownTagError(f"tags not in inventory: {unknown}")
        spans, last, words = [], 0, 0
        for m in self.pattern.finditer(text):
            words += len(text[last:m.start()].split())
            spans.append(TagSpan(self.tags[m.group(1)], m.start(), m.end(), words))
            last = m.end()
        return spans

    def strip(self, text: str, keep_fillers: bool = False) -> str:
        """Text without tags (the reference for content CER). ``keep_fillers`` writes FIL tags as their word."""

        def repl(m: re.Match) -> str:
            t = self.tags[m.group(1)]
            return f" {t.spoken} " if keep_fillers and t.category == "FIL" else " "

        return re.sub(r"\s+", " ", self.pattern.sub(repl, text)).strip()
