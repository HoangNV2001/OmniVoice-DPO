import os

import pytest

from omnivoice_dpo.tags import TagInventory, UnknownTagError

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "tags_vi.yaml")


@pytest.fixture(scope="module")
def inv():
    return TagInventory.from_yaml(CFG)


def test_inventory_size(inv):
    assert len(inv) == 68
    assert [len(inv.by_category(c)) for c in ("NV", "FIL", "EMO")] == [14, 15, 39]


def test_parse_positions_and_categories(inv):
    text = "[sighs] Thôi được… [ừm] để em kiểm tra lại [clears throat] cái booking nhé. [sad]"
    spans = inv.parse(text)
    assert [s.tag.name for s in spans] == ["sighs", "ừm", "clears throat", "sad"]
    assert [s.tag.category for s in spans] == ["NV", "FIL", "NV", "EMO"]
    assert [s.word_index for s in spans] == [0, 2, 7, 10]
    assert text[spans[2].start:spans[2].end] == "[clears throat]"


def test_longest_match_wins(inv):
    assert [s.tag.name for s in inv.parse("[uh huh] vâng [uh] ạ")] == ["uh huh", "uh"]


def test_strip(inv):
    text = "[sighs] Thôi được… [ừm] để em xem nhé."
    assert inv.strip(text) == "Thôi được… để em xem nhé."
    assert inv.strip(text, keep_fillers=True) == "Thôi được… ừm để em xem nhé."


def test_unknown_tag(inv):
    with pytest.raises(UnknownTagError):
        inv.parse("[rage] KHÔNG THỂ CHẤP NHẬN ĐƯỢC!")
    assert inv.parse("[rage] KHÔNG THỂ [sighs]", strict=False)[0].tag.name == "sighs"
