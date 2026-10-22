from omnivoice_dpo.prosody.text import has_unalignable, syllables, token_to_syllable


def test_syllables_skip_tags_ipa_and_split_hyphen():
    t = "[sad] Mình bật bốn-gờ rồi /ˈkæp/ nhé."
    got = [t[a:b] for a, b in syllables(t)]
    assert got == ["Mình", "bật", "bốn", "gờ", "rồi", "nhé"]
    assert has_unalignable(t) and has_unalignable("phòng 908") and not has_unalignable("Thật sao?")


def test_token_to_syllable_with_prefix():
    text = "Anh đi nhé."
    s = "<|text_start|>" + text + "<|text_end|>"
    sh = len("<|text_start|>")
    # tokens: special, " Anh"-like pieces, punctuation, special
    offsets = [(0, sh), (sh, sh + 3), (sh + 3, sh + 6), (sh + 6, sh + 8), (sh + 8, sh + 10), (sh + 10, sh + 11),
               (sh + 11, len(s))]
    assert token_to_syllable(offsets, syllables(text), sh) == [-1, 0, 1, 2, 2, -1, -1]
