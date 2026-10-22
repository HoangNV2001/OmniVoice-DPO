import os

import pytest
import torch

pytest.importorskip("omnivoice")
transformers = pytest.importorskip("transformers")

from omnivoice.models.omnivoice import OmniVoiceConfig  # noqa: E402

from omnivoice_dpo.prosody.data import collate_prosody  # noqa: E402
from omnivoice_dpo.prosody.model import FSQ, OmniVoiceProsody, scatter_to_positions  # noqa: E402

C = 8


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    llm = transformers.Qwen3Config(vocab_size=1100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    cfg = OmniVoiceConfig(llm_config=llm)
    cfg.prosody = {"dim": 16, "n_blocks": 2}
    return OmniVoiceProsody(cfg).eval()


def pros_batch(L=20, T=30):
    # one utterance, 3 syllables on positions 4..9
    feats = torch.randn(1, T, 11)
    syl = torch.tensor([[0, 0, 10], [0, 10, 18], [0, 18, 30]])
    tok = torch.full((1, L), -1)
    tok[0, 4:6], tok[0, 6:8], tok[0, 8:10] = 0, 1, 2
    return dict(pros_feats=feats, pros_flen=torch.tensor([T]), pros_syl=syl, pros_extra=torch.zeros(3, 2),
                pros_keep=torch.ones(3), pros_tok=tok)


def test_fsq_levels_and_indices():
    q = FSQ([8, 5])
    z = torch.randn(1000, 2) * 5
    c = q(z)
    idx = q.indices(c)
    assert idx[:, 0].min() >= 0 and idx[:, 0].max() <= 7 and idx[:, 1].max() <= 4
    assert len(torch.unique(idx[:, 0])) == 8 and len(torch.unique(idx[:, 1])) == 5
    assert torch.allclose(q.from_indices(idx), c)


def test_zero_init_keeps_base_output(tiny):
    ids = torch.randint(0, 300, (1, C, 20))
    am = torch.zeros(1, 20, dtype=torch.bool)
    am[0, 12:] = True
    with torch.no_grad():
        base = tiny(input_ids=ids, audio_mask=am).logits
        withp = tiny(input_ids=ids, audio_mask=am, **pros_batch()).logits
    assert torch.equal(base, withp)


def test_dropped_syllables_add_nothing_and_grad_flows(tiny):
    torch.nn.init.normal_(tiny.prosody_encoder.out.weight)
    b = pros_batch()
    b["pros_keep"] = torch.tensor([1.0, 0.0, 1.0])
    add = tiny.prosody_add(**b)
    assert add[0, 6:8].abs().sum() == 0 and add[0, 4:6].abs().sum() > 0 and add[0, :4].abs().sum() == 0
    ids = torch.randint(0, 300, (1, C, 20))
    am = torch.zeros(1, 20, dtype=torch.bool)
    am[0, 12:] = True
    labels = torch.full((1, C, 20), -100)
    labels[0, :, 12:] = torch.randint(0, 1024, (C, 8))
    tiny.train()
    tiny(input_ids=ids, audio_mask=am, labels=labels, **pros_batch()).loss.backward()
    assert tiny.prosody_encoder.inp.weight.grad.abs().sum() > 0
    tiny.zero_grad()
    tiny.eval()
    torch.nn.init.zeros_(tiny.prosody_encoder.out.weight)


def test_scatter_positions():
    emb = torch.arange(6.0).view(3, 2)
    tok = torch.tensor([[-1, 2, 0, -1]])
    assert scatter_to_positions(emb, tok).tolist() == [[[0, 0], [4, 5], [0, 1], [0, 0]]]


def test_collate_packed_offsets():
    def s(length, n_syl, T):
        tok = torch.full((length,), -1)
        tok[1:1 + n_syl] = torch.arange(n_syl)
        return {"length": length, "pros": {"feats": torch.ones(T, 11), "span": torch.tensor([[i, i + 1] for i in range(n_syl)]),
                                           "extra": torch.zeros(n_syl, 2), "keep": torch.ones(n_syl), "tok": tok}}
    out = collate_prosody([s(5, 2, 4), {"length": 3}, s(6, 3, 7)], 1, 20, packed=True)
    assert out["pros_feats"].shape == (2, 7, 11) and out["pros_flen"].tolist() == [4, 7]
    assert out["pros_syl"][:, 0].tolist() == [0, 0, 1, 1, 1]
    assert out["pros_tok"][0, :14].tolist() == [-1, 0, 1, -1, -1, -1, -1, -1, -1, 2, 3, 4, -1, -1]


@pytest.mark.skipif(not os.environ.get("OV_TOKENIZER"), reason="set OV_TOKENIZER to an OmniVoice checkpoint dir")
def test_offsets_match_processor_tokens():
    from transformers import AutoTokenizer

    from omnivoice_dpo.prosody.text import syllables, token_to_syllable

    tok = AutoTokenizer.from_pretrained(os.environ["OV_TOKENIZER"])
    text = "Thật sao? Chiều nay con về sớm, cả nhà mình ăn mừng nhé."
    s = f"<|text_start|>{text}<|text_end|>"
    enc = tok(s, return_offsets_mapping=True)
    assert enc["input_ids"] == tok(s).input_ids
    t2s = token_to_syllable(enc["offset_mapping"], syllables(text), len("<|text_start|>"))
    spans = syllables(text)
    for (a, b), k in zip(enc["offset_mapping"], t2s):
        piece = s[a:b].strip()
        if k >= 0:
            assert piece and piece in text[spans[k][0]:spans[k][1]]
    assert sorted(set(k for k in t2s if k >= 0)) == list(range(len(spans)))


def test_batch_without_prosody_touches_encoder(tiny):
    ids = torch.randint(0, 300, (1, C, 20))
    am = torch.zeros(1, 20, dtype=torch.bool)
    am[0, 12:] = True
    labels = torch.full((1, C, 20), -100)
    labels[0, :, 12:] = torch.randint(0, 1024, (C, 8))
    tiny.train()
    tiny(input_ids=ids, audio_mask=am, labels=labels).loss.backward()
    assert all(p.grad is not None for p in tiny.prosody_encoder.parameters())
    tiny.zero_grad()
    tiny.eval()
