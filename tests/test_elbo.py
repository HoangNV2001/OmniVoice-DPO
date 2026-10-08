import math

import pytest
import torch

omnivoice = pytest.importorskip("omnivoice")
transformers = pytest.importorskip("transformers")

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig  # noqa: E402

from omnivoice_dpo import elbo  # noqa: E402
from omnivoice_dpo.losses import preference_loss  # noqa: E402

C, MASK, PAD = 8, 1024, 0


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    llm = transformers.Qwen3Config(vocab_size=1100,  # >= 1025: text embeddings are looked up at audio positions too
                                   hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   attn_implementation="eager")
    return OmniVoice(OmniVoiceConfig(llm_config=llm)).eval()


def rec(T, Lp=12, a0=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(0, 300, (C, Lp), generator=g)
    prefix[:, a0:] = torch.randint(0, 1024, (C, Lp - a0), generator=g)
    return prefix, a0, torch.randint(0, 1024, (C, T), generator=g), torch.randint(0, 1024, (C, T), generator=g)


def run(model, items, cors, **kw):
    pre, a0s, tg, cs = [], [], [], []
    for (p, a0, x), c in zip(items, cors):
        pre.append(p); a0s.append(a0); tg.append(x); cs.append(c)
    batch, meta = elbo.build_batch(pre, a0s, tg, cs, MASK, PAD)
    with torch.no_grad():
        return elbo.score(model, batch, meta, cs, len(tg), **kw)


def test_labels_only_on_masked_target():
    p, a0, x, _ = rec(10)
    cor = elbo.sample_corruption(C, 10, 3, 0.1, torch.Generator().manual_seed(1))
    batch, meta = elbo.build_batch([p], [a0], [x], [cor], MASK, PAD)
    assert batch["input_ids"].shape == (3, C, 12 + 10)
    for b, (_, k, lp, lab) in enumerate(meta):
        m = cor.masks[k]
        assert torch.equal(lab != -100, m)
        assert torch.equal(batch["input_ids"][b, :, lp:][m], torch.full((int(m.sum()),), MASK))
        assert batch["audio_mask"][b, a0:].all() and not batch["audio_mask"][b, :a0].any()


def test_padding_invariance(tiny):
    short, long_ = rec(6, seed=1), rec(20, seed=2)
    c1 = elbo.sample_corruption(C, 6, 2, 0.1, torch.Generator().manual_seed(3))
    c2 = elbo.sample_corruption(C, 20, 2, 0.1, torch.Generator().manual_seed(4))
    alone, _ = run(tiny, [short[:3]], [c1])
    both, _ = run(tiny, [short[:3], long_[:3]], [c1, c2])
    assert alone.shape == (1,) and both.shape == (2,)
    assert torch.allclose(alone[0], both[0], atol=1e-4)


def test_identical_ref_gives_log2_and_reversal_flips(tiny):
    p, a0, xc, xr = rec(8, seed=5)
    cor = elbo.sample_corruption(C, 8, 4, 0.1, torch.Generator().manual_seed(6))
    s, _ = run(tiny, [(p, a0, xc), (p, a0, xr)], [cor, cor])
    loss, m = preference_loss(s[:1], s[1:], s[:1], s[1:], beta=10.0)
    assert abs(loss.item() - math.log(2)) < 1e-6 and m["z_mean"] == 0
    l1, m1 = preference_loss(s[:1], s[1:], s[:1] * 0, s[1:] * 0, beta=1.0)
    l2, m2 = preference_loss(s[1:], s[:1], s[1:] * 0, s[:1] * 0, beta=1.0)
    assert abs(m1["z_mean"] + m2["z_mean"]) < 1e-6


def test_reference_gets_no_grad(tiny):
    ref = tiny
    policy = OmniVoice(tiny.config)
    policy.load_state_dict(tiny.state_dict())
    ref.requires_grad_(False)
    p, a0, xc, xr = rec(8, seed=7)
    cor = elbo.sample_corruption(C, 8, 2, 0.1, torch.Generator().manual_seed(8))
    batch, meta = elbo.build_batch([p, p], [a0, a0], [xc, xr], [cor, cor], MASK, PAD)
    pc_pr, _ = elbo.score(policy, batch, meta, [cor, cor], 2)
    with torch.no_grad():
        rc_rr, _ = elbo.score(ref, batch, meta, [cor, cor], 2)
    loss, _ = preference_loss(pc_pr[:1], pc_pr[1:], rc_rr[:1], rc_rr[1:], beta=5.0, lam_nll=0.1)
    loss.backward()
    assert all(q.grad is None for q in ref.parameters())
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in policy.parameters())
