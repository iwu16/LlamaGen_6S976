"""
Sanity check for CGZ watermarking logic — runs on CPU with no model weights.

Tests (matching the proposal's Step 5 validation):
  1.  PRF / get_green_mask:  half-size, determinism, position & key sensitivity.
  2.  Detection score on a hand-crafted perfectly-watermarked sequence → L - m.
  3.  Detection score on random sequences → mean ≈ (L - m) / 2.
  4.  Threshold formula and detect() decision on watermarked vs. random.
  5.  cgz_sample_from_logits: sampled token always in green set; probs sum to 1.
  6.  Full mock generation loop (model replaced by random logits) → score = L - m.
  7.  Wrong key: watermarked sequence scores near (L - m) / 2 under a different key.

Run:
    python watermark/sanity_check.py
"""

import sys
import os
import math
import torch

# Allow running from repo root or from watermark/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark.cgz_watermark import (
    get_green_mask,
    cgz_sample_from_logits,
    compute_detection_score,
    compute_threshold,
    detect,
    VOCAB_SIZE,
    DEFAULT_CONTEXT_LEN,
)

SECRET_KEY = b"test_secret_key_cgz_6S976"
CONTEXT_LEN = DEFAULT_CONTEXT_LEN   # 4
L = 256       # 16×16 image token grid (256×256 / ds16)
ALPHA = 0.01  # target false-positive rate for Theorem 2 threshold


# ---------------------------------------------------------------------------
# Helper: build a perfectly-watermarked sequence without a model.
# For each position t >= CONTEXT_LEN, pick the first green token.
# The detection score will be exactly L - CONTEXT_LEN (Theorem 3 completeness).
# ---------------------------------------------------------------------------

def build_watermarked_sequence(
    secret_key: bytes = SECRET_KEY,
    context_len: int = CONTEXT_LEN,
    length: int = L,
    vocab_size: int = VOCAB_SIZE,
) -> list:
    tokens = [0] * context_len   # arbitrary initialization (Definition 1)
    for t in range(context_len, length):
        context = tuple(tokens[t - context_len : t])
        mask = get_green_mask(secret_key, position=t, context=context, vocab_size=vocab_size)
        tokens.append(mask.nonzero()[0].item())   # first green token
    return tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prf():
    print("=== Test 1: PRF / get_green_mask ===")

    mask = get_green_mask(SECRET_KEY, position=5, context=(1, 2, 3, 4))

    # Exactly half the vocab is green
    n_green = mask.sum().item()
    assert n_green == VOCAB_SIZE // 2, f"expected {VOCAB_SIZE//2} green tokens, got {n_green}"
    print(f"  green tokens: {n_green} / {VOCAB_SIZE}  ✓")

    # Deterministic: same inputs → same mask
    mask2 = get_green_mask(SECRET_KEY, position=5, context=(1, 2, 3, 4))
    assert torch.equal(mask, mask2), "mask is not deterministic"
    print("  determinism  ✓")

    # Different position → different mask (nearly independent partitions)
    mask3 = get_green_mask(SECRET_KEY, position=6, context=(1, 2, 3, 4))
    overlap = (mask & mask3).sum().item()
    expected_overlap = VOCAB_SIZE // 4
    print(f"  overlap with different position: {overlap}  (expected ≈ {expected_overlap})")
    assert abs(overlap - expected_overlap) < VOCAB_SIZE // 8, "masks too correlated across positions"
    print("  position sensitivity  ✓")

    # Wrong key → different mask
    mask4 = get_green_mask(b"wrong_key_xyz", position=5, context=(1, 2, 3, 4))
    assert not torch.equal(mask, mask4), "wrong key produced same mask"
    print("  key sensitivity  ✓")


def test_detection_score_watermarked():
    print("\n=== Test 2: Detection score on perfectly-watermarked sequence ===")
    tokens = build_watermarked_sequence()
    score = compute_detection_score(tokens, SECRET_KEY, CONTEXT_LEN)
    expected = L - CONTEXT_LEN
    print(f"  score = {score}  /  expected = {expected}  (L - m)")
    assert score == expected, f"expected perfect score {expected}, got {score}"
    print("  Theorem 3 completeness holds  ✓")


def test_detection_score_random():
    print("\n=== Test 3: Detection score on random (unwatermarked) sequences ===")
    torch.manual_seed(42)
    N = 30
    scores = []
    for _ in range(N):
        tokens = torch.randint(0, VOCAB_SIZE, (L,)).tolist()
        scores.append(compute_detection_score(tokens, SECRET_KEY, CONTEXT_LEN))

    mean_score = sum(scores) / len(scores)
    expected_mean = (L - CONTEXT_LEN) / 2.0
    std_expected = math.sqrt((L - CONTEXT_LEN) * 0.25)   # std of Binomial(n, 0.5)
    print(f"  scores: {scores}")
    print(f"  mean = {mean_score:.2f},  expected ≈ {expected_mean:.2f} ± {std_expected:.2f}")
    # Allow 4 standard deviations for safety (failure probability < 0.01%)
    assert abs(mean_score - expected_mean) < 4 * std_expected, (
        f"mean {mean_score:.1f} too far from expected {expected_mean:.1f}"
    )
    print("  Theorem 2 soundness: random scores cluster near (L-m)/2  ✓")


def test_threshold_and_detect():
    print("\n=== Test 4: Threshold formula and detect() ===")
    tau = compute_threshold(L, CONTEXT_LEN, ALPHA)
    expected_clean = (L - CONTEXT_LEN) / 2.0
    expected_wm = L - CONTEXT_LEN
    print(f"  L={L}, m={CONTEXT_LEN}, alpha={ALPHA}")
    print(f"  tau = {tau:.2f}")
    print(f"  expected_clean = {expected_clean:.1f},  expected_wm = {expected_wm}")
    assert tau < expected_wm, "threshold must be below perfect watermark score (completeness)"
    assert tau > expected_clean, "threshold must exceed random-chance mean (soundness separation)"
    print("  threshold sits between clean mean and perfect score  ✓")

    # Watermarked sequence → detected
    wm_tokens = build_watermarked_sequence()
    res_wm = detect(wm_tokens, SECRET_KEY, CONTEXT_LEN, ALPHA)
    print(f"\n  watermarked: {res_wm}")
    assert res_wm["detected"], "watermarked sequence must be detected"
    assert res_wm["fraction_green"] == 1.0, "fraction_green should be 1.0 for perfect sequence"
    print("  watermarked sequence detected  ✓")

    # Random sequence → should not be detected
    torch.manual_seed(99)
    rand_tokens = torch.randint(0, VOCAB_SIZE, (L,)).tolist()
    res_rand = detect(rand_tokens, SECRET_KEY, CONTEXT_LEN, ALPHA)
    print(f"\n  random:      {res_rand}")
    # Theorem 2: each individual random sequence is detected with probability <= alpha = 1%.
    # We just print the result rather than asserting, since ~1% chance it could trigger.
    print(f"  random detected = {res_rand['detected']}  (expected False with prob ≥ 99%)")


def test_cgz_sample_always_green():
    print("\n=== Test 5: cgz_sample_from_logits — sampled token always in green set ===")
    torch.manual_seed(0)
    N = 100
    for trial in range(N):
        logits = torch.randn(1, 1, VOCAB_SIZE)
        position = CONTEXT_LEN + trial
        context = tuple(range(CONTEXT_LEN))   # dummy context

        next_token, green_probs = cgz_sample_from_logits(
            logits,
            SECRET_KEY,
            position=position,
            context=context,
            temperature=1.0,
            top_k=2000,   # LlamaGen default
            top_p=1.0,
        )

        token_id = next_token[0, 0].item()
        green_mask = get_green_mask(SECRET_KEY, position, context)
        assert green_mask[token_id].item(), (
            f"trial {trial}: token {token_id} not in green set at position {position}"
        )
        assert abs(green_probs.sum().item() - 1.0) < 1e-4, (
            f"trial {trial}: green_probs sums to {green_probs.sum().item()}, not 1"
        )

    print(f"  {N} trials: every sample in green set  ✓")
    print(f"  {N} trials: green_probs sums to 1      ✓")

    # Also test with top_k=0 (no filtering) and verify
    torch.manual_seed(1)
    for trial in range(N):
        logits = torch.randn(1, 1, VOCAB_SIZE)
        position = CONTEXT_LEN + trial
        context = (trial % 100, trial % 200, trial % 300, trial % 400)
        next_token, _ = cgz_sample_from_logits(
            logits, SECRET_KEY, position=position, context=context,
            temperature=0.9, top_k=0, top_p=1.0,
        )
        token_id = next_token[0, 0].item()
        green_mask = get_green_mask(SECRET_KEY, position, context)
        assert green_mask[token_id].item(), f"no-filter trial {trial}: not green"
    print(f"  {N} trials with top_k=0: every sample in green set  ✓")


def test_full_mock_generation():
    print("\n=== Test 6: Mock generation loop — full pipeline without model weights ===")
    torch.manual_seed(7)

    # Simulate the cgz_generate decode loop using random logits in place of the model.
    # This tests that the interaction of cgz_sample_from_logits + compute_detection_score
    # gives a perfect score, exactly as predicted by Theorem 3.
    generated = []
    for i in range(L):
        logits = torch.randn(1, 1, VOCAB_SIZE)   # mock model output
        if i < CONTEXT_LEN:
            # initialization phase: clean sampling
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            tok = torch.multinomial(probs, 1)[0, 0].item()
        else:
            context = tuple(generated[i - CONTEXT_LEN : i])
            tok_tensor, _ = cgz_sample_from_logits(
                logits, SECRET_KEY, position=i, context=context,
                temperature=1.0, top_k=0, top_p=1.0,
            )
            tok = tok_tensor[0, 0].item()
        generated.append(tok)

    score = compute_detection_score(generated, SECRET_KEY, CONTEXT_LEN)
    expected = L - CONTEXT_LEN
    fraction = score / expected
    print(f"  score = {score} / {expected}  (fraction = {fraction:.3f})")
    assert score == expected, f"expected perfect score {expected}, got {score}"
    print("  mock watermarked sequence: score = L - m  ✓  (Theorem 3)")

    # Also verify detect() agrees
    result = detect(generated, SECRET_KEY, CONTEXT_LEN, ALPHA)
    assert result["detected"], "detect() should return True for the mock watermarked sequence"
    print(f"  detect() result: {result}  ✓")


def test_wrong_key():
    print("\n=== Test 7: Wrong key — watermarked sequence scores near (L-m)/2 ===")
    wm_tokens = build_watermarked_sequence()

    # Under the correct key: perfect score
    score_correct = compute_detection_score(wm_tokens, SECRET_KEY, CONTEXT_LEN)
    tau = compute_threshold(L, CONTEXT_LEN, ALPHA)

    # Under a wrong key: score should be near (L-m)/2
    score_wrong = compute_detection_score(wm_tokens, b"attacker_key_does_not_know", CONTEXT_LEN)
    expected_wrong = (L - CONTEXT_LEN) / 2.0

    print(f"  correct key score: {score_correct}  (= L - m = {L - CONTEXT_LEN})")
    print(f"  wrong   key score: {score_wrong}  (expected ≈ {expected_wrong:.1f})")
    print(f"  threshold tau:     {tau:.2f}")
    assert score_correct > tau, "correct key must exceed threshold"
    # Wrong key score near half — not strictly asserting < tau since it's probabilistic,
    # but the gap should be obvious.
    gap = score_correct - score_wrong
    print(f"  gap between correct and wrong key: {gap}")
    assert gap > (L - CONTEXT_LEN) // 3, (
        f"gap {gap} unexpectedly small — PRF may not be providing enough independence"
    )
    print("  key specificity verified  ✓")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_scheme_parameters():
    tau = compute_threshold(L, CONTEXT_LEN, ALPHA)
    n = L - CONTEXT_LEN
    print("\n--- CGZ Scheme Parameters (LlamaGen c2i, 256×256 image) ---")
    print(f"  vocab size V     = {VOCAB_SIZE}")
    print(f"  sequence length L = {L}  (16×16 token grid)")
    print(f"  context length m  = {CONTEXT_LEN}")
    print(f"  watermarked steps = L - m = {n}")
    print(f"  false-positive rate alpha = {ALPHA}")
    print(f"  threshold tau     = {tau:.2f}")
    print(f"  expected score (clean):      {n/2:.1f}")
    print(f"  expected score (watermarked): {n}")
    print(f"  separation: {n - tau:.2f} std-devs above threshold")
    print()


if __name__ == "__main__":
    print("CGZ Watermark Sanity Check")
    print(f"  vocab_size={VOCAB_SIZE}, L={L}, m={CONTEXT_LEN}, alpha={ALPHA}")
    print("=" * 65)
    print_scheme_parameters()

    test_prf()
    test_detection_score_watermarked()
    test_detection_score_random()
    test_threshold_and_detect()
    test_cgz_sample_always_green()
    test_full_mock_generation()
    test_wrong_key()

    print("\n" + "=" * 65)
    print("All tests passed.")
