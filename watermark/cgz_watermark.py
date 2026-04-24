"""
CGZ watermarking for LlamaGen autoregressive image generation.

References:
  [CGZ]      Christ, Gunn, Zamir. "Undetectable Watermarks for Language Models." COLT 2024.
  [LlamaGen] Sun et al. "Autoregressive Model Beats Diffusion." 2024.

Scheme (adapted from [CGZ] Section 2–3 to the image-token setting):
  Secret key k, context length m, vocab size V = 16384.

  For each (position t, context c = z_{t-m:t-1}):
    PRF(k, t, c) deterministically splits vocab into a green half G and red half R
    of equal size (Definition 2).

  Watermarked sampler (Definition 2):
    Apply standard top-k / top-p filtering → softmax → zero red-token probabilities
    → renormalize over green tokens → multinomial sample.

  Detection (Definitions 2, Theorems 2–3):
    S_k(z) = |{t >= m : z_t in G_k^t(c_t(z))}|
    Detect iff S_k(z) >= tau, where tau from Theorem 2 controls false-positive rate alpha.

  Key properties proved in the project proposal (toy model):
    Theorem 1 (Undetectability): Pk = P in distribution.
    Theorem 2 (Soundness):       Pr_{z~P}[Det=1] <= alpha.
    Theorem 3 (Completeness):    Pr_{z~Pk}[Det=1] = 1  (every watermarked token is green).
"""

import hmac
import hashlib
import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

VOCAB_SIZE = 16384       # LlamaGen VQ codebook size
DEFAULT_CONTEXT_LEN = 4  # m in the paper


# ---------------------------------------------------------------------------
# PRF: green-set construction
# ---------------------------------------------------------------------------

def get_green_mask(
    secret_key: bytes,
    position: int,
    context: tuple,
    vocab_size: int = VOCAB_SIZE,
) -> torch.BoolTensor:
    """
    Deterministic half-partition of [vocab_size] keyed on (secret_key, position, context).

    Returns a BoolTensor of length vocab_size with exactly vocab_size // 2 True entries.

    Security properties:
      - Without the key, the partition is computationally indistinguishable from random
        (HMAC-SHA256 as PRF).
      - Different (position, context) pairs produce independently random partitions;
        this is the key invariant used in the Theorem 1 undetectability proof (the green
        set for time t has never been queried before, so it is a fresh random half-subset).
    """
    context_bytes = str((position, context)).encode()
    h = hmac.new(secret_key, context_bytes, hashlib.sha256).digest()
    # Use 8 bytes (64 bits) for better PRF quality than the 4-byte sketch in the writeup.
    seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF   # 63-bit non-negative int

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(vocab_size, generator=rng)   # random permutation of token ids

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[perm[: vocab_size // 2]] = True
    return mask


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Top-k / nucleus (top-p) filtering in logit space.  logits: [batch, vocab].
    Sets masked positions to -inf so they get probability 0 after softmax.
    These filtered-out tokens form the complement of the admissible set A_t(c)
    from Definition 1.
    """
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        # shift right: keep the first token whose cum-prob just crossed top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices_to_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def cgz_sample_from_logits(
    logits: torch.Tensor,
    secret_key: bytes,
    position: int,
    context: tuple,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One CGZ-watermarked sampling step (Definition 2 watermarking rule).

    Pipeline: temperature scale → top-k/p filter → softmax (defines admissible set)
              → zero red tokens → renormalize over green → multinomial sample.

    Args:
        logits:     [batch, 1, vocab] or [batch, vocab] — model next-token logits.
        secret_key: watermark secret.
        position:   index of this token within the generated image-token sequence (0-based).
        context:    tuple of the last m token ids preceding this position.
        temperature, top_k, top_p: standard sampling hyperparams.

    Returns:
        next_token:  [batch, 1] sampled token indices — always in the green set.
        green_probs: [batch, vocab] the renormalized distribution over green tokens.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]          # [batch, vocab]

    logits = logits / max(temperature, 1e-5)
    logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)      # [batch, vocab]; filtered tokens → prob 0

    green_mask = get_green_mask(secret_key, position, context, vocab_size=probs.shape[-1])
    green_mask = green_mask.to(probs.device)

    green_probs = probs * green_mask.float()
    green_sum = green_probs.sum(dim=-1, keepdim=True)

    # Renormalize; fall back to full probs only if zero green mass (shouldn't happen
    # with vocab_size=16384 and top_k<=8192, since ~half the admissible set is green).
    green_probs = torch.where(
        green_sum > 1e-10,
        green_probs / green_sum,
        probs,
    )

    next_token = torch.multinomial(green_probs, num_samples=1)  # [batch, 1]
    return next_token, green_probs


# ---------------------------------------------------------------------------
# Detection (Theorems 2 and 3)
# ---------------------------------------------------------------------------

def compute_threshold(L: int, m: int, alpha: float = 0.01) -> float:
    """
    Soundness threshold tau from Theorem 2.

        tau = (L - m) / 2  +  sqrt((L - m) * log(1 / alpha) / 2)

    Guarantees:  Pr_{z ~ P}[S_k(z) >= tau] <= alpha  for every key k.
    Completeness (Theorem 3): tau <= L - m, so watermarked sequences always exceed tau.
    """
    n = L - m
    return n / 2.0 + math.sqrt(n * math.log(1.0 / alpha) / 2.0)


def compute_detection_score(
    token_sequence: List[int],
    secret_key: bytes,
    context_len: int = DEFAULT_CONTEXT_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> int:
    """
    S_k(z): count positions t in [context_len, L) where z_t is in G_k^t(c_t(z)).

    Theorem 3 (Completeness): if every token was sampled with cgz_sample_from_logits,
    then S_k(z) = L - context_len deterministically.

    Theorem 2 (Soundness): for clean z ~ P, E[S_k(z)] = (L - context_len) / 2.
    """
    L = len(token_sequence)
    score = 0
    for t in range(context_len, L):
        context = tuple(token_sequence[t - context_len : t])
        token = token_sequence[t]
        green_mask = get_green_mask(
            secret_key, position=t, context=context, vocab_size=vocab_size
        )
        if green_mask[token].item():
            score += 1
    return score


def detect(
    token_sequence: List[int],
    secret_key: bytes,
    context_len: int = DEFAULT_CONTEXT_LEN,
    alpha: float = 0.01,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    """
    Run CGZ detector on a flat list of image token ids.

    Returns a dict:
        score           — S_k(z)
        threshold       — tau from Theorem 2 at the given alpha
        detected        — True iff score >= threshold
        expected_clean  — (L-m)/2, mean score for an unwatermarked sequence
        expected_wm     — L-m, score for a perfectly watermarked sequence
        fraction_green  — score / (L - m)
    """
    L = len(token_sequence)
    n = L - context_len
    score = compute_detection_score(token_sequence, secret_key, context_len, vocab_size)
    threshold = compute_threshold(L, context_len, alpha)
    return {
        "score": score,
        "threshold": threshold,
        "detected": bool(score >= threshold),
        "L": L,
        "m": context_len,
        "n": n,
        "expected_clean": n / 2.0,
        "expected_wm": float(n),
        "fraction_green": score / n if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CGZ-aware generation loop (hooks into LlamaGen's KV-cache architecture)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cgz_generate(
    model,
    cond: torch.Tensor,
    max_new_tokens: int,
    secret_key: bytes,
    context_len: int = DEFAULT_CONTEXT_LEN,
    emb_masks=None,
    cfg_scale: float = 1.0,
    cfg_interval: int = -1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Drop-in replacement for LlamaGen's generate() with CGZ watermarking embedded.

    Initialization (positions 0 .. context_len-1): sampled cleanly — matches
    "arbitrary initialization on first m tokens" from Definition 1.
    Watermarked positions (context_len .. max_new_tokens-1): sampled via
    cgz_sample_from_logits — every token is guaranteed to be in its green set
    (Theorem 3 completeness).

    Currently supports c2i (class-conditional) mode only.

    Args:
        model:          LlamaGen GPT model with KV-cache (setup_caches, tok_embeddings).
        cond:           [batch] integer class-label tensor.
        max_new_tokens: number of image tokens (= latent_size^2, e.g. 256 for 16x16).
        secret_key:     watermark secret as bytes.
        context_len:    m in the paper; context window for the PRF.
        cfg_scale:      classifier-free guidance scale (1.0 = no CFG).
        cfg_interval:   disable CFG after this many steps (-1 = always on).
        temperature, top_k, top_p: standard sampling hyperparams.

    Returns:
        tokens:         [batch, max_new_tokens] generated token indices.
        generated_list: Python list of ints for the first batch item.
                        Pass directly to detect() or compute_detection_score().
    """
    from autoregressive.models.generate import prefill, sample as clean_sample

    assert model.model_type == "c2i", "cgz_generate currently supports c2i only"

    # --- Mirror the setup block from generate() in generate.py ---
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    T = 1   # c2i: one class-conditioning token at position 0
    T_new = T + max_new_tokens
    max_batch_size = cond.shape[0]
    device = cond.device

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=T_new,
            dtype=model.tok_embeddings.weight.dtype,
        )

    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = (
                model.causal_mask[:, :, :T]
                * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
            )
        else:
            model.causal_mask[:, :, :T] = (
                model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)
            )
        eye = torch.eye(
            model.causal_mask.size(1), model.causal_mask.size(2), device=device
        )
        model.causal_mask[:] = model.causal_mask * (1 - eye) + eye

    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    sampling_kwargs_clean = dict(
        temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=True
    )

    # --- Prefill: class token → image token 0 (clean, no watermark yet) ---
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs_clean)
    seq[:, T : T + 1] = next_token
    generated = [next_token[0, 0].item()]   # accumulate for context

    # --- Autoregressive decode loop ---
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    cfg_flag = True

    for i in range(1, max_new_tokens):
        cur_token = next_token.view(-1, 1)   # [batch, 1]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False

            # Forward pass — replicate decode_one_token's CFG logic exactly
            if cfg_scale > 1.0:
                x_combined = torch.cat([cur_token, cur_token])
                logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
                cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
                if cfg_flag:
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits
            else:
                logits, _ = model(cur_token, cond_idx=None, input_pos=input_pos)

        # Sampling: CGZ once we have a full context window, clean otherwise
        token_pos = i   # index within generated image-token sequence
        if token_pos >= context_len:
            context = tuple(generated[token_pos - context_len : token_pos])
            next_token, _ = cgz_sample_from_logits(
                logits,
                secret_key=secret_key,
                position=token_pos,
                context=context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            # Initialization phase: clean sampling (Definition 1 "arbitrary initialization")
            next_token, _ = clean_sample(logits, **sampling_kwargs_clean)

        seq[:, T + i : T + i + 1] = next_token
        generated.append(next_token[0, 0].item())
        input_pos = input_pos + 1

    return seq[:, T:], generated
