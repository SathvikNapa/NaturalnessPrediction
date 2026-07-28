#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast as _autocast

    def autocast(enabled: bool = True):
        return _autocast(device_type="cuda", enabled=enabled)

except Exception:
    from torch.cuda.amp import autocast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MIN_REAL_PAIRS = 10

_CHUNK_RE = re.compile(
    r"(?P<base>.+?)(?P<tag>(?:__ch|_ch|__chunk|_chunk))(?P<idx>\d+)$",
    re.IGNORECASE,
)


def _parse_base_and_idx(path: Path) -> Optional[Tuple[str, int]]:
    stem = path.stem
    m = _CHUNK_RE.match(stem)
    if not m:
        return None
    return m.group("base").lower(), int(m.group("idx"))


def build_embed_index(embed_root: Path) -> Dict[str, List[str]]:
    idx_map: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(embed_root):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            if not fn.lower().endswith(".npy"):
                continue
            p = Path(dirpath) / fn
            parsed = _parse_base_and_idx(p)
            if parsed is None:
                continue
            base, ch_idx = parsed
            idx_map[base].append((ch_idx, str(p)))

    out: Dict[str, List[str]] = {}
    for base, pairs in idx_map.items():
        pairs.sort(key=lambda x: x[0])
        out[base] = [p for _, p in pairs]
    return out



def _np_load_safe(path: str, mmap_mode: Optional[str]) -> np.ndarray:
    if mmap_mode:
        try:
            return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        except Exception:
            pass
    return np.load(path, allow_pickle=False)


def _pool_chunk(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr) + 1e-6
        return (arr / norm).astype(np.float32)
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-6
        arr = arr / norms
        return arr.mean(axis=0).astype(np.float32)
    raise ValueError(f"Unexpected embedding shape {arr.shape}")



def load_speaker_sequence_from_cache(base_id: str, cache_root: Path) -> Optional[np.ndarray]:
    cached_path = cache_root / "speaker" / f"{base_id}.npy"
    if cached_path.exists():
        try:
            arr = _np_load_safe(str(cached_path), mmap_mode="r")
            if arr.ndim == 2:
                return arr.astype(np.float32, copy=False)
        except Exception:
            pass
    return None


def load_speaker_sequence_from_chunks(
    base_id: str,
    embed_index: Dict[str, List[str]],
    max_pairs: Optional[int] = None,
) -> np.ndarray:
    files = embed_index.get(base_id, [])
    if not files:
        raise FileNotFoundError(f"No chunks found for base_id={base_id}")

    if max_pairs is not None:
        files = files[:max_pairs]

    pooled = []
    for fp in files:
        try:
            arr = _np_load_safe(fp, mmap_mode="r")
            pooled.append(_pool_chunk(arr))
        except Exception:
            continue

    if len(pooled) < MIN_REAL_PAIRS:
        raise RuntimeError(
            f"base_id={base_id}: only {len(pooled)} valid chunks (need {MIN_REAL_PAIRS})"
        )

    seq = np.stack(pooled, axis=0)
    return np.clip(seq, -10.0, 10.0).astype(np.float32)


def get_speaker_sequence(
    base_id: str,
    embed_index: Dict[str, List[str]],
    cache_root: Optional[Path] = None,
    max_pairs: Optional[int] = None,
) -> np.ndarray:
    if cache_root is not None:
        cached = load_speaker_sequence_from_cache(base_id, cache_root)
        if cached is not None:
            if max_pairs is not None:
                cached = cached[:max_pairs]
            return cached.astype(np.float32, copy=False)
    return load_speaker_sequence_from_chunks(base_id, embed_index, max_pairs)


def interleave_with_pad(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    La, d = seq_a.shape
    Lb = seq_b.shape[0]
    L = max(La, Lb)
    zero = np.zeros((d,), dtype=np.float32)
    out = []
    for i in range(L):
        out.append(seq_a[i] if i < La else zero)
        out.append(seq_b[i] if i < Lb else zero)
    return np.stack(out, axis=0)


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    text = (text or "").strip().lower()
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    v = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-6
    return v


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.lower() == "nan":
        return ""
    return s.strip()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        hidden_dim = int(4 * d_model / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len][None, None, :, :]
        sin = self.sin_cached[:seq_len][None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, rope, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.drop = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, key_padding_mask, causal=False, return_attn=False):
        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)
        q = self.rope(q, T)
        k = self.rope(k, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if causal:
            cm = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(cm[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn_drop = self.drop(attn)
        out = (attn_drop @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), (attn if return_attn else None)


class LLaMAEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, n_heads, rope, dropout)
        self.mlp = SwiGLU(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, causal=False, return_attn=False):
        h = self.attn_norm(x)
        attn_out, attn = self.attn(h, key_padding_mask, causal=causal, return_attn=return_attn)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.ffn_norm(x)))
        return x, attn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class BaseTransformerAudioEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, num_layers: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

    def encode_pooled(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.drop(h)
        T = h.shape[1]
        kpm = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        h = self.enc(h, src_key_padding_mask=kpm)
        valid = (~kpm).float().unsqueeze(-1)
        return (h * valid).sum(1) / lengths.clamp(min=1).unsqueeze(-1).to(h.dtype)


class NaturalnessTransformer(nn.Module):
    def __init__(
        self,
        d_in,
        d_model=256,
        num_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=2048,
        causal=False,
    ):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.input_proj = nn.Linear(d_in, d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rope = RotaryEmbedding(dim=(d_model // n_heads), max_position=max_len)
        self.layers = nn.ModuleList(
            [LLaMAEncoderBlock(d_model, n_heads, self.rope, dropout) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def forward_tokens(self, x: torch.Tensor, lengths: torch.Tensor, return_attn: bool = False):
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        kpm = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, key_padding_mask=kpm, causal=self.causal, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)
        h = self.final_norm(h)
        return h, (attn_list if return_attn else None)

    def encode_pooled(self, x, lengths):
        B, T, _ = x.shape
        h, _ = self.forward_tokens(x, lengths, return_attn=False)
        kpm = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        valid = (~kpm).float().unsqueeze(-1)
        return (h * valid).sum(1) / lengths.clamp(min=1).unsqueeze(-1).to(h.dtype)


# ---------------------------------------------------------------------------
# Aux encoder
# ---------------------------------------------------------------------------

class CatCtxRelEncoder(nn.Module):
    """
    Supports two checkpoint variants:
      A) proj = [LayerNorm(in_dim), Linear(in_dim->out_dim), ReLU, Dropout]
      B) proj = [Linear(in_dim->out_dim), ReLU, Dropout]
    """
    def __init__(
        self,
        cat_cardinalities,
        cat_embed_dim,
        ctx_dim,
        rel_dim,
        out_dim,
        dropout,
        ctx_ln: bool = True,
        rel_ln: bool = True,
    ):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, cat_embed_dim) for c in cat_cardinalities])
        self.cat_proj = nn.Sequential(
            nn.Linear(len(cat_cardinalities) * cat_embed_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        ctx_layers: List[nn.Module] = []
        if ctx_ln:
            ctx_layers.append(nn.LayerNorm(ctx_dim))
        ctx_layers += [nn.Linear(ctx_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.ctx_proj = nn.Sequential(*ctx_layers)

        rel_layers: List[nn.Module] = []
        if rel_ln:
            rel_layers.append(nn.LayerNorm(rel_dim))
        rel_layers += [nn.Linear(rel_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.rel_proj = nn.Sequential(*rel_layers)

    def forward(self, cat_x, ctx_x, rel_x):
        embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embs)]
        cat = self.cat_proj(torch.cat(embs, dim=-1))
        ctx = self.ctx_proj(ctx_x)
        rel = self.rel_proj(rel_x)
        return torch.cat([cat, ctx, rel], dim=-1)


def masked_mean_pool(x, lengths):
    B, T, D = x.shape
    mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(1) / lengths.clamp(min=1).to(x.dtype).unsqueeze(-1)


class NaturalnessFusionTransformerModel(nn.Module):
    """LLaMA-ish audio encoder + concat aux -> head  (model_type=llama)"""
    def __init__(
        self,
        d_in_audio,
        audio_d_model,
        num_layers,
        n_heads,
        dropout,
        max_len,
        cat_cards,
        ctx_dim,
        rel_dim,
        use_aux,
        aux_dim=128,
        cat_embed_dim=32,
        ctx_ln: bool = True,
        rel_ln: bool = True,
        attn_mode: str = "mean_query",
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.attn_mode = attn_mode
        self.audio = NaturalnessTransformer(
            d_in_audio, audio_d_model, num_layers, n_heads, dropout, max_len, False
        )
        if self.use_aux:
            self.aux = CatCtxRelEncoder(
                cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout,
                ctx_ln=ctx_ln, rel_ln=rel_ln,
            )
            fusion_in = audio_d_model + 3 * aux_dim
        else:
            self.aux = None
            fusion_in = audio_d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    @staticmethod
    def _reduce_self_attn(attn_bhts: torch.Tensor, kpm: torch.Tensor, mode: str) -> torch.Tensor:
        """Reduce [B,H,T,T] attn to [B,T] for attention dumping."""
        a = attn_bhts.mean(dim=1)
        if mode == "last_query":
            B, T, _ = a.shape
            lengths = (~kpm).sum(dim=1).clamp_min(1)
            idx = (lengths - 1).view(B, 1, 1).expand(B, 1, T)
            aq = a.gather(1, idx).squeeze(1)
        else:
            valid_q = (~kpm).float()
            denom = valid_q.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            aq = (a * valid_q[:, :, None]).sum(dim=1) / denom
        valid_src = (~kpm).float()
        aq = aq * valid_src
        denom2 = aq.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return aq / denom2

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x, return_attn: bool = False):
        if not return_attn:
            p = self.audio.encode_pooled(x_audio, len_audio)
            if self.use_aux:
                p = torch.cat([p, self.aux(cat_x, ctx_x, rel_x)], -1)
            return self.head(p).squeeze(-1)

        tokens, attn_list = self.audio.forward_tokens(x_audio, len_audio, return_attn=True)
        assert attn_list is not None
        T = tokens.shape[1]
        kpm = torch.arange(T, device=x_audio.device)[None, :] >= len_audio[:, None]
        valid = (~kpm).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * valid).sum(dim=1) / denom
        if self.use_aux:
            pooled = torch.cat([pooled, self.aux(cat_x, ctx_x, rel_x)], -1)
        logits = self.head(pooled).squeeze(-1)
        attn_reduced = [self._reduce_self_attn(a, kpm, self.attn_mode) for a in attn_list]
        return logits, attn_reduced, kpm.detach(), len_audio.detach()


# Alias kept for any external code that imports by the old name.
NaturalnessFusionLLaMAModel = NaturalnessFusionTransformerModel


class BaseFusionTransformerModel(nn.Module):
    """Sinusoidal PE + nn.TransformerEncoder + concat aux -> head  (model_type=base_transformer)"""
    def __init__(
        self,
        d_in_audio,
        audio_d_model,
        num_layers,
        n_heads,
        dropout,
        max_len,
        cat_cards,
        ctx_dim,
        rel_dim,
        use_aux,
        aux_dim=128,
        cat_embed_dim=32,
        ctx_ln: bool = True,
        rel_ln: bool = True,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.audio = BaseTransformerAudioEncoder(
            d_in_audio, audio_d_model, num_layers, n_heads, dropout, max_len
        )
        if self.use_aux:
            self.aux = CatCtxRelEncoder(
                cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout,
                ctx_ln=ctx_ln, rel_ln=rel_ln,
            )
            fusion_in = audio_d_model + 3 * aux_dim
        else:
            self.aux = None
            fusion_in = audio_d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        p = self.audio.encode_pooled(x_audio, len_audio)
        if self.use_aux:
            p = torch.cat([p, self.aux(cat_x, ctx_x, rel_x)], -1)
        return self.head(p).squeeze(-1)


class NaturalnessFusionMLPModel(nn.Module):
    def __init__(
        self,
        d_in_audio,
        dropout,
        cat_cards,
        ctx_dim,
        rel_dim,
        use_aux,
        aux_dim=128,
        cat_embed_dim=32,
        hidden=512,
        layers=2,
        ctx_ln: bool = True,
        rel_ln: bool = True,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        if self.use_aux:
            self.aux = CatCtxRelEncoder(
                cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout,
                ctx_ln=ctx_ln, rel_ln=rel_ln,
            )
            fusion_in = d_in_audio + 3 * aux_dim
        else:
            self.aux = None
            fusion_in = d_in_audio
        blocks: list = []
        in_dim = fusion_in
        for _ in range(max(1, layers)):
            blocks += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        p = masked_mean_pool(x_audio, len_audio)
        if self.use_aux:
            p = torch.cat([p, self.aux(cat_x, ctx_x, rel_x)], -1)
        return self.head(p).squeeze(-1)


class NaturalnessFusionLogRegModel(nn.Module):
    def __init__(
        self,
        d_in_audio,
        dropout,
        cat_cards,
        ctx_dim,
        rel_dim,
        use_aux,
        aux_dim=128,
        cat_embed_dim=32,
        ctx_ln: bool = True,
        rel_ln: bool = True,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        if self.use_aux:
            self.aux = CatCtxRelEncoder(
                cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout,
                ctx_ln=ctx_ln, rel_ln=rel_ln,
            )
            fusion_in = d_in_audio + 3 * aux_dim
        else:
            self.aux = None
            fusion_in = d_in_audio
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(fusion_in, 1)

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        p = masked_mean_pool(x_audio, len_audio)
        if self.use_aux:
            p = torch.cat([p, self.aux(cat_x, ctx_x, rel_x)], -1)
        return self.linear(self.drop(p)).squeeze(-1)


class CrossAttnFusion(nn.Module):
    """One AUX token cross-attends over AUDIO tokens (multi-layer)."""
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q":  nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "attn":    nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "drop":    nn.Dropout(dropout),
                "ffn":     nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
                ),
            })
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, aux_tok: torch.Tensor, audio_tokens: torch.Tensor, kpm: torch.Tensor, return_weights: bool = False):
        q = aux_tok
        w_list = []
        for blk in self.layers:
            qn = blk["norm_q"](q)
            kvn = blk["norm_kv"](audio_tokens)
            attn_out, attn_w = blk["attn"](
                qn, kvn, kvn,
                key_padding_mask=kpm,
                need_weights=return_weights,
                average_attn_weights=False,
            )
            q = q + blk["drop"](attn_out)
            q = q + blk["ffn"](q)
            if return_weights and attn_w is not None:
                w_list.append(attn_w.detach())
        if return_weights:
            return q, w_list
        return q


class NaturalnessCrossAttnModel(nn.Module):
    """LLaMA-ish audio token encoder + aux token cross-attends to audio.  (model_type=crossattn)"""
    def __init__(
        self,
        d_in_audio,
        d_model,
        num_layers,
        n_heads,
        dropout,
        max_len,
        cross_layers,
        cross_dropout,
        cat_cards,
        ctx_dim,
        rel_dim,
        use_aux,
        aux_dim=128,
        cat_embed_dim=32,
        ctx_ln: bool = True,
        rel_ln: bool = True,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.audio_encoder = NaturalnessTransformer(
            d_in_audio, d_model, num_layers, n_heads, dropout, max_len, False
        )
        if self.use_aux:
            self.aux = CatCtxRelEncoder(
                cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout,
                ctx_ln=ctx_ln, rel_ln=rel_ln,
            )
            self.aux_to_tok = nn.Sequential(
                nn.LayerNorm(3 * aux_dim),
                nn.Linear(3 * aux_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            )
        else:
            self.aux = None
            self.aux_to_tok = None
        self.xattn = CrossAttnFusion(d_model, n_heads, cross_layers, cross_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    @staticmethod
    def _reduce_cross_attn(attn_bh1t: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        """Reduce [B,H,1,T] cross-attn weights to [B,T]."""
        a = attn_bh1t.mean(dim=1).squeeze(1)
        valid = (~kpm).float()
        a = a * valid
        denom = a.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return a / denom

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x, return_attn: bool = False):
        tokens, _ = self.audio_encoder.forward_tokens(x_audio, len_audio, return_attn=False)
        T = tokens.shape[1]
        kpm = torch.arange(T, device=x_audio.device)[None, :] >= len_audio[:, None]
        if self.use_aux:
            aux_feat = self.aux(cat_x, ctx_x, rel_x)
            aux_tok = self.aux_to_tok(aux_feat).unsqueeze(1)
        else:
            valid = (~kpm).float().unsqueeze(-1)
            pooled = (tokens * valid).sum(1) / len_audio.clamp(min=1).unsqueeze(-1).to(tokens.dtype)
            aux_tok = pooled.unsqueeze(1)

        if not return_attn:
            fused = self.xattn(aux_tok, tokens, kpm, return_weights=False)
            return self.head(fused.squeeze(1)).squeeze(-1)

        fused, w_list = self.xattn(aux_tok, tokens, kpm, return_weights=True)
        logits = self.head(fused.squeeze(1)).squeeze(-1)
        attn_reduced = [self._reduce_cross_attn(w, kpm) for w in w_list]
        return logits, attn_reduced, kpm.detach(), len_audio.detach()


def _infer_model_type_from_sd(sd: dict) -> str:
    keys = set(sd.keys())
    if any(k.startswith("audio_encoder.") for k in keys):
        return "crossattn"
    if any(k.startswith("audio.in_proj.") for k in keys):
        return "base_transformer"
    if any(k.startswith("audio.layers.") for k in keys):
        return "llama"
    if any(k.startswith("head.0.weight") for k in keys):
        return "mlp"
    if any(k.startswith("linear.weight") for k in keys):
        return "logreg"
    return "llama"


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = state.get("model_state_dict", state.get("state_dict", None))
    if sd is None:
        if isinstance(state, dict) and any(isinstance(k, str) and k.endswith(".weight") for k in state.keys()):
            sd = state
            state = {"model_state_dict": sd}
        else:
            raise KeyError("Checkpoint missing model_state_dict/state_dict")

    hp = state.get("hyperparams", {}) or {}

    model_type = state.get("model_type", hp.get("model_type", None))
    if model_type is None:
        model_type = _infer_model_type_from_sd(sd)
    model_type = model_type.lower().strip()
    if model_type == "transformer":
        model_type = "llama"

    if model_type == "crossattn":
        proj_key = "audio_encoder.input_proj.weight"
    elif model_type == "base_transformer":
        proj_key = "audio.in_proj.weight"
    else:
        proj_key = "audio.input_proj.weight"

    if proj_key not in sd:
        raise KeyError(
            f"Expected key '{proj_key}' in checkpoint for model_type='{model_type}'. "
            f"Available keys (first 20): {list(sd.keys())[:20]}"
        )

    audio_d_model = int(sd[proj_key].shape[0])
    d_in_audio    = int(sd[proj_key].shape[1])

    dropout       = float(hp.get("dropout", 0.1))
    num_layers    = int(hp.get("num_layers", 6))
    n_heads       = int(hp.get("n_heads", 8))
    max_seq_len   = int(hp.get("max_seq_len", 512) or 2048)
    mlp_hidden    = int(hp.get("mlp_hidden", 512))
    mlp_layers    = int(hp.get("mlp_layers", 2))
    cross_layers  = int(hp.get("crossattn_layers", 2))
    cross_dropout = float(hp.get("crossattn_dropout", 0.1))

    use_aux = bool(state.get("use_aux", hp.get("use_aux", False)))
    if not use_aux:
        use_aux = any(k.startswith("aux.") for k in sd.keys())

    ctx_dim       = int(state.get("ctx_dim", hp.get("ctx_dim", 384)))
    rel_dim       = int(state.get("rel_dim", hp.get("rel_dim", 384)))
    cat_cards     = state.get("cat_cards", hp.get("cat_cards", [1, 1, 1]))
    aux_dim       = int(hp.get("aux_dim", 128))
    cat_embed_dim = int(hp.get("cat_embed_dim", 32))
    ctx_ln = True
    rel_ln = True

    if use_aux:
        inferred_cards = []
        i = 0
        while True:
            k = f"aux.cat_embs.{i}.weight"
            if k not in sd:
                break
            inferred_cards.append(int(sd[k].shape[0]))
            i += 1
        if inferred_cards:
            cat_cards = inferred_cards
        if "aux.cat_embs.0.weight" in sd:
            cat_embed_dim = int(sd["aux.cat_embs.0.weight"].shape[1])

        def infer_proj(prefix: str, default_in: int) -> Tuple[int, bool]:
            w0 = sd.get(f"{prefix}.0.weight", None)
            if w0 is None:
                return default_in, True
            if w0.dim() == 1:
                return int(w0.numel()), True
            if w0.dim() == 2:
                return int(w0.shape[1]), False
            return default_in, True

        ctx_dim, ctx_ln = infer_proj("aux.ctx_proj", ctx_dim)
        rel_dim, rel_ln = infer_proj("aux.rel_proj", rel_dim)

        def infer_out_dim(prefix: str, fallback: int) -> int:
            w0 = sd.get(f"{prefix}.0.weight", None)
            if w0 is not None and w0.dim() == 2:
                return int(w0.shape[0])
            w1 = sd.get(f"{prefix}.1.weight", None)
            if w1 is not None and w1.dim() == 2:
                return int(w1.shape[0])
            return fallback

        aux_dim = infer_out_dim("aux.ctx_proj", aux_dim)

        if model_type == "crossattn":
            w = sd.get("aux_to_tok.1.weight", None)
            if w is not None and w.dim() == 2:
                audio_d_model = int(w.shape[0])

    common_aux = dict(
        cat_cards=cat_cards, ctx_dim=ctx_dim, rel_dim=rel_dim,
        use_aux=use_aux, aux_dim=aux_dim, cat_embed_dim=cat_embed_dim,
        ctx_ln=ctx_ln, rel_ln=rel_ln,
    )

    if model_type == "llama":
        model = NaturalnessFusionTransformerModel(
            d_in_audio=d_in_audio, audio_d_model=audio_d_model,
            num_layers=num_layers, n_heads=n_heads, dropout=dropout,
            max_len=max_seq_len, **common_aux,
        )
    elif model_type == "base_transformer":
        model = BaseFusionTransformerModel(
            d_in_audio=d_in_audio, audio_d_model=audio_d_model,
            num_layers=num_layers, n_heads=n_heads, dropout=dropout,
            max_len=max_seq_len, **common_aux,
        )
    elif model_type == "crossattn":
        model = NaturalnessCrossAttnModel(
            d_in_audio=d_in_audio, d_model=audio_d_model,
            num_layers=num_layers, n_heads=n_heads, dropout=dropout,
            max_len=max_seq_len, cross_layers=cross_layers, cross_dropout=cross_dropout,
            **common_aux,
        )
    elif model_type == "mlp":
        model = NaturalnessFusionMLPModel(
            d_in_audio=d_in_audio, dropout=dropout,
            hidden=mlp_hidden, layers=mlp_layers, **common_aux,
        )
    elif model_type == "logreg":
        model = NaturalnessFusionLogRegModel(
            d_in_audio=d_in_audio, dropout=dropout, **common_aux,
        )
    else:
        raise ValueError(f"Unknown model_type='{model_type}'")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]}")
    if missing:
        print(f"[WARN] Missing keys (up to 20): {missing[:20]}")

    model.to(device).eval()

    info = {
        "model_type": model_type,
        "d_in_audio": d_in_audio,
        "audio_d_model": audio_d_model,
        "ctx_dim": ctx_dim,
        "rel_dim": rel_dim,
        "cat_cards": cat_cards,
        "use_aux": use_aux,
        "aux_dim": aux_dim,
        "cat_embed_dim": cat_embed_dim,
        "ctx_ln": ctx_ln,
        "rel_ln": rel_ln,
        "epoch": state.get("epoch"),
        "score": state.get("score"),
        "hyperparams": hp,
    }
    return model, info


@torch.no_grad()
def predict_single(
    model: nn.Module,
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    cat_ids: np.ndarray,
    ctx_emb: np.ndarray,
    rel_emb: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    seq = interleave_with_pad(seq_a, seq_b)
    T = seq.shape[0]

    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    lengths = torch.tensor([T], dtype=torch.long, device=device)
    cat_t = torch.from_numpy(cat_ids).unsqueeze(0).to(device)
    ctx_t = torch.from_numpy(ctx_emb).unsqueeze(0).to(device)
    rel_t = torch.from_numpy(rel_emb).unsqueeze(0).to(device)

    logit = model(x, lengths, cat_t, ctx_t, rel_t)
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)

    return {"probability": prob, "prediction": pred, "logit": logit.item()}


def _maybe_get_cached_embed(embed_map: Dict[str, np.ndarray], key: str, dim: int) -> np.ndarray:
    v = embed_map.get(key, None)
    if v is None:
        return _hash_embedding(key, dim=dim)
    try:
        if int(v.shape[0]) != int(dim):
            return _hash_embedding(key, dim=dim)
        return v.astype(np.float32, copy=False)
    except Exception:
        return _hash_embedding(key, dim=dim)


@torch.no_grad()
def predict_csv(
    model: nn.Module,
    info: dict,
    input_csv: str,
    embed_index: Dict[str, List[str]],
    device: torch.device,
    context_embed_map: Dict[str, np.ndarray],
    rel_embed_map: Dict[str, np.ndarray],
    cache_root: Optional[Path] = None,
    max_pairs: Optional[int] = None,
    threshold: float = 0.5,
    batch_size: int = 16,
    shard_id: int = 0,
    num_shards: int = 1,
) -> List[dict]:
    hp = info.get("hyperparams", {}) or {}

    p1_col             = hp.get("p1_path_col", "participant1_relpath_abs")
    p2_col             = hp.get("p2_path_col", "participant2_relpath_abs")
    label_col          = hp.get("label_col", "naturalness")
    context_col        = hp.get("context_col", "high_level_context")
    speaker_a_role_col = hp.get("speaker_a_role_col", "speaker_a_role")
    speaker_b_role_col = hp.get("speaker_b_role_col", "speaker_b_role")
    rel_detail_col     = hp.get("rel_detail_col", "rel_detail")
    rel_template       = hp.get("rel_text_template", "{speaker_a_role} [SEP] {speaker_b_role} [SEP] {rel_detail}")
    aug_type_col       = "augmentation_type"

    use_aux = bool(info.get("use_aux", False))
    ctx_dim = int(info.get("ctx_dim", 384))
    rel_dim = int(info.get("rel_dim", 384))

    rows: List[Tuple[int, dict]] = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames_in = set(reader.fieldnames or [])
        for i, r in enumerate(reader):
            if num_shards > 1 and (i % num_shards) != shard_id:
                continue
            rows.append((i, r))

    results: List[dict] = []
    batch_data: List[dict] = []

    for row_idx, r in rows:
        p1_raw = r.get(p1_col, "") or ""
        p2_raw = r.get(p2_col, "") or ""
        a = Path(p1_raw).stem.lower()
        b = Path(p2_raw).stem.lower()

        aug_type    = _safe_str(r.get(aug_type_col, ""))
        rel_detail  = _safe_str(r.get(rel_detail_col, ""))
        high_ctx    = _safe_str(r.get(context_col, ""))

        if not embed_index.get(a) or not embed_index.get(b):
            results.append({
                "row_idx": row_idx,
                "p1_base": a, "p2_base": b,
                "p1_path": p1_raw, "p2_path": p2_raw,
                "label": r.get(label_col, ""),
                "augmentation_type": aug_type,
                "rel_detail": rel_detail,
                "high_level_context": high_ctx,
                "probability": None, "prediction": None, "logit": None,
                "error": "missing_embeddings",
            })
            continue

        try:
            seq_a = get_speaker_sequence(a, embed_index, cache_root, max_pairs)
            seq_b = get_speaker_sequence(b, embed_index, cache_root, max_pairs)
        except Exception as e:
            results.append({
                "row_idx": row_idx,
                "p1_base": a, "p2_base": b,
                "p1_path": p1_raw, "p2_path": p2_raw,
                "label": r.get(label_col, ""),
                "augmentation_type": aug_type,
                "rel_detail": rel_detail,
                "high_level_context": high_ctx,
                "probability": None, "prediction": None, "logit": None,
                "error": str(e),
            })
            continue

        seq = interleave_with_pad(seq_a, seq_b)
        cat_ids = np.zeros(3, dtype=np.int64)

        if use_aux:
            ctx_key = high_ctx
            ctx_emb = _maybe_get_cached_embed(context_embed_map, ctx_key, ctx_dim)

            ra = _safe_str(r.get(speaker_a_role_col, ""))
            rb = _safe_str(r.get(speaker_b_role_col, ""))
            rd = rel_detail
            rel_key = rel_template.format(speaker_a_role=ra, speaker_b_role=rb, rel_detail=rd).strip()
            rel_emb = _maybe_get_cached_embed(rel_embed_map, rel_key, rel_dim)
        else:
            ctx_emb = np.zeros(ctx_dim, dtype=np.float32)
            rel_emb = np.zeros(rel_dim, dtype=np.float32)

        batch_data.append({
            "row_idx": row_idx,
            "p1_base": a, "p2_base": b,
            "p1_path": p1_raw, "p2_path": p2_raw,
            "label": r.get(label_col, ""),
            "augmentation_type": aug_type,
            "rel_detail": rel_detail,
            "high_level_context": high_ctx,
            "seq": seq,
            "cat_ids": cat_ids,
            "ctx_emb": ctx_emb.astype(np.float32, copy=False),
            "rel_emb": rel_emb.astype(np.float32, copy=False),
        })

        if len(batch_data) >= batch_size:
            results.extend(_run_batch(model, batch_data, device, threshold))
            batch_data = []

    if batch_data:
        results.extend(_run_batch(model, batch_data, device, threshold))

    results.sort(key=lambda x: x["row_idx"])
    return results


def _run_batch(model, batch_data, device, threshold):
    B = len(batch_data)
    T_max = max(d["seq"].shape[0] for d in batch_data)
    d_in = batch_data[0]["seq"].shape[1]

    x_pad = np.zeros((B, T_max, d_in), dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    cats = np.zeros((B, 3), dtype=np.int64)
    ctxs = np.stack([d["ctx_emb"] for d in batch_data], axis=0).astype(np.float32)
    rels = np.stack([d["rel_emb"] for d in batch_data], axis=0).astype(np.float32)

    for i, d in enumerate(batch_data):
        T = d["seq"].shape[0]
        x_pad[i, :T] = d["seq"]
        lengths[i] = T
        cats[i] = d["cat_ids"]

    x_t   = torch.from_numpy(x_pad).to(device)
    len_t = torch.from_numpy(lengths).to(device)
    cat_t = torch.from_numpy(cats).to(device)
    ctx_t = torch.from_numpy(ctxs).to(device)
    rel_t = torch.from_numpy(rels).to(device)

    logits = model(x_t, len_t, cat_t, ctx_t, rel_t)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    out = []
    for i, d in enumerate(batch_data):
        prob = float(probs[i])
        out.append({
            "row_idx":            d["row_idx"],
            "p1_base":            d["p1_base"],
            "p2_base":            d["p2_base"],
            "p1_path":            d["p1_path"],
            "p2_path":            d["p2_path"],
            "label":              d["label"],
            "augmentation_type":  d["augmentation_type"],
            "rel_detail":         d["rel_detail"],
            "high_level_context": d["high_level_context"],
            "probability":        prob,
            "prediction":         int(prob >= threshold),
            "logit":              float(logits_np[i]),
            "error":              None,
        })
    return out


def main():
    p = argparse.ArgumentParser(description="Inference for naturalness classifier (robust checkpoint loader)")

    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt or any checkpoint")
    p.add_argument("--embed-root", required=True, help="Root dir of .npy embeddings")
    p.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, etc. If omitted, torchrun uses LOCAL_RANK.")

    p.add_argument("--input-csv", default=None, help="CSV file with dyadic pairs to score")
    p.add_argument("--output-csv", default=None, help="Where to write predictions CSV (base name; shards append .shardXX)")
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument("--p1", default=None, help="Base stem for participant 1 (single-pair mode)")
    p.add_argument("--p2", default=None, help="Base stem for participant 2 (single-pair mode)")

    p.add_argument("--index-cache", default=None, help="Pickle of embed index (reuse from training)")
    p.add_argument("--cache-root", default=None, help="Speaker cache root (reuse from training)")
    p.add_argument("--context-cache", default=None, help="Context text embed cache pickle")
    p.add_argument("--rel-cache", default=None, help="Relationship text embed cache pickle")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-seq-len", type=int, default=None, help="Override max_seq_len used to cap #chunks")

    p.add_argument("--dump-attn", action="store_true",
                   help="Save reduced attention weights (rank0). Supported for llama and crossattn models.")
    p.add_argument("--attn-out", default=None,
                   help="Directory for attention dumps (default: <checkpoint_dir>/attn)")
    p.add_argument("--attn-max-batches", type=int, default=50,
                   help="Maximum number of batches to collect attention from")
    p.add_argument("--attn-mode", default="mean_query", choices=["mean_query", "last_query"],
                   help="How to reduce query dimension in llama self-attention dump")

    p.add_argument("--num-shards", type=int, default=1,
                   help="How many parallel shards to split the CSV into (usually = #GPUs).")
    p.add_argument("--shard-id", type=int, default=-1,
                   help="Which shard this process should run (0..num_shards-1). If -1, auto from LOCAL_RANK when torchrun.")

    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.device:
        device = torch.device(args.device)
    else:
        if local_rank >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun launched but CUDA is not available")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank >= 0:
        if args.num_shards == 1:
            args.num_shards = world_size
        if args.shard_id == -1:
            args.shard_id = local_rank
    else:
        if args.shard_id == -1:
            args.shard_id = 0

    if args.num_shards < 1:
        args.num_shards = 1
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"--shard-id must be in [0, {args.num_shards-1}] but got {args.shard_id}")

    print(f"Device: {device} (LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, shard={args.shard_id}/{args.num_shards})")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, info = load_model_from_checkpoint(args.checkpoint, device)
    if args.dump_attn and hasattr(model, "attn_mode"):
        model.attn_mode = args.attn_mode
    print(
        "  model_type={mt} d_in_audio={din} audio_d_model={adm} use_aux={ua} "
        "ctx_dim={cd} rel_dim={rd} aux_dim={ad} ctx_ln={cln} rel_ln={rln} epoch={ep}".format(
            mt=info["model_type"],
            din=info["d_in_audio"],
            adm=info["audio_d_model"],
            ua=info["use_aux"],
            cd=info["ctx_dim"],
            rd=info["rel_dim"],
            ad=info["aux_dim"],
            cln=info["ctx_ln"],
            rln=info["rel_ln"],
            ep=info.get("epoch"),
        )
    )

    embed_root = Path(args.embed_root).resolve()
    if args.index_cache and Path(args.index_cache).exists():
        print(f"Loading embed index from cache: {args.index_cache}")
        with open(args.index_cache, "rb") as f:
            embed_index = pickle.load(f)
    else:
        print(f"Scanning embed_root: {embed_root}")
        embed_index = build_embed_index(embed_root)
        if args.index_cache:
            with open(args.index_cache, "wb") as f:
                pickle.dump(embed_index, f)
            print(f"  Saved index cache: {args.index_cache}")
    print(f"  base_ids={len(embed_index)} total_chunks={sum(len(v) for v in embed_index.values())}")

    cache_root = Path(args.cache_root) if args.cache_root else None

    hp = info.get("hyperparams", {}) or {}
    msl = int(args.max_seq_len or hp.get("max_seq_len", 512) or 0)
    max_pairs = None
    if msl > 0:
        max_pairs = max(1, msl // 2)

    context_embed_map: Dict[str, np.ndarray] = {}
    rel_embed_map: Dict[str, np.ndarray] = {}

    if info.get("use_aux", False):
        if args.context_cache and Path(args.context_cache).exists():
            with open(args.context_cache, "rb") as f:
                context_embed_map = pickle.load(f)
            print(f"Loaded context cache: {len(context_embed_map)} entries")
        if args.rel_cache and Path(args.rel_cache).exists():
            with open(args.rel_cache, "rb") as f:
                rel_embed_map = pickle.load(f)
            print(f"Loaded rel cache: {len(rel_embed_map)} entries")

    if args.p1 and args.p2:
        a = args.p1.lower()
        b = args.p2.lower()
        print(f"\nSingle-pair inference: p1={a} p2={b}")

        seq_a = get_speaker_sequence(a, embed_index, cache_root, max_pairs)
        seq_b = get_speaker_sequence(b, embed_index, cache_root, max_pairs)

        ctx_dim = int(info.get("ctx_dim", 384))
        rel_dim = int(info.get("rel_dim", 384))

        result = predict_single(
            model, seq_a, seq_b,
            cat_ids=np.zeros(3, dtype=np.int64),
            ctx_emb=_hash_embedding("", dim=ctx_dim),
            rel_emb=_hash_embedding("", dim=rel_dim),
            device=device,
            threshold=args.threshold,
        )
        print(f"  probability: {result['probability']:.4f}")
        print(f"  prediction:  {result['prediction']} ({'natural' if result['prediction'] == 1 else 'unnatural'})")
        print(f"  logit:       {result['logit']:.4f}")
        return

    if not args.input_csv:
        print("ERROR: Provide --input-csv for batch mode or --p1/--p2 for single-pair mode.")
        return

    dump_attn = args.dump_attn and info["model_type"] in ("llama", "crossattn")
    if args.dump_attn and not dump_attn:
        print(f"[WARN] --dump-attn is only supported for llama and crossattn models (got {info['model_type']}); skipping.")

    if dump_attn:
        attn_out_dir = Path(args.attn_out) if args.attn_out else Path(args.checkpoint).parent / "attn"
        attn_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Attention dumps → {attn_out_dir}")

    print(f"\nRunning batch inference on: {args.input_csv} (shard {args.shard_id}/{args.num_shards})")
    results = predict_csv(
        model=model,
        info=info,
        input_csv=args.input_csv,
        embed_index=embed_index,
        device=device,
        context_embed_map=context_embed_map,
        rel_embed_map=rel_embed_map,
        cache_root=cache_root,
        max_pairs=max_pairs,
        threshold=args.threshold,
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    n_ok  = sum(1 for r in results if r["error"] is None)
    n_err = sum(1 for r in results if r["error"] is not None)
    preds = [r["prediction"] for r in results if r["prediction"] is not None]
    n_natural   = sum(preds)
    n_unnatural = len(preds) - n_natural

    print(f"\nResults (this shard): {n_ok} scored, {n_err} errors")
    print(f"  natural={n_natural} unnatural={n_unnatural}")

    labeled = [(r["label"], r["prediction"]) for r in results
               if r["prediction"] is not None and r["label"] not in ("", None)]
    if labeled:
        correct = 0
        total = 0
        for lab, pred in labeled:
            try:
                correct += int(int(float(lab)) == int(pred))
                total += 1
            except Exception:
                pass
        if total:
            print(f"  accuracy={correct}/{total} = {correct/total:.4f}")

    base_out = args.output_csv or args.input_csv.replace(".csv", "_predictions.csv")
    if args.num_shards > 1:
        if base_out.lower().endswith(".csv"):
            output_csv = base_out[:-4] + f".shard{args.shard_id:02d}.csv"
        else:
            output_csv = base_out + f".shard{args.shard_id:02d}.csv"
    else:
        output_csv = base_out

    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "row_idx", "p1_base", "p2_base", "p1_path", "p2_path",
            "label", "augmentation_type", "rel_detail", "high_level_context",
            "probability", "prediction", "logit", "error",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nPredictions written to: {output_csv}")


if __name__ == "__main__":
    main()
