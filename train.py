#!/usr/bin/env python3
"""
Output Files Generated:
  <out_dir>/test_predictions.csv
  <out_dir>/test_misclassified.csv
  <out_dir>/test_metrics.json
  <out_dir>/attn/attn_batches.pt
  <out_dir>/attn/attn_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    def autocast(enabled: bool = True):
        return _autocast(device_type="cuda", enabled=enabled)

    GradScaler = _GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MIN_REAL_PAIRS = 10


def ddp_is_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_is_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_available() else 1


def is_main() -> bool:
    return get_rank() == 0


def ddp_barrier():
    if ddp_is_available():
        dist.barrier()


def setup_ddp_from_env() -> Tuple[bool, int, int]:
    """
    Returns (distributed, rank, local_rank)
    Works with torchrun which sets LOCAL_RANK, RANK, WORLD_SIZE.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank
    return False, 0, 0


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if ddp_is_available():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def broadcast_bool(flag: bool) -> bool:
    if not ddp_is_available():
        return flag
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor([1 if flag else 0], device=dev, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(t.item())


def _kpm_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
    rng = torch.arange(T, device=lengths.device)[None, :]
    return rng >= lengths[:, None]



@dataclass
class HyperParams:
    train_csv: str = ""
    test_csv: str = ""
    val_csv: str = ""

    embed_root: str = ""
    out_dir: str = ""

    model_type: str = "llama"

    p1_path_col: str = "participant1_relpath_abs"
    p2_path_col: str = "participant2_relpath_abs"
    label_col: str = "naturalness"
    context_col: str = "high_level_context"
    rel_detail_col: str = "rel_detail"
    speaker_a_role_col: str = "speaker_a_role"
    speaker_b_role_col: str = "speaker_b_role"

    use_cats: bool = True
    use_context: bool = True
    use_rel_text: bool = True
    speech_only: bool = False

    rel_text_template: str = "{speaker_a_role} [SEP] {speaker_b_role} [SEP] {rel_detail}"
    rel_text_cache: str = ""

    max_seq_len: int = 512
    window_strategy: str = "head"

    val_split: float = 0.0

    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    seed: int = 42
    patience: int = 8
    use_amp: bool = True

    d_model: int = 256
    num_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    crossattn_layers: int = 2
    crossattn_dropout: float = 0.1

    mlp_hidden: int = 512
    mlp_layers: int = 2

    index_cache: str = ""
    gather_test_preds: bool = True
    cache_items: int = 128
    mmap_load: bool = True

    cache_root: str = ""
    cache_mode: str = "speaker"
    cache_dtype: str = "float16"
    prebuild_cache: bool = False

    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    context_cache: str = ""
    prebuild_text: bool = False
    text_dim_fallback: int = 384
    text_batch_size: int = 256
    text_max_len: int = 256
    text_fp16: bool = False
    text_use_gpu_parallel: bool = True

    test_threshold: float = 0.5

    dump_attn: bool = False
    attn_out: str = ""
    attn_max_batches: int = 50
    attn_mode: str = "mean_query"

    resume_from: Optional[str] = None


_CHUNK_RE = re.compile(r"(?P<base>.+?)(?P<tag>(?:__ch|_ch|__chunk|_chunk))(?P<idx>\d+)$", re.IGNORECASE)


def _parse_base_and_idx(path: Path) -> Optional[Tuple[str, int]]:
    stem = path.stem
    m = _CHUNK_RE.match(stem)
    if not m:
        return None
    base = m.group("base").lower()
    idx = int(m.group("idx"))
    return base, idx


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


def _dtype_from_str(s: str) -> np.dtype:
    s = (s or "").lower().strip()
    if s in ("float16", "fp16", "half"):
        return np.float16
    if s in ("float32", "fp32", "single"):
        return np.float32
    raise ValueError(f"Unsupported dtype {s} (use float16 or float32)")


def _atomic_save_pickle(dst: Path, obj: Any):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    os.replace(str(tmp), str(dst))


def _atomic_npy_save(dst: Path, arr: np.ndarray):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp.npy"
    np.save(str(tmp), arr)
    os.replace(str(tmp), str(dst))


def _acquire_lock(lock_path: Path, timeout_s: float = 60.0) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                return False
            time.sleep(0.05 + random.random() * 0.15)


def _release_lock(lock_path: Path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _np_load_safe(path: str, mmap_mode: Optional[str]) -> np.ndarray:
    if mmap_mode:
        try:
            return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        except Exception:
            pass
    return np.load(path, allow_pickle=False)


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    text = (text or "").strip().lower()
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    v = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-6)
    return v


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def _encode_texts_hf(
    texts: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    out_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with autocast(enabled=(use_fp16 and device.type == "cuda")):
            o = model(**enc)
            h = o.last_hidden_state
            pooled = _mean_pool(h, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)

        out_list.append(pooled.float().cpu().numpy())

    return np.concatenate(out_list, axis=0) if out_list else np.zeros((0, 0), dtype=np.float32)


def _ddp_build_text_map_from_unique_texts(
    uniq: List[str],
    cache_path: Path,
    model_name: str,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
    fallback_dim: int,
    use_gpu_parallel: bool,
) -> Dict[str, np.ndarray]:
    if ddp_is_available():
        obj_list = [uniq]
        dist.broadcast_object_list(obj_list, src=0)
        uniq = obj_list[0]

    do_shard = ddp_is_available() and use_gpu_parallel and torch.cuda.is_available()

    if do_shard:
        world = get_world_size()
        rank = get_rank()
        shard = uniq[rank::world]

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")

        embed_map_shard: Dict[str, np.ndarray] = {}
        try:
            embs = _encode_texts_hf(
                texts=shard,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                max_len=max_len,
                use_fp16=use_fp16,
            ).astype(np.float32)
            for t, e in zip(shard, embs):
                embed_map_shard[t] = e
        except Exception as e:
            if is_main():
                print(f"[TEXT] HF encode failed ({e}); fallback to hash embeddings.")
            for t in shard:
                embed_map_shard[t] = _hash_embedding(t, dim=fallback_dim)

        gathered: List[Dict[str, np.ndarray]] = [None] * world  # type: ignore
        dist.all_gather_object(gathered, embed_map_shard)

        if is_main():
            merged: Dict[str, np.ndarray] = {}
            for part in gathered:
                merged.update(part)
            _atomic_save_pickle(cache_path, merged)
            print(f"[TEXT] cached to: {cache_path} | total={len(merged)}")
            return merged
        return {}

    if not is_main():
        return {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embed_map: Dict[str, np.ndarray] = {}
    try:
        embs = _encode_texts_hf(
            texts=uniq,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_len=max_len,
            use_fp16=use_fp16,
        ).astype(np.float32)
        for t, e in zip(uniq, embs):
            embed_map[t] = e
    except Exception as e:
        print(f"[TEXT] HF encode failed ({e}); fallback to hash embeddings.")
        for t in uniq:
            embed_map[t] = _hash_embedding(t, dim=fallback_dim)

    _atomic_save_pickle(cache_path, embed_map)
    print(f"[TEXT] cached to: {cache_path} | total={len(embed_map)}")
    return embed_map


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.lower() == "nan":
        return ""
    return s.strip()


def build_context_text_embed_map_ddp(
    csv_paths: List[Path],
    context_col: str,
    cache_path: Path,
    model_name: str,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
    fallback_dim: int,
    use_gpu_parallel: bool,
) -> Dict[str, np.ndarray]:
    if is_main():
        contexts: List[str] = []
        for csv_path in csv_paths:
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                if context_col not in (reader.fieldnames or []):
                    raise ValueError(f"CSV missing context col: {context_col} in {csv_path}")
                for r in reader:
                    contexts.append(str(r.get(context_col, "") or "").strip())

        seen: set[str] = set()
        uniq: List[str] = []
        for t in contexts:
            tt = (t or "").strip()
            if tt not in seen:
                seen.add(tt)
                uniq.append(tt)

        print(f"[CTX] unique {context_col}: {len(uniq)}")
    else:
        uniq = None

    if ddp_is_available():
        obj_list = [uniq]
        dist.broadcast_object_list(obj_list, src=0)
        uniq = obj_list[0]
    assert uniq is not None

    return _ddp_build_text_map_from_unique_texts(
        uniq=uniq,
        cache_path=cache_path,
        model_name=model_name,
        batch_size=batch_size,
        max_len=max_len,
        use_fp16=use_fp16,
        fallback_dim=fallback_dim,
        use_gpu_parallel=use_gpu_parallel,
    )


def build_relationship_text_embed_map_ddp(
    csv_paths: List[Path],
    speaker_a_role_col: str,
    speaker_b_role_col: str,
    rel_detail_col: str,
    template: str,
    cache_path: Path,
    model_name: str,
    batch_size: int,
    max_len: int,
    use_fp16: bool,
    fallback_dim: int,
    use_gpu_parallel: bool,
) -> Dict[str, np.ndarray]:
    if is_main():
        rel_texts: List[str] = []
        needed = {speaker_a_role_col, speaker_b_role_col, rel_detail_col}
        for csv_path in csv_paths:
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                missing = [c for c in needed if c not in (reader.fieldnames or [])]
                if missing:
                    raise ValueError(f"CSV missing required relationship cols {missing} in {csv_path}")

                for r in reader:
                    ra = _safe_str(r.get(speaker_a_role_col, ""))
                    rb = _safe_str(r.get(speaker_b_role_col, ""))
                    rd = _safe_str(r.get(rel_detail_col, ""))
                    txt = template.format(speaker_a_role=ra, speaker_b_role=rb, rel_detail=rd).strip()
                    rel_texts.append(txt)

        seen: set[str] = set()
        uniq: List[str] = []
        for t in rel_texts:
            tt = (t or "").strip()
            if tt not in seen:
                seen.add(tt)
                uniq.append(tt)

        print(f"[REL] unique relationship_text: {len(uniq)}")
        print(f"[REL] template={template!r}")
    else:
        uniq = None

    if ddp_is_available():
        obj_list = [uniq]
        dist.broadcast_object_list(obj_list, src=0)
        uniq = obj_list[0]
    assert uniq is not None

    return _ddp_build_text_map_from_unique_texts(
        uniq=uniq,
        cache_path=cache_path,
        model_name=model_name,
        batch_size=batch_size,
        max_len=max_len,
        use_fp16=use_fp16,
        fallback_dim=fallback_dim,
        use_gpu_parallel=use_gpu_parallel,
    )


class LRUCache:
    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, k: str) -> Optional[np.ndarray]:
        if k not in self._d:
            return None
        v = self._d.pop(k)
        self._d[k] = v
        return v

    def put(self, k: str, v: np.ndarray):
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


class DyadicNaturalnessDataset(Dataset):
    """
    Returns:
      x:      [T, d_in]    float32
      length: scalar long
      cats:   [3]          long
      ctx:    [Dc]         float32
      rel:    [Dr]         float32
      y:      scalar float32
      meta:   dict with ids for saving predictions
    """

    def __init__(
        self,
        csv_path: str,
        embed_index: Dict[str, List[str]],
        context_embed_map: Dict[str, np.ndarray],
        rel_embed_map: Dict[str, np.ndarray],
        hp: HyperParams,
        min_pairs: int = MIN_REAL_PAIRS,
    ):
        self.csv_path = Path(csv_path)
        self.embed_index = embed_index
        self.context_embed_map = context_embed_map
        self.rel_embed_map = rel_embed_map
        self.hp = hp
        self.min_pairs = int(min_pairs)

        self.window_strategy = hp.window_strategy
        self.seed = int(hp.seed)
        self.mmap_load = bool(hp.mmap_load)

        self.cache_root = Path(hp.cache_root).resolve() if hp.cache_root else None
        self.cache_mode = (hp.cache_mode or "none").lower().strip()
        if self.cache_root is None:
            self.cache_mode = "none"
        if self.cache_mode not in ("none", "speaker"):
            raise ValueError(f"--cache-mode must be none|speaker (got {self.cache_mode})")
        self.cache_dtype = _dtype_from_str(hp.cache_dtype)

        if self.cache_root is not None:
            (self.cache_root / "speaker").mkdir(parents=True, exist_ok=True)
            (self.cache_root / "locks").mkdir(parents=True, exist_ok=True)

        self.max_pairs = max(1, int(hp.max_seq_len // 2))
        if self.max_pairs < self.min_pairs:
            raise ValueError(f"max_seq_len={hp.max_seq_len} implies max_pairs={self.max_pairs} < min_pairs={self.min_pairs}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.cat_cols = [hp.speaker_a_role_col, hp.speaker_b_role_col, hp.rel_detail_col]
        self.cat_vocab: Dict[str, Dict[str, int]] = {c: {"__UNK__": 0} for c in self.cat_cols}

        def norm_cat(v: Any) -> str:
            s = str(v).strip().lower() if v is not None else "__UNK__"
            if s == "" or s == "nan":
                s = "__UNK__"
            return s

        def add_cat(col: str, v: Any):
            s = norm_cat(v)
            if s not in self.cat_vocab[col]:
                self.cat_vocab[col][s] = len(self.cat_vocab[col])

        RawRow = Tuple[str, str, int, np.ndarray, str, str, Dict[str, Any]]
        raw_rows: List[RawRow] = []

        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            needed = {
                hp.p1_path_col,
                hp.p2_path_col,
                hp.label_col,
                hp.context_col,
                hp.rel_detail_col,
                hp.speaker_a_role_col,
                hp.speaker_b_role_col,
            }
            missing = [c for c in needed if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for i, r in enumerate(reader):
                p1_raw = str(r[hp.p1_path_col])
                p2_raw = str(r[hp.p2_path_col])
                a = Path(p1_raw).stem.lower()
                b = Path(p2_raw).stem.lower()
                lab = int(float(r[hp.label_col]))

                for c in self.cat_cols:
                    add_cat(c, r.get(c, "__UNK__"))

                cat_ids = np.array([self.cat_vocab[c].get(norm_cat(r.get(c)), 0) for c in self.cat_cols], dtype=np.int64)
                ctx = str(r.get(hp.context_col, "") or "").strip()

                ra = _safe_str(r.get(hp.speaker_a_role_col, ""))
                rb = _safe_str(r.get(hp.speaker_b_role_col, ""))
                rd = _safe_str(r.get(hp.rel_detail_col, ""))
                rel_txt = hp.rel_text_template.format(speaker_a_role=ra, speaker_b_role=rb, rel_detail=rd).strip()

                meta = {
                    "row_idx_in_csv": i,
                    "p1_path": p1_raw,
                    "p2_path": p2_raw,
                    "p1_base": a,
                    "p2_base": b,
                }
                raw_rows.append((a, b, lab, cat_ids, ctx, rel_txt, meta))

        self.num_total = len(raw_rows)
        self.num_missing_audio = 0
        self.num_too_short = 0
        self.num_missing_ctx = 0
        self.num_missing_rel = 0

        self.rows: List[RawRow] = []
        bases: set[str] = set()

        for a, b, lab, cat_ids, ctx, rel_txt, meta in raw_rows:
            ca = len(self.embed_index.get(a, []))
            cb = len(self.embed_index.get(b, []))
            if ca == 0 or cb == 0:
                self.num_missing_audio += 1
                continue
            if min(ca, cb) < self.min_pairs:
                self.num_too_short += 1
                continue

            if self.hp.use_context and (ctx not in self.context_embed_map):
                self.num_missing_ctx += 1
            if self.hp.use_rel_text and (rel_txt not in self.rel_embed_map):
                self.num_missing_rel += 1

            self.rows.append((a, b, lab, cat_ids, ctx, rel_txt, meta))
            bases.add(a)
            bases.add(b)

        self.unique_bases: List[str] = sorted(bases)
        self.cache = LRUCache(max_items=hp.cache_items)

        if is_main():
            print(
                f"[Dataset:{self.csv_path.name}] rows={self.num_total}, kept={len(self.rows)} "
                f"(missing_audio={self.num_missing_audio}, too_short={self.num_too_short}, min_pairs={self.min_pairs})"
            )
            print("  cat_vocab sizes: " + ", ".join([f"{c}={len(self.cat_vocab[c])}" for c in self.cat_cols]))
            if self.hp.use_context and self.num_missing_ctx > 0:
                print(f"  WARNING: {self.num_missing_ctx} kept rows missing context embedding; zeros fallback.")
            if self.hp.use_rel_text and self.num_missing_rel > 0:
                print(f"  WARNING: {self.num_missing_rel} kept rows missing relationship embedding; zeros fallback.")
            if self.cache_mode != "none":
                print(f"  cache_mode={self.cache_mode} cache_root={self.cache_root} cache_dtype={self.cache_dtype}")
            print(f"  window_strategy={self.window_strategy} max_pairs={self.max_pairs} mmap_load={self.mmap_load}")
            print(f"  use_cats={self.hp.use_cats} use_context={self.hp.use_context} use_rel_text={self.hp.use_rel_text} speech_only={self.hp.speech_only}")

    def __len__(self) -> int:
        return len(self.rows)

    def _det_window(self, base_id: str, full_len: int) -> Tuple[int, int]:
        take_n = min(full_len, self.max_pairs)
        if take_n <= 0:
            return 0, 0
        if self.window_strategy == "head" or full_len == take_n:
            return 0, take_n
        h = hash((self.seed, base_id)) & 0xFFFFFFFF
        rng = random.Random(h)
        start = rng.randint(0, full_len - take_n)
        return start, take_n

    @staticmethod
    def _pool_chunk(arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
        if arr.ndim == 1:
            norm = np.linalg.norm(arr) + 1e-6
            return (arr / norm).astype(np.float32)
        if arr.ndim == 2:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-6
            arr = arr / norms
            v = arr.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-6)
            return v.astype(np.float32)
        raise ValueError(f"Unexpected embedding shape {arr.shape}")

    def _speaker_cache_path(self, base_id: str) -> Path:
        assert self.cache_root is not None
        return self.cache_root / "speaker" / f"{base_id}.npy"

    def _speaker_lock_path(self, base_id: str) -> Path:
        assert self.cache_root is not None
        return self.cache_root / "locks" / f"{base_id}.lock"

    def _build_speaker_cache(self, base_id: str) -> np.ndarray:
        files_all = self.embed_index.get(base_id, [])
        if not files_all:
            raise FileNotFoundError(f"No chunks for base_id={base_id}")

        pooled: List[np.ndarray] = []
        mmap_mode = "r" if self.mmap_load else None
        for fp in files_all:
            try:
                arr = _np_load_safe(fp, mmap_mode=mmap_mode)
                pooled.append(self._pool_chunk(arr))
            except Exception as e:
                if is_main():
                    print(f"[WARN][chunk_load] base_id={base_id} file={fp} err={type(e).__name__}: {e}")
                continue

        if len(pooled) < self.min_pairs:
            raise RuntimeError(f"base_id={base_id}: only {len(pooled)} valid chunks (need {self.min_pairs})")

        seq = np.stack(pooled, axis=0)
        seq = np.clip(seq, -10.0, 10.0)
        return seq.astype(self.cache_dtype, copy=False)

    def _load_speaker_sequence(self, base_id: str) -> np.ndarray:
        if self.cache_mode == "none" or self.cache_root is None:
            raise RuntimeError("Speaker cache disabled, but _load_speaker_sequence called.")

        cached_path = self._speaker_cache_path(base_id)
        lock_path = self._speaker_lock_path(base_id)

        hit = self.cache.get(base_id)
        if hit is not None:
            return hit

        if cached_path.exists():
            mmap_mode = "r" if self.mmap_load else None
            try:
                arr = _np_load_safe(str(cached_path), mmap_mode=mmap_mode)
                if arr.ndim == 2:
                    self.cache.put(base_id, arr)
                    return arr
                cached_path.unlink(missing_ok=True)
            except Exception as e:
                if is_main():
                    print(f"[WARN][cache_load] corrupt speaker cache: {cached_path} err={type(e).__name__}: {e}")
                try:
                    cached_path.unlink(missing_ok=True)
                except Exception:
                    pass

        got_lock = _acquire_lock(lock_path, timeout_s=60.0)
        try:
            if cached_path.exists():
                mmap_mode = "r" if self.mmap_load else None
                try:
                    arr = _np_load_safe(str(cached_path), mmap_mode=mmap_mode)
                    self.cache.put(base_id, arr)
                    return arr
                except Exception:
                    try:
                        cached_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            seq = self._build_speaker_cache(base_id)
            _atomic_npy_save(cached_path, seq)
        finally:
            if got_lock:
                _release_lock(lock_path)

        mmap_mode = "r" if self.mmap_load else None
        arr = _np_load_safe(str(cached_path), mmap_mode=mmap_mode)
        self.cache.put(base_id, arr)
        return arr

    def _load_sequence_window(self, base_id: str) -> np.ndarray:
        files_all = self.embed_index.get(base_id, [])
        if not files_all:
            raise FileNotFoundError(f"No chunks for base_id={base_id}")

        start, take_n = self._det_window(base_id, len(files_all))
        if take_n <= 0:
            raise FileNotFoundError(f"No files selected for base_id={base_id}")

        if self.cache_mode == "speaker" and self.cache_root is not None:
            merged = self._load_speaker_sequence(base_id)
            end = min(int(merged.shape[0]), start + take_n)
            if end - start < self.min_pairs:
                raise RuntimeError(f"Cached merged window too short for {base_id}: have {end-start}, need {self.min_pairs}")
            return np.array(merged[start:end], copy=False).astype(np.float32, copy=False)

        sel_files = files_all[start : start + take_n]
        pooled: List[np.ndarray] = []
        mmap_mode = "r" if self.mmap_load else None
        for fp in sel_files:
            try:
                arr = _np_load_safe(fp, mmap_mode=mmap_mode)
                pooled.append(self._pool_chunk(arr))
            except Exception as e:
                if is_main():
                    print(f"[WARN][chunk_load] base_id={base_id} file={fp} err={type(e).__name__}: {e}")
                continue

        if len(pooled) < self.min_pairs:
            raise RuntimeError(f"base_id={base_id}: only {len(pooled)} valid chunks in window (need {self.min_pairs})")

        seq = np.stack(pooled, axis=0)
        seq = np.clip(seq, -10.0, 10.0)
        return seq.astype(np.float32, copy=False)

    @staticmethod
    def _interleave_with_pad(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
        La, d = seq_a.shape
        Lb, _ = seq_b.shape
        L = max(La, Lb)
        zero = np.zeros((d,), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(L):
            out.append(seq_a[i] if i < La else zero)
            out.append(seq_b[i] if i < Lb else zero)
        return np.stack(out, axis=0)

    def prebuild_speaker_cache_rank0(self):
        if self.cache_mode != "speaker" or self.cache_root is None or (not is_main()):
            return
        bases = self.unique_bases
        print(f"[Cache] Prebuilding speaker cache for {len(bases)} base_ids -> {self.cache_root / 'speaker'}")
        ok, fail = 0, 0
        pbar = tqdm(bases, desc="[Cache] build speaker", unit="base")
        for base_id in pbar:
            try:
                cached_path = self._speaker_cache_path(base_id)
                if cached_path.exists():
                    try:
                        _ = _np_load_safe(str(cached_path), mmap_mode=("r" if self.mmap_load else None))
                        ok += 1
                        continue
                    except Exception:
                        try:
                            cached_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                lock_path = self._speaker_lock_path(base_id)
                got_lock = _acquire_lock(lock_path, timeout_s=5.0)
                try:
                    if cached_path.exists():
                        ok += 1
                        continue
                    seq = self._build_speaker_cache(base_id)
                    _atomic_npy_save(cached_path, seq)
                    ok += 1
                finally:
                    if got_lock:
                        _release_lock(lock_path)
            except Exception:
                fail += 1
            if (ok + fail) % 50 == 0:
                pbar.set_postfix(ok=ok, fail=fail)
        print(f"[Cache] done: ok={ok} fail={fail}")

    def __getitem__(self, idx: int):
        a, b, lab, cat_ids, ctx, rel_txt, meta = self.rows[idx]

        seq_a = self._load_sequence_window(a)
        seq_b = self._load_sequence_window(b)

        seq = self._interleave_with_pad(seq_a, seq_b)
        T = int(seq.shape[0])

        if not self.hp.use_cats:
            cat_ids_use = np.zeros((len(self.cat_cols),), dtype=np.int64)
        else:
            cat_ids_use = cat_ids.astype(np.int64, copy=False)

        if not self.hp.use_context:
            ctx_emb = np.zeros((self.hp.text_dim_fallback,), dtype=np.float32)
        else:
            ctx_emb = self.context_embed_map.get(ctx)
            if ctx_emb is None:
                ctx_emb = np.zeros((self.hp.text_dim_fallback,), dtype=np.float32)
            else:
                ctx_emb = ctx_emb.astype(np.float32, copy=False)

        if self.hp.use_rel_text:
            rel_emb = self.rel_embed_map.get(rel_txt)
            if rel_emb is None:
                rel_emb = np.zeros((self.hp.text_dim_fallback,), dtype=np.float32)
            else:
                rel_emb = rel_emb.astype(np.float32, copy=False)
        else:
            rel_emb = np.zeros((self.hp.text_dim_fallback,), dtype=np.float32)

        x = torch.from_numpy(seq.astype(np.float32, copy=False))
        lengths = torch.tensor(T, dtype=torch.long)
        cat_t = torch.from_numpy(cat_ids_use)
        ctx_t = torch.from_numpy(ctx_emb)
        rel_t = torch.from_numpy(rel_emb)
        y = torch.tensor(float(lab), dtype=torch.float32)
        return x, lengths, cat_t, ctx_t, rel_t, y, meta


def collate_dyadic_batch(batch):
    xs, lengths, cats, ctxs, rels, ys, metas = zip(*batch)
    lengths = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in lengths]).long()
    ys = torch.stack(ys).float()

    B = len(xs)
    T_max = max(int(x.shape[0]) for x in xs)
    d_in = int(xs[0].shape[1])

    x_pad = torch.zeros(B, T_max, d_in, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = x

    cat_t = torch.stack(cats).long()
    ctx_t = torch.stack(ctxs).float()
    rel_t = torch.stack(rels).float()
    return x_pad, lengths, cat_t, ctx_t, rel_t, ys, list(metas)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * x
        return self.weight * norm_x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        hidden_dim = int(4 * d_model / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos_cached[:seq_len][None, None, :, :]
        sin = self.sin_cached[:seq_len][None, None, :, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.drop = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B,T,D]
        key_padding_mask: [B,T] True=pad
        returns:
          out: [B,T,D]
          attn: [B,H,T,T] (if return_attn else None)
        """
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
            scores = scores.masked_fill(cm[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_drop = self.drop(attn)
        out = attn_drop @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)

        return out, (attn if return_attn else None)


class LLaMAEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, n_heads, rope, dropout)
        self.mlp = SwiGLU(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.attn_norm(x)
        attn_out, attn = self.attn(h, key_padding_mask, causal=causal, return_attn=return_attn)
        x = x + self.drop(attn_out)
        h2 = self.ffn_norm(x)
        x = x + self.drop(self.mlp(h2))
        return x, attn


class NaturalnessTransformer(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        num_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2048,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.input_proj = nn.Linear(d_in, d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rope = RotaryEmbedding(dim=(d_model // n_heads), max_position=max_len)
        self.layers = nn.ModuleList([LLaMAEncoderBlock(d_model, n_heads, self.rope, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

    def forward_tokens(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)

        key_padding_mask = _kpm_from_lengths(lengths, T)

        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn = layer(h, key_padding_mask=key_padding_mask, causal=self.causal, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)

        h = self.final_norm(h)
        return h, (attn_list if return_attn else None)

    def encode_pooled(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h, _ = self.forward_tokens(x, lengths, return_attn=False)
        kpm = _kpm_from_lengths(lengths, T)
        valid = (~kpm).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (h * valid).sum(dim=1) / denom


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
        T = x.shape[1]
        return x + self.pe[:T].unsqueeze(0)


class BaseTransformerAudioEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, num_layers: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

    def forward_tokens(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.drop(h)
        kpm = _kpm_from_lengths(lengths, h.shape[1])
        h = self.enc(h, src_key_padding_mask=kpm)
        return h

    def encode_pooled(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.forward_tokens(x, lengths)
        kpm = _kpm_from_lengths(lengths, h.shape[1])
        valid = (~kpm).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (h * valid).sum(dim=1) / denom


class CatCtxRelEncoder(nn.Module):
    def __init__(self, cat_cardinalities: List[int], cat_embed_dim: int, ctx_dim: int, rel_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c, cat_embed_dim) for c in cat_cardinalities])
        self.cat_proj = nn.Sequential(
            nn.Linear(len(cat_cardinalities) * cat_embed_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rel_proj = nn.Sequential(
            nn.LayerNorm(rel_dim),
            nn.Linear(rel_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, cat_x: torch.Tensor, ctx_x: torch.Tensor, rel_x: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embs)]
        cat = self.cat_proj(torch.cat(embs, dim=-1))
        ctx = self.ctx_proj(ctx_x)
        rel = self.rel_proj(rel_x)
        return torch.cat([cat, ctx, rel], dim=-1)


def masked_mean_pool(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    rng = torch.arange(T, device=x.device)[None, :]
    mask = (rng < lengths[:, None]).to(x.dtype)
    mask3 = mask.unsqueeze(-1)
    summed = (x * mask3).sum(dim=1)
    denom = lengths.clamp(min=1).to(x.dtype).unsqueeze(-1)
    return summed / denom


class NaturalnessFusionLLaMAModel(nn.Module):
    def __init__(
        self,
        d_in_audio: int,
        audio_d_model: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        max_len: int,
        cat_cards: List[int],
        ctx_dim: int,
        rel_dim: int,
        use_aux: bool,
        attn_mode: str = "mean_query",
        aux_dim: int = 128,
        cat_embed_dim: int = 32,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.attn_mode = attn_mode
        self.audio = NaturalnessTransformer(
            d_in=d_in_audio,
            d_model=audio_d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            causal=False,
        )
        if self.use_aux:
            self.aux = CatCtxRelEncoder(cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout)
            fusion_in = audio_d_model + (3 * aux_dim)
        else:
            self.aux = None
            fusion_in = audio_d_model

        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _reduce_self_attn(attn_bhts: torch.Tensor, kpm: torch.Tensor, mode: str) -> torch.Tensor:
        """
        attn_bhts: [B,H,T,T]
        kpm: [B,T] True=pad
        returns [B,T] attention over source positions after reducing heads and query positions.
        """
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
        aq = aq / denom2
        return aq

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x, return_attn: bool = False):
        if not return_attn:
            pooled_audio = self.audio.encode_pooled(x_audio, len_audio)
            if self.use_aux:
                aux = self.aux(cat_x, ctx_x, rel_x)
                z = torch.cat([pooled_audio, aux], dim=-1)
            else:
                z = pooled_audio
            return self.head(z).squeeze(-1)

        tokens, attn_list = self.audio.forward_tokens(x_audio, len_audio, return_attn=True)
        assert attn_list is not None
        kpm = _kpm_from_lengths(len_audio, tokens.shape[1])

        valid = (~kpm).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled_audio = (tokens * valid).sum(dim=1) / denom

        if self.use_aux:
            aux = self.aux(cat_x, ctx_x, rel_x)
            z = torch.cat([pooled_audio, aux], dim=-1)
        else:
            z = pooled_audio
        logits = self.head(z).squeeze(-1)

        attn_reduced: List[torch.Tensor] = []
        for a in attn_list:
            attn_reduced.append(self._reduce_self_attn(a, kpm, mode=self.attn_mode))
        return logits, attn_reduced, kpm.detach(), len_audio.detach()


class BaseFusionTransformerModel(nn.Module):
    def __init__(
        self,
        d_in_audio: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        max_len: int,
        cat_cards: List[int],
        ctx_dim: int,
        rel_dim: int,
        use_aux: bool,
        aux_dim: int = 128,
        cat_embed_dim: int = 32,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.audio = BaseTransformerAudioEncoder(
            d_in=d_in_audio,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
        )
        if self.use_aux:
            self.aux = CatCtxRelEncoder(cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout)
            fusion_in = d_model + (3 * aux_dim)
        else:
            self.aux = None
            fusion_in = d_model

        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        pooled_audio = self.audio.encode_pooled(x_audio, len_audio)
        if self.use_aux:
            aux = self.aux(cat_x, ctx_x, rel_x)
            z = torch.cat([pooled_audio, aux], dim=-1)
        else:
            z = pooled_audio
        return self.head(z).squeeze(-1)


class NaturalnessFusionMLPModel(nn.Module):
    def __init__(
        self,
        d_in_audio: int,
        dropout: float,
        cat_cards: List[int],
        ctx_dim: int,
        rel_dim: int,
        use_aux: bool,
        aux_dim: int = 128,
        cat_embed_dim: int = 32,
        hidden: int = 512,
        layers: int = 2,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        if self.use_aux:
            self.aux = CatCtxRelEncoder(cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout)
            fusion_in = d_in_audio + (3 * aux_dim)
        else:
            self.aux = None
            fusion_in = d_in_audio

        blocks: List[nn.Module] = []
        in_dim = fusion_in
        n_layers = max(1, int(layers))
        for _ in range(n_layers):
            blocks.append(nn.Linear(in_dim, hidden))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        pooled_audio = masked_mean_pool(x_audio, len_audio)
        if self.use_aux:
            aux = self.aux(cat_x, ctx_x, rel_x)
            z = torch.cat([pooled_audio, aux], dim=-1)
        else:
            z = pooled_audio
        return self.head(z).squeeze(-1)


class NaturalnessFusionLogRegModel(nn.Module):
    def __init__(
        self,
        d_in_audio: int,
        dropout: float,
        cat_cards: List[int],
        ctx_dim: int,
        rel_dim: int,
        use_aux: bool,
        aux_dim: int = 128,
        cat_embed_dim: int = 32,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        if self.use_aux:
            self.aux = CatCtxRelEncoder(cat_cards, cat_embed_dim, ctx_dim, rel_dim, aux_dim, dropout)
            fusion_in = d_in_audio + (3 * aux_dim)
        else:
            self.aux = None
            fusion_in = d_in_audio

        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(fusion_in, 1)

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x):
        pooled_audio = masked_mean_pool(x_audio, len_audio)
        if self.use_aux:
            aux = self.aux(cat_x, ctx_x, rel_x)
            z = torch.cat([pooled_audio, aux], dim=-1)
        else:
            z = pooled_audio
        z = self.drop(z)
        return self.linear(z).squeeze(-1)


class CrossAttnFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(d_model),
                        "norm_kv": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                        "drop": nn.Dropout(dropout),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * d_model, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, aux_tok: torch.Tensor, audio_tokens: torch.Tensor, kpm: torch.Tensor, return_weights: bool = False):
        q = aux_tok
        w_list: List[torch.Tensor] = []
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
            if return_weights:
                w_list.append(attn_w.detach())
        if return_weights:
            return q, w_list
        return q


class NaturalnessCrossAttnModel(nn.Module):
    def __init__(
        self,
        d_in_audio: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        max_len: int,
        cross_layers: int,
        cross_dropout: float,
        cat_cards: List[int],
        ctx_dim: int,
        rel_dim: int,
        use_aux: bool,
        aux_dim: int = 128,
        cat_embed_dim: int = 32,
    ):
        super().__init__()
        self.use_aux = bool(use_aux)
        self.audio_encoder = NaturalnessTransformer(
            d_in=d_in_audio,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            causal=False,
        )

        if self.use_aux:
            self.aux = CatCtxRelEncoder(cat_cards, cat_embed_dim, ctx_dim, rel_dim, out_dim=aux_dim, dropout=dropout)
            self.aux_to_tok = nn.Sequential(
                nn.LayerNorm(3 * aux_dim),
                nn.Linear(3 * aux_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.aux = None
            self.aux_to_tok = nn.Identity()

        self.xattn = CrossAttnFusion(d_model=d_model, n_heads=n_heads, num_layers=cross_layers, dropout=cross_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _reduce_cross_attn(attn_bh1t: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        a = attn_bh1t.mean(dim=1).squeeze(1)
        valid = (~kpm).float()
        a = a * valid
        denom = a.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return a / denom

    def forward(self, x_audio, len_audio, cat_x, ctx_x, rel_x, return_attn: bool = False):
        tokens, _ = self.audio_encoder.forward_tokens(x_audio, len_audio, return_attn=False)
        kpm = _kpm_from_lengths(len_audio, tokens.shape[1])

        if self.use_aux:
            aux_feat = self.aux(cat_x, ctx_x, rel_x)
            aux_tok = self.aux_to_tok(aux_feat).unsqueeze(1)
        else:
            valid = (~kpm).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (tokens * valid).sum(dim=1) / denom
            aux_tok = pooled.unsqueeze(1)

        if not return_attn:
            fused = self.xattn(aux_tok, tokens, kpm, return_weights=False)
            return self.head(fused.squeeze(1)).squeeze(-1)

        fused, w_list = self.xattn(aux_tok, tokens, kpm, return_weights=True)
        logits = self.head(fused.squeeze(1)).squeeze(-1)

        attn_reduced: List[torch.Tensor] = []
        for w in w_list:
            attn_reduced.append(self._reduce_cross_attn(w, kpm))
        return logits, attn_reduced, kpm.detach(), len_audio.detach()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds == labels).sum()
    total = torch.tensor(labels.numel(), device=labels.device, dtype=torch.long)
    return correct, total


@torch.no_grad()
def ddp_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool, threshold: float):
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    total_sum = torch.tensor(0.0, device=device)

    for x, lengths, cat_x, ctx_x, rel_x, labels, _metas in loader:
        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        cat_x = cat_x.to(device, non_blocking=True)
        ctx_x = ctx_x.to(device, non_blocking=True)
        rel_x = rel_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(x, lengths, cat_x, ctx_x, rel_x)
            loss = criterion(logits, labels)

        correct, total = accuracy_from_logits(logits, labels, threshold=threshold)
        bs = torch.tensor(labels.numel(), device=device, dtype=torch.float32)

        loss_sum += loss.detach() * bs
        correct_sum += correct.detach()
        total_sum += total.detach()

    all_reduce_sum(loss_sum)
    all_reduce_sum(correct_sum)
    all_reduce_sum(total_sum)

    loss_mean = (loss_sum / total_sum.clamp(min=1.0)).item()
    acc_mean = (correct_sum / total_sum.clamp(min=1.0)).item()
    return loss_mean, acc_mean


@torch.no_grad()
def ddp_test_collect_with_meta_and_attn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    dump_attn: bool,
    attn_max_batches: int,
    attn_out_dir: Path,
    attn_mode: str,
):
    model.eval()
    y_true_list, y_prob_list = [], []
    meta_list: List[Dict[str, Any]] = []

    do_attn = dump_attn and is_main()
    attn_batches: List[Dict[str, Any]] = []
    dumped_batches = 0

    supports_attn = hasattr(model, "module") and hasattr(model.module, "forward")
    raw_model = model.module if hasattr(model, "module") else model

    def _try_forward_with_attn(x, lengths, cat_x, ctx_x, rel_x):
        if hasattr(raw_model, "forward"):
            try:
                return raw_model.forward(x, lengths, cat_x, ctx_x, rel_x, return_attn=True)
            except TypeError:
                return None
        return None

    for batch_idx, (x, lengths, cat_x, ctx_x, rel_x, labels, metas) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        cat_x = cat_x.to(device, non_blocking=True)
        ctx_x = ctx_x.to(device, non_blocking=True)
        rel_x = rel_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            attn_pack = None
            if do_attn and dumped_batches < attn_max_batches:
                out = _try_forward_with_attn(x, lengths, cat_x, ctx_x, rel_x)
                if out is not None and isinstance(out, tuple) and len(out) == 4:
                    logits, attn_list, kpm, lens_out = out
                    attn_pack = (attn_list, kpm, lens_out)
                else:
                    logits = model(x, lengths, cat_x, ctx_x, rel_x)
            else:
                logits = model(x, lengths, cat_x, ctx_x, rel_x)

        probs = torch.sigmoid(logits)
        y_true_list.append(labels.detach().cpu())
        y_prob_list.append(probs.detach().cpu())
        meta_list.extend(metas)

        if do_attn and attn_pack is not None and dumped_batches < attn_max_batches:
            attn_list, kpm, lens_out = attn_pack
            attn_cpu = [a.detach().float().cpu().numpy() for a in attn_list]
            kpm_cpu = kpm.detach().cpu().numpy()
            lens_cpu = lens_out.detach().cpu().numpy()

            attn_batches.append(
                {
                    "batch_idx": batch_idx,
                    "metas": metas,
                    "lengths": lens_cpu,
                    "kpm": kpm_cpu,
                    "attn_reduced": attn_cpu,
                }
            )
            dumped_batches += 1

    y_true = torch.cat(y_true_list).numpy() if y_true_list else np.array([])
    y_prob = torch.cat(y_prob_list).numpy() if y_prob_list else np.array([])

    attn_payload = None
    if do_attn and attn_batches:
        attn_out_dir.mkdir(parents=True, exist_ok=True)
        attn_path = attn_out_dir / "attn_batches.pt"
        torch.save(
            {
                "attn_mode": attn_mode,
                "num_batches": len(attn_batches),
                "batches": attn_batches,
            },
            attn_path,
        )
        attn_payload = {"attn_path": str(attn_path)}

    return y_true, y_prob, meta_list, attn_payload


def _write_test_outputs(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, metas: List[Dict[str, Any]], threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob >= threshold).astype(np.int64)
    correct = (y_pred == y_true.astype(np.int64)).astype(np.int64)

    pred_path = out_dir / "test_predictions.csv"
    mis_path = out_dir / "test_misclassified.csv"
    metrics_path = out_dir / "test_metrics.json"

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "row_idx_in_csv",
                "p1_base",
                "p2_base",
                "p1_path",
                "p2_path",
                "label",
                "y_prob",
                "y_pred",
                "correct",
            ],
        )
        w.writeheader()
        for i in range(int(y_true.shape[0])):
            m = metas[i] if i < len(metas) else {}
            w.writerow(
                {
                    "idx": i,
                    "row_idx_in_csv": m.get("row_idx_in_csv", ""),
                    "p1_base": m.get("p1_base", ""),
                    "p2_base": m.get("p2_base", ""),
                    "p1_path": m.get("p1_path", ""),
                    "p2_path": m.get("p2_path", ""),
                    "label": int(y_true[i]),
                    "y_prob": float(y_prob[i]),
                    "y_pred": int(y_pred[i]),
                    "correct": int(correct[i]),
                }
            )

    with mis_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "row_idx_in_csv",
                "p1_base",
                "p2_base",
                "p1_path",
                "p2_path",
                "label",
                "y_prob",
                "y_pred",
            ],
        )
        w.writeheader()
        for i in range(int(y_true.shape[0])):
            if int(correct[i]) == 1:
                continue
            m = metas[i] if i < len(metas) else {}
            w.writerow(
                {
                    "idx": i,
                    "row_idx_in_csv": m.get("row_idx_in_csv", ""),
                    "p1_base": m.get("p1_base", ""),
                    "p2_base": m.get("p2_base", ""),
                    "p1_path": m.get("p1_path", ""),
                    "p2_path": m.get("p2_path", ""),
                    "label": int(y_true[i]),
                    "y_prob": float(y_prob[i]),
                    "y_pred": int(y_pred[i]),
                }
            )

    cm = confusion_matrix(y_true.astype(np.int64), y_pred.astype(np.int64)).tolist()
    out = {
        "threshold": float(threshold),
        "test_acc": float((correct.mean() if correct.size else 0.0)),
        "confusion_matrix": cm,
    }
    try:
        out["brier"] = float(brier_score_loss(y_true.astype(np.int64), y_prob.astype(np.float64)))
    except Exception:
        out["brier"] = None

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[TEST] wrote: {pred_path}")
    print(f"[TEST] wrote: {mis_path}")
    print(f"[TEST] wrote: {metrics_path}")


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _attn_summary_from_batches(attn_batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    attn_batches: list with "attn_reduced" (list of layers, each [B,T]) and "kpm"/"lengths"
    Returns simple stats per layer: mean entropy, mean top1 mass, mean top5 mass.
    """
    if not attn_batches:
        return {"note": "no attention batches"}

    L = len(attn_batches[0]["attn_reduced"])
    ent_sums = np.zeros((L,), dtype=np.float64)
    top1_sums = np.zeros((L,), dtype=np.float64)
    top5_sums = np.zeros((L,), dtype=np.float64)
    n_sums = np.zeros((L,), dtype=np.float64)

    for b in attn_batches:
        kpm = b["kpm"].astype(bool)
        valid = (~kpm).astype(np.float32)
        for li in range(L):
            a = b["attn_reduced"][li].astype(np.float64)
            ent = _entropy(a)

            top1 = np.max(a, axis=-1)
            top5 = np.sort(a, axis=-1)[:, -5:].sum(axis=-1) if a.shape[1] >= 5 else np.sort(a, axis=-1).sum(axis=-1)
            ent_sums[li] += ent.sum()
            top1_sums[li] += top1.sum()
            top5_sums[li] += top5.sum()
            n_sums[li] += a.shape[0]

    out = {"layers": []}
    for li in range(L):
        n = max(1.0, n_sums[li])
        out["layers"].append(
            {
                "layer": int(li),
                "mean_entropy": float(ent_sums[li] / n),
                "mean_top1_mass": float(top1_sums[li] / n),
                "mean_top5_mass": float(top5_sums[li] / n),
            }
        )
    return out



def parse_args() -> HyperParams:
    p = argparse.ArgumentParser()

    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True, help="Explicit test CSV (recommended).")
    p.add_argument("--val-csv", type=str, default="", help="Optional explicit val CSV. If absent, uses --val-split from train (if >0).")

    p.add_argument("--embed-root", type=str, required=True, help="Root directory of .npy embeddings (embeds_vec or embeds_seq).")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument(
        "--model-type",
        type=str,
        default="llama",
        choices=["base_transformer", "llama", "crossattn", "mlp", "logreg", "transformer"],
        help="Methods: base_transformer | llama | crossattn | mlp | logreg. Alias: transformer -> llama.",
    )

    p.add_argument("--p1-path-col", type=str, default=HyperParams.p1_path_col)
    p.add_argument("--p2-path-col", type=str, default=HyperParams.p2_path_col)
    p.add_argument("--label-col", type=str, default=HyperParams.label_col)

    p.add_argument("--context-col", type=str, default=HyperParams.context_col)
    p.add_argument("--rel-detail-col", type=str, default=HyperParams.rel_detail_col)
    p.add_argument("--speaker-a-role-col", type=str, default=HyperParams.speaker_a_role_col)
    p.add_argument("--speaker-b-role-col", type=str, default=HyperParams.speaker_b_role_col)

    p.add_argument("--speech-only", action="store_true", help="Disable ALL non-speech features (cats + context + rel-text).")
    p.add_argument("--no-cats", action="store_true", help="Disable categorical features (speaker roles + rel_detail ids).")
    p.add_argument("--no-context", action="store_true", help="Disable high_level_context text embeddings.")
    p.add_argument("--no-rel-text", action="store_true", help="Disable relationship text embeddings.")

    p.add_argument("--rel-text-template", type=str, default=HyperParams.rel_text_template)
    p.add_argument("--rel-text-cache", type=str, default="")

    p.add_argument("--max-seq-len", type=int, default=HyperParams.max_seq_len)
    p.add_argument("--window-strategy", type=str, default=HyperParams.window_strategy, choices=["head", "random"])
    p.add_argument("--val-split", type=float, default=HyperParams.val_split, help="If no --val-csv, carve this fraction from train for validation. Use 0.0 for no val.")
    p.add_argument("--patience", type=int, default=HyperParams.patience, help="Early stopping patience (only used if validation exists).")

    p.add_argument("--batch-size", type=int, default=HyperParams.batch_size)
    p.add_argument("--num-epochs", type=int, default=HyperParams.num_epochs)
    p.add_argument("--lr", type=float, default=HyperParams.lr)
    p.add_argument("--weight-decay", type=float, default=HyperParams.weight_decay)
    p.add_argument("--num-workers", type=int, default=HyperParams.num_workers)
    p.add_argument("--seed", type=int, default=HyperParams.seed)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)

    p.add_argument("--d-model", type=int, default=HyperParams.d_model)
    p.add_argument("--num-layers", type=int, default=HyperParams.num_layers)
    p.add_argument("--n-heads", type=int, default=HyperParams.n_heads)
    p.add_argument("--dropout", type=float, default=HyperParams.dropout)

    p.add_argument("--crossattn-layers", type=int, default=HyperParams.crossattn_layers)
    p.add_argument("--crossattn-dropout", type=float, default=HyperParams.crossattn_dropout)

    p.add_argument("--mlp-hidden", type=int, default=HyperParams.mlp_hidden)
    p.add_argument("--mlp-layers", type=int, default=HyperParams.mlp_layers)

    p.add_argument("--index-cache", type=str, default="")
    p.add_argument("--no-gather-test-preds", action="store_true")
    p.add_argument("--cache-items", type=int, default=HyperParams.cache_items)
    p.add_argument("--no-mmap-load", action="store_true")

    p.add_argument("--cache-root", type=str, default="")
    p.add_argument("--cache-mode", type=str, default="speaker", choices=["speaker", "none"])
    p.add_argument("--cache-dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--prebuild-cache", action="store_true")

    p.add_argument("--text-model", type=str, default=HyperParams.text_model)
    p.add_argument("--context-cache", type=str, default="")
    p.add_argument("--prebuild-text", action="store_true")
    p.add_argument("--text-batch-size", type=int, default=HyperParams.text_batch_size)
    p.add_argument("--text-max-len", type=int, default=HyperParams.text_max_len)
    p.add_argument("--text-fp16", action="store_true")
    p.add_argument("--no-text-gpu-parallel", action="store_true")

    p.add_argument("--test-threshold", type=float, default=0.5, help="Threshold on sigmoid(logit) to compute y_pred.")
    p.add_argument("--no-write-test-preds", action="store_true", help="Disable writing test_predictions/misclassified/metrics files.")

    p.add_argument("--dump-attn", action="store_true", help="Dump reduced attention during test (rank0). Only for llama/crossattn.")
    p.add_argument("--attn-out", type=str, default="", help="Directory to write attention dumps (default: <out_dir>/attn).")
    p.add_argument("--attn-max-batches", type=int, default=50, help="Limit number of test batches to dump attention for.")
    p.add_argument("--attn-mode", type=str, default="mean_query", choices=["mean_query", "last_query"], help="For llama: how to reduce query dimension.")

    a = p.parse_args()

    hp = HyperParams(
        train_csv=a.train_csv,
        test_csv=a.test_csv,
        val_csv=a.val_csv,
        embed_root=a.embed_root,
        out_dir=a.out_dir,
        model_type=a.model_type,
        p1_path_col=a.p1_path_col,
        p2_path_col=a.p2_path_col,
        label_col=a.label_col,
        context_col=a.context_col,
        rel_detail_col=a.rel_detail_col,
        speaker_a_role_col=a.speaker_a_role_col,
        speaker_b_role_col=a.speaker_b_role_col,
        use_cats=(not a.no_cats),
        use_context=(not a.no_context),
        use_rel_text=(not a.no_rel_text),
        speech_only=bool(a.speech_only),
        rel_text_template=a.rel_text_template,
        rel_text_cache=a.rel_text_cache,
        max_seq_len=a.max_seq_len,
        window_strategy=a.window_strategy,
        val_split=float(a.val_split),
        batch_size=a.batch_size,
        num_epochs=a.num_epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        num_workers=a.num_workers,
        seed=a.seed,
        patience=a.patience,
        use_amp=(not a.no_amp) and torch.cuda.is_available(),
        d_model=a.d_model,
        num_layers=a.num_layers,
        n_heads=a.n_heads,
        dropout=a.dropout,
        crossattn_layers=int(a.crossattn_layers),
        crossattn_dropout=float(a.crossattn_dropout),
        mlp_hidden=a.mlp_hidden,
        mlp_layers=a.mlp_layers,
        index_cache=a.index_cache,
        gather_test_preds=(not a.no_gather_test_preds),
        cache_items=a.cache_items,
        mmap_load=(not a.no_mmap_load),
        cache_root=a.cache_root,
        cache_mode=a.cache_mode,
        cache_dtype=a.cache_dtype,
        prebuild_cache=a.prebuild_cache,
        text_model=a.text_model,
        context_cache=a.context_cache,
        prebuild_text=a.prebuild_text,
        text_batch_size=a.text_batch_size,
        text_max_len=a.text_max_len,
        text_fp16=bool(a.text_fp16),
        text_use_gpu_parallel=(not a.no_text_gpu_parallel),
        test_threshold=float(a.test_threshold),
        dump_attn=bool(a.dump_attn),
        attn_out=str(a.attn_out),
        attn_max_batches=int(a.attn_max_batches),
        attn_mode=str(a.attn_mode),
        resume_from=a.resume_from,
    )
    hp._no_write_test_preds = bool(a.no_write_test_preds)

    if hp.speech_only:
        hp.use_cats = False
        hp.use_context = False
        hp.use_rel_text = False

    if (hp.model_type or "").lower().strip() == "transformer":
        hp.model_type = "llama"

    return hp



def main():
    distributed, rank, local_rank = setup_ddp_from_env()
    hp = parse_args()
    set_seed(hp.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(hp.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "hyperparams.json").open("w") as f:
            json.dump(asdict(hp), f, indent=2)

    ddp_barrier()

    if is_main():
        print(f"DDP: {distributed} | world_size={get_world_size()} | rank={get_rank()} | local_rank={local_rank}")
        print(f"Device: {device} | AMP: {hp.use_amp}")
        print(f"MODEL_TYPE: {hp.model_type}")
        print(f"TRAIN CSV: {hp.train_csv}")
        print(f"TEST  CSV: {hp.test_csv}")
        print(f"VAL   CSV: {hp.val_csv if hp.val_csv else '(none)'} | val_split={hp.val_split}")
        print(f"EMBED_ROOT: {hp.embed_root}")
        if hp.cache_root and hp.cache_mode != "none":
            print(f"CACHE: mode={hp.cache_mode} root={Path(hp.cache_root).resolve()} dtype={hp.cache_dtype} prebuild={hp.prebuild_cache}")
        print(f"TEXT: model={hp.text_model} prebuild_text={hp.prebuild_text} gpu_parallel={hp.text_use_gpu_parallel}")
        print(f"FEATURES: speech_only={hp.speech_only} use_cats={hp.use_cats} use_context={hp.use_context} use_rel_text={hp.use_rel_text}")
        print(f"TEST threshold={hp.test_threshold}")
        if hp.dump_attn:
            print(f"ATTN: dump_attn=True mode={hp.attn_mode} max_batches={hp.attn_max_batches} out={hp.attn_out or (str(out_dir/'attn'))}")

    embed_root = Path(hp.embed_root).resolve()
    if not embed_root.exists():
        raise FileNotFoundError(f"embed_root not found: {embed_root}")

    index_cache = Path(hp.index_cache) if hp.index_cache else (out_dir / "embed_index.pkl")
    if is_main():
        if index_cache.exists():
            print(f"[Index] Loading cache: {index_cache}")
            with index_cache.open("rb") as f:
                embed_index = pickle.load(f)
        else:
            print("[Index] Scanning embed_root recursively (one-time)...")
            embed_index = build_embed_index(embed_root)
            with index_cache.open("wb") as f:
                pickle.dump(embed_index, f)
            print(f"[Index] Cached to: {index_cache}")
        print(f"[Index] base_ids={len(embed_index)} total_chunk_files={sum(len(v) for v in embed_index.values())}")
    ddp_barrier()
    if not is_main():
        if not index_cache.exists():
            raise RuntimeError(f"Rank{rank}: expected index_cache to exist: {index_cache}")
        with index_cache.open("rb") as f:
            embed_index = pickle.load(f)
    ddp_barrier()

    context_cache = Path(hp.context_cache) if hp.context_cache else (out_dir / "context_hf_cache.pkl")
    rel_cache = Path(hp.rel_text_cache) if hp.rel_text_cache else (out_dir / "relationship_hf_cache.pkl")

    csv_paths_for_text = [Path(hp.train_csv), Path(hp.test_csv)]
    if hp.val_csv:
        csv_paths_for_text.append(Path(hp.val_csv))

    need_build_ctx = hp.use_context and (hp.prebuild_text or (not context_cache.exists()))
    need_build_rel = hp.use_rel_text and (hp.prebuild_text or (not rel_cache.exists()))

    if need_build_ctx:
        build_context_text_embed_map_ddp(
            csv_paths=csv_paths_for_text,
            context_col=hp.context_col,
            cache_path=context_cache,
            model_name=hp.text_model,
            batch_size=hp.text_batch_size,
            max_len=hp.text_max_len,
            use_fp16=hp.text_fp16,
            fallback_dim=hp.text_dim_fallback,
            use_gpu_parallel=hp.text_use_gpu_parallel,
        )
    else:
        if is_main():
            print(f"[CTX] {'Disabled' if not hp.use_context else f'Using existing cache: {context_cache}'}")

    if need_build_rel:
        build_relationship_text_embed_map_ddp(
            csv_paths=csv_paths_for_text,
            speaker_a_role_col=hp.speaker_a_role_col,
            speaker_b_role_col=hp.speaker_b_role_col,
            rel_detail_col=hp.rel_detail_col,
            template=hp.rel_text_template,
            cache_path=rel_cache,
            model_name=hp.text_model,
            batch_size=hp.text_batch_size,
            max_len=hp.text_max_len,
            use_fp16=hp.text_fp16,
            fallback_dim=hp.text_dim_fallback,
            use_gpu_parallel=hp.text_use_gpu_parallel,
        )
    else:
        if is_main():
            print(f"[REL] {'Disabled' if not hp.use_rel_text else f'Using existing cache: {rel_cache}'}")

    ddp_barrier()

    if not hp.use_context:
        context_embed_map = {}
    else:
        if not context_cache.exists():
            if is_main():
                print("[CTX] ERROR: context cache not found. Using empty map (zeros fallback).")
            context_embed_map = {}
        else:
            with context_cache.open("rb") as f:
                context_embed_map = pickle.load(f)

    if not hp.use_rel_text:
        rel_embed_map = {}
    else:
        if not rel_cache.exists():
            if is_main():
                print("[REL] ERROR: relationship cache not found. Using empty map (zeros fallback).")
            rel_embed_map = {}
        else:
            with rel_cache.open("rb") as f:
                rel_embed_map = pickle.load(f)

    ddp_barrier()

    train_dataset_full = DyadicNaturalnessDataset(
        csv_path=hp.train_csv,
        embed_index=embed_index,
        context_embed_map=context_embed_map,
        rel_embed_map=rel_embed_map,
        hp=hp,
        min_pairs=MIN_REAL_PAIRS,
    )
    test_dataset = DyadicNaturalnessDataset(
        csv_path=hp.test_csv,
        embed_index=embed_index,
        context_embed_map=context_embed_map,
        rel_embed_map=rel_embed_map,
        hp=hp,
        min_pairs=MIN_REAL_PAIRS,
    )

    if len(train_dataset_full) == 0:
        if is_main():
            print("\n[ERROR] Train dataset is empty AFTER filtering.")
            print("Most likely your embed_root files aren't indexed by the filename parser.")
            print("Check that your embedding filenames look like: <wav_stem_lower>__ch00017.npy")
        raise ValueError("Train dataset is empty.")

    if len(test_dataset) == 0:
        if is_main():
            print("\n[ERROR] Test dataset is empty AFTER filtering.")
        raise ValueError("Test dataset is empty.")

    if hp.val_csv:
        val_dataset = DyadicNaturalnessDataset(
            csv_path=hp.val_csv,
            embed_index=embed_index,
            context_embed_map=context_embed_map,
            rel_embed_map=rel_embed_map,
            hp=hp,
            min_pairs=MIN_REAL_PAIRS,
        )
    else:
        val_dataset = None

    if (val_dataset is None) and (hp.val_split and hp.val_split > 0.0):
        N = len(train_dataset_full)
        val_n = max(1, int(N * hp.val_split))
        if val_n >= N:
            val_n = max(1, N - 1)
        train_n = N - val_n

        g = torch.Generator().manual_seed(hp.seed)
        perm = torch.randperm(N, generator=g).tolist()
        train_idx = perm[:train_n]
        val_idx = perm[train_n:]

        train_set = Subset(train_dataset_full, train_idx)
        val_set = Subset(train_dataset_full, val_idx)
        if is_main():
            print(f"[Split] train={len(train_set)} val={len(val_set)} (from train_csv via val_split={hp.val_split})")
    else:
        train_set = train_dataset_full
        val_set = val_dataset
        if is_main():
            if val_set is None:
                print("[Split] No validation set (val_split=0 and no val_csv). Early stopping disabled.")
            else:
                print(f"[Split] train={len(train_set)} val={len(val_set)} (explicit val_csv)")

    if hp.prebuild_cache and hp.cache_root and hp.cache_mode == "speaker":
        if is_main():
            train_dataset_full.prebuild_speaker_cache_rank0()
        ddp_barrier()

    x0, _, ctx0, rel0, _, _m0 = None, None, None, None, None, None
    x0, _, cat0, ctx0, rel0, _, _m0 = train_dataset_full[0]
    d_in_audio = int(x0.shape[-1])
    ctx_dim = int(ctx0.shape[-1])
    rel_dim = int(rel0.shape[-1])
    cat_cards = [len(train_dataset_full.cat_vocab[c]) for c in train_dataset_full.cat_cols]

    use_aux = bool(hp.use_cats or hp.use_context or hp.use_rel_text)
    if not use_aux:
        cat_cards = [1, 1, 1]
        ctx_dim = hp.text_dim_fallback
        rel_dim = hp.text_dim_fallback

    if is_main():
        print(f"Detected: d_in_audio={d_in_audio} ctx_dim={ctx_dim} rel_dim={rel_dim} cat_cards={cat_cards} use_aux={use_aux}")
        print(f"Train usable={len(train_set)} | Test usable={len(test_dataset)}")

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if (distributed and val_set is not None) else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False) if distributed else None

    dl_common = dict(
        num_workers=hp.num_workers,
        pin_memory=True,
        collate_fn=collate_dyadic_batch,
        persistent_workers=(hp.num_workers > 0),
        prefetch_factor=2 if hp.num_workers > 0 else None,
    )
    dl_common = {k: v for k, v in dl_common.items() if v is not None}

    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **dl_common)
    val_loader = DataLoader(val_set, batch_size=hp.batch_size, shuffle=False, sampler=val_sampler, **dl_common) if val_set is not None else None
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, sampler=test_sampler, **dl_common)

    mt = (hp.model_type or "llama").lower().strip()
    if mt == "transformer":
        mt = "llama"

    if mt == "llama":
        model: nn.Module = NaturalnessFusionLLaMAModel(
            d_in_audio=d_in_audio,
            audio_d_model=hp.d_model,
            num_layers=hp.num_layers,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            max_len=hp.max_seq_len or 2048,
            cat_cards=cat_cards,
            ctx_dim=ctx_dim,
            rel_dim=rel_dim,
            use_aux=use_aux,
            attn_mode=hp.attn_mode,
            aux_dim=128,
            cat_embed_dim=32,
        )
    elif mt == "base_transformer":
        model = BaseFusionTransformerModel(
            d_in_audio=d_in_audio,
            d_model=hp.d_model,
            num_layers=hp.num_layers,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            max_len=hp.max_seq_len or 2048,
            cat_cards=cat_cards,
            ctx_dim=ctx_dim,
            rel_dim=rel_dim,
            use_aux=use_aux,
            aux_dim=128,
            cat_embed_dim=32,
        )
    elif mt == "crossattn":
        model = NaturalnessCrossAttnModel(
            d_in_audio=d_in_audio,
            d_model=hp.d_model,
            num_layers=hp.num_layers,
            n_heads=hp.n_heads,
            dropout=hp.dropout,
            max_len=hp.max_seq_len or 2048,
            cross_layers=hp.crossattn_layers,
            cross_dropout=hp.crossattn_dropout,
            cat_cards=cat_cards,
            ctx_dim=ctx_dim,
            rel_dim=rel_dim,
            use_aux=use_aux,
            aux_dim=128,
            cat_embed_dim=32,
        )
    elif mt == "mlp":
        model = NaturalnessFusionMLPModel(
            d_in_audio=d_in_audio,
            dropout=hp.dropout,
            cat_cards=cat_cards,
            ctx_dim=ctx_dim,
            rel_dim=rel_dim,
            use_aux=use_aux,
            aux_dim=128,
            cat_embed_dim=32,
            hidden=hp.mlp_hidden,
            layers=hp.mlp_layers,
        )
    elif mt == "logreg":
        model = NaturalnessFusionLogRegModel(
            d_in_audio=d_in_audio,
            dropout=hp.dropout,
            cat_cards=cat_cards,
            ctx_dim=ctx_dim,
            rel_dim=rel_dim,
            use_aux=use_aux,
            aux_dim=128,
            cat_embed_dim=32,
        )
    else:
        raise ValueError(f"Unknown --model-type={hp.model_type!r}")

    model = model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scaler = GradScaler(enabled=hp.use_amp)

    metrics_path = out_dir / "metrics.jsonl"
    best_model_path = out_dir / "best_model.pt"
    last_model_path = out_dir / "last_model.pt"
    ckpt_dir = out_dir / "checkpoints"
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_score = -1.0
    epochs_no_improve = 0

    if hp.resume_from:
        ck = Path(hp.resume_from)
        if not ck.exists():
            raise FileNotFoundError(f"--resume-from not found: {ck}")
        maploc = {"cuda:%d" % 0: "cuda:%d" % local_rank} if torch.cuda.is_available() else "cpu"
        state = torch.load(str(ck), map_location=maploc)

        sd = state["model_state_dict"]
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(sd)

        if state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if hp.use_amp and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

        start_epoch = int(state.get("epoch", 0)) + 1
        best_score = float(state.get("best_score", -1.0))
        epochs_no_improve = int(state.get("epochs_no_improve", 0))

        if is_main():
            print(f"Resumed from {ck} -> start_epoch={start_epoch}, best_score={best_score:.4f}")

    ddp_barrier()
    if is_main():
        print("Starting training...")

    has_val = val_loader is not None

    for epoch in range(start_epoch, hp.num_epochs + 1):
        if distributed:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sum_local = 0.0
        correct_local = 0.0
        total_local = 0.0

        it = tqdm(train_loader, desc=f"Epoch {epoch} [train] r{get_rank()}", leave=True, mininterval=1.0) if is_main() else train_loader

        for x, lengths, cat_x, ctx_x, rel_x, labels, _metas in it:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            cat_x = cat_x.to(device, non_blocking=True)
            ctx_x = ctx_x.to(device, non_blocking=True)
            rel_x = rel_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=hp.use_amp):
                logits = model(x, lengths, cat_x, ctx_x, rel_x)
                loss = criterion(logits, labels)

            if hp.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                corr, tot = accuracy_from_logits(logits.detach(), labels.detach(), threshold=hp.test_threshold)
                loss_sum_local += float(loss.item()) * labels.numel()
                correct_local += float(corr.item())
                total_local += float(tot.item())

            if is_main():
                it.set_postfix(loss=f"{loss.item():.4f}")

        t_loss = torch.tensor([loss_sum_local, total_local], device=device, dtype=torch.float32)
        t_corr = torch.tensor([correct_local, total_local], device=device, dtype=torch.float32)
        all_reduce_sum(t_loss)
        all_reduce_sum(t_corr)

        train_loss = (t_loss[0] / t_loss[1].clamp(min=1.0)).item()
        train_acc = (t_corr[0] / t_corr[1].clamp(min=1.0)).item()

        if has_val:
            val_loss, val_acc = ddp_eval(model, val_loader, criterion, device, hp.use_amp, threshold=hp.test_threshold)  # type: ignore[arg-type]
            score = val_acc
        else:
            val_loss, val_acc = float("nan"), float("nan")
            score = train_acc

        if is_main():
            if has_val:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | (no val)")

            with metrics_path.open("a") as mf:
                mf.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "score": score,
                        }
                    )
                    + "\n"
                )

            raw_model = model.module if isinstance(model, DDP) else model

            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if hp.use_amp else None,
                    "epoch": epoch,
                    "best_score": best_score,
                    "epochs_no_improve": epochs_no_improve,
                    "hyperparams": asdict(hp),
                },
                ckpt_path,
            )

            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "hyperparams": asdict(hp),
                },
                last_model_path,
            )

            improved = score > best_score
            if improved:
                best_score = score
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": raw_model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                        "hyperparams": asdict(hp),
                    },
                    best_model_path,
                )
                print(f"  -> New best saved: score={score:.4f}")
            else:
                epochs_no_improve += 1
                if has_val:
                    print(f"  -> No val improvement ({epochs_no_improve}/{hp.patience})")
                else:
                    print(f"  -> No improvement vs best(train_acc) ({epochs_no_improve}/{hp.patience})")

        stop_now = False
        if has_val and is_main() and epochs_no_improve >= hp.patience:
            stop_now = True
        stop_now = broadcast_bool(stop_now)
        if stop_now:
            if is_main():
                print("Early stopping triggered.")
            break

        ddp_barrier()

    if is_main():
        if has_val:
            print(f"Training done. Best val_acc={best_score:.4f}")
        else:
            print(f"Training done. Best train_acc={best_score:.4f}")

    ddp_barrier()

    ck_to_use = best_model_path if best_model_path.exists() else last_model_path
    maploc = {"cuda:%d" % 0: "cuda:%d" % local_rank} if torch.cuda.is_available() else "cpu"
    state = torch.load(str(ck_to_use), map_location=maploc)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(state["model_state_dict"])

    test_loss, test_acc = ddp_eval(model, test_loader, criterion, device, hp.use_amp, threshold=hp.test_threshold)
    if is_main():
        print(f"[TEST] ckpt={ck_to_use.name} loss={test_loss:.4f} acc={test_acc:.4f}")

    if hp.gather_test_preds:
        attn_dir = Path(hp.attn_out) if hp.attn_out else (out_dir / "attn")

        y_true_local, y_prob_local, metas_local, attn_payload_local = ddp_test_collect_with_meta_and_attn(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=hp.use_amp,
            dump_attn=hp.dump_attn and (mt in ("llama", "crossattn")),
            attn_max_batches=hp.attn_max_batches,
            attn_out_dir=attn_dir,
            attn_mode=hp.attn_mode,
        )

        if ddp_is_available():
            gathered_true: List[np.ndarray] = [None] * get_world_size()
            gathered_prob: List[np.ndarray] = [None] * get_world_size()
            gathered_meta: List[List[Dict[str, Any]]] = [None] * get_world_size()
            dist.all_gather_object(gathered_true, y_true_local)
            dist.all_gather_object(gathered_prob, y_prob_local)
            dist.all_gather_object(gathered_meta, metas_local)
            if is_main():
                y_true = np.concatenate([x for x in gathered_true if x is not None and x.size > 0], axis=0) if gathered_true else np.array([])
                y_prob = np.concatenate([x for x in gathered_prob if x is not None and x.size > 0], axis=0) if gathered_prob else np.array([])
                metas: List[Dict[str, Any]] = []
                for part in gathered_meta:
                    if part:
                        metas.extend(part)
            else:
                y_true, y_prob, metas = np.array([]), np.array([]), []
        else:
            y_true, y_prob, metas = y_true_local, y_prob_local, metas_local

        if is_main() and y_true.size > 0:
            y_pred = (y_prob >= hp.test_threshold).astype(np.int64)
            print("\n=== Classification report (test) ===")
            print(classification_report(y_true, y_pred, target_names=["unnatural (0)", "natural (1)"], digits=4))
            print("=== Confusion matrix ===")
            print(confusion_matrix(y_true, y_pred))
            try:
                brier = brier_score_loss(y_true, y_prob)
                print(f"Brier score: {brier:.6f}")
            except Exception as e:
                print(f"Brier score failed: {e}")

            if not getattr(hp, "_no_write_test_preds", False):
                _write_test_outputs(out_dir=out_dir, y_true=y_true, y_prob=y_prob, metas=metas, threshold=hp.test_threshold)

            if hp.dump_attn and (mt in ("llama", "crossattn")):
                attn_path = attn_dir / "attn_batches.pt"
                if attn_path.exists():
                    payload = torch.load(str(attn_path), map_location="cpu", weights_only=False)
                    attn_batches = payload.get("batches", [])
                    summary = _attn_summary_from_batches(attn_batches)
                    summary.update(
                        {
                            "model_type": mt,
                            "attn_mode": hp.attn_mode,
                            "attn_max_batches": hp.attn_max_batches,
                            "attn_path": str(attn_path),
                        }
                    )
                    with (attn_dir / "attn_summary.json").open("w") as f:
                        json.dump(summary, f, indent=2)
                    print(f"[ATTN] wrote: {attn_dir/'attn_summary.json'}")
                else:
                    print("[ATTN] requested but no attn_batches.pt found (model may not support return_attn).")

    else:
        if is_main():
            print("[TEST] gather_test_preds disabled -> not writing per-datapoint files.")

    ddp_barrier()
    if ddp_is_available():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
