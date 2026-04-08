#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict, defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

TARGET_SR = 16000
DYAD_RE = re.compile(r"V\d+_(S\d+)_I(\d+)_P[\w\d]+", re.I)

try:
    import librosa
except Exception:
    librosa = None

sys.path.append("./vox-profile-release")
from src.model.emotion.whisper_emotion_dim import WhisperWrapper

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def file_uid(wav_path: Path) -> str:
    """
    Unique, stable ID per input WAV file.
    Uses stem + short hash of full resolved path to avoid collisions.
    Path must already be resolved (absolute) before calling this — see read_list_file.
    """
    try:
        resolved = wav_path.resolve()
    except Exception:
        resolved = wav_path
    h = _short_hash(str(resolved))
    return f"{wav_path.stem.lower()}_{h}"


def dyad_id_for_path(wav_path: Path) -> str:
    """
    Dyad grouping:
      - Seamless-style filenames: group by S####_I######## (same as before)
      - Everything else: unique per file (stem + hash) to avoid collisions
    """
    stem = wav_path.stem
    m = DYAD_RE.match(stem)
    if m:
        return f"{m.group(1)}_I{m.group(2)}"
    return file_uid(wav_path)


def stem_for_segments(wav_path: Path) -> str:
    """
    Segment naming base:
      - Seamless-style: keep readable original stem (lowercased)
      - Otherwise: use unique per-file uid (stem + hash)
    """
    if DYAD_RE.match(wav_path.stem):
        return wav_path.stem.lower()
    return file_uid(wav_path)


def read_list_file(path: Path) -> List[Path]:
    """
    Read wav list, resolving all paths to absolute form immediately.
    This ensures hashing in file_uid() is always based on a canonical
    absolute path, making dyad IDs stable regardless of the CWD at runtime.
    """
    out: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(Path(s).resolve())
    return out


def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(str(path), "r") as f:
        sr = f.samplerate
        y = f.read(dtype="float32", always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    return y, sr


def to_16k(y: np.ndarray, sr_in: int, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    if sr_in == target_sr:
        return y.astype(np.float32, copy=False), sr_in
    if librosa is None:
        raise RuntimeError(f"Need librosa to resample {sr_in}→{target_sr}. pip install librosa")
    if len(y) == 0:
        raise RuntimeError(f"Input signal length=0 is too small to resample from {sr_in}->{target_sr}")
    y = librosa.resample(y, orig_sr=sr_in, target_sr=target_sr, res_type="kaiser_best").astype(np.float32)
    return y, target_sr


def chunk_sliding_pad(
    y: np.ndarray,
    sr: int,
    win_sec: float,
    hop_sec: float,
    pad_last: bool,
) -> List[Tuple[int, int, float, float, bool]]:
    """
    Sliding windows.
    Returns spans: (i0, i1_true, t0, t1_true, padded_flag)
    """
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("win_sec and hop_sec must be > 0")

    out = []
    N = len(y)
    i0 = 0
    while i0 < N:
        i1 = i0 + win
        if i1 <= N:
            out.append((i0, i1, i0 / sr, i1 / sr, False))
        else:
            if not pad_last:
                break
            out.append((i0, N, i0 / sr, N / sr, True))
        i0 += hop
    return out


def pad_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) >= target_len:
        return x
    return np.pad(x, (0, target_len - len(x)), mode="constant")


def pool_sequence(x_btd: torch.Tensor, how: str) -> torch.Tensor:
    if how == "mean":
        return x_btd.mean(dim=1)
    if how == "max":
        return x_btd.max(dim=1).values
    raise ValueError(f"Unknown pool '{how}'")


def npy_shape_header_only(path: Path) -> Tuple[int, ...]:
    """
    Read .npy header to get shape WITHOUT mmap (prevents leaking file descriptors).
    """
    import numpy.lib.format as fmt

    with path.open("rb") as f:
        major, minor = fmt.read_magic(f)
        if (major, minor) == (1, 0):
            shape, _fortran, _dtype = fmt.read_array_header_1_0(f)
        elif (major, minor) == (2, 0):
            shape, _fortran, _dtype = fmt.read_array_header_2_0(f)
        elif (major, minor) == (3, 0):
            shape, _fortran, _dtype = fmt.read_array_header_3_0(f)
        else:
            raise ValueError(f"Unsupported npy version {(major, minor)} for {path}")
    return tuple(shape)


def embedding_paths(
    emb_vec_root: Path,
    emb_seq_root: Path,
    dyad_id: str,
    seg_stem: str,
    save_full_seq: bool,
) -> Tuple[Path, Optional[Path]]:
    vec_path = emb_vec_root / dyad_id / f"{seg_stem}.npy"
    seq_path = (emb_seq_root / dyad_id / f"{seg_stem}.npy") if save_full_seq else None
    return vec_path, seq_path


def embeddings_exist(vec_path: Path, seq_path: Optional[Path]) -> bool:
    if not vec_path.exists():
        return False
    if seq_path is not None and not seq_path.exists():
        return False
    return True


def _expected_num_chunks_from_header(wav_path: Path, win_sec: float, hop_sec: float, pad_last: bool) -> int:
    """
    Estimate chunk count without loading/resampling audio.
    Uses ceil(frames * TARGET_SR / sr_in) to avoid underestimating (prevents false 'complete').
    Mirrors chunk_sliding_pad's while i0 < N logic (pad_last True) and i0+win<=N logic (pad_last False).
    """
    try:
        info = sf.info(str(wav_path))
        frames = int(info.frames)
        sr_in = int(info.samplerate)
    except Exception:
        return 0

    if frames <= 0 or sr_in <= 0:
        return 0

    N = (frames * TARGET_SR + sr_in - 1) // sr_in

    hop = int(round(hop_sec * TARGET_SR))
    win = int(round(win_sec * TARGET_SR))
    if hop <= 0 or win <= 0 or N <= 0:
        return 0

    if pad_last:
        return (N + hop - 1) // hop
    else:
        if N < win:
            return 0
        return 1 + (N - win) // hop


def _index_chunk_files_by_base(emb_dir: Path) -> Dict[str, set]:
    """
    Parse <seg_base>__ch#####.npy inside a dyad embedding dir into:
      { seg_base -> set(chunk_indices_int) }
    """
    out: Dict[str, set] = defaultdict(set)
    if not emb_dir.exists() or not emb_dir.is_dir():
        return out

    try:
        with os.scandir(emb_dir) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                name = ent.name
                if not name.endswith(".npy"):
                    continue
                stem = name[:-4]
                if "__ch" not in stem:
                    continue
                base, idx_s = stem.rsplit("__ch", 1)
                if not idx_s.isdigit():
                    continue
                out[base].add(int(idx_s))
    except Exception:
        return {}

    return out


def _wav_complete_by_embeddings(
    wav_path: Path,
    out_dir: Path,
    win_sec: float,
    hop_sec: float,
    pad_last: bool,
    save_full_seq: bool,
    vec_idx_by_base: Dict[str, set],
    seq_idx_by_base: Optional[Dict[str, set]],
) -> bool:
    """
    True iff all required chunk indices for this wav exist in embeds_vec (and embeds_seq if enabled),
    for the *current* (win_sec, hop_sec, pad_last).
    """
    seg_base = stem_for_segments(wav_path)
    expected = _expected_num_chunks_from_header(wav_path, win_sec, hop_sec, pad_last)
    if expected <= 0:
        return False

    vec_set = vec_idx_by_base.get(seg_base, set())
    if len(vec_set) < expected:
        return False
    for i in range(expected):
        if i not in vec_set:
            return False

    if save_full_seq:
        if seq_idx_by_base is None:
            return False
        seq_set = seq_idx_by_base.get(seg_base, set())
        if len(seq_set) < expected:
            return False
        for i in range(expected):
            if i not in seq_set:
                return False

    return True


def compute_complete_dyads_from_embeddings(
    wavs_all: List[Path],
    out_dir: Path,
    win_sec: float,
    hop_sec: float,
    pad_last: bool,
    save_full_seq: bool,
    already_skipped: Optional[set] = None,
) -> set:
    already_skipped = already_skipped or set()

    emb_vec_root = out_dir / "embeds_vec"
    emb_seq_root = out_dir / "embeds_seq"

    dyad2wavs: Dict[str, List[Path]] = defaultdict(list)
    for p in wavs_all:
        dyad2wavs[dyad_id_for_path(p)].append(p)

    complete: set = set()

    for dyad_id, plist in dyad2wavs.items():
        if dyad_id in already_skipped:
            complete.add(dyad_id)
            continue

        dyad_vec_dir = emb_vec_root / dyad_id
        if not dyad_vec_dir.exists():
            continue

        vec_idx_by_base = _index_chunk_files_by_base(dyad_vec_dir)

        seq_idx_by_base = None
        if save_full_seq:
            dyad_seq_dir = emb_seq_root / dyad_id
            if not dyad_seq_dir.exists():
                continue
            seq_idx_by_base = _index_chunk_files_by_base(dyad_seq_dir)

        ok = True
        for wav_path in plist:
            if not _wav_complete_by_embeddings(
                wav_path=wav_path,
                out_dir=out_dir,
                win_sec=win_sec,
                hop_sec=hop_sec,
                pad_last=pad_last,
                save_full_seq=save_full_seq,
                vec_idx_by_base=vec_idx_by_base,
                seq_idx_by_base=seq_idx_by_base,
            ):
                ok = False
                break

        if ok:
            complete.add(dyad_id)

    return complete


class VADCache:

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.cache: Dict[str, Dict[str, Tuple[str, str, str]]] = {}

    def get(self, dyad_id: str, seg_stem: str) -> Optional[Tuple[str, str, str]]:
        if dyad_id not in self.cache:
            self.cache[dyad_id] = self._load_dyad(dyad_id)
        return self.cache[dyad_id].get(seg_stem)

    def _row_seg_stem(self, row: Dict[str, str]) -> Optional[str]:
        if "seg_stem" in row and row["seg_stem"]:
            return row["seg_stem"].strip().lower()
        if "pair_stem" in row and "chunk_index" in row and row["pair_stem"] and row["chunk_index"]:
            try:
                ci = int(float(row["chunk_index"]))
                return f"{row['pair_stem'].strip().lower()}__ch{ci:05d}"
            except Exception:
                return None
        return None

    def _load_from_csv(self, path: Path, dyad_filter: Optional[str]) -> Dict[str, Tuple[str, str, str]]:
        out: Dict[str, Tuple[str, str, str]] = {}
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                if r.fieldnames is None:
                    return out
                for row in r:
                    if dyad_filter is not None and row.get("dyad_id", "").strip() != dyad_filter:
                        continue
                    ss = self._row_seg_stem(row)
                    if not ss:
                        continue
                    v = row.get("valence", "")
                    a = row.get("arousal", "")
                    d = row.get("dominance", "")
                    if v != "" and a != "" and d != "":
                        out[ss] = (v, a, d)
        except Exception:
            pass
        return out

    def _load_dyad(self, dyad_id: str) -> Dict[str, Tuple[str, str, str]]:
        out: Dict[str, Tuple[str, str, str]] = {}

        merged = self.out_dir / f"{dyad_id}.csv"
        if merged.exists():
            out.update(self._load_from_csv(merged, dyad_filter=None))
            if out:
                return out

        shards = self.out_dir / "shards"
        if shards.exists():
            for p in sorted(shards.glob("rank*.csv")):
                out.update(self._load_from_csv(p, dyad_filter=dyad_id))

        return out


def _ensure_whisper_expected_mel_len(model: nn.Module, input_features: torch.Tensor) -> torch.Tensor:
    max_src = getattr(getattr(model, "backbone_model", model).config, "max_source_positions", None)
    if max_src is None:
        return input_features
    expected = int(max_src) * 2
    cur = int(input_features.shape[-1])
    if cur == expected:
        return input_features
    if cur < expected:
        return F.pad(input_features, (0, expected - cur))
    return input_features[..., :expected]


def forward_fast(
    model: nn.Module,
    device: torch.device,
    batch_audio_np: List[np.ndarray],
    pool: str,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      valence[B], arousal[B], dominance[B], pooled_vecs[B,Denc], last_btd_cpu[B,Tenc,Denc]
    """
    lengths = torch.tensor([len(a) for a in batch_audio_np], device=device)

    feats = model.feature_extractor(
        [a.astype(np.float32, copy=False) for a in batch_audio_np],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )
    input_features = feats.input_features.to(device)
    input_features = _ensure_whisper_expected_mel_len(model, input_features)

    if hasattr(model, "_embed_positions_750") and not getattr(model, "_did_set_embedpos", False):
        model.backbone_model.encoder.embed_positions = nn.Embedding.from_pretrained(
            model._embed_positions_750.to(device), freeze=True
        )
        model._did_set_embedpos = True

    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                enc = model.backbone_model.encoder(input_features, output_hidden_states=True)
                last_btd = enc.hidden_states[-1]
        else:
            enc = model.backbone_model.encoder(input_features, output_hidden_states=True)
            last_btd = enc.hidden_states[-1]

        x = last_btd.transpose(1, 2)
        x = model.model_seq(x)
        x = x.transpose(1, 2)

        feat_lengths = model._get_feat_extract_output_lengths(lengths.detach().cpu()).to(device)

        pooled_h = []
        for i in range(x.size(0)):
            L = int(feat_lengths[i].item())
            pooled_h.append(x[i, : max(L, 1)].mean(dim=0))
        pooled_h = torch.stack(pooled_h, dim=0)

        arousal = model.arousal_layer(pooled_h).detach().float().cpu().numpy().reshape(-1)
        valence = model.valence_layer(pooled_h).detach().float().cpu().numpy().reshape(-1)
        dominance = model.dominance_layer(pooled_h).detach().float().cpu().numpy().reshape(-1)

        pooled_vec = pool_sequence(last_btd.detach().float(), pool).cpu().numpy().astype(np.float32)
        last_btd_cpu = last_btd.detach().float().cpu().numpy().astype(np.float32)

    return valence, arousal, dominance, pooled_vec, last_btd_cpu


CSV_FIELDS = [
    "dyad_id",
    "pair_stem",
    "speaker_id",
    "seg_stem",
    "chunk_index",
    "start",
    "end",
    "true_end",
    "duration",
    "true_duration",
    "padded",
    "valence",
    "arousal",
    "dominance",
    "vad_source",
    "emb_vec_path",
    "emb_vec_shape",
    "emb_seq_path",
    "emb_seq_shape",
    "segment_path",
    "src_wav",
    "win_sec",
    "hop_sec",
    "pool",
]


@dataclass
class WorkerArgs:
    wav_list: Path
    out_dir: Path
    win_sec: float
    hop_sec: float
    pad_last: bool
    save_segments: bool
    seg_format: str
    batch_size: int
    pool: str
    save_full_seq: bool
    amp: bool
    num_workers_io: int
    skip_existing_dyads: bool
    flush_every: int


def worker_main(rank: int, world_size: int, args: WorkerArgs):
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion-dim").to(device)
    model.eval()

    out_dir = args.out_dir
    seg_root = out_dir / "segments"
    emb_vec_root = out_dir / "embeds_vec"
    emb_seq_root = out_dir / "embeds_seq"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    emb_vec_root.mkdir(parents=True, exist_ok=True)
    if args.save_full_seq:
        emb_seq_root.mkdir(parents=True, exist_ok=True)
    if args.save_segments:
        seg_root.mkdir(parents=True, exist_ok=True)

    shard_csv = shards_dir / f"rank{rank}.csv"
    shard_exists = shard_csv.exists() and shard_csv.stat().st_size > 0
    shard_f = shard_csv.open("a", newline="", encoding="utf-8")
    shard_w = csv.DictWriter(shard_f, fieldnames=CSV_FIELDS)
    if not shard_exists:
        shard_w.writeheader()

    vad_cache = VADCache(out_dir)

    wavs_all = read_list_file(args.wav_list)
    wavs = [p for i, p in enumerate(wavs_all) if (i % max(world_size, 1)) == rank]

    existing_dyads: set = set()
    if args.skip_existing_dyads:
        existing_dyads |= {
            p.stem for p in out_dir.iterdir()
            if p.is_file() and p.suffix == ".csv"
        }

        existing_dyads |= compute_complete_dyads_from_embeddings(
            wavs_all=wavs_all,
            out_dir=out_dir,
            win_sec=args.win_sec,
            hop_sec=args.hop_sec,
            pad_last=args.pad_last,
            save_full_seq=args.save_full_seq,
            already_skipped=existing_dyads,
        )

    win_n = int(round(args.win_sec * TARGET_SR))

    pending_audio: List[np.ndarray] = []
    pending_meta: List[Dict[str, Any]] = []

    rows_written = 0

    def write_row(
        meta: Dict[str, Any],
        v: str,
        a: str,
        d: str,
        vad_source: str,
        vec_path: str,
        vec_shape: str,
        seq_path: str,
        seq_shape: str,
    ):
        nonlocal rows_written
        row = {
            "dyad_id": meta["dyad_id"],
            "pair_stem": meta["pair_stem"],
            "speaker_id": meta["speaker_id"],
            "seg_stem": meta["seg_stem"],
            "chunk_index": meta["chunk_index"],
            "start": f"{meta['start']:.3f}",
            "end": f"{meta['end']:.3f}",
            "true_end": f"{meta['true_end']:.3f}",
            "duration": f"{(meta['end'] - meta['start']):.3f}",
            "true_duration": f"{(meta['true_end'] - meta['start']):.3f}",
            "padded": "1" if meta["padded"] else "0",
            "valence": v,
            "arousal": a,
            "dominance": d,
            "vad_source": vad_source,
            "emb_vec_path": vec_path,
            "emb_vec_shape": vec_shape,
            "emb_seq_path": seq_path,
            "emb_seq_shape": seq_shape,
            "segment_path": meta["segment_path"],
            "src_wav": meta["src_wav"],
            "win_sec": str(args.win_sec),
            "hop_sec": str(args.hop_sec),
            "pool": args.pool,
        }
        shard_w.writerow(row)
        rows_written += 1
        if args.flush_every > 0 and (rows_written % args.flush_every) == 0:
            try:
                shard_f.flush()
            except Exception:
                pass

    def _save_and_write(
        meta: Dict[str, Any],
        v: float,
        a: float,
        d: float,
        pooled_vec_i: np.ndarray,
        last_btd_i: np.ndarray,
    ):
        dyad_id = meta["dyad_id"]
        seg_stem = meta["seg_stem"]

        vec_path_p, seq_path_p = embedding_paths(
            emb_vec_root, emb_seq_root, dyad_id, seg_stem, args.save_full_seq
        )
        vec_path_p.parent.mkdir(parents=True, exist_ok=True)
        if seq_path_p is not None:
            seq_path_p.parent.mkdir(parents=True, exist_ok=True)

        if not vec_path_p.exists():
            np.save(vec_path_p, pooled_vec_i)
        if seq_path_p is not None and not seq_path_p.exists():
            np.save(seq_path_p, last_btd_i)

        vec_shape = str(tuple(pooled_vec_i.shape))
        seq_shape = str(tuple(last_btd_i.shape)) if seq_path_p is not None else ""

        write_row(
            meta=meta,
            v=f"{v:.6f}",
            a=f"{a:.6f}",
            d=f"{d:.6f}",
            vad_source="computed",
            vec_path=str(vec_path_p),
            vec_shape=vec_shape,
            seq_path=str(seq_path_p) if seq_path_p is not None else "",
            seq_shape=seq_shape,
        )

    def flush_compute():
        if not pending_audio:
            return
        try:
            val, aro, dom, pooled_vec, last_btd = forward_fast(
                model=model,
                device=device,
                batch_audio_np=pending_audio,
                pool=args.pool,
                amp=args.amp,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda out of memory" in msg) and device.type == "cuda" and len(pending_audio) > 1:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                audios = pending_audio[:]
                metas = pending_meta[:]
                pending_audio.clear()
                pending_meta.clear()
                for one_audio, one_meta in zip(audios, metas):
                    v1, a1, d1, vec1, seq1 = forward_fast(
                        model=model,
                        device=device,
                        batch_audio_np=[one_audio],
                        pool=args.pool,
                        amp=args.amp,
                    )
                    _save_and_write(one_meta, float(v1[0]), float(a1[0]), float(d1[0]), vec1[0], seq1[0])
                return
            raise

        for i, meta in enumerate(pending_meta):
            _save_and_write(
                meta=meta,
                v=float(val[i]),
                a=float(aro[i]),
                d=float(dom[i]),
                pooled_vec_i=pooled_vec[i],
                last_btd_i=last_btd[i],
            )

        pending_audio.clear()
        pending_meta.clear()

    pbar = tqdm(wavs, desc=f"Rank {rank}/{world_size} WAVs", unit="wav")
    for wav_path in pbar:
        dyad_id = dyad_id_for_path(wav_path)
        seg_base = stem_for_segments(wav_path)

        if args.skip_existing_dyads and (dyad_id in existing_dyads):
            continue

        try:
            y, sr = load_mono(wav_path)
            y, sr = to_16k(y, sr, TARGET_SR)
        except Exception as e:
            print(f"[WARN][rank{rank}] {wav_path}: {e}", file=sys.stderr)
            continue

        spans = chunk_sliding_pad(y, sr, args.win_sec, args.hop_sec, pad_last=args.pad_last)

        seg_dir = (seg_root / dyad_id) if args.save_segments else None
        if seg_dir is not None:
            seg_dir.mkdir(parents=True, exist_ok=True)

        for ci, (i0, i1_true, t0, t1_true, padded) in enumerate(spans):
            ch = y[i0:i1_true]
            if padded:
                ch = pad_to_len(ch, win_n)
                t1 = t0 + args.win_sec
            else:
                t1 = t1_true

            seg_stem = f"{seg_base}__ch{ci:05d}"

            vec_path_p, seq_path_p = embedding_paths(
                emb_vec_root, emb_seq_root, dyad_id, seg_stem, args.save_full_seq
            )

            segment_path = ""
            if args.save_segments and seg_dir is not None:
                seg_file = seg_dir / f"{seg_stem}.{args.seg_format}"
                segment_path = str(seg_file)
                if not seg_file.exists():
                    try:
                        sf.write(segment_path, ch, sr)
                    except Exception as e:
                        print(f"[WARN][rank{rank}] failed to save {seg_file}: {e}", file=sys.stderr)

            meta = {
                "dyad_id": dyad_id,
                "pair_stem": seg_base,
                "speaker_id": wav_path.stem,
                "seg_stem": seg_stem,
                "chunk_index": ci,
                "start": float(t0),
                "end": float(t1),
                "true_end": float(t1_true),
                "padded": bool(padded),
                "segment_path": segment_path,
                "src_wav": str(wav_path),
            }

            if embeddings_exist(vec_path_p, seq_path_p):
                if len(pending_audio) >= args.batch_size:
                    flush_compute()

                try:
                    vshape = str(npy_shape_header_only(vec_path_p))
                except Exception:
                    vshape = ""
                sshape = ""
                if seq_path_p is not None:
                    try:
                        sshape = str(npy_shape_header_only(seq_path_p))
                    except Exception:
                        sshape = ""

                vad = vad_cache.get(dyad_id, seg_stem)
                if vad is not None:
                    v_s, a_s, d_s = vad
                    vad_source = "cache"
                else:
                    v_s = a_s = d_s = ""
                    vad_source = "missing"

                write_row(
                    meta=meta,
                    v=v_s,
                    a=a_s,
                    d=d_s,
                    vad_source=vad_source,
                    vec_path=str(vec_path_p),
                    vec_shape=vshape,
                    seq_path=str(seq_path_p) if seq_path_p is not None else "",
                    seq_shape=sshape,
                )
                continue

            pending_audio.append(ch.astype(np.float32, copy=False))
            pending_meta.append(meta)
            if len(pending_audio) >= args.batch_size:
                flush_compute()

    flush_compute()
    try:
        shard_f.flush()
        shard_f.close()
    except Exception:
        pass

class LRUCSVWriters:
    def __init__(self, max_open: int, fieldnames: List[str]):
        self.max_open = max_open
        self.fieldnames = fieldnames
        self.cache: "OrderedDict[str, Tuple[Any, csv.DictWriter]]" = OrderedDict()

    def get(self, path: Path) -> csv.DictWriter:
        key = str(path)
        if key in self.cache:
            f, w = self.cache.pop(key)
            self.cache[key] = (f, w)
            return w

        while len(self.cache) >= self.max_open:
            _old_key, (old_f, _old_w) = self.cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        f = path.open("a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=self.fieldnames)
        if not exists:
            w.writeheader()
        self.cache[key] = (f, w)
        return w

    def close_all(self):
        for f, _w in self.cache.values():
            try:
                f.close()
            except Exception:
                pass
        self.cache.clear()


def merge_shards(out_dir: Path, max_open: int = 64):
    shards_dir = out_dir / "shards"
    shard_files = sorted(shards_dir.glob("rank*.csv"))
    if not shard_files:
        return

    writers = LRUCSVWriters(max_open=max_open, fieldnames=CSV_FIELDS)

    skipped = 0
    for sfp in tqdm(shard_files, desc="Merging shards", unit="file"):
        with sfp.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if None in row:
                    skipped += 1
                    continue

                dyad_id = (row.get("dyad_id") or "").strip()
                if not dyad_id:
                    continue
                out_csv = out_dir / f"{dyad_id}.csv"
                w = writers.get(out_csv)
                w.writerow(row)

    writers.close_all()
    if skipped:
        print(f"[WARN] Skipped {skipped} malformed shard rows.")

def main():
    ap = argparse.ArgumentParser(
        description="Sliding-window segmentation + VAD + Whisper encoder final-layer embeddings (multi-GPU, resume-friendly)."
    )
    ap.add_argument("--wav-list", type=Path, required=True, help="Text file: one wav path per line.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dir (segments/, embeds_vec/, embeds_seq/, per-dyad CSVs).")

    ap.add_argument("--win-sec", type=float, default=3.0, help="Window size seconds.")
    ap.add_argument("--hop-sec", type=float, default=1.0, help="Hop size seconds (hop < win => sliding window).")
    ap.add_argument("--pad-last", action="store_true", help="Pad last short window to full length (default ON).")
    ap.add_argument("--no-pad-last", dest="pad_last", action="store_false", help="Drop trailing short window.")
    ap.set_defaults(pad_last=True)

    ap.add_argument("--save-segments", action="store_true", help="Save 16k mono segments to out-dir/segments/<dyad>/")
    ap.add_argument("--seg-format", type=str, default="wav", choices=["wav", "flac"])

    ap.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    ap.add_argument("--pool", choices=["mean", "max"], default="mean", help="Pool encoder [T,D] -> [D].")
    ap.add_argument("--save-full-seq", action="store_true", help="Also save per-window full encoder [T,D].")

    ap.add_argument("--amp", action="store_true", help="Use autocast fp16 on CUDA for faster encoder.")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)

    ap.add_argument("--num-gpus", type=int, default=0, help="0 = all visible CUDA devices.")
    ap.add_argument(
        "--skip-existing-dyads",
        action="store_true",
        help=(
            "Skip dyads that are already complete. This checks (A) out-dir/<dyad>.csv if present, "
            "AND (B) if all expected chunk embeddings exist for ALL wavs in that dyad, even if no CSV exists."
        ),
    )

    ap.add_argument("--merge-max-open", type=int, default=64, help="Max output CSV files open during merge (prevents EMFILE).")
    ap.add_argument("--flush-every", type=int, default=2000, help="Flush shard CSV every N written rows. 0 disables.")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    world_size = 1
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        world_size = n if args.num_gpus == 0 else max(1, min(args.num_gpus, n))

    wargs = WorkerArgs(
        wav_list=args.wav_list,
        out_dir=args.out_dir,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        pad_last=args.pad_last,
        save_segments=args.save_segments,
        seg_format=args.seg_format,
        batch_size=args.batch_size,
        pool=args.pool,
        save_full_seq=args.save_full_seq,
        amp=args.amp,
        num_workers_io=0,
        skip_existing_dyads=args.skip_existing_dyads,
        flush_every=args.flush_every,
    )

    if world_size <= 1:
        worker_main(rank=0, world_size=1, args=wargs)
    else:
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker_main, args=(world_size, wargs), nprocs=world_size, join=True)

    merge_shards(args.out_dir, max_open=args.merge_max_open)
    print(f"[OK] Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
