"""
tts_shared_v2.py
Shared audio post-processing utilities for all TTS conversion scripts. (v2)

v2 changes vs v1:
- Gap-expansion packer: Speaker A is NEVER trimmed, stretched, faded, or
  otherwise modified. Excess B duration expands the gap before it; total
  conversation length grows if needed rather than cutting A's speech.
- fade_in_out() applied to B segments only, prevents click/pop at
  piecing boundaries without ever touching A's waveform.
- add_background_noise() — mild shared pink/white noise floor added to
  both A and B post-assembly so they share one acoustic environment.

Handles:
- Resampling / mono conversion
- Silence trimming (B only)
- Pause squeezing (B only)
- Per-segment fade-in/out (B only, click prevention)
- RMS normalization (match Speaker A loudness)
- Spectral noise reduction (noisereduce, B only)
- Background noise floor addition (both A and B, shared seed)
- Timeline packing with gap-expansion (A never modified)
- Track assembly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import pandas as pd


# ---------------------------------------------------------------------------
# Basic audio helpers
# ---------------------------------------------------------------------------

def resample_linear(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(y, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    n_out = max(1, int(round(len(y) * sr_out / sr_in)))
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


def mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y.mean(axis=1).astype(np.float32) if y.ndim == 2 else y


def load_wav(path: str, sr_out: int) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    return resample_linear(mono(y), sr, sr_out)


def frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size < frame:
        return np.array([float(np.sqrt(np.mean(y * y) + 1e-12))], dtype=np.float32)
    n = 1 + (len(y) - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        out[i] = float(np.sqrt(np.mean(y[s:s + frame] ** 2) + 1e-12))
    return out


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------

def trim_silence(
    y: np.ndarray,
    sr: int,
    thresh_db: float = -38.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    pad_ms: float = 10.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    frame = max(16, int(sr * frame_ms / 1000))
    hop = max(8, int(sr * hop_ms / 1000))
    rms = frame_rms(y, frame, hop)
    thresh = float(np.max(rms) + 1e-12) * 10 ** (thresh_db / 20)
    idx = np.where(rms > thresh)[0]
    if idx.size == 0:
        return y
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) * hop - pad)
    end = min(len(y), int(idx[-1]) * hop + frame + pad)
    return y[start:end]


def squeeze_pauses(
    y: np.ndarray,
    sr: int,
    target_len: int,
    thresh_db: float = -38.0,
    max_pause_keep_ms: float = 70.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or len(y) <= target_len:
        return y
    frame = max(16, int(sr * 0.020))
    hop = max(8, int(sr * 0.010))
    rms = frame_rms(y, frame, hop)
    thresh = float(np.max(rms) + 1e-12) * 10 ** (thresh_db / 20)
    speech = rms > thresh
    max_keep = int(sr * max_pause_keep_ms / 1000)
    parts: List[np.ndarray] = []
    i = 0
    while i < len(speech):
        j, val = i, bool(speech[i])
        while j < len(speech) and bool(speech[j]) == val:
            j += 1
        chunk = y[i * hop: min(len(y), j * hop + frame)]
        parts.append(chunk if val else chunk[:max_keep])
        i = j
    return np.concatenate(parts) if parts else y


# ---------------------------------------------------------------------------
# Enforce exact length with fade-out (prevents hard clip)
# ---------------------------------------------------------------------------

def enforce_len(y: np.ndarray, target: int, sr: int, fade_ms: float = 40.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if target <= 0:
        return y
    if len(y) < target:
        return np.pad(y, (0, target - len(y))).astype(np.float32)
    if len(y) == target:
        return y
    fade = int(sr * fade_ms / 1000)
    out = y[:target].copy()
    if fade > 0 and target > fade:
        w = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
        out[target - fade:] *= w
    return out


def fade_in_out(
    y: np.ndarray,
    sr: int,
    fade_ms: float = 15.0,
    fade_in_ms: Optional[float] = None,
    fade_out_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a fade-in and fade-out to a segment before it's placed on the
    timeline, using a smooth cosine (equal-power-ish) curve rather than
    linear — linear ramps still leave an audible "ramping" quality on
    speech onsets, cosine sounds closer to a natural attack.

    fade_in_ms / fade_out_ms override fade_ms independently if given.
    TTS onsets are often steeper than natural speech, so fade-in is
    typically given more time than fade-out (e.g. 50ms in / 25ms out)
    to avoid sounding like a turn was "cut into."

    Safe to call on very short clips — fade length is clamped to half
    the clip so fade-in and fade-out never overlap and invert.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return y

    fi_ms = fade_in_ms if fade_in_ms is not None else fade_ms
    fo_ms = fade_out_ms if fade_out_ms is not None else fade_ms

    fade_in  = min(int(sr * fi_ms / 1000), n // 2)
    fade_out = min(int(sr * fo_ms / 1000), n // 2)

    out = y.copy()
    if fade_in > 0:
        # Cosine ramp 0 -> 1 (smoother attack than linear)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in, dtype=np.float32)))
        out[:fade_in] *= ramp
    if fade_out > 0:
        ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out, dtype=np.float32)))
        out[-fade_out:] *= ramp
    return out


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 = speed up (shorter), rate < 1 = slow down (longer)."""
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def fit_to_len(
    y: np.ndarray,
    target: int,
    sr: int,
    silence_db: float = -38.0,
    max_pause_ms: float = 70.0,
    fade_ms: float = 40.0,
    allow_stretch: bool = True,
) -> np.ndarray:
    """Full fitting pipeline: trim -> squeeze pauses -> stretch -> enforce."""
    y = trim_silence(y, sr, thresh_db=silence_db)
    if len(y) > target:
        y = squeeze_pauses(y, sr, target, thresh_db=silence_db, max_pause_keep_ms=max_pause_ms)
    if len(y) > target and allow_stretch:
        try:
            y = time_stretch(y, rate=len(y) / target)
        except Exception:
            pass
    return enforce_len(y, target, sr, fade_ms=fade_ms)


# ---------------------------------------------------------------------------
# RMS normalization: match Speaker A loudness
# ---------------------------------------------------------------------------

def rms_of(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 1e-8
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def rms_normalize(y: np.ndarray, target_rms: float, headroom: float = 0.99) -> np.ndarray:
    """Scale y so its RMS matches target_rms, with peak headroom clip."""
    y = np.asarray(y, dtype=np.float32)
    src = rms_of(y)
    if src < 1e-8:
        return y
    scaled = y * (target_rms / src)
    peak = float(np.max(np.abs(scaled)) + 1e-12)
    if peak > headroom:
        scaled *= headroom / peak
    return scaled.astype(np.float32)


def speaker_a_rms(y_a_full: np.ndarray) -> float:
    """Compute RMS over speech-active regions of Speaker A."""
    return rms_of(y_a_full[np.abs(y_a_full) > 0.001])


# ---------------------------------------------------------------------------
# Spectral noise reduction
# ---------------------------------------------------------------------------

def denoise(y: np.ndarray, sr: int, prop_decrease: float = 0.75) -> np.ndarray:
    """
    Spectral noise gate via noisereduce.
    prop_decrease: 0=no reduction, 1=full reduction. 0.75 is gentle.
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=y.astype(np.float32),
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=False,
        ).astype(np.float32)
    except ImportError:
        print("[WARN] noisereduce not installed; skipping denoising. pip install noisereduce")
        return y


def add_background_noise(
    y: np.ndarray,
    target_snr_db: float = 30.0,
    color: str = "pink",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add mild, consistent background noise to an audio track at a target
    signal-to-noise ratio. This masks small TTS/codec artifacts and gives
    A and B a shared sonic floor so they don't sound like they were
    recorded in two different acoustic spaces.

    target_snr_db: higher = quieter noise (30dB is subtle, 20dB is noticeable).
    color: "white" (flat spectrum) or "pink" (1/f, more natural-sounding room hiss).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    rng = np.random.default_rng(seed)

    n = len(y)
    if color == "pink":
        # Generate pink noise via FFT filtering (1/sqrt(f) amplitude spectrum)
        white = rng.standard_normal(n).astype(np.float32)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1] if n > 1 else 1.0   # avoid div-by-zero at DC
        spectrum = np.fft.rfft(white)
        spectrum = spectrum / np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=n).astype(np.float32)
    else:
        noise = rng.standard_normal(n).astype(np.float32)

    # Normalize noise to unit RMS, then scale to hit target SNR vs signal RMS
    noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
    noise = noise / noise_rms

    sig_rms = rms_of(y)
    if sig_rms < 1e-6:
        sig_rms = 0.02   # fallback floor so silent segments still get a touch of noise

    target_noise_rms = sig_rms / (10 ** (target_snr_db / 20.0))
    noise = noise * target_noise_rms

    out = y + noise
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# F0 / WPM estimation for caption building
# ---------------------------------------------------------------------------

def estimate_f0_mean(y: np.ndarray, sr: int) -> float:
    frame = int(sr * 0.025)
    hop = int(sr * 0.010)
    win = np.hanning(frame).astype(np.float32)
    lo, hi = max(1, int(sr / 400)), int(sr / 80)
    f0s = []
    for i in range(0, len(y) - frame, hop):
        seg = y[i:i + frame] * win
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        if corr[0] < 1e-9 or hi >= len(corr):
            continue
        corr /= corr[0]
        pk = int(np.argmax(corr[lo:hi])) + lo
        if corr[pk] > 0.4:
            f0s.append(sr / pk)
    return float(np.mean(f0s)) if f0s else 150.0


def estimate_wpm(items: List[Dict[str, Any]]) -> float:
    words = sum(len(str(it.get("transcript", "")).split()) for it in items)
    dur = sum(max(0.0, float(it["end"]) - float(it["start"])) for it in items)
    return words / dur * 60.0 if dur > 0 else 130.0


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def load_transcript(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path) as f:
        obj = json.load(f)
    out = []
    for it in obj.get("metadata:transcript", []):
        try:
            s, e, t = float(it["start"]), float(it["end"]), str(it.get("transcript", "")).strip()
        except Exception:
            continue
        if t and e > s:
            out.append({"start": s, "end": e, "transcript": t})
    return sorted(out, key=lambda x: x["start"])


def wav_to_json(wav_path: str) -> str:
    p = str(wav_path)
    return (p[:-4] if p.endswith(".wav") else p.split(".wav")[0]) + ".json"


# ---------------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------------

OPPOSITE_EMOTION: Dict[str, str] = {
    "Happiness": "Sadness", "Sadness": "Happiness",
    "Anger": "Neutral",     "Contempt": "Happiness",
    "Disgust": "Happiness", "Fear": "Happiness",
    "Surprise": "Neutral",  "Other": "Neutral",
    "Neutral": "Happiness",
}

EMOTION_TO_STYLE: Dict[str, Tuple[str, str]] = {
    "Happiness": ("expressive and animated", "slightly high-pitch"),
    "Sadness":   ("slightly expressive and animated", "low-pitched"),
    "Neutral":   ("slightly expressive and animated", "moderate pitch"),
    "Anger":     ("expressive and animated", "low-pitched"),
    "Fear":      ("slightly expressive and animated", "moderate pitch"),
    "Contempt":  ("expressive and animated", "moderate pitch"),
    "Disgust":   ("expressive and animated", "low-pitched"),
    "Surprise":  ("very expressive and animated", "slightly high-pitch"),
    "Other":     ("slightly expressive and animated", "moderate pitch"),
}

def safe_json(s: Any, default) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s.strip()) if s.strip() else default
        except Exception:
            return default
    return s if s is not None else default


def assign_emotion(t0: float, t1: float, spans, labels, fallback: str) -> str:
    best, lab = 0.0, fallback
    for l, sp in zip(labels, spans):
        try:
            ol = max(0.0, min(t1, float(sp["end"])) - max(t0, float(sp["start"])))
        except Exception:
            continue
        if ol > best:
            best, lab = ol, str(l)
    return lab


# ---------------------------------------------------------------------------
# Timeline packing (shared by all scripts)
# ---------------------------------------------------------------------------

def build_events(items_a, items_b) -> List[Dict[str, Any]]:
    events = (
        [{"spk": "A", "idx": i, "t0": float(it["start"]), "t1": float(it["end"]), "text": it["transcript"]} for i, it in enumerate(items_a)] +
        [{"spk": "B", "idx": i, "t0": float(it["start"]), "t1": float(it["end"]), "text": it["transcript"]} for i, it in enumerate(items_b)]
    )
    return sorted(events, key=lambda e: (e["t0"], 0 if e["spk"] == "B" else 1))


def pack_and_assemble(
    events: List[Dict[str, Any]],
    sr: int,
    total_sec: float,
    silence_db: float = -38.0,
    max_pause_ms: float = 70.0,
    fade_ms: float = 40.0,
    b_allow_stretch: bool = True,
    min_gap_ms: float = 100.0,
    segment_fade_ms: float = 15.0,
    segment_fade_in_ms: float = 50.0,
    segment_fade_out_ms: float = 25.0,
    add_noise: bool = True,
    noise_snr_db: float = 26.0,
    noise_color: str = "pink",
    noise_seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Pack events sequentially with gap-expansion strategy:

    - Speaker A audio is NEVER trimmed or compressed (original speech).
    - When B TTS is longer than its original window, the gap AFTER the
      preceding A utterance is expanded to absorb the overflow. If that
      gap is already at min_gap_ms, the expansion cascades forward,
      pushing all subsequent events later.
    - Total conversation duration grows if needed — we never sacrifice
      speech to hit an artificial length target.
    - B edge silence is trimmed before placement (reduces unnecessary
      expansion). B is never hard-truncated.
    - Each placed B segment gets an asymmetric cosine fade: fade-IN is
      longer than fade-OUT by default (50ms vs 25ms), because TTS onsets
      are typically steeper than natural speech and a longer fade-in
      softens that "jumping in" quality at turn transitions. A is NEVER
      faded, trimmed, or otherwise modified — it is placed exactly as
      recorded, since any modification risks audibly cutting into speech.
    - If add_noise, a mild shared background noise floor (pink or white)
      is added to both A and B tracks at the SAME seed, so the two
      speakers share one consistent acoustic environment instead of
      sounding like they were recorded in different rooms.

    Returns track_a, track_b, mix, packed_events_with_new_timings.
    """
    MIN_GAP = int(round(min_gap_ms / 1000.0 * sr))   # minimum silence between events

    # Step 1: trim edge silence from B only — A is untouched original speech.
    # Fade-in/out is applied to B ONLY. A is never modified in any way —
    # fading real recorded speech can eat into onset/offset consonants,
    # which is exactly the kind of cutoff we're trying to avoid.
    for e in events:
        y = np.asarray(e["audio"], dtype=np.float32)
        if e["spk"] == "B":
            y = trim_silence(y, sr, thresh_db=silence_db)
            y = fade_in_out(y, sr,
                            fade_in_ms=segment_fade_in_ms,
                            fade_out_ms=segment_fade_out_ms)
        e["audio"] = y
        e["dur"] = len(y)

    # Step 2: build initial gaps (silence) from original transcript timings.
    # These are the pauses inserted BETWEEN pieced-together segments.
    gaps_samp = [
        max(MIN_GAP, int(round((events[i + 1]["t0"] - events[i]["t1"]) * sr)))
        for i in range(len(events) - 1)
    ]

    # Step 3: gap-expansion pass
    # For each B utterance: if its audio is longer than its original window,
    # expand the gap BEFORE it (i.e., after the preceding event) to absorb
    # the overflow. If no preceding gap exists, expand the gap after it.
    # This pushes A utterances later rather than cutting B shorter.
    for i, e in enumerate(events):
        if e["spk"] != "B":
            continue

        orig_dur = max(0, int(round((e["t1"] - e["t0"]) * sr)))
        b_dur    = e["dur"]
        overflow = b_dur - orig_dur   # >0 means B needs more room than transcript gave it

        if overflow <= 0:
            continue   # B fits in its original window, no expansion needed

        # Try to expand the gap before this B utterance first (index i-1)
        if i > 0:
            gaps_samp[i - 1] += overflow
        elif i < len(gaps_samp):
            # B is the first event — expand gap after it instead
            gaps_samp[i] += overflow
        # If B is the only event, overflow goes into trailing silence (total grows)

    # Step 4: sequential placement — cursor moves forward through A, gap, B, gap, ...
    # This is literally how the pieces (and the pauses between them) get assembled
    # onto the single shared timeline.
    cursor = 0
    for i, e in enumerate(events):
        e["t0_new_samp"] = cursor
        e["t0_new"]      = cursor / sr
        cursor          += e["dur"]
        e["t1_new_samp"] = cursor
        e["t1_new"]      = cursor / sr
        if i < len(gaps_samp):
            cursor += gaps_samp[i]   # insert the pause before the next segment

    # Total length is whatever the content requires (may exceed original total_sec)
    total_samp = cursor
    events[-1]["_total_samp"] = total_samp   # store on last event for reference

    # Step 5: assemble tracks at the expanded total length.
    # Each segment is overlap-added at its new position; because every
    # segment was already faded in/out in Step 1, overlapping fade regions
    # (if any, from gap=0 edge cases) sum smoothly instead of clicking.
    track_a = np.zeros(total_samp, dtype=np.float32)
    track_b = np.zeros(total_samp, dtype=np.float32)
    for e in events:
        y  = np.asarray(e["audio"], dtype=np.float32)
        s0 = int(e["t0_new_samp"])
        s1 = min(total_samp, s0 + len(y))
        if s0 >= total_samp or s1 <= s0:
            continue
        if e["spk"] == "A":
            track_a[s0:s1] += y[:s1 - s0]
        else:
            track_b[s0:s1] += y[:s1 - s0]

    # Step 6: mild shared background noise floor on BOTH tracks before mixing.
    # Same seed -> same noise realization -> A and B sound like they share
    # one room instead of two different recording environments.
    if add_noise:
        track_a = add_background_noise(track_a, target_snr_db=noise_snr_db,
                                       color=noise_color, seed=noise_seed)
        track_b = add_background_noise(track_b, target_snr_db=noise_snr_db,
                                       color=noise_color,
                                       seed=(noise_seed + 1) if noise_seed is not None else None)

    track_a = np.clip(track_a, -1.0, 1.0)
    track_b = np.clip(track_b, -1.0, 1.0)
    mix     = np.clip(track_a + track_b, -1.0, 1.0)
    return track_a, track_b, mix, events


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    ensure(path.parent)
    sf.write(str(path), y.astype(np.float32), sr)


def write_meta(path: Path, events: List[Dict], extra: Dict = {}) -> None:
    rows = []
    for e in events:
        rows.append({
            "spk": e["spk"], "idx": e["idx"],
            "t0_orig": e["t0"], "t1_orig": e["t1"],
            "t0_new": e.get("t0_new", -1), "t1_new": e.get("t1_new", -1),
            "dur_orig": e["t1"] - e["t0"],
            "dur_new": e.get("t1_new", e["t1"]) - e.get("t0_new", e["t0"]),
            "text": e.get("text", ""),
            **{k: e.get(k, "") for k in ["orig_emo", "opp_emo", "caption", "tts_model"]},
            **extra,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
