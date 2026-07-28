#!/usr/bin/env python3
"""
convert_emotivoice_v2.py  --  Speaker B emotion conversion via EmotiVoice (v2)

v2 changes vs v1:
  - Gap-expansion packer: Speaker A audio is NEVER trimmed, stretched, or
    faded. When B's TTS output is longer than its original window, the
    gap before it expands (and total conversation duration may grow)
    instead of cutting into A's speech.
  - Per-segment fade-in/out applied to B only (click prevention),
    A passes through byte-identical to the source recording.
  - Mild shared background noise floor (pink/white, tunable SNR) added
    to both A and B after assembly so they share one acoustic environment.
  - Duration headroom (orig * 1.25) so model never rushes and clips consonants
  - Spectral denoising of B before mixing
  - RMS normalization: B matched to Speaker A loudness
  - nar_run uses confirmed EmoCapTTS signature directly (no introspection)
  - CPU fallback: torch.load patched with map_location=cpu

Usage:
  python convert_emotivoice_v2.py \
    --age-gender-csv /path/age_sex_predictions_merged.csv \
    --emotion-csv /path/emotion_predictions_merged.csv \
    --out-root /path/output \
    --max-rows 5
"""

from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf

# Put tts_shared.py in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from tts_shared_v2 import (
    load_wav, write_wav, write_meta, ensure,
    build_events, pack_and_assemble,
    load_transcript, wav_to_json,
    safe_json, assign_emotion,
    OPPOSITE_EMOTION, EMOTION_TO_STYLE,
    estimate_f0_mean, estimate_wpm,
    rms_normalize, speaker_a_rms, denoise,
    trim_silence,
)

DEFAULT_SR = 24000


# ---------------------------------------------------------------------------
# Caption builder (rich: encodes F0 + WPM for speaker identity)
# ---------------------------------------------------------------------------

def age_bucket(age: float) -> str:
    return "middle-aged adult" if age >= 35 else "young adult"


def build_caption(
    sex: str, age: float, emotion: str,
    monotony: str, pitch: str,
    f0_mean: float = 150.0, wpm: float = 130.0,
) -> str:
    g = sex.strip().lower()
    g = "female" if "fem" in g else "male"
    age_cat = age_bucket(age)
    ident = f"a {age_cat} {'woman' if g == 'female' else 'man'}"
    poss = "her" if g == "female" else "his"

    pace = ("slow pace" if wpm < 110 else
            "moderate pace" if wpm < 150 else
            "fast pace" if wpm < 190 else "very fast pace")
    voice = ("a deep voice" if f0_mean < 120 else
             "a medium-pitched voice" if f0_mean < 165 else
             "a bright voice" if f0_mean < 220 else "a high-pitched voice")
    pitch_str = {"slightly high-pitch": "a slightly high pitch",
                 "moderate pitch": "a moderate pitch",
                 "low-pitched": "a low-pitched voice"}.get(pitch, pitch)

    return (
        f"{ident} with {voice} delivers {poss} speech with {emotion.lower()}, "
        f"in a {monotony} manner. {poss.capitalize()} voice carries {pitch_str} "
        f"and {poss} pace is {pace}."
    )


# ---------------------------------------------------------------------------
# Model loading (with CPU fallback patch)
# ---------------------------------------------------------------------------

def load_model(device: str, seed: int = 42):
    import torch
    import capspeech.nar.generate as nar
    nar.seed_everything(seed)

    if device == "cpu" or not torch.cuda.is_available():
        _orig = torch.load
        def _patched(f, *a, **kw):
            kw.setdefault("map_location", torch.device("cpu"))
            return _orig(f, *a, **kw)
        torch.load = _patched
        try:
            return nar.load_model(device, "EmoCapTTS")
        finally:
            torch.load = _orig

    return nar.load_model(device, "EmoCapTTS")


def synthesize(model_list, device: str, duration: Optional[float],
               text: str, caption: str,
               speed: float = 1.0, steps: int = 25, cfg: float = 2.0,
               voice_seed: Optional[int] = None) -> np.ndarray:
    """
    voice_seed: if given, the RNG is reset to this exact seed immediately
    before synthesis. EmoCapTTS has no speaker/voice-embedding input — it
    infers voice purely from sampling noise + caption text. Without
    re-seeding, the RNG state drifts forward after every call, so each
    utterance for the "same" speaker samples a different point in voice
    space and sounds like a different person. Re-seeding with the same
    voice_seed for every utterance belonging to one speaker collapses
    that variance back down so voice identity stays consistent across
    a row, while caption text differences (emotion, pace) still vary
    the *style* as intended.
    """
    import capspeech.nar.generate as nar
    if voice_seed is not None:
        nar.seed_everything(voice_seed)
    return np.asarray(
        nar.run(model_list, device, duration, text, caption,
                speed=speed, steps=steps, cfg=cfg),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def process_row(row: Dict, model_list, device: str, out_root: Path, args) -> None:
    row_id = str(row.get("row_id", "NA"))
    p1_id  = str(row.get("participant1_id", "P1"))
    p2_id  = str(row.get("participant2_id", "P2"))
    wav_a  = str(row.get(args.p1_wav_col, ""))
    wav_b  = str(row.get(args.p2_wav_col, ""))
    if not wav_a or not wav_b:
        return

    row_dir = ensure(out_root / f"row_{row_id}_p1_{p1_id}_p2_{p2_id}")
    err_log = row_dir / "errors.log"

    if args.skip_existing and all((row_dir / f).exists() for f in
            ["speaker1_track.wav", "speaker2_track.wav", "mixed.wav", "metadata.csv"]):
        return

    # Load transcripts
    try:
        items_a = load_transcript(wav_to_json(wav_a))
        items_b = load_transcript(wav_to_json(wav_b))
    except Exception as e:
        (row_dir / "ERROR.txt").write_text(str(e))
        return

    if not items_a or not items_b:
        (row_dir / "ERROR.txt").write_text("Empty transcript")
        return

    # Load Speaker A audio
    try:
        y_a = load_wav(wav_a, args.sr)
    except Exception as e:
        err_log.write_text(str(e))
        return

    # Speaker B characteristics for caption
    sex_b  = str(row.get(args.sex_col, "Female"))
    age_b  = float(row.get(args.age_col, 30.0) or 30.0)
    emo_labels = safe_json(row.get(args.emo_labels_col, "[]"), [])
    emo_spans  = safe_json(row.get(args.emo_spans_col,  "[]"), [])
    fallback   = str(row.get(args.emo_fallback_col, "Neutral") or "Neutral")

    # F0 + WPM for speaker-identity caption
    y_b_raw = None
    try:
        y_b_raw = load_wav(wav_b, args.sr)
        f0_b    = estimate_f0_mean(y_b_raw, args.sr)
        wpm_b   = estimate_wpm(items_b)
    except Exception:
        f0_b, wpm_b = 150.0, 130.0

    # Compute RMS for both speakers (from original audio) and decide the
    # normalization target based on --normalize-to:
    #   "a"      -> B always matches A (legacy/default behavior)
    #   "louder" -> whichever speaker started louder becomes the target,
    #               so the quieter speaker is boosted up to match instead
    #               of always pulling B down to A
    rms_a = speaker_a_rms(y_a)
    rms_b = speaker_a_rms(y_b_raw) if y_b_raw is not None else rms_a

    if args.normalize_to == "louder":
        target_rms = max(rms_a, rms_b)
    else:
        target_rms = rms_a   # "a" (default): B matches A, same as before

    # NOTE: Speaker A's audio is never modified by this script (it is the
    # original recording — see pack_and_assemble). If rms_b > rms_a and
    # normalize_to="louder", B will be synthesized at B's own (louder)
    # original level rather than being pulled down to A. A itself is
    # left at its native recorded volume either way.

    # Deterministic per-speaker voice seed: EmoCapTTS has no speaker/voice
    # embedding input, so without re-seeding the RNG before every call,
    # each utterance samples a different point in the model's voice space
    # and Speaker B sounds like a different person every line. Hashing
    # row_id + p2_id gives a stable seed unique to THIS speaker in THIS
    # row, reused for every one of their utterances so voice identity
    # stays consistent while caption text still varies emotion/style.
    voice_seed = (
        int(hashlib.sha256(f"{row_id}_{p2_id}".encode()).hexdigest(), 16) % (2**31)
        if args.lock_voice_seed else None
    )

    # Build events
    events = build_events(items_a, items_b)
    total_sec = max(e["t1"] for e in events) + args.tail_sec

    # Attach audio
    for e in events:
        t0, t1 = e["t0"], e["t1"]
        dur = max(0.0, t1 - t0)

        if e["spk"] == "A":
            s0, s1 = int(round(t0 * args.sr)), int(round(t1 * args.sr))
            e["audio"] = y_a[s0:s1].copy() if s1 > s0 else np.zeros(0, np.float32)
            e["tts_model"] = "orig"

        else:
            orig_emo = assign_emotion(t0, t1, emo_spans, emo_labels, fallback) \
                       if len(emo_labels) == len(emo_spans) > 0 else fallback
            opp_emo  = OPPOSITE_EMOTION.get(orig_emo, "Neutral")
            mono_sty, pitch = EMOTION_TO_STYLE.get(opp_emo, ("slightly expressive and animated", "moderate pitch"))

            caption = build_caption(sex_b, age_b, opp_emo, mono_sty, pitch, f0_b, wpm_b)[:args.max_caption]

            # 25% headroom so model finishes consonants; packer trims silence not speech
            tts_dur = dur * args.duration_headroom if dur > 0 else None

            try:
                audio = synthesize(model_list, device, tts_dur, e["text"], caption,
                                   speed=args.tts_speed, steps=args.tts_steps, cfg=args.tts_cfg,
                                   voice_seed=voice_seed)
                # tail pad gives packer silence to trim
                tail = int(round(args.tail_pad_sec * args.sr))
                if tail > 0:
                    audio = np.pad(audio, (0, tail)).astype(np.float32)
                # denoise
                audio = denoise(audio, args.sr, prop_decrease=args.denoise_strength)
                # normalize to match Speaker A loudness
                audio = rms_normalize(audio, target_rms)
            except Exception as ex:
                with open(err_log, "a") as f:
                    f.write(f"[idx={e['idx']}] synth failed: {ex}\n")
                audio = np.zeros(0, np.float32)

            e["audio"]    = audio
            e["orig_emo"] = orig_emo
            e["opp_emo"]  = opp_emo
            e["caption"]  = caption
            e["tts_model"] = "emotivoice"

    # Pack timeline and assemble
    track_a, track_b, mix, packed = pack_and_assemble(
        events, args.sr, total_sec,
        silence_db=args.silence_db,
        fade_ms=args.fade_ms,
        b_allow_stretch=args.b_allow_stretch,
        min_gap_ms=args.min_gap_ms,
        segment_fade_ms=args.segment_fade_ms,
        segment_fade_in_ms=args.segment_fade_in_ms,
        segment_fade_out_ms=args.segment_fade_out_ms,
        add_noise=args.add_noise,
        noise_snr_db=args.noise_snr_db,
        noise_color=args.noise_color,
    )

    write_wav(row_dir / "speaker1_track.wav", track_a, args.sr)
    write_wav(row_dir / "speaker2_track.wav", track_b, args.sr)
    write_wav(row_dir / "mixed.wav",          mix,     args.sr)
    write_meta(row_dir / "metadata.csv", packed, extra={
        "rms_a": round(float(rms_a), 5),
        "rms_b_orig": round(float(rms_b), 5),
        "normalize_to": args.normalize_to,
        "target_rms": round(float(target_rms), 5),
    })
    print(f"[OK] {row_dir.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age-gender-csv", required=True)
    ap.add_argument("--emotion-csv",    required=True)
    ap.add_argument("--out-root",       required=True)
    ap.add_argument("--sr",             type=int,   default=DEFAULT_SR)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--max-rows",       type=int,   default=0, help="0 = all rows")
    ap.add_argument("--skip-existing",  action="store_true")
    ap.add_argument("--natural-only",   action="store_true")
    ap.add_argument("--natural-col",    default="naturalness")
    ap.add_argument("--p1-wav-col",     default="participant1_relpath_abs")
    ap.add_argument("--p2-wav-col",     default="participant2_relpath_abs")
    ap.add_argument("--sex-col",        default="sex_label_p2")
    ap.add_argument("--age-col",        default="age_years_p2")
    ap.add_argument("--emo-labels-col", default="emotion_utterance_labels_p2")
    ap.add_argument("--emo-spans-col",  default="emotion_utterance_spans_p2")
    ap.add_argument("--emo-fallback-col", default="emotion_label_p2")
    ap.add_argument("--tail-sec",       type=float, default=0.5)
    ap.add_argument("--tail-pad-sec",   type=float, default=0.15)
    ap.add_argument("--duration-headroom", type=float, default=1.25)
    ap.add_argument("--silence-db",     type=float, default=-38.0)
    ap.add_argument("--fade-ms",        type=float, default=40.0)
    ap.add_argument("--max-caption",    type=int,   default=400)
    ap.add_argument("--tts-speed",      type=float, default=1.0)
    ap.add_argument("--tts-steps",      type=int,   default=25)
    ap.add_argument("--tts-cfg",        type=float, default=2.0)
    ap.add_argument("--denoise-strength", type=float, default=0.75)
    ap.add_argument("--b-allow-stretch",  action="store_true")
    ap.add_argument("--min-gap-ms",     type=float, default=100.0,
                    help="Minimum silence (ms) preserved between any two pieced-together segments")
    ap.add_argument("--segment-fade-ms", type=float, default=15.0,
                    help="Legacy symmetric fade fallback; use --segment-fade-in-ms / --segment-fade-out-ms instead")
    ap.add_argument("--segment-fade-in-ms",  type=float, default=50.0,
                    help="Fade-in (ms) for B segments; longer than fade-out softens TTS onset attack")
    ap.add_argument("--segment-fade-out-ms", type=float, default=25.0,
                    help="Fade-out (ms) for B segments")
    ap.add_argument("--add-noise",      action="store_true", default=True,
                    help="Add mild shared background noise to A and B (default on)")
    ap.add_argument("--no-noise",       dest="add_noise", action="store_false",
                    help="Disable background noise addition")
    ap.add_argument("--noise-snr-db",   type=float, default=26.0,
                    help="Target SNR in dB; lower = louder/more present noise floor in gaps")
    ap.add_argument("--noise-color",    choices=["pink", "white"], default="pink",
                    help="pink = natural room-hiss-like; white = flat spectrum")
    ap.add_argument("--normalize-to", choices=["a", "louder"], default="a",
                    help="'a' (default): B is normalized to match Speaker A's loudness. "
                         "'louder': B is normalized to match whichever speaker (A or B) "
                         "was originally louder, so the quieter one effectively gets boosted "
                         "rather than always pulling B down to A. Speaker A's own audio is "
                         "never modified in either mode.")
    ap.add_argument("--lock-voice-seed", action="store_true", default=True,
                    help="Re-seed RNG with a per-speaker deterministic seed before every "
                         "synthesis call, so the same speaker sounds consistent across all "
                         "their utterances instead of a different voice each line (default on)")
    ap.add_argument("--no-lock-voice-seed", dest="lock_voice_seed", action="store_false",
                    help="Disable voice-seed locking (legacy behavior: voice drifts per utterance)")
    ap.add_argument("--device",         default="cuda:0")
    ap.add_argument("--num-gpus",       type=int, default=0,
                    help="Launch this many subprocess workers, one per visible GPU, "
                         "sharding rows across them. 0 = single-process mode using --device.")
    ap.add_argument("--worker-rank",    type=int, default=None,
                    help="[internal] subprocess worker rank, do not set manually")
    ap.add_argument("--world-size",     type=int, default=None,
                    help="[internal] total subprocess worker count, do not set manually")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Multi-GPU launch: if --num-gpus > 0 and we are NOT already a worker
    # subprocess, spawn one subprocess per GPU (each pinned via
    # CUDA_VISIBLE_DEVICES) and shard rows across them by row index % world_size.
    # ------------------------------------------------------------------
    if args.num_gpus > 0 and args.worker_rank is None:
        import subprocess as sp
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        gpu_list = [g.strip() for g in gpus.split(",") if g.strip()] if gpus else \
                   [str(i) for i in range(args.num_gpus)]
        n = min(args.num_gpus, len(gpu_list)) if gpu_list else args.num_gpus

        print(f"[LAUNCH] Spawning {n} worker subprocesses across GPUs {gpu_list[:n]}", flush=True)
        # Rebuild argv without --num-gpus and its value (each worker runs single-GPU internally)
        skip_next = False
        filtered = []
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a == "--num-gpus":
                skip_next = True
                continue
            filtered.append(a)

        procs = []
        for r in range(n):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[r]
            cmd = [sys.executable, __file__] + filtered + [
                "--device", "cuda:0",   # each subprocess sees only its 1 GPU as cuda:0
                "--worker-rank", str(r),
                "--world-size", str(n),
            ]
            procs.append(sp.Popen(cmd, env=env))

        rc = 0
        for p in procs:
            p.wait()
            rc = rc or p.returncode
        if rc != 0:
            raise SystemExit(rc)
        return
    # ------------------------------------------------------------------

    def is_nat(v):
        return str(v).strip().lower() in {"1", "natural", "true"}

    df_ag = pd.read_csv(args.age_gender_csv)
    df_em = pd.read_csv(args.emotion_csv)
    if args.natural_only:
        if args.natural_col in df_ag.columns:
            df_ag = df_ag[df_ag[args.natural_col].apply(is_nat)].reset_index(drop=True)
        if args.natural_col in df_em.columns:
            df_em = df_em[df_em[args.natural_col].apply(is_nat)].reset_index(drop=True)

    keys = ["row_id"]
    if all(c in df_ag.columns and c in df_em.columns for c in ["participant1_id", "participant2_id"]):
        keys += ["participant1_id", "participant2_id"]
    merged = pd.merge(df_ag, df_em, on=keys, how="inner", suffixes=("", "_emo"))
    if args.max_rows > 0:
        merged = merged.iloc[:args.max_rows].reset_index(drop=True)

    # Shard rows across workers: row index % world_size == worker_rank.
    # Each GPU subprocess only processes its own slice of the dataset.
    if args.worker_rank is not None and args.world_size is not None:
        idx = np.arange(len(merged))
        merged = merged.iloc[idx[idx % args.world_size == args.worker_rank]].reset_index(drop=True)
        tag = f"[Worker {args.worker_rank}/{args.world_size}]"
    else:
        tag = ""

    print(f"{tag} Processing {len(merged)} rows on {args.device}", flush=True)
    model_list = load_model(args.device, args.seed)
    out_root = Path(args.out_root)

    from tqdm import tqdm
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=tag or "rows"):
        try:
            process_row(row.to_dict(), model_list, args.device, out_root, args)
        except Exception as ex:
            print(f"{tag} [ERR] row {row.get('row_id')}: {ex}", flush=True)

    print(f"{tag} Done.", flush=True)


if __name__ == "__main__":
    main()
