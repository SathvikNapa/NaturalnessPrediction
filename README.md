# Dyadic Naturalness Classifier

A pipeline for predicting whether a spoken dyadic interaction sounds **natural** (1) or **unnatural** (0). It combines sliding-window Whisper encoder embeddings with text embeddings of conversational context and speaker relationship metadata.

The pipeline has three stages:

```
raw WAVs  --->  feature extraction  --->  model training  --->  inference
```

---

## Pretrained Weights

All checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing). Download the subfolder matching your desired feature set and model.

| Feature set | Input modality | `crossattn` | `llama` | `mlp` | `dyadformer` |
|---|---|:---:|:---:|:---:|:---:|
| `speech_only` | Audio only | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | — | — |
| `speech_rel` | Audio + relationship text | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | — | — |
| `speech_context` | Audio + context + roles | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) |
| `speech_context_rel` | Audio + context + roles + relationship | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) | [↓](https://drive.google.com/drive/folders/1PeePtN-5fC_UTlnOH5Q0zsok8Mv8lQdR?usp=sharing) |

Each folder contains `best_model.pt` (best validation accuracy) and `last_model.pt` (final epoch). The Drive root is `SLT_naturalness_ablation/`; subdirectory names map as: `trace_crossencoder` → `crossattn`, `trace_latefusion` → `llama`.

---

## Dataset

**TRACE** (Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech) is available on [Hugging Face](https://huggingface.co/datasets/SathvikNapa/TRACE). 8,434 labeled dyadic pairs split into train (5,131) and test (3,303). License: CC BY-NC 4.0.

| Augmentation type | Description | Label | Train pairs | Test pairs |
|---|---|:---:|:---:|:---:|
| [`original`](https://huggingface.co/datasets/SathvikNapa/TRACE/tree/main/audio/original) | Unmodified naturalistic speech | 1 — natural | 1,295 | 830 |
| [`original_vc`](https://huggingface.co/datasets/SathvikNapa/TRACE/tree/main/audio/original_vc) | Naturalistic speech with voice conversion applied | 0 — unnatural | 1,288 | 825 |
| [`emotivoice_tts`](https://huggingface.co/datasets/SathvikNapa/TRACE/tree/main/audio/emotivoice_tts) | Speaker B replaced with contrastive-emotion TTS | 0 — unnatural | 1,295 | 830 |
| [`emotivoice_tts_vc`](https://huggingface.co/datasets/SathvikNapa/TRACE/tree/main/audio/emotivoice_tts_vc) | TTS Speaker B additionally voice-converted | 0 — unnatural | 1,253 | 818 |

---

## Dependencies

**Python packages**

```
torch torchvision torchaudio
soundfile
librosa
numpy
scikit-learn
transformers
sentence-transformers
tqdm
```

**External dependency**

`vox-profile-release` must be cloned into the **same directory** as the pipeline scripts. `extract.py` adds `./vox-profile-release` to `sys.path` at runtime and imports `WhisperWrapper` from it directly.

```bash
git clone https://github.com/UttaranB127/vox-profile-release.git
```

The required file inside the repo is:

```
vox-profile-release/src/model/emotion/whisper_emotion_dim.py
```

---

## Repository layout

```
working_dir/
├── extract.py              # Stage 1: feature extraction
├── train.py                # Stage 2: model training (LLaMA/CrossAttn/MLP/LogReg, DDP)
├── infer.py                # Stage 3: inference (batch CSV + optional attn dump)
├── emotion_conversion/
│   ├── convert_emotivoice_v2.py   # Speaker-B emotion resynthesis via EmotiVoice
│   └── tts_shared_v2.py           # Shared audio/TTS utilities
└── vox-profile-release/    # cloned alongside the scripts
    └── src/
        └── model/
            └── emotion/
                └── whisper_emotion_dim.py
```

---

## Stage 1: Feature Extraction

`extract.py` segments each WAV file into overlapping windows, runs each window through a Whisper-large-v3 encoder fine-tuned for dimensional emotion prediction, and saves the resulting embeddings to disk.

**What it produces**

```
out_dir/
├── embeds_vec/<dyad_id>/<seg_stem>__ch#####.npy   # pooled [D] vector per chunk
├── embeds_seq/<dyad_id>/<seg_stem>__ch#####.npy   # full [T, D] sequence (optional)
├── shards/rank*.csv                                # per-GPU metadata (merged after)
└── <dyad_id>.csv                                   # per-dyad merged CSV (valence/arousal/dominance)
```

**Usage**

```bash
python extract.py \
    --wav-list  /path/to/wavs.txt \
    --out-dir   /path/to/embeddings \
    --win-sec   3.0 \
    --hop-sec   1.0 \
    --batch-size 64 \
    --amp
```

Multi-GPU (parallel — no DDP):

```bash
python extract.py \
    --wav-list  /path/to/wavs.txt \
    --out-dir   /path/to/embeddings \
    --num-gpus  4
```

**Key arguments**

| Argument | Default | Description |
|---|---|---|
| `--wav-list` | required | Text file with one WAV path per line |
| `--out-dir` | required | Output directory |
| `--win-sec` | `3.0` | Window length in seconds |
| `--hop-sec` | `1.0` | Hop size in seconds |
| `--pad-last` / `--no-pad-last` | pad on | Pad the last short window to full length |
| `--batch-size` | `64` | Chunks per GPU per forward pass |
| `--pool` | `mean` | How to pool `[T, D]` → `[D]`: `mean` or `max` |
| `--save-full-seq` | off | Also save the full `[T, D]` sequence (needed for `crossattn` model) |
| `--amp` / `--no-amp` | amp on | fp16 autocast on CUDA |
| `--num-gpus` | `0` (all) | Number of GPUs to use |
| `--skip-existing-dyads` | off | Resume: skip dyads whose chunk files are already complete |

**Resuming**

Pass `--skip-existing-dyads`. The script checks both the merged CSV and the completeness of all expected `.npy` chunk files before skipping a dyad.

**Filename conventions**

The script groups files into dyads by parsing Seamless-corpus-style filenames (`V\d+_(S\d+)_I(\d+)_P...`). Files that do not match this pattern are assigned a unique ID based on a hash of their resolved path, preventing cross-file collisions.

---

## Stage 2: Model Training

`train.py` reads dyadic pair CSVs, loads speaker embeddings produced by Stage 1, and trains a binary classifier.

**Input CSV columns**

| Column | Description |
|---|---|
| `participant1_relpath_abs` | Path whose stem is the base ID for speaker A |
| `participant2_relpath_abs` | Path whose stem is the base ID for speaker B |
| `naturalness` | Binary label: 0 = unnatural, 1 = natural |
| `high_level_context` | Free-text description of the conversational context |
| `speaker_a_role` | Role label for speaker A |
| `speaker_b_role` | Role label for speaker B |
| `rel_detail` | Free-text description of the relationship between speakers |

**Usage**

```bash
python train.py \
    --train-csv   data/train.csv \
    --val-csv     data/val.csv \
    --test-csv    data/test.csv \
    --embed-root  /path/to/embeddings/embeds_vec \
    --out-dir     runs/exp01 \
    --model-type  llama \
    --batch-size  32 \
    --num-epochs  30 \
    --lr          1e-4
```

Multi-GPU with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py \
    --train-csv  data/train.csv \
    --test-csv   data/test.csv \
    --embed-root /path/to/embeddings/embeds_vec \
    --out-dir    runs/exp01
```

**Model types**

| `--model-type` | Description |
|---|---|
| `llama` | LLaMA-style encoder: RMSNorm + RoPE attention + SwiGLU FFN |
| `base_transformer` | Standard PyTorch `TransformerEncoder` with sinusoidal PE |
| `crossattn` | LLaMA encoder for audio; auxiliary features cross-attend into audio tokens |
| `mlp` | Mean-pooled audio + MLP head |
| `logreg` | Mean-pooled audio + single linear layer (logistic regression baseline) |

All models fuse audio embeddings with three auxiliary feature streams: categorical speaker role embeddings, a context text embedding, and a relationship text embedding (encoded via `sentence-transformers/all-MiniLM-L6-v2` by default). Pass `--speech-only` to disable all auxiliary features.

**Key training arguments**

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `llama` | See table above |
| `--d-model` | `256` | Transformer hidden dimension |
| `--num-layers` | `6` | Number of encoder layers |
| `--n-heads` | `8` | Number of attention heads |
| `--dropout` | `0.1` | Dropout rate |
| `--max-seq-len` | `512` | Maximum interleaved sequence length (= 2× max chunks per speaker) |
| `--window-strategy` | `head` | `head` (first N chunks) or `random` (deterministic random window by speaker ID hash) |
| `--batch-size` | `16` | Training batch size |
| `--num-epochs` | `20` | Maximum training epochs |
| `--lr` | `1e-4` | AdamW learning rate |
| `--patience` | `8` | Early stopping patience (val accuracy) |
| `--speech-only` | off | Disable all non-speech auxiliary features |
| `--no-cats` | off | Disable categorical role embeddings |
| `--no-context` | off | Disable context text embeddings |
| `--no-rel-text` | off | Disable relationship text embeddings |
| `--cache-root` | — | Directory for speaker-level embedding cache (speeds up multi-epoch training) |
| `--prebuild-cache` | off | Pre-build speaker cache before training begins |
| `--resume-from` | — | Path to checkpoint to resume training from |
| `--dump-attn` | off | Save reduced attention weights at test time (llama and crossattn only) |
| `--text-model` | `all-MiniLM-L6-v2` | HuggingFace model for encoding context and relationship text |

**Outputs**

```
runs/exp01/
├── best_model.pt          # checkpoint with best validation accuracy
├── last_model.pt          # checkpoint after the final epoch
├── checkpoints/
│   └── epoch_N.pt         # per-epoch checkpoints
├── metrics.jsonl          # per-epoch train/val loss and accuracy
├── hyperparams.json
├── test_predictions.csv   # per-row probability and prediction on test set
├── test_misclassified.csv # misclassified test rows only
├── test_metrics.json      # accuracy, confusion matrix, Brier score
└── embed_index.pkl        # cached embedding index (reusable in inference)
```

If `--dump-attn` is passed:

```
runs/exp01/attn/
├── attn_batches.pt        # raw reduced attention tensors
└── attn_summary.json      # per-layer entropy and top-mass statistics
```

---

## Stage 3: Inference

`infer.py` loads a trained checkpoint and scores new dyadic pairs. It supports both single-pair and batch CSV modes and is robust to checkpoints from older training runs (it infers model architecture from state dict keys when hyperparameters are missing).

**Single-pair mode**

```bash
python infer.py \
    --checkpoint  runs/exp01/best_model.pt \
    --embed-root  /path/to/embeddings/embeds_vec \
    --p1          speaker_a_stem \
    --p2          speaker_b_stem
```

**Batch CSV mode**

```bash
python infer.py \
    --checkpoint   runs/exp01/best_model.pt \
    --embed-root   /path/to/embeddings/embeds_vec \
    --input-csv    data/test.csv \
    --output-csv   results/predictions.csv \
    --batch-size   32 \
    --context-cache runs/exp01/context_hf_cache.pkl \
    --rel-cache     runs/exp01/relationship_hf_cache.pkl
```

Multi-GPU sharded inference (each GPU processes every N-th row):

```bash
torchrun --nproc_per_node=4 infer.py \
    --checkpoint  runs/exp01/best_model.pt \
    --embed-root  /path/to/embeddings/embeds_vec \
    --input-csv   data/test.csv \
    --output-csv  results/predictions.csv
```

**Key inference arguments**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `best_model.pt` or any checkpoint |
| `--embed-root` | required | Root directory of `.npy` embeddings |
| `--input-csv` | — | CSV of dyadic pairs to score (batch mode) |
| `--output-csv` | — | Output predictions CSV |
| `--p1` / `--p2` | — | Speaker base stems for single-pair mode |
| `--threshold` | `0.5` | Decision threshold on sigmoid output |
| `--batch-size` | `16` | Inference batch size |
| `--index-cache` | — | Reuse embed index pickle from training |
| `--cache-root` | — | Reuse speaker cache from training |
| `--context-cache` | — | Reuse context text embed pickle from training |
| `--rel-cache` | — | Reuse relationship text embed pickle from training |
| `--max-seq-len` | from checkpoint | Override max chunks per speaker |
| `--num-shards` | `1` | Number of parallel shards (set automatically from `WORLD_SIZE` with torchrun) |

**Output CSV columns**

`row_idx`, `p1_base`, `p2_base`, `p1_path`, `p2_path`, `label`, `augmentation_type`, `rel_detail`, `high_level_context`, `probability`, `prediction`, `logit`, `error`

Rows where embeddings are missing receive `error="missing_embeddings"` rather than crashing the run.

---

## Emotion Resynthesis

`emotion_conversion/` contains scripts for generating AI-resynthesised speech with speaker-appropriate emotion conditioning via EmotiVoice.

```
emotion_conversion/
├── convert_emotivoice_v2.py   # Speaker-B emotion conversion pipeline (v2)
└── tts_shared_v2.py           # Shared audio/TTS utilities
```

**Usage**

```bash
python emotion_conversion/convert_emotivoice_v2.py \
    --age-gender-csv /path/age_sex_predictions_merged.csv \
    --emotion-csv    /path/emotion_predictions_merged.csv \
    --out-root       /path/output \
    --max-rows       5
```

**What it does**

For each dyadic interaction, Speaker B's turns are replaced with EmotiVoice TTS output conditioned on:
- F0 range and WPM from the original Speaker B recording (speaker identity cues)
- The *opposite* emotion from Speaker A (contrastive emotion conditioning)
- Age/gender metadata for caption building

Gap-expansion packing is used so Speaker A's audio is never trimmed or stretched — if B's TTS output is longer than its original window, the gap before it expands rather than cutting into A's speech.

---

## End-to-end example

```bash
# 1. Extract embeddings
python extract.py \
    --wav-list data/wavs.txt \
    --out-dir  embeddings/ \
    --win-sec  3.0 --hop-sec 1.0 --amp

# 2. Train
python train.py \
    --train-csv  data/train.csv \
    --val-csv    data/val.csv \
    --test-csv   data/test.csv \
    --embed-root embeddings/embeds_vec \
    --out-dir    runs/exp01 \
    --model-type llama \
    --cache-root embeddings/speaker_cache \
    --prebuild-cache

# 3. Infer
python infer.py \
    --checkpoint    runs/exp01/best_model.pt \
    --embed-root    embeddings/embeds_vec \
    --index-cache   runs/exp01/embed_index.pkl \
    --context-cache runs/exp01/context_hf_cache.pkl \
    --rel-cache     runs/exp01/relationship_hf_cache.pkl \
    --input-csv     data/test.csv \
    --output-csv    results/predictions.csv
```