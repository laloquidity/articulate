# 🗣️ Articulate

**Turn any text into a voice-cloned narrated video with word-level captions.**

Point it at a voice recording, give it text, get a professionally captioned video — powered by [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) voice cloning and [Whisper](https://github.com/linto-ai/whisper-timestamped) word-level timestamps.

---

## ✨ Features

- **🎤 Auto Voice Detection** — Point at any audio file (podcast, interview, etc.) and it automatically finds the cleanest segment for voice cloning
- **📝 Smart Text Preprocessing** — Fixes abbreviations (`e.g.` → "for example"), numbered lists, ALL-CAPS headers, and other TTS pitfalls
- **🔊 Voice-Cloned TTS** — Uses Chatterbox for high-quality voice cloning from just a 20-second sample
- **✂️ Silence Trimming** — Automatically detects and trims excessive pauses for natural pacing
- **📊 Word-Level Captions** — Whisper-powered word timestamps with real-time highlighting
- **🎬 Video Export** — Outputs a polished MP4 in 16:9, 9:16, or 1:1 format
- **⚡ GPU Accelerated** — Supports CUDA, Apple Silicon (MPS), and CPU fallback

## 🚀 Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/laloquidity/articulate.git
cd articulate

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e .
```

> **Note:** Requires **Python 3.10–3.11** and [ffmpeg](https://ffmpeg.org/) installed on your system. Python 3.12 has compatibility issues with chatterbox-tts's numpy requirement.

### 2. Run

```bash
# Full pipeline: voice file + text → captioned video
articulate run \
    --voice my_recording.mp3 \
    --text article.txt \
    --output-dir output
```

That's it. The pipeline will:
1. 🔍 Analyze your audio and find the best 20s clip for voice cloning
2. 📝 Preprocess the text for natural speech
3. 🎙️ Generate voice-cloned narration segment by segment
4. ✂️ Trim excessive silences
5. 📊 Extract word-level timestamps
6. 🎬 Build the final captioned video

### 3. Output

```
output/
├── reference.wav       # Auto-detected reference clip
├── text_processed.txt  # Preprocessed text
├── audio.wav           # Raw TTS audio
├── audio_trimmed.wav   # Silence-trimmed audio
├── timestamps.json     # Word-level timestamps
└── final_video.mp4     # 🎬 Your video!
```

## 📖 CLI Reference

### `articulate run` — Full Pipeline

```bash
articulate run --voice <audio> --text <file> [options]
```

| Option | Default | Description |
|---|---|---|
| `--voice` | *required* | Source audio for voice cloning (any format) |
| `--text` | *required* | Text file to narrate |
| `--reference` | *auto* | Pre-extracted reference clip (skips auto-detection) |
| `-o, --output-dir` | `output/` | Output directory |
| `--aspect-ratio` | `16:9` | Video format: `16:9`, `9:16`, `1:1` |
| `--speed` | `1.0` | Playback speed (e.g., `0.85` for slower) |
| `--exaggeration` | `0.5` | Emotion level 0–1 |
| `--cfg-weight` | `0.5` | Voice similarity weight |
| `--whisper-model` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `--words-per-group` | `5` | Words shown at once in captions |
| `--skip-preprocess` | | Skip text preprocessing |
| `--skip-tts` | | Skip TTS (reuse existing audio) |
| `--skip-trim` | | Skip silence trimming |
| `--skip-timestamps` | | Skip Whisper (reuse existing timestamps) |

### `articulate find-ref` — Find Best Voice Clip

Analyzes audio to find the segment with the cleanest speech for voice cloning.

```bash
articulate find-ref podcast.mp3 -o reference.wav
articulate find-ref interview.wav -d 15 -s 3  # 15s clips, 3s stride
```

### `articulate preprocess` — Fix Text for TTS

```bash
articulate preprocess article.txt -o article_clean.txt
```

Fixes:
- `e.g.` → "for example", `i.e.` → "that is", `etc.` → "etcetera"
- `U.S.` → "US", `U.K.` → "UK", `USSR` → "the Soviet Union"
- `1) foo, 2) bar` → "first, foo, second, bar"
- `ALL CAPS HEADERS` → "Title Case Headers"

### Individual Steps

```bash
articulate tts --audio-prompt ref.wav --text article.txt
articulate trim audio.wav
articulate timestamps --audio audio.wav
articulate video --audio audio.wav --timestamps ts.json
```

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Source Audio │────▶│  Find Best   │────▶│  Reference  │
│  (any file) │     │  Reference   │     │  Clip (WAV) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│  Raw Text   │────▶│  Preprocess  │            │
│             │     │  for TTS     │            │
└─────────────┘     └──────┬───────┘            │
                           │                    │
                           ▼                    ▼
                    ┌──────────────────────────────┐
                    │   Chatterbox TTS Generation  │
                    │   (sentence-by-sentence)     │
                    └──────────────┬───────────────┘
                                   │
                           ┌───────▼──────┐
                           │ Trim Silence │
                           └───────┬──────┘
                                   │
                           ┌───────▼──────────┐
                           │ Whisper Timestamps│
                           └───────┬──────────┘
                                   │
                           ┌───────▼──────┐     ┌─────────────┐
                           │ Build Video  │────▶│ final.mp4   │
                           └──────────────┘     └─────────────┘
```

## ⚙️ System Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.10–3.11 recommended (3.12 has numpy compatibility issues) |
| **ffmpeg** | Required for audio/video processing |
| **GPU** | CUDA or Apple Silicon recommended (CPU works but slow) |
| **RAM** | 8GB+ recommended |
| **Disk** | ~2GB for model weights (downloaded on first run) |

### GPU Support

| Device | Performance | Notes |
|---|---|---|
| NVIDIA CUDA | ⚡ Fastest | Any modern NVIDIA GPU |
| Apple Silicon MPS | ⚡ Fast | M1/M2/M3/M4 Macs |
| CPU | 🐢 Slow | Works but 5-10x slower |

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## 📜 License

MIT — see [LICENSE](LICENSE) for details.
