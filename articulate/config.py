"""
Shared configuration, constants, and device detection.
"""
import os
import sys


# ─── Device Detection ────────────────────────────────────────────────

def get_device() -> str:
    """Get best available PyTorch device (cuda > mps > cpu)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Audio Defaults ──────────────────────────────────────────────────

SAMPLE_RATE = 24000              # Chatterbox native sample rate
SILENCE_THRESHOLD_DB = -40       # dB threshold for silence detection
MAX_SILENCE_MS = 150             # Max allowed silence gap in ms
INTER_SENTENCE_GAP_S = 0.15     # Silence inserted between TTS segments


# ─── Video Defaults ──────────────────────────────────────────────────

RESOLUTIONS = {
    "9:16": (1080, 1920),        # TikTok, Reels, Shorts (portrait)
    "16:9": (1920, 1080),        # YouTube, X (landscape)
    "1:1":  (1080, 1080),        # Instagram square
}

FONT_SIZE = 64
FONT_COLOR = (255, 255, 255)     # White
HIGHLIGHT_COLOR = (0, 200, 255)  # Cyan for current word
BG_COLOR = (0, 0, 0)            # Black background
WORDS_PER_GROUP = 5
FPS = 30

# Cross-platform font search paths
FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/SFPro-Bold.otf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


# ─── Reference Clip Defaults ────────────────────────────────────────

REF_CLIP_DURATION_S = 20         # Target reference clip length
REF_CLIP_MIN_S = 10              # Minimum acceptable clip
REF_CLIP_MAX_S = 30              # Maximum clip length
REF_WINDOW_STRIDE_S = 5          # Sliding window stride for analysis


# ─── TTS Defaults ────────────────────────────────────────────────────

TTS_EXAGGERATION = 0.5           # Emotion exaggeration (0=neutral, 1=max)
TTS_CFG_WEIGHT = 0.5             # Voice similarity weight


def get_font(size: int = FONT_SIZE):
    """Get a clean font. Searches system paths, falls back to default."""
    from PIL import ImageFont
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()
