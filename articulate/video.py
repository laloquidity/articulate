"""
Build a captioned video with synced word-level highlighting.

Renders text captions over a black background with the current word
highlighted in cyan, driven by word-level timestamps from Whisper.

Usage:
    python -m articulate.video --audio audio.wav --timestamps ts.json
"""
import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from .config import (
    RESOLUTIONS, FONT_SIZE, FONT_COLOR, HIGHLIGHT_COLOR,
    BG_COLOR, WORDS_PER_GROUP, FPS, get_font,
)


def render_text_frame(
    width: int,
    height: int,
    words: list[dict],
    current_time: float,
    font,
    words_per_group: int = WORDS_PER_GROUP,
) -> np.ndarray:
    """Render a single frame with the current word group highlighted."""
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    if not words:
        return np.array(img)

    # Find current word index
    current_word_idx = 0
    for i, w in enumerate(words):
        if w["start"] <= current_time <= w["end"]:
            current_word_idx = i
            break
        elif w["start"] > current_time:
            current_word_idx = max(0, i - 1)
            break
    else:
        current_word_idx = len(words) - 1

    # Select word group
    group_start = (current_word_idx // words_per_group) * words_per_group
    group_end = min(group_start + words_per_group, len(words))
    group_words = words[group_start:group_end]

    if not group_words:
        return np.array(img)

    # Layout: wrap words to fit within margins
    margin = 80
    max_width = width - margin * 2

    lines = []
    current_line = []
    current_line_width = 0

    for w in group_words:
        word_text = w["word"] + " "
        bbox = draw.textbbox((0, 0), word_text, font=font)
        word_width = bbox[2] - bbox[0]

        if current_line_width + word_width > max_width and current_line:
            lines.append(current_line)
            current_line = [w]
            current_line_width = word_width
        else:
            current_line.append(w)
            current_line_width += word_width

    if current_line:
        lines.append(current_line)

    # Center text block vertically
    line_height = FONT_SIZE + 20
    total_height = len(lines) * line_height
    start_y = (height - total_height) // 2

    # Draw each line with per-word highlighting
    for line_idx, line_words in enumerate(lines):
        line_text = " ".join(w["word"] for w in line_words)
        bbox = draw.textbbox((0, 0), line_text, font=font)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) // 2
        y = start_y + line_idx * line_height

        for w in line_words:
            word_text = w["word"]
            is_current = w["start"] <= current_time <= w["end"]
            color = HIGHLIGHT_COLOR if is_current else FONT_COLOR

            draw.text((x, y), word_text, fill=color, font=font)
            bbox = draw.textbbox((0, 0), word_text + " ", font=font)
            x += bbox[2] - bbox[0]

    return np.array(img)


def build_video(
    audio_path: str,
    timestamps_path: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    words_per_group: int = WORDS_PER_GROUP,
) -> str:
    """Build the final captioned video.

    Args:
        audio_path: Path to narration audio WAV.
        timestamps_path: Path to word timestamps JSON.
        output_path: Output MP4 path.
        aspect_ratio: Video format — "9:16", "16:9", or "1:1".
        words_per_group: Number of words shown at once.

    Returns:
        Path to the output video file.
    """
    from moviepy import AudioFileClip, VideoClip

    with open(timestamps_path, "r") as f:
        data = json.load(f)

    words = data["words"]
    total_duration = data["total_duration"]
    width, height = RESOLUTIONS[aspect_ratio]
    font = get_font(FONT_SIZE)

    print(f"📐 Resolution: {aspect_ratio} → {width}×{height}")
    print(f"🎬 Duration: {total_duration:.1f}s")
    print(f"📝 Words: {len(words)}")

    def make_frame(t):
        return render_text_frame(width, height, words, t, font, words_per_group)

    print("🎞️  Rendering video frames...")
    video = VideoClip(make_frame, duration=total_duration + 1.0).with_fps(FPS)

    print("🔊 Adding audio track...")
    audio = AudioFileClip(audio_path)
    video = video.with_audio(audio)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"💾 Exporting to {output_path}...")
    video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger="bar",
    )

    audio.close()
    video.close()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'═' * 50}")
    print(f"✓ Video exported: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Resolution: {width}×{height}")
    print(f"  Duration: {total_duration:.1f}s")
    print(f"{'═' * 50}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a captioned video from audio + timestamps",
    )
    parser.add_argument("--audio", required=True, help="Path to audio WAV file")
    parser.add_argument("--timestamps", required=True, help="Path to timestamps JSON")
    parser.add_argument("-o", "--output", default="output/final_video.mp4",
                        help="Output MP4 path (default: output/final_video.mp4)")
    parser.add_argument("--aspect-ratio", default="16:9",
                        choices=["9:16", "16:9", "1:1"],
                        help="Video aspect ratio (default: 16:9)")
    parser.add_argument("--words-per-group", type=int, default=5,
                        help="Words shown at once (default: 5)")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Audio: {args.audio} not found")
        sys.exit(1)
    if not os.path.exists(args.timestamps):
        print(f"❌ Timestamps: {args.timestamps} not found")
        sys.exit(1)

    build_video(args.audio, args.timestamps, args.output,
                args.aspect_ratio, args.words_per_group)


if __name__ == "__main__":
    main()
