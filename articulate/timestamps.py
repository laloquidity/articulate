"""
Extract word-level timestamps from audio using Whisper.

Uses whisper-timestamped for accurate word-level timing data,
which drives the caption highlighting in the final video.

Usage:
    python -m articulate.timestamps --audio audio.wav -o timestamps.json
"""
import argparse
import json
import os
import sys


def extract_timestamps(
    audio_path: str,
    output_path: str,
    model_size: str = "base",
) -> str:
    """Extract word-level timestamps using whisper-timestamped.

    Args:
        audio_path: Path to audio WAV file.
        output_path: Output JSON path.
        model_size: Whisper model size (tiny/base/small/medium/large).

    Returns:
        Path to the timestamps JSON file.
    """
    import whisper_timestamped as whisper

    print(f"🎧 Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size, device="cpu")
    print("✓ Model loaded")

    print(f"🔍 Transcribing: {audio_path}")
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True)

    # Word-level data
    words = []
    for segment in result["segments"]:
        for word_data in segment.get("words", []):
            entry = {
                "word": word_data["text"].strip(),
                "start": round(word_data["start"], 3),
                "end": round(word_data["end"], 3),
                "confidence": round(word_data.get("confidence", 1.0), 3),
            }
            if entry["word"] and not entry["word"].startswith("[*"):
                words.append(entry)

    # Segment-level data (fallback)
    segments = [
        {
            "text": seg["text"].strip(),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
        }
        for seg in result["segments"]
    ]

    output_data = {
        "audio_path": audio_path,
        "total_duration": round(segments[-1]["end"], 3) if segments else 0,
        "word_count": len(words),
        "segment_count": len(segments),
        "words": words,
        "segments": segments,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 50}")
    print(f"✓ Extracted {len(words)} words across {len(segments)} segments")
    print(f"  Total duration: {output_data['total_duration']:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'═' * 50}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract word-level timestamps from audio",
    )
    parser.add_argument("--audio", required=True, help="Path to audio WAV file")
    parser.add_argument("-o", "--output", default="output/timestamps.json",
                        help="Output JSON path (default: output/timestamps.json)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)

    extract_timestamps(args.audio, args.output, args.model)


if __name__ == "__main__":
    main()
