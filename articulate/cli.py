"""
CLI entry point for Articulate.

Provides subcommands for the full pipeline and individual steps:
    articulate run          Full pipeline
    articulate find-ref     Find best reference clip
    articulate preprocess   Preprocess text
    articulate tts          Generate TTS audio
    articulate trim         Trim silence
    articulate timestamps   Extract timestamps
    articulate video        Build captioned video
"""
import argparse
import os
import sys

from . import __version__


def cmd_run(args):
    """Run the full pipeline."""
    from .pipeline import run_pipeline
    run_pipeline(
        voice_source=args.voice,
        text_file=args.text,
        output_dir=args.output_dir,
        aspect_ratio=args.aspect_ratio,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        whisper_model=args.whisper_model,
        words_per_group=args.words_per_group,
        reference_clip=args.reference,
        skip_preprocess=args.skip_preprocess,
        skip_tts=args.skip_tts,
        skip_trim=args.skip_trim,
        skip_timestamps=args.skip_timestamps,
        speed=args.speed,
    )


def cmd_find_ref(args):
    """Find best reference clip from source audio."""
    from .find_reference import find_best_reference
    find_best_reference(
        args.audio, args.output,
        clip_duration=args.duration,
        stride=args.stride,
        top_n=args.top_n,
    )


def cmd_preprocess(args):
    """Preprocess text for TTS."""
    from .preprocess import main as preprocess_main
    sys.argv = ["preprocess", args.input]
    if args.output:
        sys.argv.extend(["-o", args.output])
    preprocess_main()


def cmd_tts(args):
    """Generate TTS audio."""
    from .tts import generate_tts
    generate_tts(
        args.audio_prompt, args.text, args.output,
        args.exaggeration, args.cfg_weight, args.sentence_gap,
    )


def cmd_trim(args):
    """Trim silence."""
    from .silence import trim_silence
    output = args.output
    if not output:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_trimmed{ext}"
    trim_silence(args.input, output, args.max_silence, args.threshold)


def cmd_timestamps(args):
    """Extract timestamps."""
    from .timestamps import extract_timestamps
    extract_timestamps(args.audio, args.output, args.model)


def cmd_video(args):
    """Build video."""
    from .video import build_video
    build_video(args.audio, args.timestamps, args.output,
                args.aspect_ratio, args.words_per_group)


def main():
    parser = argparse.ArgumentParser(
        prog="articulate",
        description="🗣️ Turn any text into a voice-cloned narrated video with word-level captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: audio file + text → video
  articulate run --voice podcast.mp3 --text article.txt

  # Find the best voice clip from a recording
  articulate find-ref interview.wav -o my_voice.wav

  # Just preprocess text (fix abbreviations, headers, etc.)
  articulate preprocess article.txt -o article_clean.txt
        """,
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ───────────────────────────────────────────────────────
    p_run = subparsers.add_parser(
        "run", help="Run the full article-to-video pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  articulate run --voice podcast.mp3 --text article.txt
  articulate run --voice podcast.mp3 --text article.txt --speed 0.85
  articulate run --voice podcast.mp3 --text article.txt --aspect-ratio 9:16
        """,
    )
    p_run.add_argument("--voice", required=True,
                       help="Source audio for voice cloning (any format)")
    p_run.add_argument("--text", required=True,
                       help="Text file to narrate")
    p_run.add_argument("--reference",
                       help="Pre-extracted reference clip (skips auto-detection)")
    p_run.add_argument("-o", "--output-dir", default="output",
                       help="Output directory (default: output/)")
    p_run.add_argument("--aspect-ratio", default="16:9",
                       choices=["9:16", "16:9", "1:1"],
                       help="Video aspect ratio (default: 16:9)")
    p_run.add_argument("--speed", type=float, default=None,
                       help="Playback speed, e.g. 0.85 for slower (default: 1.0)")
    p_run.add_argument("--exaggeration", type=float, default=0.5,
                       help="Emotion exaggeration 0-1 (default: 0.5)")
    p_run.add_argument("--cfg-weight", type=float, default=0.5,
                       help="Voice similarity weight (default: 0.5)")
    p_run.add_argument("--whisper-model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    p_run.add_argument("--words-per-group", type=int, default=5,
                       help="Caption words shown at once (default: 5)")
    p_run.add_argument("--skip-preprocess", action="store_true",
                       help="Skip text preprocessing")
    p_run.add_argument("--skip-tts", action="store_true",
                       help="Skip TTS (use existing audio)")
    p_run.add_argument("--skip-trim", action="store_true",
                       help="Skip silence trimming")
    p_run.add_argument("--skip-timestamps", action="store_true",
                       help="Skip timestamp extraction")
    p_run.set_defaults(func=cmd_run)

    # ── find-ref ─────────────────────────────────────────────────
    p_ref = subparsers.add_parser(
        "find-ref", help="Find the best voice reference clip from audio",
    )
    p_ref.add_argument("audio", help="Source audio file")
    p_ref.add_argument("-o", "--output", default="reference.wav",
                       help="Output clip path (default: reference.wav)")
    p_ref.add_argument("-d", "--duration", type=float, default=20.0,
                       help="Target clip duration in seconds (default: 20)")
    p_ref.add_argument("-s", "--stride", type=float, default=5.0,
                       help="Analysis window stride (default: 5)")
    p_ref.add_argument("-n", "--top-n", type=int, default=5,
                       help="Top candidates to show (default: 5)")
    p_ref.set_defaults(func=cmd_find_ref)

    # ── preprocess ───────────────────────────────────────────────
    p_pre = subparsers.add_parser(
        "preprocess", help="Preprocess text for natural TTS",
    )
    p_pre.add_argument("input", help="Input text file")
    p_pre.add_argument("-o", "--output", help="Output text file")
    p_pre.set_defaults(func=cmd_preprocess)

    # ── tts ──────────────────────────────────────────────────────
    p_tts = subparsers.add_parser(
        "tts", help="Generate TTS audio with voice cloning",
    )
    p_tts.add_argument("--audio-prompt", required=True,
                       help="Reference audio WAV (10-30s)")
    p_tts.add_argument("--text", required=True,
                       help="Text file to narrate")
    p_tts.add_argument("-o", "--output", default="output/audio.wav",
                       help="Output WAV path")
    p_tts.add_argument("--exaggeration", type=float, default=0.5)
    p_tts.add_argument("--cfg-weight", type=float, default=0.5)
    p_tts.add_argument("--sentence-gap", type=float, default=0.15)
    p_tts.set_defaults(func=cmd_tts)

    # ── trim ─────────────────────────────────────────────────────
    p_trim = subparsers.add_parser(
        "trim", help="Trim excessive silence from audio",
    )
    p_trim.add_argument("input", help="Input audio file")
    p_trim.add_argument("-o", "--output", help="Output audio file")
    p_trim.add_argument("--max-silence", type=int, default=150,
                        help="Max silence in ms (default: 150)")
    p_trim.add_argument("--threshold", type=int, default=-40,
                        help="Silence threshold in dB (default: -40)")
    p_trim.set_defaults(func=cmd_trim)

    # ── timestamps ───────────────────────────────────────────────
    p_ts = subparsers.add_parser(
        "timestamps", help="Extract word timestamps from audio",
    )
    p_ts.add_argument("--audio", required=True, help="Audio WAV file")
    p_ts.add_argument("-o", "--output", default="output/timestamps.json")
    p_ts.add_argument("--model", default="base",
                      choices=["tiny", "base", "small", "medium", "large"])
    p_ts.set_defaults(func=cmd_timestamps)

    # ── video ────────────────────────────────────────────────────
    p_vid = subparsers.add_parser(
        "video", help="Build captioned video from audio + timestamps",
    )
    p_vid.add_argument("--audio", required=True, help="Audio WAV file")
    p_vid.add_argument("--timestamps", required=True, help="Timestamps JSON")
    p_vid.add_argument("-o", "--output", default="output/final_video.mp4")
    p_vid.add_argument("--aspect-ratio", default="16:9",
                       choices=["9:16", "16:9", "1:1"])
    p_vid.add_argument("--words-per-group", type=int, default=5)
    p_vid.set_defaults(func=cmd_video)

    # ── Parse and dispatch ───────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
