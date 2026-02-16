"""
Pipeline orchestrator — chains all steps into a single command.

Steps:
  1. (Optional) Find best reference clip from source audio
  2. (Optional) Preprocess text for TTS
  3. Generate TTS audio with Chatterbox voice cloning
  4. Trim excessive silence
  5. Extract word-level timestamps with Whisper
  6. Build captioned video

Usage:
    articulate run --voice audio.mp3 --text article.txt
"""
import os
import sys
import time


def banner(msg: str):
    width = max(len(msg) + 4, 50)
    print(f"\n{'╔' + '═' * width + '╗'}")
    print(f"{'║'} {msg.ljust(width - 1)}{'║'}")
    print(f"{'╚' + '═' * width + '╝'}\n")


def run_pipeline(
    voice_source: str,
    text_file: str,
    output_dir: str = "output",
    aspect_ratio: str = "16:9",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    whisper_model: str = "base",
    words_per_group: int = 5,
    reference_clip: str | None = None,
    skip_preprocess: bool = False,
    skip_tts: bool = False,
    skip_trim: bool = False,
    skip_timestamps: bool = False,
    speed: float | None = None,
) -> dict:
    """Run the full article-to-video pipeline.

    Args:
        voice_source: Path to source audio for voice cloning.
                      Can be a full recording (auto-finds best clip) or a
                      pre-extracted reference WAV.
        text_file: Path to text file with content to narrate.
        output_dir: Output directory for all generated files.
        aspect_ratio: Video aspect ratio (9:16, 16:9, 1:1).
        exaggeration: TTS emotion 0-1.
        cfg_weight: TTS voice similarity weight.
        whisper_model: Whisper model size.
        words_per_group: Caption words shown at once.
        reference_clip: Path to pre-extracted reference clip (skips find-ref).
        skip_preprocess: Skip text preprocessing.
        skip_tts: Skip TTS (use existing audio).
        skip_trim: Skip silence trimming.
        skip_timestamps: Skip Whisper (use existing timestamps).
        speed: Optional playback speed multiplier (e.g., 0.85 for slower).

    Returns:
        Dict with paths to all output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    ref_path = os.path.join(output_dir, "reference.wav")
    text_processed = os.path.join(output_dir, "text_processed.txt")
    audio_path = os.path.join(output_dir, "audio.wav")
    audio_trimmed = os.path.join(output_dir, "audio_trimmed.wav")
    timestamps_path = os.path.join(output_dir, "timestamps.json")
    video_path = os.path.join(output_dir, "final_video.mp4")

    pipeline_start = time.time()

    # ── Step 1: Find reference clip ──────────────────────────────
    if reference_clip:
        ref_path = reference_clip
        print(f"\n⏭  Using provided reference: {ref_path}")
    else:
        banner("Step 1: Finding Best Voice Reference Clip")
        from .find_reference import find_best_reference
        find_best_reference(voice_source, ref_path)

    if not os.path.exists(ref_path):
        print(f"❌ Reference clip not found: {ref_path}")
        sys.exit(1)

    # ── Step 2: Preprocess text ──────────────────────────────────
    if not skip_preprocess:
        banner("Step 2: Preprocessing Text for TTS")
        from .preprocess import preprocess_for_tts

        with open(text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        processed = preprocess_for_tts(raw_text)
        with open(text_processed, "w", encoding="utf-8") as f:
            f.write(processed)

        print(f"📝 {len(raw_text):,} → {len(processed):,} chars")
        tts_text = text_processed
    else:
        print(f"\n⏭  Skipping preprocessing")
        tts_text = text_file

    # ── Step 3: Generate TTS ─────────────────────────────────────
    if not skip_tts:
        banner("Step 3: Generating TTS Audio (Chatterbox)")
        from .tts import generate_tts
        generate_tts(ref_path, tts_text, audio_path, exaggeration, cfg_weight)
    else:
        print(f"\n⏭  Skipping TTS (using existing audio)")
        if not os.path.exists(audio_path):
            print(f"❌ Audio not found: {audio_path}")
            sys.exit(1)

    # ── Step 4: Trim silence ─────────────────────────────────────
    if not skip_trim:
        banner("Step 4: Trimming Excess Silence")
        from .silence import trim_silence
        trim_silence(audio_path, audio_trimmed)
        final_audio = audio_trimmed
    else:
        print(f"\n⏭  Skipping silence trimming")
        final_audio = audio_path

    # ── Step 5: Extract timestamps ───────────────────────────────
    if not skip_timestamps:
        banner("Step 5: Extracting Word Timestamps (Whisper)")
        from .timestamps import extract_timestamps
        extract_timestamps(final_audio, timestamps_path, whisper_model)
    else:
        print(f"\n⏭  Skipping timestamps (using existing JSON)")
        if not os.path.exists(timestamps_path):
            print(f"❌ Timestamps not found: {timestamps_path}")
            sys.exit(1)

    # ── Step 6: Build video ──────────────────────────────────────
    banner("Step 6: Building Captioned Video")
    from .video import build_video
    build_video(final_audio, timestamps_path, video_path,
                aspect_ratio, words_per_group)

    # ── Step 7: Speed adjustment (optional) ──────────────────────
    final_output = video_path
    if speed and speed != 1.0:
        banner(f"Step 7: Adjusting Speed to {speed}x")
        speed_path = os.path.join(output_dir, f"final_video_{speed}x.mp4")
        pts_factor = 1.0 / speed
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter_complex",
            f"[0:v]setpts={pts_factor}*PTS[v];[0:a]atempo={speed}[a]",
            "-map", "[v]", "-map", "[a]",
            speed_path,
        ]
        print(f"   Running: ffmpeg speed adjustment...")
        subprocess.run(cmd, capture_output=True, check=True)
        file_size = os.path.getsize(speed_path) / (1024 * 1024)
        print(f"   ✓ {speed_path} ({file_size:.1f} MB)")
        final_output = speed_path

    # ── Done ─────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print(f"\n{'🎉' * 20}")
    banner("Pipeline Complete!")
    print(f"  📁 Output:       {output_dir}/")
    print(f"  🎤 Reference:    {ref_path}")
    print(f"  🔊 Audio:        {final_audio}")
    print(f"  📝 Timestamps:   {timestamps_path}")
    print(f"  🎬 Video:        {final_output}")
    print(f"  ⏱️  Total time:   {minutes}m {seconds}s")
    print(f"\n  Ready to post! 🚀")

    return {
        "reference": ref_path,
        "audio": final_audio,
        "timestamps": timestamps_path,
        "video": final_output,
        "output_dir": output_dir,
    }
