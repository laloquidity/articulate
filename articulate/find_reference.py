"""
Auto-detect the best reference clip from a source audio file.

Analyzes the audio with a sliding window to find the segment with
the cleanest speech — lowest silence percentage, steadiest energy,
and least background noise — ideal for Chatterbox voice cloning.

Usage:
    python -m articulate.find_reference audio.mp3 -o reference.wav
"""
import argparse
import os
import sys
import numpy as np


def analyze_window(samples: np.ndarray, sr: int) -> dict:
    """Compute quality metrics for a candidate audio window."""
    duration = len(samples) / sr

    # 1. Silence percentage (lower = more speech)
    threshold = 10 ** (-40 / 20.0)  # -40 dB
    window_size = int(sr * 0.02)    # 20ms frames
    silent_frames = 0
    total_frames = 0

    for i in range(0, len(samples) - window_size, window_size):
        chunk = samples[i : i + window_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < threshold:
            silent_frames += 1
        total_frames += 1

    silence_pct = (silent_frames / max(total_frames, 1)) * 100

    # 2. RMS energy consistency (lower std/mean = steadier voice)
    frame_energies = []
    for i in range(0, len(samples) - window_size, window_size):
        chunk = samples[i : i + window_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > threshold:  # only non-silent frames
            frame_energies.append(rms)

    if len(frame_energies) > 1:
        energy_cv = np.std(frame_energies) / max(np.mean(frame_energies), 1e-10)
    else:
        energy_cv = 1.0  # penalize if almost no speech

    # 3. Peak-to-RMS ratio (lower = less clipping/noise)
    overall_rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))
    crest_factor = peak / max(overall_rms, 1e-10)

    # Composite score: lower is better
    # Weight silence heavily, energy consistency moderately, crest factor lightly
    score = (silence_pct * 2.0) + (energy_cv * 30.0) + (crest_factor * 2.0)

    return {
        "silence_pct": round(silence_pct, 1),
        "energy_cv": round(energy_cv, 3),
        "crest_factor": round(crest_factor, 1),
        "score": round(score, 1),
        "speech_frames": len(frame_energies),
        "total_frames": total_frames,
    }


def find_best_reference(
    audio_path: str,
    output_path: str,
    clip_duration: float = 20.0,
    stride: float = 5.0,
    top_n: int = 5,
) -> str:
    """
    Analyze source audio and extract the best segment for voice cloning.

    Args:
        audio_path: Path to source audio (any format ffmpeg supports).
        output_path: Where to save the extracted reference WAV.
        clip_duration: Target clip length in seconds (default: 20).
        stride: Window stride in seconds (default: 5).
        top_n: Number of top candidates to show in report.

    Returns:
        Path to the extracted reference clip.
    """
    import torch
    import torchaudio

    print(f"🔍 Analyzing: {audio_path}")

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    samples = waveform[0].numpy()
    total_duration = len(samples) / sr
    print(f"   Duration: {total_duration:.1f}s, Sample rate: {sr}")

    # Resample to 24kHz if needed (Chatterbox native rate)
    target_sr = 24000
    if sr != target_sr:
        print(f"   Resampling {sr} → {target_sr} Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        samples = waveform[0].numpy()
        sr = target_sr

    # Sliding window analysis
    clip_samples = int(clip_duration * sr)
    stride_samples = int(stride * sr)
    candidates = []

    print(f"   Scanning with {clip_duration}s windows, {stride}s stride...")

    for start_sample in range(0, len(samples) - clip_samples, stride_samples):
        end_sample = start_sample + clip_samples
        window = samples[start_sample:end_sample]
        metrics = analyze_window(window, sr)

        start_time = start_sample / sr
        end_time = end_sample / sr
        candidates.append({
            "start_time": start_time,
            "end_time": end_time,
            "start_sample": start_sample,
            "end_sample": end_sample,
            **metrics,
        })

    if not candidates:
        print("❌ Audio too short for analysis")
        sys.exit(1)

    # Sort by composite score (lower = better)
    candidates.sort(key=lambda c: c["score"])

    # Report top candidates
    print(f"\n{'═' * 60}")
    print(f"  Top {min(top_n, len(candidates))} candidates (lower score = better)")
    print(f"{'═' * 60}")
    print(f"  {'Rank':<5} {'Time':<16} {'Silence%':<10} {'Energy CV':<10} {'Crest':<8} {'Score':<8}")
    print(f"  {'─' * 55}")

    for i, c in enumerate(candidates[:top_n]):
        marker = " ★" if i == 0 else ""
        start_m, start_s = divmod(int(c["start_time"]), 60)
        end_m, end_s = divmod(int(c["end_time"]), 60)
        time_str = f"{start_m}:{start_s:02d}–{end_m}:{end_s:02d}"
        print(
            f"  {i+1:<5} {time_str:<16} {c['silence_pct']:<10} "
            f"{c['energy_cv']:<10} {c['crest_factor']:<8} {c['score']:<8}{marker}"
        )

    # Extract best clip
    best = candidates[0]
    clip_waveform = waveform[:, best["start_sample"] : best["end_sample"]]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, clip_waveform, sr)

    start_m, start_s = divmod(int(best["start_time"]), 60)
    end_m, end_s = divmod(int(best["end_time"]), 60)
    print(f"\n✅ Best clip: {start_m}:{start_s:02d}–{end_m}:{end_s:02d}")
    print(f"   Silence:  {best['silence_pct']}%")
    print(f"   Score:    {best['score']}")
    print(f"   Saved:    {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Find the best voice reference clip from a source audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a podcast and extract the best 20s clip
  python -m solar_constellation.find_reference podcast.mp3

  # Extract a 15s clip with tighter analysis
  python -m solar_constellation.find_reference interview.wav -d 15 -s 3 -o my_voice.wav
        """,
    )
    parser.add_argument("audio", help="Source audio file (any format ffmpeg supports)")
    parser.add_argument(
        "-o", "--output",
        default="reference.wav",
        help="Output path for reference clip (default: reference.wav)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=20.0,
        help="Target clip duration in seconds (default: 20)",
    )
    parser.add_argument(
        "-s", "--stride",
        type=float,
        default=5.0,
        help="Window stride in seconds (default: 5)",
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=5,
        help="Number of top candidates to show (default: 5)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)

    find_best_reference(
        args.audio,
        args.output,
        clip_duration=args.duration,
        stride=args.stride,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
