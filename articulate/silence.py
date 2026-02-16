"""
Trim excessive silence from TTS-generated audio.

Detects silent regions and shortens any gap longer than the configured
maximum down to that maximum, preserving natural sentence pauses.

Usage:
    python -m articulate.silence --input audio.wav -o trimmed.wav
"""
import argparse
import os
import sys
import numpy as np


def detect_silence_regions(
    samples: np.ndarray,
    sr: int,
    threshold_db: int = -40,
    min_silence_ms: int = 100,
) -> list[tuple[int, int, float]]:
    """Find contiguous silent regions in audio.

    Returns:
        List of (start_sample, end_sample, duration_ms) tuples.
    """
    threshold = 10 ** (threshold_db / 20.0)
    window_size = int(sr * 0.02)  # 20ms frames

    is_silent = []
    for i in range(0, len(samples) - window_size, window_size):
        chunk = samples[i : i + window_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        is_silent.append(rms < threshold)

    regions = []
    in_silence = False
    start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i * window_size
            in_silence = True
        elif not silent and in_silence:
            end = i * window_size
            duration_ms = (end - start) / sr * 1000
            if duration_ms >= min_silence_ms:
                regions.append((start, end, duration_ms))
            in_silence = False

    if in_silence:
        end = len(samples)
        duration_ms = (end - start) / sr * 1000
        if duration_ms >= min_silence_ms:
            regions.append((start, end, duration_ms))

    return regions


def trim_silence(
    input_path: str,
    output_path: str,
    max_silence_ms: int = 150,
    threshold_db: int = -40,
) -> str:
    """Trim silence regions longer than max_silence_ms.

    Args:
        input_path: Input audio file.
        output_path: Output audio file.
        max_silence_ms: Maximum allowed silence in milliseconds.
        threshold_db: Silence threshold in dB.

    Returns:
        Path to the trimmed audio file.
    """
    import torch
    import torchaudio

    print(f"🔍 Loading audio: {input_path}")
    waveform, sr = torchaudio.load(input_path)
    samples = waveform[0].numpy()

    print(f"   Sample rate: {sr}, Duration: {len(samples)/sr:.1f}s")

    regions = detect_silence_regions(samples, sr, threshold_db=threshold_db)
    print(f"   Found {len(regions)} silent regions ≥ 100ms")

    long_regions = [r for r in regions if r[2] > max_silence_ms]
    print(f"   Regions exceeding {max_silence_ms}ms: {len(long_regions)}")
    if long_regions:
        durations = [r[2] for r in long_regions]
        print(f"   Duration range: {min(durations):.0f}ms – {max(durations):.0f}ms")
        print(f"   Average: {np.mean(durations):.0f}ms")

    # Build trimmed audio
    max_silence_samples = int(sr * max_silence_ms / 1000)
    output_parts = []
    prev_end = 0
    trimmed_total = 0

    for start, end, duration_ms in regions:
        output_parts.append(samples[prev_end:start])
        if duration_ms > max_silence_ms:
            output_parts.append(samples[start : start + max_silence_samples])
            trimmed_total += (end - start) - max_silence_samples
        else:
            output_parts.append(samples[start:end])
        prev_end = end

    output_parts.append(samples[prev_end:])

    new_samples = np.concatenate(output_parts)
    new_waveform = torch.from_numpy(new_samples).unsqueeze(0)

    old_duration = len(samples) / sr
    new_duration = len(new_samples) / sr

    print(f"\n✂️  Trimmed {trimmed_total/sr:.1f}s of excess silence")
    print(f"   Original: {old_duration:.1f}s → New: {new_duration:.1f}s")
    print(f"   Reduction: {(old_duration - new_duration):.1f}s "
          f"({(1 - new_duration/old_duration)*100:.1f}%)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, new_waveform, sr)
    print(f"   Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Trim excessive silence from TTS audio",
    )
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file (default: <input>_trimmed.wav)")
    parser.add_argument("--max-silence", type=int, default=150,
                        help="Max allowed silence in ms (default: 150)")
    parser.add_argument("--threshold", type=int, default=-40,
                        help="Silence threshold in dB (default: -40)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Audio file not found: {args.input}")
        sys.exit(1)

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_trimmed{ext}"

    trim_silence(args.input, args.output, args.max_silence, args.threshold)


if __name__ == "__main__":
    main()
