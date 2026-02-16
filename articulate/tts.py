"""
Generate TTS audio from text using Chatterbox voice cloning.

Splits text into sentences, generates audio for each segment using a
reference voice, and concatenates with natural pauses between segments.

Usage:
    python -m articulate.tts --audio-prompt ref.wav --text article.txt
"""
import argparse
import os
import re
import sys
import time


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS generation.

    - Splits on sentence-ending punctuation
    - Merges very short fragments (<10 chars) with the previous sentence
    - Splits very long sentences (>300 chars) at commas/semicolons
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]

    # Merge short fragments
    merged = []
    for s in sentences:
        if merged and len(s) < 10:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    # Split long sentences at natural break points
    final = []
    for s in merged:
        if len(s) > 300:
            parts = re.split(r'(?<=[,;])\s+', s)
            chunk = ""
            for p in parts:
                if len(chunk) + len(p) > 250 and chunk:
                    final.append(chunk.strip())
                    chunk = p
                else:
                    chunk = (chunk + " " + p).strip()
            if chunk:
                final.append(chunk)
        else:
            final.append(s)

    return final


def generate_tts(
    audio_prompt: str,
    text_file: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    sentence_gap: float = 0.15,
) -> str:
    """Generate TTS audio from text file using Chatterbox voice cloning.

    Args:
        audio_prompt: Path to reference WAV for voice cloning (10-30s).
        text_file: Path to text file with content to narrate.
        output_path: Output WAV path.
        exaggeration: Emotion exaggeration, 0=neutral to 1=expressive.
        cfg_weight: Voice similarity weight (higher=closer to reference).
        sentence_gap: Silence between sentences in seconds.

    Returns:
        Path to the generated audio file.
    """
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    from .config import get_device

    # Read text
    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"📄 Text loaded: {len(full_text):,} characters")

    sentences = split_into_sentences(full_text)
    print(f"📝 Split into {len(sentences)} segments")

    # Load model
    device = get_device()
    print(f"🔧 Loading Chatterbox on {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"✓ Model loaded (sample rate: {model.sr})")

    # Generate each segment
    all_wavs = []
    total_start = time.time()

    for i, sentence in enumerate(sentences):
        seg_start = time.time()
        preview = sentence[:60] + ("..." if len(sentence) > 60 else "")
        print(f"  🎙️  [{i+1}/{len(sentences)}] {preview}")

        try:
            wav = model.generate(
                sentence,
                audio_prompt_path=audio_prompt,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            all_wavs.append(wav)

            duration = wav.shape[1] / model.sr
            elapsed = time.time() - seg_start
            print(f"         ✓ {duration:.1f}s audio in {elapsed:.1f}s")

        except Exception as e:
            print(f"         ✗ Error: {e}")
            silence = torch.zeros(1, int(model.sr * 0.5))
            all_wavs.append(silence)

    # Concatenate with gaps
    silence_gap = torch.zeros(1, int(model.sr * sentence_gap))
    combined_parts = []
    for i, wav in enumerate(all_wavs):
        combined_parts.append(wav)
        if i < len(all_wavs) - 1:
            combined_parts.append(silence_gap)

    combined = torch.cat(combined_parts, dim=1)
    total_duration = combined.shape[1] / model.sr
    total_elapsed = time.time() - total_start

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torchaudio.save(output_path, combined, model.sr)

    print(f"\n{'═' * 50}")
    print(f"✓ Generated {total_duration:.1f}s of audio in {total_elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Sample rate: {model.sr}")
    print(f"{'═' * 50}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio with Chatterbox voice cloning",
    )
    parser.add_argument("--audio-prompt", required=True,
                        help="Reference audio for voice cloning (WAV, 10-30s)")
    parser.add_argument("--text", required=True,
                        help="Text file with content to narrate")
    parser.add_argument("-o", "--output", default="output/audio.wav",
                        help="Output WAV path (default: output/audio.wav)")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Emotion exaggeration 0-1 (default: 0.5)")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="Voice similarity weight (default: 0.5)")
    parser.add_argument("--sentence-gap", type=float, default=0.15,
                        help="Silence between sentences in seconds (default: 0.15)")

    args = parser.parse_args()

    if not os.path.exists(args.audio_prompt):
        print(f"❌ Reference audio not found: {args.audio_prompt}")
        sys.exit(1)
    if not os.path.exists(args.text):
        print(f"❌ Text file not found: {args.text}")
        sys.exit(1)

    generate_tts(
        args.audio_prompt, args.text, args.output,
        args.exaggeration, args.cfg_weight, args.sentence_gap,
    )


if __name__ == "__main__":
    main()
