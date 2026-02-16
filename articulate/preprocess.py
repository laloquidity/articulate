"""
Preprocess article text to improve TTS naturalness.

Replaces abbreviations, fixes formatting, converts numbered lists to
ordinals, and normalizes chapter headings for better speech synthesis.

Usage:
    python -m articulate.preprocess input.txt -o output.txt
"""
import argparse
import os
import re
import sys


def preprocess_for_tts(text: str) -> str:
    """
    Transform article text for natural-sounding TTS.

    Handles:
      - Period-based abbreviations (e.g., i.e., U.S., etc.)
      - Inline and line-start numbered lists → ordinal words
      - ALL-CAPS chapter headings → title case with pauses
      - Escaped quotes, multiple spaces/newlines
    """

    # ── 1. Abbreviations with periods ────────────────────────────
    text = re.sub(r'\be\.g\.\s*,?\s*', 'for example, ', text)
    text = re.sub(r'\bi\.e\.\s*,?\s*', 'that is, ', text)

    text = re.sub(r'\betc\.\)', 'etcetera)', text)
    text = re.sub(r'\betc\.\s', 'etcetera. ', text)
    text = re.sub(r'\betc\.$', 'etcetera.', text)

    text = re.sub(r'\bU\.S\.', 'US', text)
    text = re.sub(r'\bU\.K\.', 'UK', text)
    text = re.sub(r'\bU\.N\.', 'UN', text)
    text = re.sub(r'\bD\.C\.', 'DC', text)
    text = re.sub(r'\bUSSR\b', 'the Soviet Union', text)

    # ── 2. Inline numbered lists:  "1) foo, 2) bar" ─────────────
    for digit, word in [("1", "first"), ("2", "second"), ("3", "third"),
                         ("4", "fourth"), ("5", "fifth")]:
        text = re.sub(rf'\b{digit}\)\s*', f'{word}, ', text)

    # ── 3. Line-start numbered lists: "1. Foo" → "First: Foo" ───
    ordinals = {
        '1': 'First', '2': 'Second', '3': 'Third', '4': 'Fourth',
        '5': 'Fifth', '6': 'Sixth', '7': 'Seventh', '8': 'Eighth',
    }
    for num, word in ordinals.items():
        text = re.sub(rf'^{num}\.\s+', f'{word}: ', text, flags=re.MULTILINE)

    # ── 4. ALL-CAPS chapter headings → title case + pause ────────
    def _title_case_header(match):
        title = match.group(0).strip().title()
        return f"\n\n{title}.\n\n"

    text = re.sub(r"^[A-Z][A-Z ':]+$", _title_case_header, text, flags=re.MULTILINE)

    # ── 5. Cleanup ───────────────────────────────────────────────
    text = text.replace('\\"', '"')                      # escaped quotes
    text = re.sub(r'(\d+)\s*percent', r'\1 percent', text)
    text = re.sub(r'\n{3,}', '\n\n', text)               # max 2 newlines
    text = re.sub(r'  +', ' ', text)                     # double spaces

    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess text for natural TTS output",
    )
    parser.add_argument("input", help="Input text file")
    parser.add_argument("-o", "--output", help="Output text file (default: <input>_tts.txt)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_tts{ext}"

    with open(args.input, "r", encoding="utf-8") as f:
        original = f.read()

    processed = preprocess_for_tts(original)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(processed)

    # Summary
    print(f"📝 Preprocessed for TTS")
    print(f"   Input:  {args.input} ({len(original):,} chars)")
    print(f"   Output: {args.output} ({len(processed):,} chars)")

    # Change report
    checks = [
        ('e.g.', 'for example'), ('i.e.', 'that is'),
        ('U.S.', 'US'), ('etc.', 'etcetera'), ('USSR', 'Soviet Union'),
    ]
    changes = []
    for pattern, replacement in checks:
        count = original.count(pattern) - processed.count(pattern)
        if count > 0:
            changes.append(f"   ✓ {pattern} → {replacement} ({count}×)")

    caps = len(re.findall(r"^[A-Z][A-Z ':]+$", original, re.MULTILINE))
    if caps:
        changes.append(f"   ✓ ALL-CAPS headers → title case ({caps}×)")

    nums = len(re.findall(r'^\d+\.', original, re.MULTILINE))
    if nums:
        changes.append(f"   ✓ Numbered lists → ordinal words ({nums}×)")

    if changes:
        print("\n   Changes:")
        for c in changes:
            print(c)


if __name__ == "__main__":
    main()
