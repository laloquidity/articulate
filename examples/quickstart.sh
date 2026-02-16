#!/usr/bin/env bash
# Quick Start — Solar Constellation
#
# Usage:
#   ./quickstart.sh <path-to-voice-audio> [article.txt]
#
# Examples:
#   ./quickstart.sh ~/Downloads/podcast.mp3
#   ./quickstart.sh ~/Downloads/interview.wav my_article.txt

set -euo pipefail

VOICE="${1:?Usage: ./quickstart.sh <voice-audio> [text-file]}"
TEXT="${2:-examples/sample_article.txt}"

echo "🌌 Solar Constellation — Quick Start"
echo "   Voice: $VOICE"
echo "   Text:  $TEXT"
echo ""

articulate run \
    --voice "$VOICE" \
    --text "$TEXT" \
    --output-dir output \
    --aspect-ratio 16:9 \
    --speed 0.85
