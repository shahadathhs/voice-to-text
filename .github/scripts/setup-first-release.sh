#!/bin/bash
set -e

echo "Setting up first release for voice-to-text..."
echo ""
echo "This script will create an initial v1.0.0 tag to bootstrap semantic-release."
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if there are existing tags
if git rev-parse v1.0.0 >/dev/null 2>&1; then
    echo "✓ Tag v1.0.0 already exists"
    echo "Current setup is complete!"
    exit 0
fi

echo "Creating initial v1.0.0 tag..."
git tag -a v1.0.0 -m "Initial release v1.0.0

- FastAPI voice-to-text API with OpenAI Whisper
- Multiple audio formats support
- Translation and speaker diarization
- CLI and API interfaces
- Comprehensive documentation"

echo "✓ Tag v1.0.0 created"
echo ""
echo "Next steps:"
echo "  1. Push the tag to GitHub:"
echo "     git push origin v1.0.0"
echo ""
echo "  2. The next semantic-release run will work from this baseline"
echo ""
echo "Note: Future releases will be automated by GitHub Actions"
