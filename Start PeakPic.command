#!/bin/bash
# Start PeakPic — macOS/Linux launcher, the Mac equivalent of "Start PeakPic.bat".
#
# Double-clickable from Finder (macOS treats a +x .command file as a runnable
# script, opened in Terminal) or runnable directly from a terminal:
#   ./"Start PeakPic.command"
#
# Unlike the Windows .bat (which hardcodes C:\Code Projects\PickLikeMe),
# this resolves the project directory from the script's own location, so it
# keeps working no matter where the repository was cloned.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "No virtual environment found at .venv/ — see docs/Developer_Onboarding_Mac.md Section 5" >&2
    exit 1
fi

source .venv/bin/activate
python -m picklikeme.desktop
