#!/bin/bash
# Usage: ./archive_fit.sh <run_name>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/past_fits/$1"

if [ -d "$DEST" ]; then
    echo "Error: '$DEST' already exists. Choose a different name."
    exit 1
fi

mkdir -p "$DEST"

find "$SCRIPT_DIR/outfiles" -name "*tempfile*" -delete

mv "$SCRIPT_DIR/noise_files" "$DEST/noise_files"
mv "$SCRIPT_DIR/outfiles" "$DEST/outfiles"

mkdir -p "$SCRIPT_DIR/noise_files"
mkdir -p "$SCRIPT_DIR/outfiles"

rm -f "$SCRIPT_DIR/slurm_out"/*

echo "Archived to past_fits/$1 and cleared slurm_out."
