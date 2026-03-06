#!/bin/bash
# Build an essay from Markdown to HTML using pandoc.
# Usage: ./build.sh <essay-folder>
# Example: ./build.sh sample-essay

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ESSAY_DIR="$SCRIPT_DIR/$1"

if [ -z "$1" ] || [ ! -d "$ESSAY_DIR" ]; then
    echo "Usage: ./build.sh <essay-folder>"
    echo "Example: ./build.sh sample-essay"
    exit 1
fi

if [ ! -f "$ESSAY_DIR/essay.md" ]; then
    echo "Error: $ESSAY_DIR/essay.md not found"
    exit 1
fi

pandoc "$ESSAY_DIR/essay.md" \
    -o "$ESSAY_DIR/index.html" \
    --template="$SCRIPT_DIR/template.html" \
    --katex \
    --syntax-highlighting=kate \
    --standalone

echo "Built: $ESSAY_DIR/index.html"
