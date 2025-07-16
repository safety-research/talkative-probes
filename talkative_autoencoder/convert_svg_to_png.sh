#!/usr/bin/env bash

# Convert SVG to PNG using the Python script
# Usage: ./convert_svg_to_png.sh <svg_file> [png_file]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if svg file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <svg_file> [png_file]"
    echo "  svg_file: Path to the SVG file to convert"
    echo "  png_file: Optional output PNG path (defaults to same name with .png extension)"
    exit 1
fi

SVG_FILE="$1"
PNG_FILE="${2:-}"

# Check if SVG file exists
if [ ! -f "$SVG_FILE" ]; then
    echo "Error: SVG file not found: $SVG_FILE"
    exit 1
fi

# Source the environment helper
if [ -f "$SCRIPT_DIR/scripts/ensure_env.sh" ]; then
    source "$SCRIPT_DIR/scripts/ensure_env.sh"
fi

# Install cairosvg if not present
if ! uv pip show cairosvg >/dev/null 2>&1; then
    echo "Installing cairosvg..."
    uv add cairosvg
fi

# Run the Python conversion script
if [ -n "$PNG_FILE" ]; then
    uv run python "$SCRIPT_DIR/examples/scripts/convert_svg_to_png.py" "$SVG_FILE" "$PNG_FILE"
else
    uv run python "$SCRIPT_DIR/examples/scripts/convert_svg_to_png.py" "$SVG_FILE"
fi