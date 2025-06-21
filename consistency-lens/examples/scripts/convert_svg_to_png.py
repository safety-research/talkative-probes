#!/usr/bin/env python3
"""
Convert SVG files to PNG for viewing
"""

import sys
from pathlib import Path
import cairosvg

def convert_svg_to_png(svg_path: str, png_path: str = None):
    """Convert SVG to PNG using cairosvg"""
    svg_path = Path(svg_path)
    
    if not svg_path.exists():
        print(f"Error: SVG file not found: {svg_path}")
        return False
    
    if png_path is None:
        png_path = svg_path.with_suffix('.png')
    
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=150)
        print(f"PNG saved to: {png_path}")
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_svg_to_png.py <svg_file> [png_file]")
        sys.exit(1)
    
    svg_file = sys.argv[1]
    png_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_svg_to_png(svg_file, png_file)