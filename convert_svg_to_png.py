#!/usr/bin/env python3
"""Convert SVG diagrams to PNG format for presentations."""

import subprocess
import sys
from pathlib import Path

def convert_svg_to_png(svg_file, png_file, width=None, dpi=300):
    """Convert SVG to PNG using Inkscape or cairosvg."""
    
    # Try Inkscape first
    try:
        cmd = ['inkscape', svg_file, '-o', png_file, f'--export-dpi={dpi}']
        if width:
            cmd.extend([f'--export-width={width}'])
        subprocess.run(cmd, check=True)
        print(f"✓ Converted {svg_file} to {png_file} using Inkscape")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try cairosvg as fallback
    try:
        import cairosvg
        kwargs = {'dpi': dpi}
        if width:
            kwargs['output_width'] = width
        cairosvg.svg2png(url=svg_file, write_to=png_file, **kwargs)
        print(f"✓ Converted {svg_file} to {png_file} using cairosvg")
        return True
    except ImportError:
        print("cairosvg not installed. Install with: pip install cairosvg")
    except Exception as e:
        print(f"Error with cairosvg: {e}")
    
    print("Failed to convert. Install Inkscape or cairosvg.")
    return False

if __name__ == "__main__":
    # Convert all SVG files
    svg_files = [
        "talkative_probes_diagram.svg",
        "talkative_probes_minimal.svg",
        "talkative_probes_animated.svg",
        "talkative_probes_kl_focus.svg",
        "talkative_probes_redesigned.svg"
    ]
    
    for svg_file in svg_files:
        if Path(svg_file).exists():
            png_file = svg_file.replace('.svg', '.png')
            convert_svg_to_png(svg_file, png_file, width=2000)  # High res for presentations 