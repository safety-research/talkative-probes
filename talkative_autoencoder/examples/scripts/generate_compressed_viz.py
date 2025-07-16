#!/usr/bin/env python3
"""
Generate compressed SVG visualizations for transcript analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

def get_metric_color(value: float, metric_type: str) -> str:
    """
    Get color for metric value based on type and range
    """
    colors = {
        'good': '#28a745',
        'medium': '#ff8c00',  # Darker orange for better readability
        'bad': '#dc3545'
    }
    
    if metric_type == 'kl_divergence':
        if value < 1:
            return colors['good']
        elif value < 3:
            return colors['medium']
        else:
            return colors['bad']
    elif metric_type == 'mse':
        if value < 2:
            return colors['good']
        elif value < 5:
            return colors['medium']
        else:
            return colors['bad']
    elif metric_type == 'relative_rmse':
        if value < 0.3:
            return colors['good']
        elif value < 0.6:
            return colors['medium']
        else:
            return colors['bad']
    return colors['medium']

def create_svg_element(tag: str, attrib: Dict = None, text: str = None) -> ET.Element:
    """Helper to create SVG elements"""
    elem = ET.Element(tag, attrib or {})
    if text:
        elem.text = text
    return elem

def wrap_text(text: str, max_width: int = 100) -> List[str]:
    """Simple text wrapping"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_width:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_transcript_visualization(data: Dict, output_path: str):
    """
    Generate compressed SVG visualization from transcript data
    """
    # Extract data
    original_text = data['original_text']
    total_tokens = data['total_tokens']
    tokens = data['tokens']
    
    # Calculate dimensions - MORE COMPRESSED
    row_height = 24  # Reduced from 30
    table_height = (len(tokens) + 1) * row_height  # +1 for header
    
    # Calculate header height based on text
    text_lines = wrap_text(original_text, 110)  # Wider wrapping
    text_box_height = max(40, 15 + len(text_lines) * 16)  # Tighter line spacing
    header_height = 110 + text_box_height  # Reduced from 130
    
    total_height = header_height + table_height + 10  # Minimal margin
    
    # Create SVG root
    svg = ET.Element('svg', {
        'width': '1200',
        'height': str(total_height),
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f'0 0 1200 {total_height}'
    })
    
    # Background
    svg.append(create_svg_element('rect', {
        'width': '100%',
        'height': '100%',
        'fill': '#ffffff'
    }))
    
    # Header group
    header = create_svg_element('g', {'id': 'header'})
    
    # Header background - adjust height dynamically
    header_bg_height = 75 + text_box_height  # Reduced from 90
    header.append(create_svg_element('rect', {
        'x': '15', 'y': '10',  # Reduced margins
        'width': '1170', 'height': str(header_bg_height),
        'fill': '#f8f9fa',
        'stroke': '#dee2e6',
        'stroke-width': '1',
        'rx': '6'
    }))
    
    # Title
    header.append(create_svg_element('text', {
        'x': '30', 'y': '35',  # Adjusted positions
        'font-family': 'Arial, sans-serif',
        'font-size': '20',  # Slightly smaller
        'font-weight': 'bold',
        'fill': '#212529'
    }, 'Token-by-Token Analysis'))
    
    # Metadata
    header.append(create_svg_element('text', {
        'x': '30', 'y': '55',
        'font-family': 'Arial, sans-serif',
        'font-size': '12',  # Smaller
        'fill': '#6c757d'
    }, f'Total Tokens: {total_tokens} | Analysis Type: Token-by-token breakdown'))
    
    # Original text box
    text_box_y = 70  # Reduced from 95
    
    header.append(create_svg_element('rect', {
        'x': '30', 'y': str(text_box_y),
        'width': '1140', 'height': str(text_box_height),
        'fill': '#e9ecef',
        'stroke': '#dee2e6',
        'stroke-width': '1',
        'rx': '4'
    }))
    
    # Add wrapped text lines with tighter spacing
    for i, line in enumerate(text_lines):
        header.append(create_svg_element('text', {
            'x': '45', 'y': str(text_box_y + 18 + i * 16),  # Tighter spacing
            'font-family': 'Georgia, serif',
            'font-size': '14',  # Smaller font
            'fill': '#212529'
        }, line))
    
    svg.append(header)
    
    # Token table
    table_y = header_height
    table = create_svg_element('g', {
        'id': 'token-table',
        'transform': f'translate(15, {table_y})'  # Reduced margin
    })
    
    # Table header
    table.append(create_svg_element('rect', {
        'x': '0', 'y': '0',
        'width': '1170', 'height': str(row_height),
        'fill': '#343a40'
    }))
    
    # Column headers - adjusted positions for better spacing
    headers = [
        ('10', 'Pos'),
        ('70', 'Token'),
        ('240', 'Explanation'),
        ('785', 'KL Div'),
        ('885', 'MSE'),
        ('985', 'Rel RMSE')
    ]
    
    header_y = str(row_height - 7)  # Center text vertically
    for x, text in headers:
        table.append(create_svg_element('text', {
            'x': x, 'y': header_y,
            'font-family': 'Arial, sans-serif',
            'font-size': '12',  # Smaller
            'fill': 'white',
            'font-weight': 'bold'
        }, text))
    
    # Table rows
    for idx, token_data in enumerate(tokens):
        row_y = (idx + 1) * row_height
        
        # Alternating row colors
        row_fill = '#ffffff' if idx % 2 == 0 else '#f8f9fa'
        
        # Row background
        table.append(create_svg_element('rect', {
            'x': '0', 'y': str(row_y),
            'width': '1170', 'height': str(row_height),
            'fill': row_fill,
            'stroke': '#dee2e6',
            'stroke-width': '0.5'
        }))
        
        text_y = str(row_y + row_height - 7)  # Center text vertically
        
        # Position
        table.append(create_svg_element('text', {
            'x': '10', 'y': text_y,
            'font-family': 'Courier, monospace',
            'font-size': '10',  # Smaller
            'fill': '#6c757d'
        }, str(token_data['position'])))
        
        # Token
        table.append(create_svg_element('text', {
            'x': '70', 'y': text_y,
            'font-family': 'Courier, monospace',
            'font-size': '11',  # Smaller
            'font-weight': 'bold',
            'fill': '#212529'
        }, token_data['token']))
        
        # Explanation (truncate if too long)
        explanation = token_data['explanation']
        if len(explanation) > 68:
            explanation = explanation[:65] + '...'
        table.append(create_svg_element('text', {
            'x': '240', 'y': text_y,
            'font-family': 'Arial, sans-serif',
            'font-size': '10',  # Smaller
            'font-style': 'italic',
            'fill': '#495057'
        }, explanation))
        
        # Metrics with color coding
        kl_color = get_metric_color(token_data['kl_divergence'], 'kl_divergence')
        table.append(create_svg_element('text', {
            'x': '785', 'y': text_y,
            'font-family': 'Courier, monospace',
            'font-size': '10',  # Smaller
            'fill': kl_color,
            'font-weight': 'bold'
        }, f"{token_data['kl_divergence']:.3f}"))
        
        mse_color = get_metric_color(token_data['mse'], 'mse')
        table.append(create_svg_element('text', {
            'x': '885', 'y': text_y,
            'font-family': 'Courier, monospace',
            'font-size': '10',  # Smaller
            'fill': mse_color,
            'font-weight': 'bold'
        }, f"{token_data['mse']:.1f}"))
        
        rmse_color = get_metric_color(token_data['relative_rmse'], 'relative_rmse')
        table.append(create_svg_element('text', {
            'x': '985', 'y': text_y,
            'font-family': 'Courier, monospace',
            'font-size': '10',  # Smaller
            'fill': rmse_color,
            'font-weight': 'bold'
        }, f"{token_data['relative_rmse']:.3f}"))
    
    svg.append(table)
    
    # Convert to string with proper formatting
    tree = ET.ElementTree(svg)
    
    # Pretty print
    xml_str = ET.tostring(svg, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    final_xml = '\n'.join(lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(final_xml)
    
    print(f"SVG visualization saved to: {output_path}")