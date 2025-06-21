# Transcript Formatting Guide

This guide provides instructions for creating consistent, professional visualizations of experimental transcripts using SVG elements.

## Overview

The goal is to create visually appealing, standardized presentations of token-by-token analysis that include:
- Original text display
- Token breakdown with explanations
- Metrics visualization (KL divergence, MSE, relative RMSE)
- Professional styling with consistent formatting

## Directory Structure

```
examples/
├── transcript_formatting_guide.md (this file)
├── templates/
│   ├── transcript_template.svg
│   └── styles.css
├── scripts/
│   └── generate_transcript_viz.py
└── outputs/
    └── [generated transcript visualizations]
```

## Formatting Components

### 1. Header Section
- **Title**: Experiment name/identifier
- **Metadata**: Date, model version, dataset info
- **Original Text**: Display in a prominent box with proper typography

### 2. Token Analysis Table
Structure each row with:
- **Position**: Token index (0-based)
- **Token**: The actual token (monospace font)
- **Explanation**: Model's interpretation (italics)
- **Metrics**: KL divergence, MSE, relative RMSE (color-coded)

### 3. Visual Elements

#### Color Scheme
```css
/* Background */
--bg-primary: #ffffff;
--bg-secondary: #f8f9fa;
--bg-accent: #e9ecef;

/* Text */
--text-primary: #212529;
--text-secondary: #6c757d;
--text-accent: #007bff;

/* Metrics (gradient from good to bad) */
--metric-good: #28a745;
--metric-medium: #ffc107;
--metric-bad: #dc3545;

/* Borders and dividers */
--border-color: #dee2e6;
```

#### Typography
- **Headers**: Sans-serif, bold, 16-20pt
- **Body text**: Sans-serif, regular, 12-14pt
- **Tokens**: Monospace, 12pt
- **Explanations**: Sans-serif, italic, 11pt
- **Metrics**: Monospace, 10pt

### 4. SVG Structure Template

```svg
<svg width="1200" height="auto" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#ffffff"/>
  
  <!-- Header -->
  <g id="header">
    <rect x="20" y="20" width="1160" height="120" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1" rx="8"/>
    <text x="40" y="60" font-family="Arial, sans-serif" font-size="24" font-weight="bold">Transcript Analysis</text>
    <text x="40" y="90" font-family="Arial, sans-serif" font-size="14" fill="#6c757d">Dataset: [Name] | Model: [Version] | Date: [YYYY-MM-DD]</text>
    
    <!-- Original text box -->
    <rect x="40" y="110" width="1120" height="auto" fill="#e9ecef" stroke="#dee2e6" stroke-width="1" rx="4"/>
    <text x="60" y="130" font-family="Georgia, serif" font-size="16" fill="#212529">
      <!-- Original text here with proper line wrapping -->
    </text>
  </g>
  
  <!-- Token Analysis Table -->
  <g id="token-table" transform="translate(20, 180)">
    <!-- Table header -->
    <rect x="0" y="0" width="1160" height="40" fill="#343a40"/>
    <text x="20" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">Position</text>
    <text x="120" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">Token</text>
    <text x="320" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">Explanation</text>
    <text x="720" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">KL Div</text>
    <text x="850" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">MSE</text>
    <text x="980" y="28" font-family="Arial, sans-serif" font-size="14" fill="white" font-weight="bold">Rel RMSE</text>
    
    <!-- Table rows (generated dynamically) -->
    <!-- Each row follows this pattern: -->
    <g class="token-row" transform="translate(0, 40)">
      <rect x="0" y="0" width="1160" height="40" fill="#ffffff" stroke="#dee2e6" stroke-width="0.5"/>
      <text x="20" y="28" font-family="Courier, monospace" font-size="12">0</text>
      <text x="120" y="28" font-family="Courier, monospace" font-size="12" font-weight="bold">Israel</text>
      <text x="320" y="28" font-family="Arial, sans-serif" font-size="11" font-style="italic">Referred oun vacated Ging[Israel]</text>
      <text x="720" y="28" font-family="Courier, monospace" font-size="10" fill="#dc3545">8.896</text>
      <text x="850" y="28" font-family="Courier, monospace" font-size="10" fill="#dc3545">11591.7</text>
      <text x="980" y="28" font-family="Courier, monospace" font-size="10" fill="#dc3545">0.985</text>
    </g>
  </g>
  
  <!-- Optional: Visualization of metrics -->
  <g id="metrics-viz" transform="translate(20, [dynamic-y])">
    <!-- Bar charts or heat maps for metrics -->
  </g>
</svg>
```

## Best Practices

### 1. Consistent Spacing
- Use 20px margins around the entire visualization
- 40px padding for main content areas
- 10-15px spacing between elements
- Consistent row heights (40px recommended)

### 2. Color Coding for Metrics
Apply gradient coloring based on value ranges:
- **KL Divergence**: 0-1 (green), 1-3 (yellow), >3 (red)
- **MSE**: 0-2 (green), 2-5 (yellow), >5 (red)  
- **Relative RMSE**: 0-0.3 (green), 0.3-0.6 (yellow), >0.6 (red)

### 3. Text Formatting
- Truncate long explanations with ellipsis (...)
- Escape special characters in XML/SVG
- Use proper text anchoring for alignment
- Consider text wrapping for long original texts

### 4. Responsive Design
- Set viewBox for proper scaling
- Use percentage-based widths where appropriate
- Consider mobile/tablet viewing

### 5. Accessibility
- Include proper ARIA labels
- Use sufficient color contrast
- Provide text alternatives for visual elements

## Implementation Steps

1. **Parse the transcript data** into structured format
2. **Calculate visual dimensions** based on content length
3. **Generate SVG header** with metadata
4. **Create token table** with proper formatting
5. **Apply color coding** to metrics
6. **Add any additional visualizations** (charts, graphs)
7. **Export as SVG** and optionally convert to PNG

## Example Python Script Structure

```python
def create_transcript_visualization(transcript_data, output_path):
    """
    Generate SVG visualization from transcript data
    
    Args:
        transcript_data: Dict containing tokens, explanations, and metrics
        output_path: Path to save the SVG file
    """
    # 1. Parse and validate data
    # 2. Calculate dimensions
    # 3. Create SVG structure
    # 4. Add styling
    # 5. Populate with data
    # 6. Save to file
```

## Tips for Quality Output

1. **Test with various data lengths** to ensure layout stability
2. **Use consistent number formatting** (e.g., 3 decimal places)
3. **Include hover effects** for interactive viewing (if applicable)
4. **Consider adding a legend** for metric interpretations
5. **Export at high resolution** for presentations

## Common Issues and Solutions

- **Text overflow**: Use dynamic sizing or text wrapping
- **Large datasets**: Implement pagination or scrolling
- **Special characters**: Properly escape for XML/SVG
- **Performance**: Consider server-side rendering for large visualizations

## SVG to PNG Conversion

Since SVG files are text-based and may not be easily viewable in all contexts, we provide a conversion script to generate PNG images:

### Setup
```bash
# Add cairosvg to project dependencies
source scripts/ensure_env.sh && uv add cairosvg
```

### Usage
```bash
# Generate SVG visualization
source scripts/ensure_env.sh && uv run python examples/scripts/generate_transcript_viz.py

# Convert SVG to PNG for viewing
source scripts/ensure_env.sh && uv run python examples/scripts/convert_svg_to_png.py examples/outputs/example_transcript.svg

# Or specify custom output path
source scripts/ensure_env.sh && uv run python examples/scripts/convert_svg_to_png.py input.svg output.png
```

### Why Both Formats?
- **SVG**: Vector format, scalable, editable, smaller file size, ideal for web display
- **PNG**: Raster format, universally viewable, better for presentations and documentation
- The conversion script uses 150 DPI for high-quality output suitable for presentations

### Viewing Options
- **Browser**: Open the SVG file directly in any modern web browser
- **Image Viewer**: Use the PNG version for standard image viewing applications
- **Code Editor**: Many editors (VSCode, etc.) can preview SVG files inline

This guide should help maintain consistency across all transcript visualizations while providing flexibility for different experiment types.