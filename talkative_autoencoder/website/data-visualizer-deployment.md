# Data Visualizer Deployment Guide

Deploy the standalone Talkative Autoencoder data visualizer to kitft.com/data-visualizer

## Overview

The enhanced visualiser.html is a completely standalone file that can be hosted anywhere as a static site. It includes:
- All visualization functionality from the original
- Embedded CSS and JavaScript (no external dependencies except CDN)
- File upload and data sharing via Pantry/JSONBin
- Full feature parity with the original visualizer

## Deployment Options

### Option 1: Direct Upload to kitft.com

1. **Rename the file**:
   ```bash
   cp visualiser-enhanced.html index.html
   ```

2. **Upload via your hosting provider**:
   - Create a `data-visualizer` directory on your server
   - Upload `index.html` to that directory
   - Access at: https://kitft.com/data-visualizer/

### Option 2: GitHub Pages Subdirectory

1. **In your kitft.github.io repo**:
   ```bash
   mkdir -p data-visualizer
   cp /path/to/visualiser-enhanced.html data-visualizer/index.html
   git add data-visualizer/
   git commit -m "Add standalone data visualizer"
   git push
   ```

2. **Access at**: https://kitft.github.io/data-visualizer/

### Option 3: Netlify Drop

1. **Create a folder**:
   ```bash
   mkdir talkative-visualizer
   cp visualiser-enhanced.html talkative-visualizer/index.html
   ```

2. **Drag and drop** the folder to https://app.netlify.com/drop

3. **Get instant URL** like: https://amazing-name-123.netlify.app

## Features Available

- ✅ Load JSON data from Consistency Lens analysis
- ✅ Upload files (JSON/TXT)
- ✅ Share data via Pantry or JSONBin.io
- ✅ Navigate multiple transcripts
- ✅ Table and transposed views
- ✅ Salience coloring
- ✅ Column visibility toggles
- ✅ Click-to-copy cells
- ✅ Rich tooltips with token details
- ✅ Adjustable spacing and column widths
- ✅ Settings persistence in localStorage
- ✅ URL-based data loading (?bin=id)

## Customization

### Change Default API Keys

Edit these lines in the HTML:
```javascript
elements.apiKeyInput.value = localStorage.getItem('logViewerPantryId') || 'YOUR-PANTRY-ID';
elements.collectionIdInput.value = localStorage.getItem('logViewerPantryBasket') || 'YOUR-BASKET';
```

### Update Branding

Change the title and header:
```html
<title>Your Custom Title</title>
<h1 class="text-3xl font-bold text-[#5D4037] mb-2">Your Custom Header</h1>
```

### Modify Color Scheme

Update the Tailwind classes or add custom CSS in the `<style>` section.

## Testing Locally

```bash
# Python 3
python -m http.server 8000

# Node.js
npx http-server -p 8000

# Open in browser
open http://localhost:8000/visualiser-enhanced.html
```

## Integration with Main Site

### Add to Navigation

```html
<!-- In your main site navigation -->
<a href="/data-visualizer/">Talkative Autoencoder Visualizer</a>
```

### Embed in iframe

```html
<iframe src="/data-visualizer/" width="100%" height="800px" frameborder="0"></iframe>
```

### Direct Link from Results

When your main app generates results, provide a link:
```javascript
const shareableUrl = `https://kitft.com/data-visualizer/?bin=${uploadedBinId}`;
```

## Security Considerations

- The Pantry/JSONBin keys are public and scoped for write-only access
- No server-side code or database required
- All data processing happens client-side
- Shared data is stored on third-party services (Pantry/JSONBin)

## Maintenance

The visualizer is self-contained and requires no maintenance unless you want to:
- Update the visualization logic
- Change storage providers
- Add new features

To update, simply replace the index.html file on your server.