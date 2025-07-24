# Talkative Autoencoder Frontend Architecture

## Modular Structure

### Shared Components
1. **visualizer-core.js** (421 lines)
   - Core visualization functions
   - Rendering logic for tables, transposed views, full text
   - Utility functions (salience coloring, formatting)
   - Column management
   - Copy to clipboard functionality

2. **visualizer-styles.css** (190 lines)
   - All shared CSS styles
   - Token box styling
   - Tooltip styling
   - Button styling
   - Service switcher styling

### Standalone Visualizer
For static hosting at kitft.com/data-visualizer

1. **visualizer-standalone.html** (151 lines)
   - Clean HTML structure
   - Links to shared CSS
   - Loads core + standalone JS

2. **visualizer-standalone.js** (581 lines)
   - Data parsing and adapters
   - Storage service integration (Pantry/JSONBin)
   - File upload handling
   - Navigation between transcripts
   - Uses VisualizationCore as a global

**Total for standalone**: ~1,343 lines (compared to 1,224 lines in the old all-in-one file)

### Enhanced Main Frontend
For WebSocket connection to inference server

1. **index-enhanced.html** (439 lines)
   - Full UI with tabs (Analyze, Load Data, Share)
   - Parameter controls for analysis
   - Links to shared CSS

2. **app-enhanced.js** (761 lines)
   - WebSocket integration
   - Real-time analysis
   - All visualizer features
   - Imports VisualizationCore as ES6 module

## Deployment

### For Standalone (kitft.github.io/data-viewer/)
In the kitft.github.io repository, place `data-viewer-index.html` as:
```
data-viewer/
└── index.html
```

This references the assets from the talkative-autoencoder subdirectory:
- `../talkative-autoencoder/visualizer-core.js`
- `../talkative-autoencoder/visualizer-standalone.js`
- `../talkative-autoencoder/visualizer-styles.css`

### For Main Frontend (in this repo)
The main frontend is already set up:
```
frontend/
├── index.html
├── app.js
├── visualizer-core.js
└── visualizer-styles.css
```

## Benefits of This Architecture

1. **True Modularization**: Core visualization logic is shared
2. **Maintainability**: Single source of truth for visualization code
3. **Flexibility**: Each version can have unique features
4. **Clean Separation**: 
   - Core logic: 421 lines
   - Styles: 190 lines
   - Standalone app: 732 lines (HTML + JS)
   - Enhanced app: 1,200 lines (HTML + JS)

## Optional: Single-File Build

If you need a truly single HTML file for the standalone version:

```bash
#!/bin/bash
# build-standalone.sh

cat visualizer-standalone.html | \
  sed '/<link rel="stylesheet" href="visualizer-styles.css">/r visualizer-styles.css' | \
  sed 's/<link rel="stylesheet" href="visualizer-styles.css">/<style>/' | \
  sed '/<script src=".\/visualizer-core.js"><\/script>/r visualizer-core.js' | \
  sed 's/<script src=".\/visualizer-core.js"><\/script>/<script>/' | \
  sed '/<script src=".\/visualizer-standalone.js"><\/script>/r visualizer-standalone.js' | \
  sed 's/<script src=".\/visualizer-standalone.js"><\/script>/<script>/' \
  > visualizer-bundled.html
```

This would create a single ~1,343 line HTML file with everything embedded.