// Standalone Visualizer for Talkative Autoencoder
// This file uses the VisualizationCore loaded as a global from visualizer-core.js

// Data adapters
const DataAdapters = {
    // Transform column-oriented data to row format
    transformColumnData: (colData) => {
        const data = [];
        const keys = Object.keys(colData);
        if (keys.length === 0) return [];

        const numRows = Object.keys(colData[keys[0]]).length;

        for (let i = 0; i < numRows; i++) {
            const row = {};
            keys.forEach(key => {
                if (colData[key] && colData[key][String(i)] !== undefined) {
                    row[key] = colData[key][String(i)];
                }
            });
            data.push(row);
        }
        return data;
    },

    // Parse text input (supports multiple JSON objects)
    parseText: (text) => {
        if (!text || text.trim() === '') return [];
        try {
            const jsonBlocks = text.trim().replace(/}(?:\s|\n)*{/g, '}|||JSON_DELIMITER|||{').split('|||JSON_DELIMITER|||');
            const transcripts = [];

            for (const block of jsonBlocks) {
                if (block.trim() === "") continue;
                const parsed = JSON.parse(block);
                
                let dataPart = parsed;
                let metadata = null;

                // Check for new format with metadata
                if (parsed && parsed.metadata && parsed.data) {
                    dataPart = parsed.data;
                    metadata = parsed.metadata;
                }

                let transcriptData;
                if (dataPart && typeof dataPart === 'object' && !Array.isArray(dataPart) && dataPart.position) {
                    // Column-oriented data
                    transcriptData = DataAdapters.transformColumnData(dataPart);
                } else if (Array.isArray(dataPart)) {
                    // Row-oriented data
                    transcriptData = dataPart;
                }

                if (transcriptData && transcriptData.length > 0) {
                    // Ensure explanation field exists
                    transcriptData.forEach(row => {
                        if (!row.explanation && Array.isArray(row.explanation_structured)) {
                            row.explanation = row.explanation_structured.join('');
                        }
                    });
                    transcripts.push({ data: transcriptData, metadata: metadata });
                }
            }
            return transcripts;
        } catch (e) {
            console.error("Invalid JSON input:", e);
            alert("There was an error parsing the JSON data. Please check the console for details.");
            return [];
        }
    }
};

// Storage adapters
const StorageAdapters = {
    Pantry: {
        upload: async (content, apiKey, collectionId) => {
            const uniqueId = 'log-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            
            const res = await fetch(`https://getpantry.cloud/apiv1/pantry/${apiKey}/basket/${collectionId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    [uniqueId]: {
                        data: content,
                        timestamp: new Date().toISOString()
                    }
                })
            });
            
            if (!res.ok) {
                const errorText = await res.text();
                
                if (errorText.includes('does not exist')) {
                    // Try creating basket
                    const createRes = await fetch(`https://getpantry.cloud/apiv1/pantry/${apiKey}/basket/${collectionId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            [uniqueId]: {
                                data: content,
                                timestamp: new Date().toISOString()
                            }
                        })
                    });
                    
                    if (!createRes.ok) {
                        throw new Error('Failed to create basket');
                    }
                } else {
                    throw new Error(errorText);
                }
            }
            
            return `${collectionId}/${uniqueId}`;
        },
        
        fetch: async (binId, apiKey) => {
            const [basketName, uniqueId] = binId.split('/');
            const res = await fetch(`https://getpantry.cloud/apiv1/pantry/${apiKey}/basket/${basketName}`);
            
            if (!res.ok) throw new Error('Failed to fetch basket');
            
            const result = await res.json();
            if (uniqueId && result[uniqueId]) {
                return result[uniqueId].data;
            }
            return result;
        }
    },
    
    JSONBin: {
        upload: async (content, apiKey, collectionId) => {
            const res = await fetch('https://api.jsonbin.io/v3/b', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Access-Key': apiKey,
                    'X-Bin-Name': 'LogViewer-Data-' + new Date().toISOString(),
                    'X-Collection-Id': collectionId,
                    'X-Bin-Private': 'false'
                },
                body: JSON.stringify({ data: content })
            });
            
            if (!res.ok) throw new Error('Failed to upload');
            
            const result = await res.json();
            return result.metadata.id;
        },
        
        fetch: async (binId) => {
            const res = await fetch(`https://api.jsonbin.io/v3/b/${binId}/latest`);
            if (!res.ok) throw new Error('Failed to fetch bin');
            
            const result = await res.json();
            return result.record.data;
        }
    }
};

// Main application
document.addEventListener('DOMContentLoaded', () => {
    // State
    let allTranscripts = [];
    let currentTranscriptIndex = 0;
    let isTransposed = false;
    let salienceColoringEnabled = false;
    let currentService = 'pantry';
    let columnVisibility = VisualizationCore.getDefaultVisibility();

    // DOM elements
    const elements = {
        // Input elements
        textInput: document.getElementById('text-input'),
        parseBtn: document.getElementById('parse-btn'),
        uploadBtn: document.getElementById('upload-btn'),
        fileUpload: document.getElementById('file-upload'),
        
        // Service settings
        pantryBtn: document.getElementById('pantry-btn'),
        jsonbinBtn: document.getElementById('jsonbin-btn'),
        apiKeyInput: document.getElementById('api-key-input'),
        collectionIdInput: document.getElementById('collection-id-input'),
        
        // Navigation
        navigationContainer: document.getElementById('navigation-container'),
        prevBtn: document.getElementById('prev-btn'),
        nextBtn: document.getElementById('next-btn'),
        navCounter: document.getElementById('nav-counter'),
        bottomNavigationContainer: document.getElementById('bottom-navigation-container'),
        bottomPrevBtn: document.getElementById('bottom-prev-btn'),
        bottomNextBtn: document.getElementById('bottom-next-btn'),
        bottomNavCounter: document.getElementById('bottom-nav-counter'),
        sidePrevBtn: document.getElementById('side-prev-btn'),
        sideNextBtn: document.getElementById('side-next-btn'),
        
        // Display controls
        displayControls: document.getElementById('display-controls'),
        controlsSeparator: document.getElementById('controls-separator'),
        columnToggleContainer: document.getElementById('column-toggle-container'),
        transposeBtn: document.getElementById('transpose-btn'),
        lineSpacingSlider: document.getElementById('line-spacing-slider'),
        colWidthControl: document.getElementById('col-width-control'),
        colWidthSlider: document.getElementById('col-width-slider'),
        salienceToggle: document.getElementById('salience-toggle'),
        salienceColorbar: document.getElementById('salience-colorbar'),
        salienceDefinition: document.getElementById('salience-definition'),
        
        // Output elements
        outputPlaceholder: document.getElementById('output-placeholder'),
        outputTable: document.getElementById('output-table'),
        tableHead: document.getElementById('table-head'),
        tableBody: document.getElementById('table-body'),
        transposedView: document.getElementById('transposed-view'),
        metadataDisplay: document.getElementById('metadata-display'),
        fullTextContainer: document.getElementById('full-text-container'),
        fullTextPlaceholder: document.getElementById('full-text-placeholder')
    };

    // Service management
    const updateServiceUI = (service) => {
        currentService = service;
        
        elements.pantryBtn.classList.toggle('active-service', service === 'pantry');
        elements.jsonbinBtn.classList.toggle('active-service', service === 'jsonbin');

        const apiKeyLabel = document.querySelector('label[for="api-key-input"]');
        const collectionIdLabel = document.querySelector('label[for="collection-id-input"]');

        if (service === 'pantry') {
            apiKeyLabel.textContent = 'Pantry ID:';
            elements.apiKeyInput.placeholder = 'Paste your Pantry ID here';
            elements.apiKeyInput.value = localStorage.getItem('logViewerPantryId') || '88947592-e047-4e50-bfc7-d55c93fb6f35';

            collectionIdLabel.textContent = 'Basket Name:';
            elements.collectionIdInput.placeholder = 'e.g., data-viewer-storage';
            elements.collectionIdInput.value = localStorage.getItem('logViewerPantryBasket') || 'data-viewer-storage';
        } else {
            apiKeyLabel.textContent = 'Your Public, Scoped API Key:';
            elements.apiKeyInput.placeholder = 'Paste your public API key here';
            elements.apiKeyInput.value = localStorage.getItem('logViewerJsonbinApiKey') || '$2a$10$cYXiC7n7tURzBeNd7E2yx.NsMNqmaYgWCoAYTmiFfGHjZKC54V.Sq';

            collectionIdLabel.textContent = 'Your Public Collection ID:';
            elements.collectionIdInput.placeholder = 'Paste your public Collection ID here';
            elements.collectionIdInput.value = localStorage.getItem('logViewerJsonbinCollectionId') || '6867e9e58561e97a5031776b';
        }
    };

    const saveSettings = () => {
        if (currentService === 'pantry') {
            localStorage.setItem('logViewerPantryId', elements.apiKeyInput.value);
            localStorage.setItem('logViewerPantryBasket', elements.collectionIdInput.value);
        } else {
            localStorage.setItem('logViewerJsonbinApiKey', elements.apiKeyInput.value);
            localStorage.setItem('logViewerJsonbinCollectionId', elements.collectionIdInput.value);
        }
        localStorage.setItem('logViewerService', currentService);
    };

    // Setup service switcher
    elements.pantryBtn.addEventListener('click', () => updateServiceUI('pantry'));
    elements.jsonbinBtn.addEventListener('click', () => updateServiceUI('jsonbin'));
    elements.apiKeyInput.addEventListener('change', saveSettings);
    elements.collectionIdInput.addEventListener('change', saveSettings);

    const lastService = localStorage.getItem('logViewerService') || 'pantry';
    updateServiceUI(lastService);

    // Upload/fetch functions
    const uploadToBin = async (content) => {
        const apiKey = elements.apiKeyInput.value;
        const collectionId = elements.collectionIdInput.value;
        
        if (!apiKey || !collectionId) {
            alert(currentService === 'pantry'
                ? 'Please provide both a Pantry ID and a Basket Name in the Settings section.'
                : 'Please provide both a JSONBin.io API Key and a Collection ID in the Settings section.');
            return null;
        }
        
        elements.uploadBtn.disabled = true;
        elements.uploadBtn.textContent = 'Uploading...';
        
        try {
            const adapter = currentService === 'pantry' ? StorageAdapters.Pantry : StorageAdapters.JSONBin;
            return await adapter.upload(content, apiKey, collectionId);
        } catch (error) {
            console.error('Upload failed:', error);
            alert(`Upload failed: ${error.message}`);
            return null;
        } finally {
            elements.uploadBtn.disabled = false;
            elements.uploadBtn.textContent = 'Upload & Get Link';
        }
    };

    const fetchFromBin = async (binId) => {
        const isPantry = binId.includes('/');
        updateServiceUI(isPantry ? 'pantry' : 'jsonbin');
        
        try {
            if (isPantry) {
                const apiKey = elements.apiKeyInput.value;
                return await StorageAdapters.Pantry.fetch(binId, apiKey);
            } else {
                return await StorageAdapters.JSONBin.fetch(binId);
            }
        } catch (error) {
            console.error('Fetch failed:', error);
            alert(`Could not load data: ${error.message}`);
            return null;
        }
    };

    // Rendering functions
    const updateColourbar = () => {
        if (!salienceColoringEnabled) {
            elements.salienceColorbar.classList.add('hidden');
            elements.salienceDefinition.classList.add('hidden');
            return;
        }
        elements.salienceColorbar.classList.remove('hidden');
        elements.salienceDefinition.classList.remove('hidden');
        elements.salienceColorbar.style.background = 'linear-gradient(to right, rgb(0,200,0) 0%, #ffffff 50%, rgb(255,0,0) 100%)';
    };

    const applySpacing = () => {
        const spacing = elements.lineSpacingSlider.value;
        if (isTransposed) {
            document.querySelectorAll('.explanation-word').forEach(word => {
                word.style.paddingBottom = `${spacing}px`;
            });
        } else {
            document.querySelectorAll('#table-body td').forEach(td => {
                td.style.paddingTop = `${spacing}px`;
                td.style.paddingBottom = `${spacing}px`;
            });
        }
    };

    const updateColumnVisibility = () => {
        for (const columnName in columnVisibility) {
            const visible = columnVisibility[columnName];
            document.querySelectorAll(`[data-column-name="${columnName}"]`).forEach(cell => {
                cell.style.display = visible ? '' : 'none';
            });
        }
    };

    const render = () => {
        const currentItem = allTranscripts[currentTranscriptIndex];
        
        if (!currentItem || allTranscripts.length === 0) {
            elements.outputPlaceholder.classList.remove('hidden');
            elements.outputTable.classList.add('hidden');
            elements.transposedView.classList.add('hidden');
            elements.fullTextPlaceholder.classList.remove('hidden');
            elements.navigationContainer.classList.add('hidden');
            elements.displayControls.classList.add('hidden');
            elements.controlsSeparator.classList.add('hidden');
            elements.metadataDisplay.classList.add('hidden');
            elements.bottomNavigationContainer.classList.add('hidden');
            elements.sidePrevBtn.classList.add('hidden');
            elements.sideNextBtn.classList.add('hidden');
            return;
        }
        
        elements.outputPlaceholder.classList.add('hidden');
        elements.fullTextContainer.classList.remove('hidden');
        elements.fullTextPlaceholder.classList.add('hidden');
        elements.displayControls.style.display = 'flex';
        elements.controlsSeparator.classList.remove('hidden');
        
        // Render components
        VisualizationCore.renderMetadata({
            metadata: currentItem.metadata,
            container: elements.metadataDisplay
        });
        
        // Only show column toggles in table view
        if (!isTransposed) {
            VisualizationCore.createColumnToggleUI({
                container: elements.columnToggleContainer,
                columnVisibility: columnVisibility,
                onChange: (column, visible) => {
                    updateColumnVisibility();
                }
            });
        } else {
            elements.columnToggleContainer.innerHTML = '';
        }
        
        VisualizationCore.renderFullTextBox({
            data: currentItem.data,
            container: elements.fullTextContainer,
            salienceColoringEnabled: salienceColoringEnabled
        });
        
        if (isTransposed) {
            elements.outputTable.classList.add('hidden');
            elements.colWidthControl.classList.remove('hidden');
            elements.transposedView.classList.remove('hidden');
            
            VisualizationCore.renderTransposedView({
                data: currentItem.data,
                container: elements.transposedView,
                colWidth: parseInt(elements.colWidthSlider.value, 10),
                salienceColoringEnabled: salienceColoringEnabled
            });
        } else {
            elements.outputTable.classList.remove('hidden');
            elements.colWidthControl.classList.add('hidden');
            elements.transposedView.classList.add('hidden');
            
            VisualizationCore.renderTable({
                data: currentItem.data,
                tableHead: elements.tableHead,
                tableBody: elements.tableBody,
                columnVisibility: columnVisibility,
                salienceColoringEnabled: salienceColoringEnabled
            });
        }
        
        // Navigation visibility
        if (allTranscripts.length > 1) {
            elements.navigationContainer.classList.remove('hidden');
            elements.navCounter.textContent = `${currentTranscriptIndex + 1} / ${allTranscripts.length}`;
            elements.prevBtn.disabled = currentTranscriptIndex === 0;
            elements.nextBtn.disabled = currentTranscriptIndex === allTranscripts.length - 1;
            
            elements.bottomNavigationContainer.classList.remove('hidden');
            elements.bottomNavCounter.textContent = `${currentTranscriptIndex + 1} / ${allTranscripts.length}`;
            elements.bottomPrevBtn.disabled = currentTranscriptIndex === 0;
            elements.bottomNextBtn.disabled = currentTranscriptIndex === allTranscripts.length - 1;
            
            if (isTransposed) {
                elements.sidePrevBtn.classList.add('hidden');
                elements.sideNextBtn.classList.add('hidden');
            } else {
                elements.sidePrevBtn.classList.remove('hidden');
                elements.sideNextBtn.classList.remove('hidden');
                elements.sidePrevBtn.style.visibility = (currentTranscriptIndex === 0) ? 'hidden' : 'visible';
                elements.sideNextBtn.style.visibility = (currentTranscriptIndex === allTranscripts.length - 1) ? 'hidden' : 'visible';
            }
        } else {
            elements.navigationContainer.classList.add('hidden');
            elements.bottomNavigationContainer.classList.add('hidden');
            elements.sidePrevBtn.classList.add('hidden');
            elements.sideNextBtn.classList.add('hidden');
        }
        
        applySpacing();
    };

    const processData = (text) => {
        allTranscripts = DataAdapters.parseText(text);
        currentTranscriptIndex = 0;
        
        // Default salience toggle to true if data present
        const hasSalienceData = allTranscripts.some(t => t.data.some(r => r.token_salience !== undefined));
        salienceColoringEnabled = hasSalienceData;
        elements.salienceToggle.checked = hasSalienceData;
        if (hasSalienceData) {
            updateColourbar();
        }
        
        render();
    };

    // Event handlers
    elements.parseBtn.addEventListener('click', () => {
        elements.uploadBtn.disabled = elements.textInput.value.trim() === '';
        processData(elements.textInput.value);
    });

    elements.uploadBtn.addEventListener('click', async () => {
        const binId = await uploadToBin(elements.textInput.value);
        if (binId) {
            const url = new URL(window.location);
            url.search = '';
            url.searchParams.set('bin', binId);
            window.history.pushState({}, '', url);
            alert('Link created! The URL has been updated. You can now copy it from your address bar.');
        }
    });

    elements.transposeBtn.addEventListener('click', () => {
        isTransposed = !isTransposed;
        document.querySelectorAll('#column-toggle-dropdown').forEach(d => d.classList.add('hidden'));
        render();
    });

    elements.salienceToggle.addEventListener('change', () => {
        salienceColoringEnabled = elements.salienceToggle.checked;
        updateColourbar();
        render();
    });

    // Navigation
    const navigate = (direction) => {
        if (direction === 'prev' && currentTranscriptIndex > 0) {
            currentTranscriptIndex--;
            render();
        } else if (direction === 'next' && currentTranscriptIndex < allTranscripts.length - 1) {
            currentTranscriptIndex++;
            render();
        }
    };

    elements.prevBtn.addEventListener('click', () => navigate('prev'));
    elements.nextBtn.addEventListener('click', () => navigate('next'));
    elements.bottomPrevBtn.addEventListener('click', () => navigate('prev'));
    elements.bottomNextBtn.addEventListener('click', () => navigate('next'));
    elements.sidePrevBtn.addEventListener('click', () => navigate('prev'));
    elements.sideNextBtn.addEventListener('click', () => navigate('next'));

    // Spacing controls
    elements.lineSpacingSlider.addEventListener('input', applySpacing);
    elements.colWidthSlider.addEventListener('input', (e) => {
        const newWidth = e.target.value;
        document.querySelectorAll('.transpose-col').forEach(col => {
            col.style.width = `${newWidth}px`;
        });
    });

    // File upload
    elements.fileUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async (e) => {
                const fileContent = e.target.result;
                elements.textInput.value = fileContent;
                elements.uploadBtn.disabled = elements.textInput.value.trim() === '';
                processData(fileContent);
                const binId = await uploadToBin(fileContent);
                if (binId) {
                    const url = new URL(window.location);
                    url.search = '';
                    url.searchParams.set('bin', binId);
                    window.history.pushState({}, '', url);
                    alert('File uploaded and link created! The URL has been updated.');
                }
            };
            reader.readAsText(file);
        }
    });

    elements.textInput.addEventListener('input', () => {
        elements.uploadBtn.disabled = elements.textInput.value.trim() === '';
    });

    // Click handlers
    document.body.addEventListener('click', (e) => {
        const target = e.target;
        
        // Copy to clipboard
        if (target.tagName === 'TD' && target.title === 'Click to copy') {
            VisualizationCore.copyToClipboard(target.textContent, target);
        }
        
        // Close dropdown
        const dropdown = document.querySelector('#column-toggle-dropdown');
        if (dropdown && !dropdown.classList.contains('hidden') && 
            !e.target.closest('#column-toggle-container')) {
            dropdown.classList.add('hidden');
        }
    });

    // Load from URL
    (async () => {
        const urlParams = new URLSearchParams(window.location.search);
        const binId = urlParams.get('bin');

        if (binId) {
            const content = await fetchFromBin(binId);
            if (content) {
                elements.textInput.value = content;
                elements.uploadBtn.disabled = elements.textInput.value.trim() === '';
                processData(content);
            }
        } else {
            // Initialize empty state
            render();
        }
    })();
});