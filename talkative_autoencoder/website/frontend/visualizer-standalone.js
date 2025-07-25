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
    Supabase: {
        url: 'https://rnncsfomwqvrdorznqrt.supabase.co',
        anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJubmNzZm9td3F2cmRvcnpucXJ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0NDIzNDgsImV4cCI6MjA2OTAxODM0OH0.fiEbMVxebWE2vpUMh4EJ24xTd6YHIrSqqG2_RpcohtY',
        
        upload: async (content) => {
            const res = await fetch(`${StorageAdapters.Supabase.url}/rest/v1/analyses`, {
                method: 'POST',
                headers: {
                    'apikey': StorageAdapters.Supabase.anonKey,
                    'Authorization': `Bearer ${StorageAdapters.Supabase.anonKey}`,
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                body: JSON.stringify({
                    data: content,
                    created_at: new Date().toISOString()
                })
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.message || 'Failed to upload to Supabase');
            }
            
            const result = await res.json();
            return result[0].id;
        },
        
        fetch: async (id) => {
            const res = await fetch(`${StorageAdapters.Supabase.url}/rest/v1/analyses?id=eq.${id}`, {
                headers: {
                    'apikey': StorageAdapters.Supabase.anonKey,
                    'Authorization': `Bearer ${StorageAdapters.Supabase.anonKey}`
                }
            });
            
            if (!res.ok) throw new Error('Failed to fetch from Supabase');
            
            const result = await res.json();
            if (!result || result.length === 0) throw new Error('Analysis not found');
            
            return result[0].data;
        }
    },
    
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
                    'X-Bin-Name': 'NaturalLanguageAutoencoder-' + new Date().toISOString(),
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
console.log('visualizer-standalone.js loaded');
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded event fired');
    // State
    let allTranscripts = [];
    let currentTranscriptIndex = 0;
    let isTransposed = false;
    let salienceColoringEnabled = false;
    let currentService = 'supabase';
    let columnVisibility = VisualizationCore.getDefaultVisibility();

    // DOM elements
    const elements = {
        // Input elements
        textInput: document.getElementById('text-input'),
        parseBtn: document.getElementById('parse-btn'),
        uploadBtn: document.getElementById('upload-btn'),
        fileUpload: document.getElementById('file-upload'),
        
        // Service settings
        supabaseBtn: document.getElementById('supabase-btn'),
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
        
        elements.supabaseBtn.classList.toggle('active-service', service === 'supabase');
        elements.pantryBtn.classList.toggle('active-service', service === 'pantry');
        elements.jsonbinBtn.classList.toggle('active-service', service === 'jsonbin');

        const apiKeyLabel = document.querySelector('label[for="api-key-input"]');
        const collectionIdLabel = document.querySelector('label[for="collection-id-input"]');
        const apiKeyDiv = elements.apiKeyInput.closest('div');
        const collectionIdDiv = elements.collectionIdInput.closest('div');

        if (service === 'supabase') {
            // Hide both fields for Supabase as they're configured in the adapter
            apiKeyDiv.style.display = 'none';
            collectionIdDiv.style.display = 'none';
        } else if (service === 'pantry') {
            apiKeyDiv.style.display = 'block';
            apiKeyLabel.textContent = 'Pantry ID:';
            elements.apiKeyInput.placeholder = 'Paste your Pantry ID here';
            elements.apiKeyInput.value = localStorage.getItem('logViewerPantryId') || '88947592-e047-4e50-bfc7-d55c93fb6f35';

            collectionIdLabel.textContent = 'Basket Name:';
            elements.collectionIdInput.placeholder = 'e.g., data-viewer-storage';
            elements.collectionIdInput.value = localStorage.getItem('logViewerPantryBasket') || 'data-viewer-storage';
            collectionIdDiv.style.display = 'block';
        } else {
            apiKeyLabel.textContent = 'Your Public, Scoped API Key:';
            elements.apiKeyInput.placeholder = 'Paste your public API key here';
            elements.apiKeyInput.value = localStorage.getItem('logViewerJsonbinApiKey') || '$2a$10$cYXiC7n7tURzBeNd7E2yx.NsMNqmaYgWCoAYTmiFfGHjZKC54V.Sq';

            collectionIdLabel.textContent = 'Your Public Collection ID:';
            elements.collectionIdInput.placeholder = 'Paste your public Collection ID here';
            elements.collectionIdInput.value = localStorage.getItem('logViewerJsonbinCollectionId') || '6867e9e58561e97a5031776b';
            collectionIdDiv.style.display = 'block';
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
    elements.supabaseBtn.addEventListener('click', () => updateServiceUI('supabase'));
    elements.pantryBtn.addEventListener('click', () => updateServiceUI('pantry'));
    elements.jsonbinBtn.addEventListener('click', () => updateServiceUI('jsonbin'));
    elements.apiKeyInput.addEventListener('change', saveSettings);
    elements.collectionIdInput.addEventListener('change', saveSettings);

    // Don't default to any service yet - let the URL loading or user choice determine it

    // Upload/fetch functions
    const uploadToBin = async (content) => {
        const apiKey = elements.apiKeyInput.value;
        const collectionId = elements.collectionIdInput.value;
        
        if (currentService !== 'supabase' && (!apiKey || !collectionId)) {
            let errorMsg = 'Please provide ';
            if (currentService === 'pantry') {
                errorMsg += 'both a Pantry ID and a Basket Name';
            } else {
                errorMsg += 'both a JSONBin.io API Key and a Collection ID';
            }
            alert(errorMsg + ' in the Settings section.');
            return null;
        }
        
        elements.uploadBtn.disabled = true;
        elements.uploadBtn.textContent = 'Uploading...';
        
        try {
            let binId;
            if (currentService === 'supabase') {
                binId = await StorageAdapters.Supabase.upload(content);
            } else if (currentService === 'pantry') {
                binId = await StorageAdapters.Pantry.upload(content, apiKey, collectionId);
            } else {
                binId = await StorageAdapters.JSONBin.upload(content, apiKey, collectionId);
            }
            return binId;
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
        console.log('fetchFromBin called with binId:', binId);
        // Detect service type from binId format
        let service, content;
        try {
            if (binId.includes('/')) {
                // Pantry format: basket/id
                service = 'pantry';
                updateServiceUI('pantry');
                const apiKey = elements.apiKeyInput.value;
                return await StorageAdapters.Pantry.fetch(binId, apiKey);
            } else if (binId.match(/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/)) {
                // UUID format for Supabase
                service = 'supabase';
                updateServiceUI('supabase');
                return await StorageAdapters.Supabase.fetch(binId);
            } else {
                // JSONBin format
                service = 'jsonbin';
                updateServiceUI('jsonbin');
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
        
        // Update columnVisibility to include all columns from data
        const dataColumns = [...new Set(currentItem.data.flatMap(row => Object.keys(row)))];
        dataColumns.forEach(col => {
            if (!(col in columnVisibility)) {
                columnVisibility[col] = false; // New columns default to hidden
            }
        });
        
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
            
            // Check if we're in an iframe
            const inIframe = window.self !== window.top;
            
            if (inIframe) {
                // We're in an iframe, so create a full URL to share
                const shareUrl = `https://kitft.com/data-viewer/?bin=${encodeURIComponent(binId)}`;
                
                // Create a more helpful message with the actual link
                const message = `Link created!\n\nShareable URL:\n${shareUrl}\n\nClick OK to copy to clipboard.`;
                
                if (confirm(message)) {
                    navigator.clipboard.writeText(shareUrl).then(() => {
                        alert('Link copied to clipboard!');
                    }).catch(() => {
                        prompt('Copy this link:', shareUrl);
                    });
                }
            } else {
                // Not in iframe, update URL normally
                window.history.pushState({}, '', url);
                alert('Link created! The URL has been updated. You can now copy it from your address bar.');
            }
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
                    
                    // Check if we're in an iframe
                    const inIframe = window.self !== window.top;
                    
                    if (inIframe) {
                        // We're in an iframe, so create a full URL to share
                        const shareUrl = `https://kitft.com/data-viewer/?bin=${encodeURIComponent(binId)}`;
                        
                        // Create a more helpful message with the actual link
                        const message = `File uploaded and link created!\n\nShareable URL:\n${shareUrl}\n\nClick OK to copy to clipboard.`;
                        
                        if (confirm(message)) {
                            navigator.clipboard.writeText(shareUrl).then(() => {
                                alert('Link copied to clipboard!');
                            }).catch(() => {
                                prompt('Copy this link:', shareUrl);
                            });
                        }
                    } else {
                        // Not in iframe, update URL normally
                        window.history.pushState({}, '', url);
                        alert('File uploaded and link created! The URL has been updated.');
                    }
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
        // Check both the current window and parent window for URL parameters
        let urlParams = new URLSearchParams(window.location.search);
        let binId = urlParams.get('bin');
        
        // If no bin parameter in iframe URL, check parent window
        if (!binId && window.parent !== window) {
            try {
                urlParams = new URLSearchParams(window.parent.location.search);
                binId = urlParams.get('bin');
                console.log('Checking parent window for bin parameter:', binId);
            } catch (e) {
                // Cross-origin restriction, can't access parent
                console.log('Cannot access parent window URL (cross-origin)');
            }
        }

        if (binId) {
            console.log('URL has bin parameter:', binId);
            try {
                const content = await fetchFromBin(binId);
                console.log('Fetched content:', content);
                if (content) {
                    // Convert to string if it's an object (from Supabase)
                    const contentStr = typeof content === 'string' ? content : JSON.stringify(content);
                    elements.textInput.value = contentStr;
                    elements.uploadBtn.disabled = elements.textInput.value.trim() === '';
                    processData(contentStr);
                } else {
                    console.log('No content returned from fetchFromBin');
                }
            } catch (error) {
                console.error('Error in URL loading:', error);
            }
        } else {
            // Initialize empty state and default to Supabase only when no bin is provided
            updateServiceUI('supabase');
            render();
        }
    })();
});