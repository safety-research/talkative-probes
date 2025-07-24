// Import visualization core
import { VisualizationCore } from './visualizer-core.js';

// Configuration
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;

const WS_URL = API_URL.replace('http', 'ws') + '/ws';

// State management
const state = {
    ws: null,
    currentRequestId: null,
    isTransposed: false,
    salienceColoringEnabled: false,
    allTranscripts: [],
    currentTranscriptIndex: 0,
    currentService: 'pantry',
    columnVisibility: VisualizationCore.getDefaultVisibility(),
    activeTab: 'analyze'
};

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
            showError("There was an error parsing the JSON data. Please check the format.");
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
                    'X-Bin-Name': 'ConsistencyLens-' + new Date().toISOString(),
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

// DOM Elements
const elements = {
    // Tabs
    analyzeTab: document.getElementById('analyzeTab'),
    loadDataTab: document.getElementById('loadDataTab'),
    shareTab: document.getElementById('shareTab'),
    analyzeContent: document.getElementById('analyzeContent'),
    loadDataContent: document.getElementById('loadDataContent'),
    shareContent: document.getElementById('shareContent'),
    
    // Connection
    connectionStatus: document.getElementById('connectionStatus'),
    statusText: document.getElementById('statusText'),
    
    // Input
    inputText: document.getElementById('inputText'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    jsonInput: document.getElementById('json-input'),
    loadJsonBtn: document.getElementById('load-json-btn'),
    fileUpload: document.getElementById('file-upload'),
    
    // Status
    loading: document.getElementById('loading'),
    error: document.getElementById('error'),
    queuePosition: document.getElementById('queuePosition'),
    
    // Parameters
    kRollouts: document.getElementById('kRollouts'),
    kRolloutsValue: document.getElementById('kRolloutsValue'),
    autoBatchSize: document.getElementById('autoBatchSize'),
    batchSize: document.getElementById('batchSize'),
    batchSizeInfo: document.getElementById('batchSizeInfo'),
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperatureValue'),
    
    // Advanced settings
    advancedToggle: document.getElementById('advancedToggle'),
    advancedSettings: document.getElementById('advancedSettings'),
    calculateSalience: document.getElementById('calculateSalience'),
    tunedLens: document.getElementById('tunedLens'),
    logitLens: document.getElementById('logitLens'),
    noEval: document.getElementById('noEval'),
    noKL: document.getElementById('noKL'),
    doHardTokens: document.getElementById('doHardTokens'),
    seed: document.getElementById('seed'),
    rolloutBatchSize: document.getElementById('rolloutBatchSize'),
    numSamples: document.getElementById('numSamples'),
    
    // Sharing
    pantryBtn: document.getElementById('pantry-btn'),
    jsonbinBtn: document.getElementById('jsonbin-btn'),
    apiKeyInput: document.getElementById('api-key-input'),
    collectionIdInput: document.getElementById('collection-id-input'),
    uploadBtn: document.getElementById('upload-btn'),
    exportJsonBtn: document.getElementById('export-json-btn'),
    
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
    
    // Visualization
    outputPlaceholder: document.getElementById('output-placeholder'),
    outputTable: document.getElementById('output-table'),
    tableHead: document.getElementById('table-head'),
    tableBody: document.getElementById('table-body'),
    transposedView: document.getElementById('transposed-view'),
    columnToggleContainer: document.getElementById('column-toggle-container'),
    transposeBtn: document.getElementById('transpose-btn'),
    displayControls: document.getElementById('display-controls'),
    lineSpacingSlider: document.getElementById('line-spacing-slider'),
    colWidthControl: document.getElementById('col-width-control'),
    colWidthSlider: document.getElementById('col-width-slider'),
    salienceToggle: document.getElementById('salience-toggle'),
    salienceColorbar: document.getElementById('salience-colorbar'),
    salienceDefinition: document.getElementById('salience-definition'),
    metadataDisplay: document.getElementById('metadata-display'),
    fullTextContainer: document.getElementById('full-text-container'),
    fullTextPlaceholder: document.getElementById('full-text-placeholder')
};

// Helper functions
const showError = (message) => {
    elements.error.querySelector('p').textContent = message;
    elements.error.classList.remove('hidden');
    setTimeout(() => elements.error.classList.add('hidden'), 5000);
};

const showLoading = (show) => {
    elements.loading.classList.toggle('hidden', !show);
};

const showQueuePosition = (position) => {
    if (position !== null) {
        elements.queuePosition.querySelector('p').textContent = `Queue position: ${position}`;
        elements.queuePosition.classList.remove('hidden');
    } else {
        elements.queuePosition.classList.add('hidden');
    }
};

// Tab management
const switchTab = (tabName) => {
    state.activeTab = tabName;
    
    // Update tab buttons
    const tabs = { analyze: elements.analyzeTab, loadData: elements.loadDataTab, share: elements.shareTab };
    const contents = { analyze: elements.analyzeContent, loadData: elements.loadDataContent, share: elements.shareContent };
    
    Object.entries(tabs).forEach(([name, tab]) => {
        if (name === tabName) {
            tab.classList.add('border-orange-500', 'text-gray-900');
            tab.classList.remove('border-transparent', 'text-gray-500');
        } else {
            tab.classList.remove('border-orange-500', 'text-gray-900');
            tab.classList.add('border-transparent', 'text-gray-500');
        }
    });
    
    Object.entries(contents).forEach(([name, content]) => {
        content.classList.toggle('hidden', name !== tabName);
    });
};

// Service management
const updateServiceUI = (service) => {
    state.currentService = service;
    
    elements.pantryBtn.classList.toggle('active-service', service === 'pantry');
    elements.jsonbinBtn.classList.toggle('active-service', service === 'jsonbin');

    const apiKeyLabel = document.querySelector('label[for="api-key-input"]');
    const collectionIdLabel = document.querySelector('label[for="collection-id-input"]');

    if (service === 'pantry') {
        apiKeyLabel.textContent = 'Pantry ID:';
        elements.apiKeyInput.placeholder = 'Paste your Pantry ID here';
        elements.apiKeyInput.value = localStorage.getItem('consistencyLensPantryId') || '';

        collectionIdLabel.textContent = 'Basket Name:';
        elements.collectionIdInput.placeholder = 'e.g., consistency-lens-data';
        elements.collectionIdInput.value = localStorage.getItem('consistencyLensPantryBasket') || 'consistency-lens-data';
    } else {
        apiKeyLabel.textContent = 'JSONBin.io API Key:';
        elements.apiKeyInput.placeholder = 'Paste your API key here';
        elements.apiKeyInput.value = localStorage.getItem('consistencyLensJsonbinApiKey') || '';

        collectionIdLabel.textContent = 'Collection ID:';
        elements.collectionIdInput.placeholder = 'Paste your collection ID here';
        elements.collectionIdInput.value = localStorage.getItem('consistencyLensJsonbinCollectionId') || '';
    }
};

const saveSettings = () => {
    if (state.currentService === 'pantry') {
        localStorage.setItem('consistencyLensPantryId', elements.apiKeyInput.value);
        localStorage.setItem('consistencyLensPantryBasket', elements.collectionIdInput.value);
    } else {
        localStorage.setItem('consistencyLensJsonbinApiKey', elements.apiKeyInput.value);
        localStorage.setItem('consistencyLensJsonbinCollectionId', elements.collectionIdInput.value);
    }
    localStorage.setItem('consistencyLensService', state.currentService);
};

// WebSocket management
const connectWebSocket = () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
    
    state.ws = new WebSocket(WS_URL);
    
    state.ws.onopen = () => {
        console.log('WebSocket connected');
        elements.statusText.textContent = 'Connected';
        elements.connectionStatus.classList.remove('bg-gray-100', 'bg-red-100');
        elements.connectionStatus.classList.add('bg-green-100');
        elements.analyzeBtn.disabled = false;
    };
    
    state.ws.onclose = () => {
        console.log('WebSocket disconnected');
        elements.statusText.textContent = 'Disconnected';
        elements.connectionStatus.classList.remove('bg-green-100');
        elements.connectionStatus.classList.add('bg-red-100');
        elements.analyzeBtn.disabled = true;
        
        // Reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('Connection error. Please check if the backend is running.');
    };
    
    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
};

const handleWebSocketMessage = (data) => {
    switch (data.type) {
        case 'queued':
            state.currentRequestId = data.request_id;
            showQueuePosition(data.queue_position);
            showLoading(true);
            break;
            
        case 'processing':
            showQueuePosition(null);
            showLoading(true);
            break;
            
        case 'result':
            showLoading(false);
            showQueuePosition(null);
            processResults(data.result);
            break;
            
        case 'error':
            showLoading(false);
            showQueuePosition(null);
            showError(data.error || 'An error occurred during analysis');
            break;
    }
};

// Analysis functions
const analyze = () => {
    const text = elements.inputText.value.trim();
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        showError('Not connected to server. Please wait...');
        connectWebSocket();
        return;
    }
    
    // Build options
    const options = {
        temperature: parseFloat(elements.temperature.value),
        optimize_explanations_config: {
            just_do_k_rollouts: parseInt(elements.kRollouts.value),
            batch_size_for_rollouts: elements.autoBatchSize.checked 
                ? Math.max(1, Math.floor(256 / parseInt(elements.kRollouts.value)))
                : parseInt(elements.batchSize.value),
            use_batched: true
        },
        calculate_salience: elements.calculateSalience.checked,
        use_tuned_lens: elements.tunedLens.checked,
        use_logit_lens: elements.logitLens.checked,
        no_eval: elements.noEval.checked,
        no_kl: elements.noKL.checked,
        do_hard_tokens: elements.doHardTokens.checked
    };
    
    if (elements.seed.value) {
        options.seed = parseInt(elements.seed.value);
    }
    
    if (elements.rolloutBatchSize.value) {
        options.optimize_explanations_config.rollout_batch_size = parseInt(elements.rolloutBatchSize.value);
    }
    
    if (elements.numSamples.value && elements.numSamples.value !== '1') {
        options.optimize_explanations_config.num_samples = parseInt(elements.numSamples.value);
    }
    
    // Send request
    state.ws.send(JSON.stringify({
        type: 'analyze',
        text: text,
        options: options
    }));
};

// Process results
const processResults = (result) => {
    // Create transcript in expected format
    const transcript = {
        data: result.data || [],
        metadata: result.metadata || {}
    };
    
    // Reset transcripts for new analysis
    state.allTranscripts = [transcript];
    state.currentTranscriptIndex = 0;
    
    // Auto-enable salience if present
    const hasSalienceData = transcript.data.some(r => r.token_salience !== undefined);
    state.salienceColoringEnabled = hasSalienceData;
    elements.salienceToggle.checked = hasSalienceData;
    
    // Enable export buttons
    elements.uploadBtn.disabled = false;
    elements.exportJsonBtn.disabled = false;
    
    render();
};

// Rendering
const updateColourbar = () => {
    if (!state.salienceColoringEnabled) {
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
    if (state.isTransposed) {
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
    for (const columnName in state.columnVisibility) {
        const visible = state.columnVisibility[columnName];
        document.querySelectorAll(`[data-column-name="${columnName}"]`).forEach(cell => {
            cell.style.display = visible ? '' : 'none';
        });
    }
};

const render = () => {
    const currentItem = state.allTranscripts[state.currentTranscriptIndex];
    
    if (!currentItem || state.allTranscripts.length === 0) {
        elements.outputPlaceholder.classList.remove('hidden');
        elements.outputTable.classList.add('hidden');
        elements.transposedView.classList.add('hidden');
        elements.fullTextContainer.classList.add('hidden');
        elements.navigationContainer.classList.add('hidden');
        elements.displayControls.classList.add('hidden');
        elements.metadataDisplay.classList.add('hidden');
        elements.bottomNavigationContainer.classList.add('hidden');
        elements.sidePrevBtn.classList.add('hidden');
        elements.sideNextBtn.classList.add('hidden');
        return;
    }
    
    elements.outputPlaceholder.classList.add('hidden');
    elements.fullTextContainer.classList.remove('hidden');
    elements.displayControls.style.display = 'flex';
    
    // Render components using VisualizationCore
    VisualizationCore.renderMetadata({
        metadata: currentItem.metadata,
        container: elements.metadataDisplay
    });
    
    VisualizationCore.createColumnToggleUI({
        container: elements.columnToggleContainer,
        columnVisibility: state.columnVisibility,
        onChange: (column, visible) => {
            updateColumnVisibility();
        }
    });
    
    VisualizationCore.renderFullTextBox({
        data: currentItem.data,
        container: elements.fullTextContainer,
        salienceColoringEnabled: state.salienceColoringEnabled
    });
    
    if (state.isTransposed) {
        elements.outputTable.classList.add('hidden');
        elements.colWidthControl.classList.remove('hidden');
        elements.transposedView.classList.remove('hidden');
        
        VisualizationCore.renderTransposedView({
            data: currentItem.data,
            container: elements.transposedView,
            colWidth: parseInt(elements.colWidthSlider.value, 10),
            salienceColoringEnabled: state.salienceColoringEnabled
        });
    } else {
        elements.outputTable.classList.remove('hidden');
        elements.colWidthControl.classList.add('hidden');
        elements.transposedView.classList.add('hidden');
        
        VisualizationCore.renderTable({
            data: currentItem.data,
            tableHead: elements.tableHead,
            tableBody: elements.tableBody,
            columnVisibility: state.columnVisibility,
            salienceColoringEnabled: state.salienceColoringEnabled
        });
    }
    
    // Navigation visibility
    if (state.allTranscripts.length > 1) {
        elements.navigationContainer.classList.remove('hidden');
        elements.navCounter.textContent = `${state.currentTranscriptIndex + 1} / ${state.allTranscripts.length}`;
        elements.prevBtn.disabled = state.currentTranscriptIndex === 0;
        elements.nextBtn.disabled = state.currentTranscriptIndex === state.allTranscripts.length - 1;
        
        elements.bottomNavigationContainer.classList.remove('hidden');
        elements.bottomNavCounter.textContent = `${state.currentTranscriptIndex + 1} / ${state.allTranscripts.length}`;
        elements.bottomPrevBtn.disabled = state.currentTranscriptIndex === 0;
        elements.bottomNextBtn.disabled = state.currentTranscriptIndex === state.allTranscripts.length - 1;
        
        if (state.isTransposed) {
            elements.sidePrevBtn.classList.add('hidden');
            elements.sideNextBtn.classList.add('hidden');
        } else {
            elements.sidePrevBtn.classList.remove('hidden');
            elements.sideNextBtn.classList.remove('hidden');
            elements.sidePrevBtn.style.visibility = (state.currentTranscriptIndex === 0) ? 'hidden' : 'visible';
            elements.sideNextBtn.style.visibility = (state.currentTranscriptIndex === state.allTranscripts.length - 1) ? 'hidden' : 'visible';
        }
    } else {
        elements.navigationContainer.classList.add('hidden');
        elements.bottomNavigationContainer.classList.add('hidden');
        elements.sidePrevBtn.classList.add('hidden');
        elements.sideNextBtn.classList.add('hidden');
    }
    
    applySpacing();
    updateColourbar();
};

// Navigation
const navigate = (direction) => {
    if (direction === 'prev' && state.currentTranscriptIndex > 0) {
        state.currentTranscriptIndex--;
        render();
    } else if (direction === 'next' && state.currentTranscriptIndex < state.allTranscripts.length - 1) {
        state.currentTranscriptIndex++;
        render();
    }
};

// Upload/Export functions
const uploadData = async () => {
    const apiKey = elements.apiKeyInput.value;
    const collectionId = elements.collectionIdInput.value;
    
    if (!apiKey || !collectionId) {
        showError(state.currentService === 'pantry'
            ? 'Please provide both a Pantry ID and a Basket Name in the Share tab.'
            : 'Please provide both a JSONBin.io API Key and a Collection ID in the Share tab.');
        switchTab('share');
        return;
    }
    
    if (state.allTranscripts.length === 0) {
        showError('No data to upload. Please analyze some text first.');
        return;
    }
    
    elements.uploadBtn.disabled = true;
    elements.uploadBtn.textContent = 'Uploading...';
    
    try {
        const dataToUpload = JSON.stringify(state.allTranscripts.length === 1 
            ? state.allTranscripts[0] 
            : state.allTranscripts);
            
        const adapter = state.currentService === 'pantry' ? StorageAdapters.Pantry : StorageAdapters.JSONBin;
        const binId = await adapter.upload(dataToUpload, apiKey, collectionId);
        
        if (binId) {
            const shareUrl = `${window.location.origin}/data-visualizer/?bin=${binId}`;
            
            // Copy to clipboard
            navigator.clipboard.writeText(shareUrl).then(() => {
                alert(`Link created and copied to clipboard!\n\n${shareUrl}\n\nYou can share this link with others to view the analysis.`);
            }).catch(() => {
                alert(`Link created!\n\n${shareUrl}\n\nPlease copy this link to share the analysis.`);
            });
        }
    } catch (error) {
        console.error('Upload failed:', error);
        showError(`Upload failed: ${error.message}`);
    } finally {
        elements.uploadBtn.disabled = false;
        elements.uploadBtn.textContent = 'Upload & Get Link';
    }
};

const exportJSON = () => {
    if (state.allTranscripts.length === 0) {
        showError('No data to export. Please analyze some text first.');
        return;
    }
    
    const dataStr = JSON.stringify(state.allTranscripts.length === 1 
        ? state.allTranscripts[0] 
        : state.allTranscripts, null, 2);
    
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `consistency-lens-analysis-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// Load data functions
const loadJSON = () => {
    const jsonText = elements.jsonInput.value.trim();
    if (!jsonText) {
        showError('Please paste JSON data to load.');
        return;
    }
    
    const transcripts = DataAdapters.parseText(jsonText);
    if (transcripts.length > 0) {
        state.allTranscripts = transcripts;
        state.currentTranscriptIndex = 0;
        
        // Enable export buttons
        elements.uploadBtn.disabled = false;
        elements.exportJsonBtn.disabled = false;
        
        // Auto-enable salience if present
        const hasSalienceData = transcripts.some(t => t.data.some(r => r.token_salience !== undefined));
        state.salienceColoringEnabled = hasSalienceData;
        elements.salienceToggle.checked = hasSalienceData;
        
        render();
        
        // Switch to analyze tab to see results
        switchTab('analyze');
    }
};

const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const fileContent = e.target.result;
            elements.jsonInput.value = fileContent;
            loadJSON();
        };
        reader.readAsText(file);
    }
};

// Initialize event listeners
const initializeEventListeners = () => {
    // Tab navigation
    elements.analyzeTab.addEventListener('click', () => switchTab('analyze'));
    elements.loadDataTab.addEventListener('click', () => switchTab('loadData'));
    elements.shareTab.addEventListener('click', () => switchTab('share'));
    
    // Analysis
    elements.analyzeBtn.addEventListener('click', analyze);
    elements.inputText.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) analyze();
    });
    
    // Parameters
    elements.kRollouts.addEventListener('input', (e) => {
        const k = parseInt(e.target.value);
        elements.kRolloutsValue.textContent = k;
        if (elements.autoBatchSize.checked) {
            const autoBatch = Math.max(1, Math.floor(256 / k));
            elements.batchSize.value = autoBatch;
            elements.batchSizeInfo.textContent = `${autoBatch} (auto)`;
        }
    });
    
    elements.autoBatchSize.addEventListener('change', (e) => {
        elements.batchSize.disabled = e.target.checked;
        if (e.target.checked) {
            const k = parseInt(elements.kRollouts.value);
            const autoBatch = Math.max(1, Math.floor(256 / k));
            elements.batchSize.value = autoBatch;
            elements.batchSizeInfo.textContent = `${autoBatch} (auto)`;
        } else {
            elements.batchSizeInfo.textContent = elements.batchSize.value;
        }
    });
    
    elements.batchSize.addEventListener('input', (e) => {
        if (!elements.autoBatchSize.checked) {
            elements.batchSizeInfo.textContent = e.target.value;
        }
    });
    
    elements.temperature.addEventListener('input', (e) => {
        elements.temperatureValue.textContent = e.target.value;
    });
    
    // Service management
    elements.pantryBtn.addEventListener('click', () => updateServiceUI('pantry'));
    elements.jsonbinBtn.addEventListener('click', () => updateServiceUI('jsonbin'));
    elements.apiKeyInput.addEventListener('change', saveSettings);
    elements.collectionIdInput.addEventListener('change', saveSettings);
    
    // Upload/Export
    elements.uploadBtn.addEventListener('click', uploadData);
    elements.exportJsonBtn.addEventListener('click', exportJSON);
    
    // Load data
    elements.loadJsonBtn.addEventListener('click', loadJSON);
    elements.fileUpload.addEventListener('change', handleFileUpload);
    
    // Visualization controls
    elements.transposeBtn.addEventListener('click', () => {
        state.isTransposed = !state.isTransposed;
        document.querySelectorAll('#column-toggle-dropdown').forEach(d => d.classList.add('hidden'));
        render();
    });
    
    elements.salienceToggle.addEventListener('change', () => {
        state.salienceColoringEnabled = elements.salienceToggle.checked;
        render();
    });
    
    // Navigation
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
};

// Load from URL parameters
const loadFromURL = async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const binId = urlParams.get('bin');

    if (binId) {
        const isPantry = binId.includes('/');
        updateServiceUI(isPantry ? 'pantry' : 'jsonbin');
        
        try {
            let content;
            if (isPantry) {
                const apiKey = elements.apiKeyInput.value;
                content = await StorageAdapters.Pantry.fetch(binId, apiKey);
            } else {
                content = await StorageAdapters.JSONBin.fetch(binId);
            }
            
            if (content) {
                const transcripts = DataAdapters.parseText(typeof content === 'string' ? content : JSON.stringify(content));
                if (transcripts.length > 0) {
                    state.allTranscripts = transcripts;
                    state.currentTranscriptIndex = 0;
                    elements.uploadBtn.disabled = false;
                    elements.exportJsonBtn.disabled = false;
                    render();
                    switchTab('analyze');
                }
            }
        } catch (error) {
            console.error('Failed to load from URL:', error);
            showError(`Could not load shared data: ${error.message}`);
        }
    }
};

// Initialize application
const initialize = () => {
    // Load saved settings
    const lastService = localStorage.getItem('consistencyLensService') || 'pantry';
    updateServiceUI(lastService);
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Connect WebSocket
    connectWebSocket();
    
    // Load from URL if present
    loadFromURL();
};

// Start the application
initialize();