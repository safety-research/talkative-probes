// VisualizationCore will be available as a global from visualizer-core.js
// Wait for it to be loaded
if (typeof VisualizationCore === 'undefined') {
    console.error('VisualizationCore not loaded. Make sure visualizer-core.js is loaded before app.js');
}

// Configuration
// Handle Cursor port forwarding - it uses special URLs
const isCursorForwarded = window.location.hostname.includes('app.github.dev') || 
                          window.location.hostname.includes('vscode-remote') ||
                          window.location.port === '3001';  // Cursor forwards 3000 to 3001

let API_URL;
if (window.location.port === '3001') {
    // Cursor is forwarding 3000->3001 and 8000->8001
    API_URL = window.location.origin.replace(':3001', ':8001');
} else if (window.location.hostname === 'localhost') {
    API_URL = 'http://localhost:8000';
} else if (window.location.hostname.includes('proxy.runpod.net')) {
    // RunPod proxy - frontend is on port 3000, backend is on port 8000
    API_URL = window.location.origin.replace('-3000.proxy', '-8000.proxy').replace(':3000', ':8000');
} else if (window.location.hostname.includes('kitft')) {
    // kitft.github.io deployment - needs to connect to RunPod backend
    // This will be replaced by deploy-frontend make target with actual RunPod URL
    API_URL = 'https://lzl87kptd5n85n-8000.proxy.runpod.net';
} else {
    API_URL = window.location.origin;
}

const WS_URL = API_URL.replace('http', 'ws').replace('https', 'wss') + '/ws';

console.log('Detected environment:', {
    hostname: window.location.hostname,
    port: window.location.port,
    isCursorForwarded,
    API_URL,
    WS_URL
});

// Check if debug mode is enabled
const isDebugMode = () => {
    return window.location.hostname === 'localhost' || 
           window.location.hostname === '127.0.0.1' ||
           new URLSearchParams(window.location.search).has('debug');
};

// Request logger for local storage (debug mode only)
const RequestLogger = {
    LOG_KEY: 'talkative_autoencoder_requests',
    MAX_LOGS: 1000, // Keep last 1000 requests
    enabled: isDebugMode(),
    
    log(type, data) {
        if (!this.enabled) return;
        
        try {
            const logs = this.getLogs();
            const entry = {
                id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(),
                timestamp: new Date().toISOString(),
                type: type, // 'analysis' or 'generation'
                ...data
            };
            
            logs.unshift(entry); // Add to beginning
            
            // Keep only last MAX_LOGS entries
            if (logs.length > this.MAX_LOGS) {
                logs.splice(this.MAX_LOGS);
            }
            
            localStorage.setItem(this.LOG_KEY, JSON.stringify(logs));
            console.log(`[RequestLogger] Logged ${type} request:`, entry);
        } catch (e) {
            console.error('[RequestLogger] Failed to log request:', e);
        }
    },
    
    getLogs() {
        try {
            const stored = localStorage.getItem(this.LOG_KEY);
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('[RequestLogger] Failed to read logs:', e);
            return [];
        }
    },
    
    clear() {
        localStorage.removeItem(this.LOG_KEY);
        console.log('[RequestLogger] Logs cleared');
    },
    
    export() {
        const logs = this.getLogs();
        const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `talkative_requests_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
};

// State management
const state = {
    ws: null,
    currentRequestId: null,
    isTransposed: false,
    salienceColoringEnabled: false,
    allTranscripts: [],
    currentTranscriptIndex: 0,
    currentService: 'supabase',
    columnVisibility: {},  // Will be initialized after VisualizationCore loads
    activeTab: 'analyze',
    loadingTimeout: null,
    skipSalienceRender: false,  // Flag to prevent render on programmatic checkbox changes
    generations: [],  // Store multiple generations
    currentGenerationIndex: 0,  // Track which generation is being viewed
    modelInfo: null,  // Store loaded model information
    autoBatchSizeMax: 256,  // Default value, will be updated from backend
    modelSwitcher: null  // Model switcher component
};

// Initialize column visibility after VisualizationCore is available
if (typeof VisualizationCore !== 'undefined') {
    state.columnVisibility = VisualizationCore.getDefaultVisibility();
} else {
    console.error('VisualizationCore not available when initializing state');
    // Fallback default visibility
    state.columnVisibility = {
        'position': true,
        'predicted': false,
        'target': true,
        'rank': false,
        'loss': false,
        'mse': true,
        'ce': false,
        'kl': false,
        'max_predicted_prob': false,
        'predicted_prob': false,
        'explanation': true,
        'decoder_completions': false,
        'pred_token': false,
        'salience_plot': false,
        'token_salience': false,
        'explanation_concatenated': false
    };
}

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
                const error = await res.text();
                throw new Error(error || 'Failed to upload to Supabase');
            }
            
            const result = await res.json();
            return Array.isArray(result) ? result[0].id : result.id;
        },
        
        fetch: async (id) => {
            const res = await fetch(`${StorageAdapters.Supabase.url}/rest/v1/analyses?id=eq.${id}&select=data`, {
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
    resetBtn: document.getElementById('resetBtn'),
    newAnalysisBtn: document.getElementById('newAnalysisBtn'),
    
    // Input
    inputText: document.getElementById('inputText'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    isChatFormatted: document.getElementById('isChatFormatted'),
    chatWarning: document.getElementById('chatWarning'),
    
    // Continuation generation
    generateBtn: document.getElementById('generateBtn'),
    numTokens: document.getElementById('numTokens'),
    numCompletions: document.getElementById('numCompletions'),
    genTemperature: document.getElementById('genTemperature'),
    generationStatus: document.getElementById('generationStatus'),
    continuationResults: document.getElementById('continuationResults'),
    continuationsList: document.getElementById('continuationsList'),
    jsonInput: document.getElementById('json-input'),
    loadJsonBtn: document.getElementById('load-json-btn'),
    fileUpload: document.getElementById('file-upload'),
    
    // Status
    error: document.getElementById('error'),
    
    // Generation loading
    generationLoading: document.getElementById('generationLoading'),
    
    // GPU Stats
    gpuStats: document.getElementById('gpu-stats'),
    gpuUtilBar: document.getElementById('gpu-util-bar'),
    gpuUtilText: document.getElementById('gpu-util-text'),
    gpuMemBar: document.getElementById('gpu-mem-bar'),
    gpuMemText: document.getElementById('gpu-mem-text'),
    gpuBusyIndicator: document.getElementById('gpu-busy-indicator'),
    generationLoadingMessage: document.getElementById('generationLoadingMessage'),
    interruptGenerationBtn: document.getElementById('interruptGenerationBtn'),
    
    // Analysis loading
    analysisLoading: document.getElementById('analysisLoading'),
    analysisLoadingMessage: document.getElementById('analysisLoadingMessage'),
    analysisLoadingProgress: document.getElementById('analysisLoadingProgress'),
    analysisProgressBar: document.getElementById('analysisProgressBar'),
    analysisProgressText: document.getElementById('analysisProgressText'),
    interruptAnalysisBtn: document.getElementById('interruptAnalysisBtn'),
    
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
    supabaseBtn: document.getElementById('supabase-btn'),
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
    fullTextPlaceholder: document.getElementById('full-text-placeholder'),
    columnExplanations: document.getElementById('column-explanations'),
    generationTabs: document.getElementById('generationTabs'),
    tabNavigation: document.getElementById('tabNavigation')
};

// Helper functions
const showError = (message) => {
    elements.error.querySelector('p').textContent = message;
    elements.error.classList.remove('hidden');
    setTimeout(() => elements.error.classList.add('hidden'), 5000);
};

const updateConnectionStatus = () => {
    if (state.modelInfo && state.modelInfo.display_name) {
        const modelName = state.modelInfo.display_name || 'Unknown Model';
        const checkpointFile = state.modelInfo.checkpoint_filename || '';
        
        if (state.modelInfo.loaded) {
            // Model is loaded - GREEN
            elements.statusText.innerHTML = `Connected: <strong>${modelName}</strong><br><span style="font-size: 0.65em; opacity: 0.8;">${checkpointFile}</span>`;
            elements.connectionStatus.classList.remove('bg-gray-100', 'bg-red-100', 'bg-yellow-100');
            elements.connectionStatus.classList.add('bg-green-100');
        } else {
            // Model not loaded yet, but we know what will be loaded - YELLOW
            elements.statusText.innerHTML = `Connected (will load: <strong>${modelName}</strong>)<br><span style="font-size: 0.65em; opacity: 0.8;">${checkpointFile}</span>`;
            elements.connectionStatus.classList.remove('bg-gray-100', 'bg-red-100', 'bg-green-100');
            elements.connectionStatus.classList.add('bg-yellow-100');
        }
    } else {
        // Connected but no model info - YELLOW
        elements.statusText.textContent = 'Connected (no model configured)';
        elements.connectionStatus.classList.remove('bg-gray-100', 'bg-red-100', 'bg-green-100');
        elements.connectionStatus.classList.add('bg-yellow-100');
    }
};

// Tab management functions
const createGenerationTabs = () => {
    console.log('Creating generation tabs. Generations:', state.generations.length, 'Transcripts:', state.allTranscripts.length);
    
    elements.tabNavigation.innerHTML = '';
    
    // Show tabs if we have at least one analysis result or any generations
    if (state.generations.length === 0 && state.allTranscripts.length === 0) {
        elements.generationTabs.classList.add('hidden');
        return;
    }
    
    // Always show tabs after first analysis
    elements.generationTabs.classList.remove('hidden');
    
    // No need to add generation here - it's already handled in processResults
    
    // Render all generation tabs
    state.generations.forEach((gen, index) => {
        const tab = document.createElement('button');
        tab.className = `py-2 px-4 text-sm font-medium border-b-2 ${
            index === state.currentGenerationIndex 
                ? 'border-orange-500 text-gray-900' 
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
        }`;
        tab.textContent = `Analysis ${index + 1}`;
        tab.onclick = () => switchGeneration(index);
        elements.tabNavigation.appendChild(tab);
    });
    
    console.log('Created', state.generations.length, 'tabs');
};

const switchGeneration = (index) => {
    if (index < 0 || index >= state.generations.length) return;
    
    state.currentGenerationIndex = index;
    state.allTranscripts = state.generations[index].transcripts;
    state.currentTranscriptIndex = 0;
    
    createGenerationTabs();
    render();
};

const showLoading = (show, message = 'Processing...', progress = null, context = 'analysis', showInterrupt = false) => {
    // Determine which loading box to use based on context
    const loadingEl = context === 'generation' ? elements.generationLoading : elements.analysisLoading;
    const messageEl = context === 'generation' ? elements.generationLoadingMessage : elements.analysisLoadingMessage;
    const progressEl = context === 'generation' ? null : elements.analysisLoadingProgress;
    const progressBarEl = context === 'generation' ? null : elements.analysisProgressBar;
    const progressTextEl = context === 'generation' ? null : elements.analysisProgressText;
    const interruptBtn = context === 'generation' ? elements.interruptGenerationBtn : elements.interruptAnalysisBtn;
    
    loadingEl.classList.toggle('hidden', !show);
    
    if (show && message) {
        messageEl.textContent = message;
        
        // Show/hide interrupt button
        interruptBtn.classList.toggle('hidden', !showInterrupt);
        
        // Only analysis context supports progress bar
        if (context === 'analysis' && progress !== null) {
            progressEl.classList.remove('hidden');
            progressBarEl.style.width = `${progress}%`;
            progressTextEl.textContent = `${Math.round(progress)}%`;
        } else if (progressEl) {
            progressEl.classList.add('hidden');
        }
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
    
    elements.supabaseBtn.classList.toggle('active-service', service === 'supabase');
    elements.pantryBtn.classList.toggle('active-service', service === 'pantry');
    elements.jsonbinBtn.classList.toggle('active-service', service === 'jsonbin');

    const apiKeyLabel = document.querySelector('label[for="api-key-input"]');
    const collectionIdLabel = document.querySelector('label[for="collection-id-input"]');
    const collectionIdDiv = elements.collectionIdInput.closest('div');
    const apiKeyDiv = elements.apiKeyInput.closest('div');

    if (service === 'supabase') {
        // Hide both fields for Supabase (uses centralized credentials)
        apiKeyDiv.style.display = 'none';
        collectionIdDiv.style.display = 'none';
    } else if (service === 'pantry') {
        apiKeyDiv.style.display = 'block';
        apiKeyLabel.textContent = 'Pantry ID:';
        elements.apiKeyInput.placeholder = 'Paste your Pantry ID here';
        elements.apiKeyInput.value = localStorage.getItem('consistencyLensPantryId') || '88947592-e047-4e50-bfc7-d55c93fb6f35';

        collectionIdLabel.textContent = 'Basket Name:';
        elements.collectionIdInput.placeholder = 'e.g., consistency-lens-data';
        elements.collectionIdInput.value = localStorage.getItem('consistencyLensPantryBasket') || 'data-viewer-storage';
        collectionIdDiv.style.display = 'block';
    } else {
        apiKeyLabel.textContent = 'JSONBin.io API Key:';
        elements.apiKeyInput.placeholder = 'Paste your API key here';
        elements.apiKeyInput.value = localStorage.getItem('consistencyLensJsonbinApiKey') || '$2a$10$cYXiC7n7tURzBeNd7E2yx.NsMNqmaYgWCoAYTmiFfGHjZKC54V.Sq';

        collectionIdLabel.textContent = 'Collection ID:';
        elements.collectionIdInput.placeholder = 'Paste your collection ID here';
        elements.collectionIdInput.value = localStorage.getItem('consistencyLensJsonbinCollectionId') || '6867e9e58561e97a5031776b';
        collectionIdDiv.style.display = 'block';
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

// Reset function
const resetModel = async () => {
    const confirmed = confirm(
        '⚠️ WARNING: This will reset the model for ALL users!\n\n' +
        'This action will:\n' +
        '• Clear the loaded model from memory\n' +
        '• Cancel any ongoing analyses\n' +
        '• Require the model to be reloaded on next use\n\n' +
        'Are you sure you want to continue?'
    );
    
    if (!confirmed) return;
    
    try {
        showLoading(true, 'Resetting model...', null, 'analysis');
        
        const response = await fetch(`${API_URL}/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to reset model');
        }
        
        const result = await response.json();
        
        // Clear local state
        state.modelInfo = null;
        state.allTranscripts = [];
        state.currentTranscriptIndex = 0;
        state.generations = [];
        state.currentGenerationIndex = 0;
        state.currentRequestId = null;
        
        // Update UI
        updateConnectionStatus();
        elements.outputPlaceholder.classList.remove('hidden');
        elements.fullTextContainer.classList.add('hidden');
        elements.displayControls.style.display = 'none';
        elements.outputTable.classList.add('hidden');
        elements.transposedView.classList.add('hidden');
        elements.navigationContainer.classList.add('hidden');
        elements.bottomNavigationContainer.classList.add('hidden');
        elements.metadataDisplay.classList.add('hidden');
        elements.sidePrevBtn.classList.add('hidden');
        elements.sideNextBtn.classList.add('hidden');
        elements.columnExplanations.classList.add('hidden');
        elements.generationTabs.classList.add('hidden');
        
        showLoading(false, '', null, 'analysis');
        showLoading(false, '', null, 'generation');
        showError('Model reset successfully. The model will be reloaded on next use.');
        
    } catch (error) {
        console.error('Reset error:', error);
        showLoading(false, '', null, 'analysis');
        showLoading(false, '', null, 'generation');
        showError('Failed to reset model: ' + error.message);
    }
};

// WebSocket management
const connectWebSocket = () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
    
    console.log('Attempting to connect to WebSocket:', WS_URL);
    state.ws = new WebSocket(WS_URL);
    
    state.ws.onopen = () => {
        console.log('WebSocket connected successfully');
        updateConnectionStatus();
        elements.analyzeBtn.disabled = false;
        elements.generateBtn.disabled = false;
        
        // Update model switcher WebSocket
        if (state.modelSwitcher) {
            state.modelSwitcher.setWebSocket(state.ws);
            // Request model list immediately after connection
            state.modelSwitcher.requestModelList();
        }
        
        // Start GPU stats polling when connected
        startGPUStatsPolling();
    };
    
    state.ws.onclose = () => {
        console.log('WebSocket disconnected');
        elements.statusText.textContent = 'Disconnected';
        elements.connectionStatus.classList.remove('bg-green-100');
        elements.connectionStatus.classList.add('bg-red-100');
        elements.analyzeBtn.disabled = true;
        elements.generateBtn.disabled = true;
        
        // Stop GPU stats polling when disconnected
        stopGPUStatsPolling();
        
        // Reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        console.error('Failed to connect to:', WS_URL);
        showError('Connection error. Please check if the backend is running.');
    };
    
    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
};

// GPU Stats Functions
let gpuStatsInterval = null;

const updateGPUStats = async () => {
    try {
        const response = await fetch(`${API_URL}/api/gpu_stats`);
        if (!response.ok) return;
        
        const stats = await response.json();
        
        // Show GPU stats panel
        elements.gpuStats.classList.remove('hidden');
        
        // Update utilization
        elements.gpuUtilBar.style.width = `${stats.utilization}%`;
        elements.gpuUtilText.textContent = `${stats.utilization}%`;
        
        // Show peak utilization if significantly different from current
        if (stats.peak_utilization && stats.peak_utilization > stats.utilization + 10) {
            elements.gpuUtilText.textContent = `${stats.utilization}% (peak: ${stats.peak_utilization}%)`;
        }
        
        // Update memory
        elements.gpuMemBar.style.width = `${stats.memory_percent}%`;
        elements.gpuMemText.textContent = `${stats.memory_used}GB / ${stats.memory_total}GB`;
        
        // Update busy indicator
        if (stats.utilization > 5) {
            elements.gpuBusyIndicator.classList.remove('hidden');
            elements.gpuUtilBar.classList.remove('bg-blue-500');
            elements.gpuUtilBar.classList.add('bg-orange-500');
        } else {
            elements.gpuBusyIndicator.classList.add('hidden');
            elements.gpuUtilBar.classList.remove('bg-orange-500');
            elements.gpuUtilBar.classList.add('bg-blue-500');
        }
        
        // Color memory bar based on usage
        if (stats.memory_percent > 80) {
            elements.gpuMemBar.classList.remove('bg-green-500', 'bg-yellow-500');
            elements.gpuMemBar.classList.add('bg-red-500');
        } else if (stats.memory_percent > 60) {
            elements.gpuMemBar.classList.remove('bg-green-500', 'bg-red-500');
            elements.gpuMemBar.classList.add('bg-yellow-500');
        } else {
            elements.gpuMemBar.classList.remove('bg-yellow-500', 'bg-red-500');
            elements.gpuMemBar.classList.add('bg-green-500');
        }
    } catch (error) {
        console.error('Failed to fetch GPU stats:', error);
    }
};

const startGPUStatsPolling = () => {
    // Update immediately
    updateGPUStats();
    
    // Then update every 2 seconds
    if (gpuStatsInterval) clearInterval(gpuStatsInterval);
    gpuStatsInterval = setInterval(updateGPUStats, 2000);
};

const stopGPUStatsPolling = () => {
    if (gpuStatsInterval) {
        clearInterval(gpuStatsInterval);
        gpuStatsInterval = null;
    }
};

const handleWebSocketMessage = (data) => {
    // Handle global notifications for model switching
    if (data.type === 'model_switch_status' && typeof globalNotifications !== 'undefined') {
        globalNotifications.handleModelSwitchStatus(data);
    }
    
    // Pass messages to model switcher if relevant
    if (state.modelSwitcher && [
        'models_list', 
        'model_switch_status', 
        'model_switch_acknowledged',
        'model_switch_complete',
        'model_switch_error',
        'model_info_update'
    ].includes(data.type)) {
        state.modelSwitcher.handleMessage(data);
    }
    
    switch (data.type) {
        case 'model_info':
            state.modelInfo = data;
            // Update auto batch size max if provided
            if (data.auto_batch_size_max) {
                state.autoBatchSizeMax = data.auto_batch_size_max;
                // Update the label in the UI
                const autoBatchLabel = document.querySelector('label[for="autoBatchSize"]');
                if (autoBatchLabel) {
                    autoBatchLabel.textContent = `Auto-calculate (${state.autoBatchSizeMax}÷N)`;
                }
                // Update batch size if auto-calculate is enabled
                if (typeof updateAutoBatchSize === 'function') {
                    updateAutoBatchSize();
                }
            }
            // Update cached models in model switcher
            if (state.modelSwitcher && data.cached_models) {
                state.modelSwitcher.cachedModels = data.cached_models;
                state.modelSwitcher.renderModelList();
            }
            updateConnectionStatus();
            // Clear any loading messages when model info is received
            showLoading(false, '', null, 'generation');
            showLoading(false, '', null, 'analysis');
            break;
            
        case 'queued':
            state.currentRequestId = data.request_id;
            // Store position info for later use in queue updates
            if (!state.activeRequests) state.activeRequests = {};
            state.activeRequests[data.request_id] = {
                position: data.queue_position,
                context: data.context || 'analysis'
            };
            
            const queueContext = data.context || 'analysis';
            const positionText = data.queue_position 
                ? `position: ${data.queue_position}${data.queue_size ? '/' + data.queue_size : ''}`
                : 'queued';
            const queueMessage = queueContext === 'generation' 
                ? `Queued for generation (${positionText})`
                : `Queued for analysis (${positionText})`;
            showLoading(true, queueMessage, null, queueContext);
            break;
            
        case 'status':
            // Handle model loading or other status updates
            // Determine context based on message content
            const context = data.message && data.message.includes('generation') ? 'generation' : 'analysis';
            showLoading(true, data.message || 'Processing...', null, context);
            break;
            
        case 'processing':
            // Use the actual message from backend and determine context
            const processingMessage = data.message || 'Processing...';
            // Use explicit context if provided, otherwise infer from message
            const processingContext = data.context || (processingMessage.includes('Generating') ? 'generation' : 'analysis');
            // Show interrupt button when processing
            showLoading(true, processingMessage, null, processingContext, true);
            break;
            
        case 'progress':
            // Handle progress updates during analysis
            const percent = (data.current / data.total) * 100;
            console.log('Progress update:', data.current, '/', data.total, '=', percent + '%', 'Message:', data.message);
            
            // Format ETA if available
            let messageWithETA = data.message || 'Analyzing tokens...';
            if (data.eta_seconds !== null && data.eta_seconds !== undefined) {
                const etaMinutes = Math.floor(data.eta_seconds / 60);
                const etaSeconds = Math.floor(data.eta_seconds % 60);
                if (etaMinutes > 0) {
                    messageWithETA += ` (ETA: ${etaMinutes}m ${etaSeconds}s)`;
                } else {
                    messageWithETA += ` (ETA: ${etaSeconds}s)`;
                }
            }
            
            showLoading(true, messageWithETA, percent, 'analysis', true);
            break;
            
        case 'result':
            showLoading(false, '', null, 'analysis');
            if (state.loadingTimeout) {
                clearTimeout(state.loadingTimeout);
                state.loadingTimeout = null;
            }
            console.log('Received result from WebSocket:', data);
            // Check if this result is for the current request
            if (data.request_id && state.currentRequestId && data.request_id !== state.currentRequestId) {
                console.log('Ignoring result for old request:', data.request_id, 'current:', state.currentRequestId);
                return;
            }
            // Check if result has valid data
            if (!data.result || !data.result.data || data.result.data.length === 0) {
                console.error('Received empty or invalid result:', data.result);
                showError('Received empty result from server');
                return;
            }
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            processResults(data.result);
            break;
            
        case 'error':
            showLoading(false, '', null, 'analysis');
            if (state.loadingTimeout) {
                clearTimeout(state.loadingTimeout);
                state.loadingTimeout = null;
            }
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            showError(data.error || 'An error occurred during analysis');
            break;
            
        case 'generation_result':
            console.log('Received generation result:', data);
            showLoading(false, '', null, 'generation');
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            handleGenerationResult(data.result);
            break;
            
        case 'generation_complete':
            console.log('Received generation complete:', data);
            showLoading(false, '', null, 'generation');
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            handleGenerationResult(data.result);
            break;
            
        case 'completed':
            // Handle analysis completion (new format from inference_service)
            console.log('Received analysis completed:', data);
            showLoading(false, '', null, 'analysis');
            if (state.loadingTimeout) {
                clearTimeout(state.loadingTimeout);
                state.loadingTimeout = null;
            }
            // Check if this result is for the current request
            if (data.request_id && state.currentRequestId && data.request_id !== state.currentRequestId) {
                console.log('Ignoring result for old request:', data.request_id, 'current:', state.currentRequestId);
                return;
            }
            // Check if result has valid data
            if (!data.result || !data.result.data || data.result.data.length === 0) {
                console.error('Received empty or invalid result:', data.result);
                showError('Received empty result from server');
                return;
            }
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            
            processResults(data.result);
            break;
            
        case 'generation_error':
            showLoading(false, '', null, 'generation');
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            showGenerationStatus('Error: ' + (data.error || 'Generation failed'), 'error');
            elements.generateBtn.disabled = false;
            break;
            
        case 'clear_loading':
            // Clear loading state for specified context
            if (data.context) {
                showLoading(false, '', null, data.context);
            }
            break;
            
        case 'interrupted':
            // Handle interrupt confirmation
            const interruptContext = data.context || 'analysis';
            showLoading(false, '', null, interruptContext);
            // Clean up request tracking
            if (data.request_id && state.activeRequests) {
                delete state.activeRequests[data.request_id];
            }
            showError(`${interruptContext === 'generation' ? 'Generation' : 'Analysis'} interrupted by user`);
            
            // Re-enable buttons
            if (interruptContext === 'generation') {
                elements.generateBtn.disabled = false;
            } else {
                elements.analyzeBtn.disabled = false;
            }
            break;
            
        case 'queue_update':
            // Update queue indicator
            const queueIndicator = document.getElementById('queueIndicator');
            const queueText = document.getElementById('queueText');
            
            if (queueIndicator && queueText) {
                if (data.queue_size > 0 || data.processing_requests > 0) {
                    queueIndicator.classList.remove('hidden');
                    
                    // Calculate user's position from the queued_ids list
                    let userPosition = null;
                    if (state.currentRequestId && data.queued_ids && Array.isArray(data.queued_ids)) {
                        const position = data.queued_ids.indexOf(state.currentRequestId);
                        if (position !== -1) {
                            userPosition = position + 1; // Convert to 1-indexed
                        }
                    }
                    
                    if (userPosition) {
                        queueText.textContent = `${userPosition}/${data.queue_size}`;
                    } else {
                        queueText.textContent = data.queue_size.toString();
                        if (data.processing_requests > 0) {
                            queueText.textContent += ` (${data.processing_requests} processing)`;
                        }
                    }
                } else {
                    queueIndicator.classList.add('hidden');
                }
            }
            break;
    }
};

// Interrupt function
const interruptComputation = (context) => {
    if (!state.currentRequestId) {
        console.error('No active request to interrupt');
        return;
    }
    
    console.log(`Interrupting ${context} for request:`, state.currentRequestId);
    
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            type: 'interrupt',
            request_id: state.currentRequestId,
            context: context
        }));
        
        // Update UI immediately
        showLoading(true, 'Interrupting...', null, context, false);
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
    
    console.log('Starting new analysis, current state:', {
        hasTranscripts: state.allTranscripts.length > 0,
        currentRequestId: state.currentRequestId,
        transcriptsLength: state.allTranscripts.length
    });
    
    // Reset UI for new analysis but preserve generations
    // Don't clear generations - we want to keep history of analyses
    state.allTranscripts = [];
    state.currentTranscriptIndex = 0;
    // state.generations = []; // REMOVED - keep previous analyses
    // state.currentGenerationIndex = 0; // REMOVED - will be set when results arrive
    elements.outputPlaceholder.classList.remove('hidden');
    elements.fullTextContainer.classList.add('hidden');
    elements.displayControls.style.display = 'none';
    elements.outputTable.classList.add('hidden');
    elements.transposedView.classList.add('hidden');
    elements.navigationContainer.classList.add('hidden');
    elements.bottomNavigationContainer.classList.add('hidden');
    elements.metadataDisplay.classList.add('hidden');
    elements.sidePrevBtn.classList.add('hidden');
    elements.sideNextBtn.classList.add('hidden');
    elements.columnExplanations.classList.add('hidden');
    // Don't hide generation tabs - keep them visible if we have previous analyses
    // elements.generationTabs.classList.add('hidden'); // REMOVED
    
    // Show initial loading state
    showLoading(true, 'Preparing analysis...', null, 'analysis');
    
    // Set a timeout to show a message if loading takes too long
    if (state.loadingTimeout) clearTimeout(state.loadingTimeout);
    state.loadingTimeout = setTimeout(() => {
        if (elements.analysisLoading.classList.contains('hidden') === false) {
            elements.analysisLoadingMessage.textContent = 'This is taking longer than usual. The model may be loading for the first time...';
        }
    }, 10000); // 10 seconds
    
    // Build options
    const options = {
        temperature: parseFloat(elements.temperature.value),
        optimize_explanations_config: {
            best_of_k: parseInt(elements.kRollouts.value),
            n_groups_per_rollout: elements.autoBatchSize.checked 
                ? Math.max(1, Math.floor(state.autoBatchSizeMax / parseInt(elements.kRollouts.value)))
                : parseInt(elements.batchSize.value),
            use_batched: true,
            temperature: parseFloat(elements.temperature.value)  // Use the same temperature as main
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
    
    // Always set rollout_batch_size to match the main batch size
    const calculatedBatchSize = elements.autoBatchSize.checked 
        ? Math.max(1, Math.floor(state.autoBatchSizeMax / parseInt(elements.kRollouts.value)))
        : parseInt(elements.batchSize.value);
    options.optimize_explanations_config.rollout_batch_size = calculatedBatchSize;
    
    if (elements.numSamples.value && elements.numSamples.value !== '1') {
        options.optimize_explanations_config.num_samples = parseInt(elements.numSamples.value);
    }
    
    // Add is_chat flag if chat-formatted
    if (elements.isChatFormatted.checked) {
        options.is_chat = true;
    }
    
    // Log the request
    RequestLogger.log('analysis', {
        text: text,
        options: options,
        model: state.modelInfo ? state.modelInfo.model_name : 'unknown'
    });
    
    // Send request
    state.ws.send(JSON.stringify({
        type: 'analyze',
        text: text,
        options: options
    }));
};

// Generation functions
const showGenerationStatus = (message, type = 'info') => {
    elements.generationStatus.textContent = message;
    elements.generationStatus.classList.remove('hidden', 'text-gray-600', 'text-green-600', 'text-red-600');
    elements.generationStatus.classList.add(
        type === 'error' ? 'text-red-600' : 
        type === 'success' ? 'text-green-600' : 
        'text-gray-600'
    );
};

const generate = () => {
    const text = elements.inputText.value.trim();
    if (!text) {
        showError('Please enter text or chat messages to generate from');
        return;
    }
    
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        showError('Not connected to server');
        return;
    }
    
    elements.generateBtn.disabled = true;
    showGenerationStatus('');
    showLoading(true, 'Generating continuations...', null, 'generation');
    
    const options = {
        num_tokens: parseInt(elements.numTokens.value),
        num_completions: parseInt(elements.numCompletions.value),
        temperature: parseFloat(elements.genTemperature.value),
        is_chat: elements.isChatFormatted.checked,
        return_full_text: true
    };
    
    // Log the request
    RequestLogger.log('generation', {
        text: text,
        options: options,
        model: state.modelInfo ? state.modelInfo.model_name : 'unknown'
    });
    
    state.ws.send(JSON.stringify({
        type: 'generate',
        text: text,
        options: options
    }));
};

const handleGenerationResult = (result) => {
    console.log('Generation result:', result);
    elements.generateBtn.disabled = false;
    
    // Handle both 'continuations' and 'completions' field names
    const continuations = result?.continuations || result?.completions;
    
    if (!result || !continuations || continuations.length === 0) {
        showGenerationStatus('No continuations generated', 'error');
        return;
    }
    
    showGenerationStatus(`Generated ${continuations.length} continuation(s)`, 'success');
    
    // Display continuations
    elements.continuationResults.classList.remove('hidden');
    elements.continuationsList.innerHTML = '';
    
    continuations.forEach((cont, idx) => {
        const div = document.createElement('div');
        div.className = 'bg-gray-50 p-3 rounded border border-gray-200';
        
        const header = document.createElement('div');
        header.className = 'flex justify-between items-center mb-2';
        header.innerHTML = `
            <span class="font-semibold text-sm text-gray-700">Continuation ${idx + 1}</span>
            <button class="use-continuation-btn text-sm px-3 py-1 bg-orange-500 text-white rounded hover:bg-orange-600" data-idx="${idx}">
                Use this
            </button>
        `;
        
        const content = document.createElement('div');
        content.className = 'text-sm text-gray-800 font-mono whitespace-pre-wrap';
        // Display with visible newlines but store original
        const displayContent = cont.replace(/\n/g, '\\n');
        content.textContent = displayContent;
        content.dataset.originalContent = cont; // Store original with real newlines
        
        div.appendChild(header);
        div.appendChild(content);
        elements.continuationsList.appendChild(div);
    });
    
    // Store continuations for later use
    state.generatedContinuations = continuations;
    
    // Add click handlers for "Use this" buttons
    document.querySelectorAll('.use-continuation-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.dataset.idx);
            const continuation = state.generatedContinuations[idx];
            
            console.log(`Using continuation ${idx}, length: ${continuation.length}`);
            
            // Update button to show it was clicked
            e.target.textContent = 'Using...';
            e.target.disabled = true;
            
            // Set the text with a small delay to let the UI update
            setTimeout(() => {
                elements.inputText.value = continuation;
                
                // Scroll to top of input
                elements.inputText.scrollTop = 0;
                elements.isChatFormatted.checked = false;
                elements.chatWarning.classList.add('hidden');
                
                // Check for chat format with another small delay
                setTimeout(() => {
                    checkForChatFormat();
                    
                    // Re-enable button
                    e.target.textContent = 'Use this';
                    e.target.disabled = false;
                    
                    // Show success message
                    showGenerationStatus('Continuation loaded into input box', 'success');
                }, 100);
            }, 10);
        });
    });
};

// Process results
const processResults = (result) => {
    console.log('Processing results:', result);
    console.log('Result data length:', result?.data?.length);
    console.log('Current allTranscripts before processing:', state.allTranscripts.length);
    
    // Validate result
    if (!result || typeof result !== 'object') {
        console.error('Invalid result object:', result);
        showError('Invalid result received from server');
        return;
    }
    
    // Check if data array is empty
    if (!result.data || result.data.length === 0) {
        console.error('Result has empty data array:', result);
        showError('No data received from server');
        return;
    }
    
    // Create transcript in expected format
    const transcript = {
        data: result.data || [],
        metadata: result.metadata || {}
    };
    
    console.log('Transcript created:', transcript);
    console.log('Data length:', transcript.data.length);
    console.log('First few data items:', transcript.data.slice(0, 3));
    
    // Store previous state for debugging
    const previousState = {
        allTranscripts: [...state.allTranscripts],
        currentTranscriptIndex: state.currentTranscriptIndex
    };
    
    // Handle multiple analyses properly
    if (state.generations.length > 0) {
        // We have previous analyses - add this as a new generation
        console.log('Adding new analysis as generation', state.generations.length + 1);
        
        // Add the new analysis as a new generation
        state.generations.push({
            transcripts: [transcript],
            timestamp: new Date()
        });
        state.currentGenerationIndex = state.generations.length - 1;
        
        // Set transcripts to the new analysis
        state.allTranscripts = [transcript];
        state.currentTranscriptIndex = 0;
    } else {
        // This is the first analysis
        console.log('Processing first analysis');
        state.allTranscripts = [transcript];
        state.currentTranscriptIndex = 0;
        
        // Create first generation
        state.generations.push({
            transcripts: [transcript],
            timestamp: new Date()
        });
        state.currentGenerationIndex = 0;
    }
    
    console.log('State updated from:', previousState, 'to:', {
        allTranscripts: state.allTranscripts,
        currentTranscriptIndex: state.currentTranscriptIndex,
        generations: state.generations.length
    });
    
    // Auto-enable salience if present
    const hasSalienceData = transcript.data.some(r => r.token_salience !== undefined);
    state.salienceColoringEnabled = hasSalienceData;
    
    // Set flag to skip render when setting checkbox programmatically
    state.skipSalienceRender = true;
    elements.salienceToggle.checked = hasSalienceData;
    // Reset flag after a short delay
    setTimeout(() => {
        state.skipSalienceRender = false;
    }, 50);
    
    // Enable export buttons
    elements.uploadBtn.disabled = false;
    elements.exportJsonBtn.disabled = false;
    
    // Create generation tabs if needed
    createGenerationTabs();
    
    console.log('About to render, allTranscripts:', state.allTranscripts);
    console.log('Generations after processing:', state.generations.length, 'Current index:', state.currentGenerationIndex);
    console.log('State before render:', JSON.stringify(state, null, 2));
    render(true); // Use immediate render for processResults
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
    updateColumnExplanations();
};

const updateColumnExplanations = () => {
    const explanations = {
        position: 'Token index in the input sequence',
        token: 'The actual text token being analysed',
        explanation: 'Human-readable interpretation of what the model is "thinking" at this token',
        mse: 'Mean Squared Error - measures how well the explanation reconstructs the model\'s internal state (lower is better)',
        relative_rmse: 'Root mean square error divided by the norm of the activation vector.',
        token_probability: 'Model\'s confidence in predicting this token (0-100%)',
        perplexity: 'Uncertainty measure - lower values indicate the model is more confident',
        cross_entropy: 'Information-theoretic measure of prediction quality (lower is better)',
        token_salience: 'Importance of this token - % increase in MSE when replaced with a space (higher = more important)',
        kl_divergence: 'KL divergence between original and reconstructed distributions',
        tuned_lens_top: 'Top prediction from tuned lens analysis',
        logit_lens_top: 'Top prediction from logit lens analysis'
    };
    
    const container = elements.columnExplanations.querySelector('.text-sm.text-gray-700.space-y-2');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Only show explanations for visible columns
    for (const [column, explanation] of Object.entries(explanations)) {
        if (state.columnVisibility[column]) {
            const p = document.createElement('p');
            p.innerHTML = `<strong>${column.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${explanation}`;
            container.appendChild(p);
        }
    }
    
    // Show/hide the explanations box if there are any visible columns
    const hasVisibleColumns = Object.values(state.columnVisibility).some(v => v);
    if (hasVisibleColumns && state.allTranscripts.length > 0) {
        elements.columnExplanations.classList.remove('hidden');
    } else {
        elements.columnExplanations.classList.add('hidden');
    }
};

let renderCount = 0;
let renderTimeout = null;
const render = (immediate = false) => {
    // Clear any pending render
    if (renderTimeout) {
        clearTimeout(renderTimeout);
        renderTimeout = null;
    }
    
    // If not immediate, debounce the render
    if (!immediate) {
        renderTimeout = setTimeout(() => {
            render(true);
        }, 10);
        return;
    }
    
    renderCount++;
    console.log(`Render called (${renderCount}), currentIndex:`, state.currentTranscriptIndex, 'transcripts:', state.allTranscripts);
    console.log('Render stack trace:', new Error().stack);
    
    // Don't render if we have no transcripts
    if (!state.allTranscripts || state.allTranscripts.length === 0) {
        console.log('No transcripts to render');
        return;
    }
    
    // Validate currentTranscriptIndex
    if (state.currentTranscriptIndex < 0 || state.currentTranscriptIndex >= state.allTranscripts.length) {
        console.error('Invalid currentTranscriptIndex:', state.currentTranscriptIndex, 'for transcripts length:', state.allTranscripts.length);
        state.currentTranscriptIndex = Math.max(0, Math.min(state.allTranscripts.length - 1, state.currentTranscriptIndex));
    }
    
    const currentItem = state.allTranscripts[state.currentTranscriptIndex];
    
    if (!currentItem || !currentItem.data || currentItem.data.length === 0) {
        console.log('Current item has no data, skipping render');
        console.log('currentItem:', currentItem);
        return;
    }
    
    console.log('Showing results containers');
    elements.outputPlaceholder.classList.add('hidden');
    elements.fullTextContainer.classList.remove('hidden');
    elements.displayControls.style.display = 'flex';
    
    console.log('Current item data:', currentItem);
    console.log('Elements:', {
        fullTextContainer: elements.fullTextContainer,
        outputTable: elements.outputTable,
        transposedView: elements.transposedView
    });
    
    // Render components using VisualizationCore
    VisualizationCore.renderMetadata({
        metadata: currentItem.metadata,
        container: elements.metadataDisplay
    });
    
    VisualizationCore.createColumnToggleUI({
        container: elements.columnToggleContainer,
        columnVisibility: state.columnVisibility,
        onChange: (column, visible) => {
            state.columnVisibility[column] = visible;
            updateColumnVisibility();
        }
    });
    
    // Apply column visibility settings
    updateColumnVisibility();
    
    // Update column explanations after rendering
    updateColumnExplanations();
    
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
        
        console.log('Rendering table with data:', currentItem.data);
        console.log('Table elements:', elements.tableHead, elements.tableBody);
        VisualizationCore.renderTable({
            data: currentItem.data,
            tableHead: elements.tableHead,
            tableBody: elements.tableBody,
            columnVisibility: state.columnVisibility,
            salienceColoringEnabled: state.salienceColoringEnabled
        });
        console.log('Table rendered');
    console.log('Table visibility after render:', {
        outputTable: elements.outputTable.classList.contains('hidden') ? 'hidden' : 'visible',
        outputTableDisplay: window.getComputedStyle(elements.outputTable).display,
        fullTextContainer: elements.fullTextContainer.classList.contains('hidden') ? 'hidden' : 'visible'
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
    
    if (state.currentService !== 'supabase' && (!apiKey || !collectionId)) {
        let errorMsg = 'Please provide ';
        if (state.currentService === 'pantry') {
            errorMsg += 'both a Pantry ID and a Basket Name';
        } else {
            errorMsg += 'both a JSONBin.io API Key and a Collection ID';
        }
        showError(errorMsg + ' in the Share tab.');
        switchTab('share');
        return;
    }
    
    if (state.allTranscripts.length === 0) {
        showError('No data to upload. Please analyse some text first.');
        return;
    }
    
    elements.uploadBtn.disabled = true;
    elements.uploadBtn.textContent = 'Uploading...';
    
    try {
        const dataToUpload = JSON.stringify(state.allTranscripts.length === 1 
            ? state.allTranscripts[0] 
            : state.allTranscripts);
            
        let binId;
        if (state.currentService === 'supabase') {
            binId = await StorageAdapters.Supabase.upload(dataToUpload);
        } else if (state.currentService === 'pantry') {
            binId = await StorageAdapters.Pantry.upload(dataToUpload, apiKey, collectionId);
        } else {
            binId = await StorageAdapters.JSONBin.upload(dataToUpload, apiKey, collectionId);
        }
        
        if (binId) {
            // Create URL for the standalone data viewer
            const shareUrl = `${window.location.origin}/data-viewer/?bin=${binId}`;
            
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
        showError('No data to export. Please analyse some text first.');
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
        
        // Set flag to skip render when setting checkbox programmatically
        state.skipSalienceRender = true;
        elements.salienceToggle.checked = hasSalienceData;
        setTimeout(() => {
            state.skipSalienceRender = false;
        }, 50);
        
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

// Check if input looks like chat format
const checkForChatFormat = () => {
    const text = elements.inputText.value.trim();
    
    // Skip check for very large texts to avoid hanging
    if (text.length > 50000) {
        console.log('Text too large for chat format check');
        return;
    }
    
    if (text.startsWith('[') && text.includes('"role"') && text.includes('"content"')) {
        // Only try parsing if it's not too large
        if (text.length < 10000) {
            try {
                const parsed = JSON.parse(text);
                if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].role) {
                    elements.chatWarning.classList.remove('hidden');
                    return;
                }
            } catch (e) {
                // Not valid JSON, but still looks like it might be chat format
            }
        }
        // Show warning based on pattern matching for large texts
        elements.chatWarning.classList.remove('hidden');
    } else {
        elements.chatWarning.classList.add('hidden');
    }
};

// Helper function to update batch size
const updateAutoBatchSize = () => {
    if (elements.autoBatchSize && elements.autoBatchSize.checked) {
        const k = parseInt(elements.kRollouts.value);
        const autoBatch = Math.max(1, Math.floor(state.autoBatchSizeMax / k));
        elements.batchSize.value = autoBatch;
        elements.batchSizeInfo.textContent = `${autoBatch} (auto)`;
        elements.batchSize.disabled = true;
    } else if (elements.batchSize) {
        elements.batchSize.disabled = false;
        elements.batchSizeInfo.textContent = elements.batchSize.value;
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
    
    // Check for chat format on input with debouncing
    let chatFormatCheckTimeout;
    elements.inputText.addEventListener('input', () => {
        clearTimeout(chatFormatCheckTimeout);
        chatFormatCheckTimeout = setTimeout(checkForChatFormat, 300); // 300ms debounce
    });
    
    elements.isChatFormatted.addEventListener('change', () => {
        if (elements.isChatFormatted.checked) {
            elements.chatWarning.classList.add('hidden');
            
            // Auto-format text to chat format if it's not already
            const text = elements.inputText.value.trim();
            if (text && !text.startsWith('[')) {
                // Check if it's already valid JSON chat format
                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].role) {
                        // Already valid chat format
                        return;
                    }
                } catch (e) {
                    // Not JSON, convert to chat format
                }
                
                // Convert plain text to chat format
                const chatFormat = [
                    {
                        "role": "user",
                        "content": text
                    }
                ];
                elements.inputText.value = JSON.stringify(chatFormat, null, 2);
                
                // Show a brief success message
                const oldText = elements.generationStatus.textContent;
                const oldClass = elements.generationStatus.className;
                showGenerationStatus('Converted to chat format', 'success');
                setTimeout(() => {
                    if (elements.generationStatus.textContent === 'Converted to chat format') {
                        elements.generationStatus.textContent = oldText;
                        elements.generationStatus.className = oldClass;
                    }
                }, 2000);
            }
        }
    });
    
    // Generation
    elements.generateBtn.addEventListener('click', generate);
    
    // Parameters
    elements.kRollouts.addEventListener('input', (e) => {
        const k = parseInt(e.target.value);
        elements.kRolloutsValue.textContent = k;
        updateAutoBatchSize();
        
        // Auto-adjust temperature based on k_rollouts
        if (k === 1) {
            elements.temperature.value = '0.1';
            elements.temperatureValue.textContent = '0.1';
        } else {
            elements.temperature.value = '1.0';
            elements.temperatureValue.textContent = '1.0';
        }
    });
    
    elements.autoBatchSize.addEventListener('change', (e) => {
        updateAutoBatchSize();
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
    elements.supabaseBtn.addEventListener('click', () => updateServiceUI('supabase'));
    elements.pantryBtn.addEventListener('click', () => updateServiceUI('pantry'));
    elements.jsonbinBtn.addEventListener('click', () => updateServiceUI('jsonbin'));
    elements.apiKeyInput.addEventListener('change', saveSettings);
    elements.collectionIdInput.addEventListener('change', saveSettings);
    
    // Reset button
    elements.resetBtn.addEventListener('click', resetModel);
    
    // Interrupt buttons
    elements.interruptGenerationBtn.addEventListener('click', () => interruptComputation('generation'));
    elements.interruptAnalysisBtn.addEventListener('click', () => interruptComputation('analysis'));
    
    // New Analysis button
    if (elements.newAnalysisBtn) {
        elements.newAnalysisBtn.addEventListener('click', () => {
            // Switch to analyze tab
            switchTab('analyze');
            // Scroll to top
            window.scrollTo(0, 0);
            // Focus on input text
            elements.inputText.focus();
        });
    }
    
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
        // Skip render if flag is set (programmatic change)
        if (!state.skipSalienceRender) {
            render();
        }
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
        // Detect service type from binId format
        let service, content;
        try {
            if (binId.includes('/')) {
                // Pantry format: basket/id
                service = 'pantry';
                updateServiceUI('pantry');
                const apiKey = elements.apiKeyInput.value;
                content = await StorageAdapters.Pantry.fetch(binId, apiKey);
            } else if (binId.match(/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/)) {
                // UUID format for Supabase
                service = 'supabase';
                updateServiceUI('supabase');
                content = await StorageAdapters.Supabase.fetch(binId);
            } else {
                // JSONBin format
                service = 'jsonbin';
                updateServiceUI('jsonbin');
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

// Expose RequestLogger to window for debugging
window.RequestLogger = RequestLogger;

// Initialize application
const initialize = () => {
    // Always default to Supabase
    updateServiceUI('supabase');
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Initialize model switcher
    const modelSwitcherContainer = document.getElementById('modelSwitcherContainer');
    if (modelSwitcherContainer && typeof ModelSwitcher !== 'undefined') {
        state.modelSwitcher = new ModelSwitcher(state.ws, modelSwitcherContainer);
        
        // Add event listeners
        state.modelSwitcher.addEventListener('switch-started', (data) => {
            console.log('Model switch started:', data);
            showLoading(true, 'Model is switching. All requests are queued...', null, 'analysis');
        });
        
        state.modelSwitcher.addEventListener('switch-completed', (data) => {
            console.log('Model switch completed:', data);
            showLoading(false, '', null, 'analysis');
            // Use a temporary success message (showError with green styling would be better)
            const successMsg = `Model switched successfully to ${data.model_name || data.model_id}!`;
            console.log(successMsg);
            // Could also update the status text
            if (elements.statusText) {
                const currentStatus = elements.statusText.textContent;
                elements.statusText.textContent = successMsg;
                setTimeout(() => {
                    elements.statusText.textContent = currentStatus;
                }, 3000);
            }
        });
        
        state.modelSwitcher.addEventListener('switch-failed', (data) => {
            console.log('Model switch failed:', data);
            showLoading(false, '', null, 'analysis');
            showError(`Model switch failed: ${data.error}`);
        });
    } else {
        console.warn('ModelSwitcher not available or container not found');
    }
    
    // Connect WebSocket
    connectWebSocket();
    
    // Initialize batch size if auto-calculate is enabled
    updateAutoBatchSize();
    
    // Load from URL if present
    loadFromURL();
};

// Start the application
initialize();
