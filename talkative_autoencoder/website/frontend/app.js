// Configuration
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin; // Use same origin for production

const WS_URL = API_URL.replace('http', 'ws') + '/ws';

// State
let ws = null;
let currentRequestId = null;
let isTransposed = false;
let salienceColoringEnabled = false;
let currentResultData = null; // Store the last received result for re-rendering

// Column configuration
const columnNames = ["position", "token", "explanation", "kl_divergence", "mse", "relative_rmse", "tuned_lens_top", "explanation_structured", "token_salience", "logit_lens_top", "layer"];
let columnVisibility = {};

// Initialize column visibility
const setDefaultVisibility = () => {
    const defaultVisibleCols = ['token', 'explanation', 'mse'];
    columnNames.forEach(colName => {
        columnVisibility[colName] = defaultVisibleCols.includes(colName);
    });
};
setDefaultVisibility();

// DOM Elements
const elements = {
    inputText: document.getElementById('inputText'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    connectionStatus: document.getElementById('connectionStatus'),
    loading: document.getElementById('loading'),
    error: document.getElementById('error'),
    queuePosition: document.getElementById('queuePosition'),
    
    // Parameter controls
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
    
    // Visualization elements
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
    metadataDisplay: document.getElementById('metadata-display'),
    fullTextContainer: document.getElementById('full-text-container'),
    fullTextPlaceholder: document.getElementById('full-text-placeholder')
};

// Helper functions
const formatNewlines = (str) => {
    if (typeof str !== 'string') return str;
    return str.replace(/\\n|\n/g, '↵ ');
};

const getSalienceColor = (value, min = -0.1, max = 0.3) => {
    if (typeof value !== 'number' || isNaN(value)) return 'transparent';
    
    const v = Math.max(-1, Math.min(max, value));
    
    if (v >= 0) {
        const beta = 50;
        const norm = max > 0 ? Math.log10(1 + beta * v) / Math.log10(1 + beta * max) : 0;
        const g = Math.round(255 * (1 - norm));
        const b = g;
        return `rgb(255,${g},${b})`;
    } else {
        const norm = (-v) / 1;
        const r = Math.round(255 * (1 - norm));
        const b = r;
        return `rgb(${r},255,${b})`;
    }
};

// WebSocket connection
const connectWebSocket = () => {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        elements.connectionStatus.textContent = 'Connected';
        elements.connectionStatus.className = 'text-sm text-green-600';
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        elements.connectionStatus.textContent = 'Disconnected';
        elements.connectionStatus.className = 'text-sm text-red-600';
        setTimeout(connectWebSocket, 5000); // Auto-reconnect
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        
        if (data.type === 'queued') {
            elements.queuePosition.textContent = `Queue position: ${data.queue_position}`;
        } else if (data.type === 'status') {
            elements.queuePosition.textContent = data.message;
        } else if (data.type === 'result') {
            elements.loading.classList.remove('active');
            if (data.status === 'completed' && data.result) {
                currentResultData = data.result; // Store the result for re-rendering
                renderResults(data.result);
            } else if (data.error) {
                showError(data.error);
            }
        } else if (data.type === 'error') {
            elements.loading.classList.remove('active');
            showError(data.error);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        elements.connectionStatus.textContent = 'Error';
        elements.connectionStatus.className = 'text-sm text-red-600';
    };
};

// Error display
const showError = (message) => {
    elements.error.textContent = message;
    elements.error.style.display = 'block';
    setTimeout(() => {
        elements.error.style.display = 'none';
    }, 5000);
};

// Calculate batch size
const calculateBatchSize = () => {
    const k = parseInt(elements.kRollouts.value);
    const batchSize = Math.max(1, Math.floor(256 / k));
    elements.batchSize.value = batchSize;
    elements.batchSizeInfo.textContent = `Auto-calculated: ${batchSize}`;
    return batchSize;
};

// Analyze text
const analyzeText = () => {
    const text = elements.inputText.value.trim();
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showError('WebSocket not connected. Please wait...');
        return;
    }
    
    elements.loading.classList.add('active');
    elements.error.style.display = 'none';
    
    const k = parseInt(elements.kRollouts.value);
    const batchSize = elements.autoBatchSize.checked 
        ? calculateBatchSize() 
        : parseInt(elements.batchSize.value);
    
    const request = {
        type: 'analyze',
        text: text,
        options: {
            batch_size: batchSize,
            seed: parseInt(elements.seed.value),
            no_eval: elements.noEval.checked,
            tuned_lens: elements.tunedLens.checked,
            logit_lens_analysis: elements.logitLens.checked,
            temperature: parseFloat(elements.temperature.value),
            do_hard_tokens: elements.doHardTokens.checked,
            return_structured: true,
            no_kl: elements.noKL.checked,
            calculate_salience: elements.calculateSalience.checked,
            calculate_token_salience: elements.calculateSalience.checked,
            optimize_explanations_config: {
                use_batched: true,
                just_do_k_rollouts: k,
                batch_size_for_rollouts: parseInt(elements.rolloutBatchSize.value),
                temperature: parseFloat(elements.temperature.value),
                num_samples_per_iteration: parseInt(elements.numSamples.value)
            }
        }
    };
    
    console.log('Sending analysis request:', request);
    ws.send(JSON.stringify(request));
};

// Render results
const renderResults = (result) => {
    console.log('Rendering results:', result);
    
    if (result.metadata) {
        renderMetadata(result.metadata);
    }
    
    if (result.data && result.data.length > 0) {
        elements.outputPlaceholder.classList.add('hidden');
        elements.displayControls.style.display = 'flex';
        
        // Default salience toggle to true if data is present
        const hasSalienceData = result.data.some(r => r.token_salience !== undefined);
        salienceColoringEnabled = hasSalienceData;
        elements.salienceToggle.checked = hasSalienceData;
        
        renderFullTextBox(result.data);
        renderColumnToggles();
        
        if (isTransposed) {
            renderTransposedView(result.data);
        } else {
            renderTable(result.data);
        }
        
        updateColourbar();
    }
};

// Render metadata
const renderMetadata = (metadata) => {
    if (metadata && Object.keys(metadata).length > 0) {
        elements.metadataDisplay.classList.remove('hidden');
        let html = '<ul class="list-disc list-inside space-y-1">';
        if (metadata.model_name) {
            html += `<li><strong>Model:</strong> <span class="font-semibold">${metadata.model_name}</span></li>`;
        }
        if (metadata.device) {
            html += `<li><strong>Device:</strong> <span class="font-semibold">${metadata.device}</span></li>`;
        }
        if (metadata.batch_size) {
            html += `<li><strong>Batch Size:</strong> <span class="font-semibold">${metadata.batch_size}</span></li>`;
        }
        html += '</ul>';
        elements.metadataDisplay.innerHTML = html;
    } else {
        elements.metadataDisplay.classList.add('hidden');
    }
};

// Render full text box
const renderFullTextBox = (data) => {
    elements.fullTextContainer.innerHTML = '';
    if (!data || data.length === 0) {
        elements.fullTextContainer.appendChild(elements.fullTextPlaceholder);
        return;
    }
    elements.fullTextPlaceholder.classList.add('hidden');
    
    data.forEach((row) => {
        const tokenString = String(row.token || '');
        const parts = tokenString.split(/\\n|\n/);
        
        parts.forEach((part, index) => {
            if (part === '' && index === parts.length - 1 && parts.length > 1) return;
            
            const container = document.createElement('div');
            container.className = 'token-box-container';
            
            const tokenSpan = document.createElement('span');
            tokenSpan.className = 'token-box';
            
            const textContent = (index < parts.length - 1) ? part + ' ↵' : part;
            tokenSpan.textContent = textContent;
            
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            
            // Build colour-coded explanation for tooltip
            if (row.explanation_structured && Array.isArray(row.explanation_structured)) {
                tooltip.innerHTML = '';
                row.explanation_structured.forEach((tok, idx) => {
                    const span = document.createElement('span');
                    span.textContent = formatNewlines(tok) + ' ';
                    if (Array.isArray(row.token_salience) && idx < row.token_salience.length) {
                        const sv = row.token_salience[idx];
                        if (salienceColoringEnabled && typeof sv === 'number') {
                            span.style.backgroundColor = getSalienceColor(sv);
                        }
                    }
                    tooltip.appendChild(span);
                });
            } else {
                tooltip.textContent = formatNewlines(row.explanation || 'No explanation available.');
            }
            
            container.appendChild(tokenSpan);
            container.appendChild(tooltip);
            elements.fullTextContainer.appendChild(container);
            
            if (index < parts.length - 1) {
                elements.fullTextContainer.appendChild(document.createElement('br'));
            }
        });
    });
};

// Render table
const renderTable = (data) => {
    elements.tableHead.innerHTML = '';
    elements.tableBody.innerHTML = '';
    elements.outputTable.classList.remove('hidden');
    elements.transposedView.classList.add('hidden');
    
    // Create header
    const headerRow = document.createElement('tr');
    headerRow.className = "bg-[#EFEBE9]";
    columnNames.forEach((colName) => {
        const th = document.createElement('th');
        th.className = "px-2 py-2 text-left text-sm font-semibold text-[#5D4037] uppercase tracking-wider";
        th.dataset.columnName = colName;
        th.textContent = colName.replace(/_/g, ' ');
        
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle';
        th.appendChild(resizeHandle);
        
        headerRow.appendChild(th);
    });
    elements.tableHead.appendChild(headerRow);
    
    // Create rows
    data.forEach((row) => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-[#EFEBE9]/80 border-b border-[#EFEBE9]';
        
        columnNames.forEach(colName => {
            const td = document.createElement('td');
            td.className = "px-2 py-2 align-top font-mono text-sm text-[#5D4037]";
            td.dataset.columnName = colName;
            
            let value = row[colName] || '';
            
            if (['kl_divergence', 'mse', 'relative_rmse'].includes(colName)) {
                const num = parseFloat(value);
                value = isNaN(num) ? 'N/A' : num.toFixed(6);
            }
            
            if (colName === 'token_salience') {
                if (Array.isArray(value)) {
                    value = value.map(v => typeof v === 'number' ? v.toPrecision(3) : v).join(', ');
                } else if (typeof value === 'number') {
                    value = value.toPrecision(3);
                }
            }
            
            if (colName === 'explanation' && Array.isArray(row.explanation_structured)) {
                td.textContent = '';
                row.explanation_structured.forEach((w, idx) => {
                    const span = document.createElement('span');
                    span.className = 'explanation-word';
                    span.textContent = formatNewlines(w) + ' ';
                    
                    if (Array.isArray(row.token_salience) && idx < row.token_salience.length) {
                        const sv = row.token_salience[idx];
                        if (salienceColoringEnabled && typeof sv === 'number') {
                            span.style.backgroundColor = getSalienceColor(sv);
                        }
                    }
                    td.appendChild(span);
                });
            } else {
                td.textContent = formatNewlines(String(value));
            }
            
            if (colName === 'token') {
                td.className = "px-2 py-2 align-top font-mono text-sm font-semibold text-[#BF360C]";
            }
            
            tr.appendChild(td);
        });
        
        elements.tableBody.appendChild(tr);
    });
    
    updateColumnVisibility();
    applySpacing();
};

// Render transposed view
const renderTransposedView = (data) => {
    elements.transposedView.innerHTML = '';
    elements.outputTable.classList.add('hidden');
    elements.transposedView.classList.remove('hidden');
    elements.colWidthControl.classList.remove('hidden');
    
    const container = document.createElement('div');
    container.className = 'flex flex-row flex-wrap gap-x-2';
    const colWidth = parseInt(elements.colWidthSlider.value, 10);
    
    data.forEach((row) => {
        const colDiv = document.createElement('div');
        colDiv.className = 'flex flex-col items-start font-mono text-sm transpose-col';
        colDiv.style.width = `${colWidth}px`;
        
        const tokenWrapper = document.createElement('div');
        tokenWrapper.className = 'rotated-token-wrapper';
        
        const tokenDiv = document.createElement('div');
        tokenDiv.className = 'font-semibold text-base rotated-token text-[#BF360C]';
        tokenDiv.textContent = formatNewlines(row.token);
        
        tokenWrapper.appendChild(tokenDiv);
        colDiv.appendChild(tokenWrapper);
        
        let explanationWords = [];
        if (row.explanation_structured && Array.isArray(row.explanation_structured)) {
            explanationWords = row.explanation_structured;
        } else if(row.explanation) {
            explanationWords = formatNewlines(row.explanation).split(/[\s↵]+/);
        }
        
        explanationWords.forEach((word, wordIndex) => {
            const wordDiv = document.createElement('div');
            wordDiv.className = 'text-[#6D4C41] font-sans w-full text-left explanation-word';
            wordDiv.textContent = formatNewlines(word);
            
            if (salienceColoringEnabled && Array.isArray(row.token_salience) && wordIndex < row.token_salience.length) {
                const salienceValue = row.token_salience[wordIndex];
                if (typeof salienceValue === 'number' && !isNaN(salienceValue)) {
                    wordDiv.style.backgroundColor = getSalienceColor(salienceValue);
                    wordDiv.style.padding = '2px 4px';
                    wordDiv.style.borderRadius = '2px';
                    wordDiv.style.margin = '1px 0';
                }
            }
            
            colDiv.appendChild(wordDiv);
        });
        
        container.appendChild(colDiv);
    });
    
    elements.transposedView.appendChild(container);
    applySpacing();
};

// Column toggles
const renderColumnToggles = () => {
    elements.columnToggleContainer.innerHTML = '';
    if (isTransposed) return;
    
    const toggleButton = document.createElement('button');
    toggleButton.className = "px-4 py-2 text-sm secondary-btn font-semibold rounded-lg shadow-sm transition-all duration-200 transform hover:scale-105 flex items-center gap-2";
    toggleButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18"/></svg> Toggle Columns`;
    
    const dropdown = document.createElement('div');
    dropdown.className = "absolute mt-2 w-56 origin-top-right bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none hidden z-20 border border-slate-200";
    
    let checkboxesHTML = '';
    columnNames.forEach(colName => {
        checkboxesHTML += `
            <label class="flex items-center px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">
                <input type="checkbox" data-column="${colName}" class="h-4 w-4 rounded border-slate-300 text-orange-600 focus:ring-orange-500" ${columnVisibility[colName] ? 'checked' : ''}>
                <span class="ml-3 select-none">${colName.replace(/_/g, ' ')}</span>
            </label>
        `;
    });
    dropdown.innerHTML = checkboxesHTML;
    
    toggleButton.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.classList.toggle('hidden');
    });
    
    dropdown.addEventListener('change', (e) => {
        if (e.target.type === 'checkbox') {
            const columnName = e.target.dataset.column;
            columnVisibility[columnName] = e.target.checked;
            updateColumnVisibility();
        }
    });
    
    elements.columnToggleContainer.appendChild(toggleButton);
    elements.columnToggleContainer.appendChild(dropdown);
    
    // Close dropdown when clicking outside
    document.addEventListener('click', () => {
        dropdown.classList.add('hidden');
    });
};

// Update column visibility
const updateColumnVisibility = () => {
    for (const columnName in columnVisibility) {
        const visible = columnVisibility[columnName];
        document.querySelectorAll(`[data-column-name="${columnName}"]`).forEach(cell => {
            cell.style.display = visible ? '' : 'none';
        });
    }
};

// Apply spacing
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

// Update colorbar
const updateColourbar = () => {
    if (!salienceColoringEnabled) {
        elements.salienceColorbar.classList.add('hidden');
        return;
    }
    elements.salienceColorbar.classList.remove('hidden');
    elements.salienceColorbar.style.background = 'linear-gradient(to right, rgb(0,200,0) 0%, #ffffff 50%, rgb(255,0,0) 100%)';
};

// Event listeners
elements.analyzeBtn.addEventListener('click', analyzeText);

elements.inputText.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeText();
    }
});

// Parameter controls
elements.kRollouts.addEventListener('input', () => {
    elements.kRolloutsValue.textContent = elements.kRollouts.value;
    if (elements.autoBatchSize.checked) {
        calculateBatchSize();
    }
});

elements.temperature.addEventListener('input', () => {
    elements.temperatureValue.textContent = elements.temperature.value;
});

elements.autoBatchSize.addEventListener('change', () => {
    elements.batchSize.disabled = elements.autoBatchSize.checked;
    if (elements.autoBatchSize.checked) {
        calculateBatchSize();
    }
});

// Advanced settings toggle
elements.advancedToggle.addEventListener('click', () => {
    const isExpanded = elements.advancedSettings.classList.contains('expanded');
    if (isExpanded) {
        elements.advancedSettings.classList.remove('expanded');
        elements.advancedSettings.classList.add('collapsed');
        elements.advancedToggle.innerHTML = '▶ Show Advanced Settings';
    } else {
        elements.advancedSettings.classList.remove('collapsed');
        elements.advancedSettings.classList.add('expanded');
        elements.advancedToggle.innerHTML = '▼ Hide Advanced Settings';
    }
});

// Visualization controls
elements.transposeBtn.addEventListener('click', () => {
    isTransposed = !isTransposed;
    if (currentResultData) {
        renderResults(currentResultData);
    }
});

elements.salienceToggle.addEventListener('change', () => {
    salienceColoringEnabled = elements.salienceToggle.checked;
    updateColourbar();
    // Re-render current results
    if (currentResultData) {
        renderResults(currentResultData);
    }
});

elements.lineSpacingSlider.addEventListener('input', applySpacing);

elements.colWidthSlider.addEventListener('input', (e) => {
    const newWidth = e.target.value;
    document.querySelectorAll('.transpose-col').forEach(col => {
        col.style.width = `${newWidth}px`;
    });
});

// Initialize
connectWebSocket();
calculateBatchSize();