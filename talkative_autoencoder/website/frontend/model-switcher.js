// Model Switcher Component for Talkative Autoencoder

class ModelSwitcher {
    constructor(wsConnection, containerElement) {
        this.ws = wsConnection;
        this.container = containerElement;
        this.currentModel = null;
        this.models = {};
        this.isSwitching = false;
        this.listeners = [];
        this.cachedModels = [];
        this.isCollapsed = true; // Start collapsed by default
        
        this.render();
    }
    
    // Add event listener
    addEventListener(event, callback) {
        this.listeners.push({ event, callback });
    }
    
    // Emit event
    emit(event, data) {
        this.listeners
            .filter(l => l.event === event)
            .forEach(l => l.callback(data));
    }
    
    // Update WebSocket connection
    setWebSocket(ws) {
        this.ws = ws;
    }
    
    // Render the UI
    render() {
        this.container.innerHTML = `
            <div class="model-switcher bg-white rounded-lg shadow-md p-4 mb-4">
                <div class="flex items-center justify-between ${this.isCollapsed ? '' : 'mb-3'}">
                    <h3 class="text-lg font-semibold text-gray-800 flex items-center gap-2 cursor-pointer" id="modelSwitcherHeader">
                        <svg class="w-4 h-4 transform transition-transform ${this.isCollapsed ? '-rotate-90' : ''}" id="toggleIcon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                        <span>Model Selection</span>
                        <span id="collapsedModelInfo" class="text-sm font-normal text-gray-600 ${this.isCollapsed ? '' : 'hidden'}"></span>
                    </h3>
                    <button id="refreshModelsBtn" class="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 ${this.isCollapsed ? 'hidden' : ''}" ${this.isSwitching ? 'disabled' : ''}>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                        Refresh
                    </button>
                </div>
                
                <div id="modelStatus" class="${this.isCollapsed ? 'hidden' : 'mb-3'}">
                    <div class="text-sm text-gray-600">Loading models...</div>
                </div>
                
                <div id="modelSwitcherContent" class="${this.isCollapsed ? 'hidden' : ''}">
                    <div id="modelList" class="space-y-2 mb-3">
                        <!-- Model options will be inserted here -->
                    </div>
                    
                    <div id="switchWarning" class="hidden mb-3 space-y-2">
                        <div class="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                            <div class="flex items-start">
                                <svg class="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                                </svg>
                                <div class="text-sm text-yellow-800">
                                    <strong>Warning:</strong> Switching models will affect all users and may take a few minutes. Any in-progress requests will be queued.
                                </div>
                            </div>
                        </div>
                        <div class="p-3 bg-blue-50 border border-blue-200 rounded-md">
                            <div class="flex items-start">
                                <svg class="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                </svg>
                                <div class="text-sm text-blue-800">
                                    <strong>Note:</strong> Different models use different tokenizers. Chat formatting may need adjustment based on the selected model (Qwen vs Gemma).
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="switchProgress" class="hidden mb-3">
                        <div class="flex items-center justify-between text-sm text-gray-600 mb-1">
                            <span>Switching model...</span>
                            <span id="switchEta"></span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div id="switchProgressBar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <button id="confirmSwitchBtn" class="hidden w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed">
                        Confirm Model Switch
                    </button>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
        this.requestModelList();
    }
    
    // Attach event listeners
    attachEventListeners() {
        const refreshBtn = this.container.querySelector('#refreshModelsBtn');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        const header = this.container.querySelector('#modelSwitcherHeader');
        
        refreshBtn?.addEventListener('click', () => this.requestModelList());
        confirmBtn?.addEventListener('click', () => this.confirmSwitch());
        
        // Toggle collapse on header click
        header?.addEventListener('click', () => this.toggleCollapse());
    }
    
    // Toggle collapse state
    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
        
        const content = this.container.querySelector('#modelSwitcherContent');
        const status = this.container.querySelector('#modelStatus');
        const refreshBtn = this.container.querySelector('#refreshModelsBtn');
        const toggleIcon = this.container.querySelector('#toggleIcon');
        const headerDiv = this.container.querySelector('.flex.items-center.justify-between');
        const collapsedInfo = this.container.querySelector('#collapsedModelInfo');
        
        if (this.isCollapsed) {
            content?.classList.add('hidden');
            status?.classList.add('hidden');
            refreshBtn?.classList.add('hidden');
            toggleIcon?.classList.add('-rotate-90');
            toggleIcon?.classList.remove('rotate-0');
            headerDiv?.classList.remove('mb-3');
            collapsedInfo?.classList.remove('hidden');
        } else {
            content?.classList.remove('hidden');
            status?.classList.remove('hidden');
            refreshBtn?.classList.remove('hidden');
            toggleIcon?.classList.remove('-rotate-90');
            toggleIcon?.classList.add('rotate-0');
            headerDiv?.classList.add('mb-3');
            collapsedInfo?.classList.add('hidden');
        }
    }
    
    // Request model list from server
    requestModelList() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showStatus('Not connected to server', 'error');
            return;
        }
        
        this.ws.send(JSON.stringify({ type: 'list_models' }));
    }
    
    // Reload model registry (requires authentication)
    reloadModelRegistry() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showStatus('Not connected to server', 'error');
            return;
        }
        
        // Get API key
        let apiKey = localStorage.getItem('talkative_backend_api_key');
        if (!apiKey) {
            apiKey = prompt('Please enter the backend API key to reload model registry:');
            if (!apiKey) {
                alert('API key is required to reload model registry');
                return;
            }
            // Save for future use
            localStorage.setItem('talkative_backend_api_key', apiKey);
        }
        
        this.ws.send(JSON.stringify({
            type: 'reload_models',
            api_key: apiKey
        }));
    }
    
    // Handle model list response
    handleModelsList(data) {
        this.models = data.models;
        this.currentModel = data.current_model;
        this.isSwitching = data.is_switching || false;
        this.queueStats = data.queue_stats || null;
        
        this.renderModelList();
        this.updateStatus();
        
        // If we have a pending warning (from model selection), show it now
        if (this.pendingWarning) {
            this.pendingWarning = false;
            this.showWarning();
        }
    }
    
    // Check if a model is cached
    isCached(modelId) {
        return this.cachedModels.includes(modelId);
    }
    
    // Render model list
    renderModelList() {
        const modelListEl = this.container.querySelector('#modelList');
        if (!modelListEl) return;
        
        const modelEntries = Object.entries(this.models);
        if (modelEntries.length === 0) {
            modelListEl.innerHTML = '<div class="text-sm text-gray-500 italic">No models available</div>';
            return;
        }
        
        modelListEl.innerHTML = modelEntries.map(([modelId, info]) => {
            // Check against selectedModel if we have one (user selection), otherwise currentModel
            const isSelected = this.selectedModel ? modelId === this.selectedModel : modelId === this.currentModel;
            const isDisabled = this.isSwitching;
            
            return `
                <label class="flex items-start p-3 border rounded-md cursor-pointer transition-colors ${
                    isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                } ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}">
                    <input type="radio" name="model" value="${modelId}" 
                        ${isSelected ? 'checked' : ''} 
                        ${isDisabled ? 'disabled' : ''}
                        class="mt-1 mr-3 text-blue-600">
                    <div class="flex-1">
                        <div class="font-medium text-gray-900">
                            ${info.display_name}
                            ${this.isCached(modelId) ? '<span class="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">Cached</span>' : ''}
                        </div>
                        <div class="text-sm text-gray-600">${info.description}</div>
                        <div class="text-xs text-gray-500 mt-1">
                            Layer: ${info.layer} | 
                            GPU Memory: ~${info.estimated_gpu_memory}GB | 
                            Batch Size: ${info.batch_size}
                        </div>
                        ${info.checkpoint_filename ? `<div class="text-xs text-gray-400 mt-0.5 font-mono" style="font-size: 0.65rem; word-break: break-all;">${info.checkpoint_filename}</div>` : ''}
                    </div>
                </label>
            `;
        }).join('');
        
        // Add change listeners
        const radios = modelListEl.querySelectorAll('input[type="radio"]');
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => this.handleModelSelect(e.target.value));
        });
    }
    
    // Handle model selection
    handleModelSelect(modelId) {
        if (modelId === this.currentModel) {
            this.hideWarning();
            return;
        }
        
        this.selectedModel = modelId;
        
        // Refresh model list to get latest queue stats before showing warning
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'list_models' }));
            // Show warning will be called when we receive the updated list
            this.pendingWarning = true;
        } else {
            this.showWarning();
        }
    }
    
    // Show warning
    showWarning() {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        
        // Update warning content if there are active requests
        if (this.queueStats && (this.queueStats.queue_size > 0 || this.queueStats.processing_requests > 0)) {
            const activeRequestsDiv = warningEl.querySelector('.active-requests-warning');
            if (!activeRequestsDiv) {
                // Add active requests warning
                const newWarning = document.createElement('div');
                newWarning.className = 'active-requests-warning p-3 bg-red-50 border border-red-300 rounded-md mt-2';
                newWarning.innerHTML = `
                    <div class="flex items-start">
                        <svg class="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <div class="text-sm text-red-800">
                            <strong class="text-base">⚠️ ACTIVE REQUESTS IN PROGRESS!</strong><br>
                            There are currently <strong>${this.queueStats.queue_size} queued</strong> and 
                            <strong>${this.queueStats.processing_requests} processing</strong> requests.<br>
                            <span class="font-medium">Switching models now will interrupt these operations!</span>
                        </div>
                    </div>
                `;
                warningEl.appendChild(newWarning);
                
                // Make the button more prominent
                confirmBtn.textContent = `⚠️ Switch Anyway (${this.queueStats.queue_size + this.queueStats.processing_requests} active requests)`;
                confirmBtn.classList.add('bg-red-600', 'hover:bg-red-700');
                confirmBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
            }
        } else {
            // Remove active requests warning if it exists
            const activeRequestsDiv = warningEl.querySelector('.active-requests-warning');
            if (activeRequestsDiv) {
                activeRequestsDiv.remove();
            }
            // Reset button to normal
            confirmBtn.textContent = 'Confirm Model Switch';
            confirmBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
            confirmBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
        }
        
        warningEl?.classList.remove('hidden');
        confirmBtn?.classList.remove('hidden');
        
        // Scroll the warning into view
        warningEl?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Hide warning
    hideWarning() {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        
        // Remove any active requests warning
        const activeRequestsDiv = warningEl?.querySelector('.active-requests-warning');
        if (activeRequestsDiv) {
            activeRequestsDiv.remove();
        }
        
        // Reset button to normal state
        if (confirmBtn) {
            confirmBtn.textContent = 'Confirm Model Switch';
            confirmBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
            confirmBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
        }
        
        warningEl?.classList.add('hidden');
        confirmBtn?.classList.add('hidden');
        
        // Clear selected model and re-render to reset radio buttons
        this.selectedModel = null;
        this.renderModelList();
    }
    
    // Confirm switch
    confirmSwitch() {
        if (!this.selectedModel || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Get API key from localStorage or prompt user
        let apiKey = localStorage.getItem('talkative_backend_api_key');
        if (!apiKey) {
            apiKey = prompt('Please enter the backend API key to switch models:');
            if (!apiKey) {
                alert('API key is required to switch models');
                return;
            }
            // Save for future use
            localStorage.setItem('talkative_backend_api_key', apiKey);
        }
        
        this.ws.send(JSON.stringify({
            type: 'switch_model',
            model_id: this.selectedModel,
            api_key: apiKey
        }));
        
        this.hideWarning();
        this.showProgress();
    }
    
    // Show progress
    showProgress() {
        const progressEl = this.container.querySelector('#switchProgress');
        progressEl?.classList.remove('hidden');
        
        // Disable all inputs
        const inputs = this.container.querySelectorAll('input, button');
        inputs.forEach(input => input.disabled = true);
    }
    
    // Hide progress
    hideProgress() {
        const progressEl = this.container.querySelector('#switchProgress');
        progressEl?.classList.add('hidden');
        
        // Re-enable inputs
        const inputs = this.container.querySelectorAll('input, button');
        inputs.forEach(input => input.disabled = false);
    }
    
    // Update status
    updateStatus() {
        const statusEl = this.container.querySelector('#modelStatus');
        const collapsedInfo = this.container.querySelector('#collapsedModelInfo');
        
        if (!statusEl) return;
        
        if (this.isSwitching) {
            statusEl.innerHTML = `
                <div class="flex items-center text-sm text-orange-600">
                    <svg class="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Model is switching... All operations are queued.
                </div>
            `;
            if (collapsedInfo) {
                collapsedInfo.textContent = '(switching...)';
            }
        } else if (this.currentModel) {
            const modelInfo = this.models[this.currentModel];
            statusEl.innerHTML = `
                <div class="text-sm">
                    <span class="text-gray-600">Current model:</span>
                    <span class="font-medium text-gray-900 ml-1">${modelInfo?.display_name || this.currentModel}</span>
                    ${modelInfo?.layer ? `<span class="text-gray-500 ml-2">(Layer ${modelInfo.layer})</span>` : ''}
                    ${modelInfo?.checkpoint_filename ? `<div class="text-xs text-gray-400 mt-0.5 font-mono" style="font-size: 0.65rem;">${modelInfo.checkpoint_filename}</div>` : ''}
                </div>
            `;
            if (collapsedInfo) {
                collapsedInfo.textContent = modelInfo?.display_name ? `(${modelInfo.display_name})` : `(${this.currentModel})`;
            }
        } else {
            statusEl.innerHTML = '<div class="text-sm text-gray-600">No model loaded</div>';
            if (collapsedInfo) {
                collapsedInfo.textContent = '(no model)';
            }
        }
    }
    
    // Show status message
    showStatus(message, type = 'info') {
        const statusEl = this.container.querySelector('#modelStatus');
        if (!statusEl) return;
        
        const colors = {
            info: 'text-gray-600',
            error: 'text-red-600',
            success: 'text-green-600'
        };
        
        statusEl.innerHTML = `<div class="text-sm ${colors[type]}">${message}</div>`;
    }
    
    // Handle switch status updates
    handleSwitchStatus(data) {
        switch (data.status) {
            case 'starting':
                this.showStatus('Starting model switch...', 'info');
                this.emit('switch-started', data);
                break;
                
            case 'completed':
                this.showStatus('Model switch completed!', 'success');
                this.hideProgress();
                this.selectedModel = null;
                this.requestModelList(); // Refresh the list
                this.emit('switch-completed', data);
                break;
                
            case 'failed':
                this.showStatus(`Model switch failed: ${data.error}`, 'error');
                this.hideProgress();
                this.emit('switch-failed', data);
                break;
        }
    }
    
    // Handle WebSocket messages
    handleMessage(data) {
        switch (data.type) {
            case 'models_list':
                this.handleModelsList(data);
                break;
                
            case 'model_switch_status':
                this.handleSwitchStatus(data);
                break;
                
            case 'model_switch_acknowledged':
                this.showStatus(data.message, 'info');
                break;
                
            case 'model_switch_complete':
                this.handleSwitchStatus({ status: 'completed', ...data });
                break;
                
            case 'model_switch_error':
                this.handleSwitchStatus({ status: 'failed', ...data });
                break;
                
            case 'models_reloaded':
                this.showStatus(data.message || 'Model registry reloaded successfully', 'success');
                this.handleModelsList(data);  // Update the model list
                break;
                
            case 'error':
                // Handle authentication errors and other errors
                if (data.error && data.error.includes('Authentication required')) {
                    // Clear saved API key if authentication failed
                    localStorage.removeItem('talkative_backend_api_key');
                    this.showStatus('Authentication failed. Please check your API key.', 'error');
                } else {
                    this.showStatus(data.error || 'An error occurred', 'error');
                }
                break;
                
            case 'model_info_update':
                // Update cached models list if provided
                if (data.cached_models) {
                    this.cachedModels = data.cached_models;
                }
                this.updateStatus();
                break;
        }
    }
}

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelSwitcher;
}