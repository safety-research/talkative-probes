// Model Switcher Component for Talkative Autoencoder

class ModelSwitcher {
    constructor(wsConnection, containerElement) {
        this.ws = wsConnection;
        this.container = containerElement;
        this.currentModel = null;
        this.models = {};
        this.isSwitching = false;
        this.listeners = [];
        
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
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-lg font-semibold text-gray-800">Model Selection</h3>
                    <button id="refreshModelsBtn" class="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1" ${this.isSwitching ? 'disabled' : ''}>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                        Refresh
                    </button>
                </div>
                
                <div id="modelStatus" class="mb-3">
                    <div class="text-sm text-gray-600">Loading models...</div>
                </div>
                
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
        `;
        
        this.attachEventListeners();
        this.requestModelList();
    }
    
    // Attach event listeners
    attachEventListeners() {
        const refreshBtn = this.container.querySelector('#refreshModelsBtn');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        
        refreshBtn?.addEventListener('click', () => this.requestModelList());
        confirmBtn?.addEventListener('click', () => this.confirmSwitch());
    }
    
    // Request model list from server
    requestModelList() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showStatus('Not connected to server', 'error');
            return;
        }
        
        this.ws.send(JSON.stringify({ type: 'list_models' }));
    }
    
    // Handle model list response
    handleModelsList(data) {
        this.models = data.models;
        this.currentModel = data.current_model;
        this.isSwitching = data.is_switching || false;
        
        this.renderModelList();
        this.updateStatus();
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
            const isSelected = modelId === this.currentModel;
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
                        <div class="font-medium text-gray-900">${info.display_name}</div>
                        <div class="text-sm text-gray-600">${info.description}</div>
                        <div class="text-xs text-gray-500 mt-1">
                            Layer: ${info.layer} | 
                            GPU Memory: ~${info.estimated_gpu_memory}GB | 
                            Batch Size: ${info.batch_size}
                        </div>
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
        this.showWarning();
    }
    
    // Show warning
    showWarning() {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        
        warningEl?.classList.remove('hidden');
        confirmBtn?.classList.remove('hidden');
    }
    
    // Hide warning
    hideWarning() {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        
        warningEl?.classList.add('hidden');
        confirmBtn?.classList.add('hidden');
    }
    
    // Confirm switch
    confirmSwitch() {
        if (!this.selectedModel || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        this.ws.send(JSON.stringify({
            type: 'switch_model',
            model_id: this.selectedModel
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
        } else if (this.currentModel) {
            const modelInfo = this.models[this.currentModel];
            statusEl.innerHTML = `
                <div class="text-sm">
                    <span class="text-gray-600">Current model:</span>
                    <span class="font-medium text-gray-900 ml-1">${modelInfo?.display_name || this.currentModel}</span>
                    ${modelInfo?.layer ? `<span class="text-gray-500 ml-2">(Layer ${modelInfo.layer})</span>` : ''}
                </div>
            `;
        } else {
            statusEl.innerHTML = '<div class="text-sm text-gray-600">No model loaded</div>';
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
                
            case 'model_info_update':
                this.updateStatus();
                break;
        }
    }
}

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelSwitcher;
}