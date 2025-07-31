// Enhanced Model Switcher Component with Group Support

class GroupedModelSwitcher {
    constructor(wsConnection, containerElement, apiVersion = 'v2') {
        this.ws = wsConnection;
        this.container = containerElement;
        this.apiVersion = apiVersion; // 'v1' for legacy, 'v2' for grouped
        this.currentModel = null;
        this.currentGroup = null;
        this.modelGroups = [];
        this.isSwitching = false;
        this.listeners = [];
        this.cachedModels = [];
        this.isCollapsed = true;
        
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
                    <div class="flex items-center gap-2">
                        <button id="preloadGroupBtn" class="text-sm text-green-600 hover:text-green-800 flex items-center gap-1 ${this.isCollapsed ? 'hidden' : ''}" ${this.isSwitching ? 'disabled' : ''}>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
                            </svg>
                            Preload Group
                        </button>
                        <button id="refreshModelsBtn" class="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 ${this.isCollapsed ? 'hidden' : ''}" ${this.isSwitching ? 'disabled' : ''}>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh
                        </button>
                    </div>
                </div>
                
                <div id="modelStatus" class="${this.isCollapsed ? 'hidden' : 'mb-3'}">
                    <div class="text-sm text-gray-600">Loading models...</div>
                </div>
                
                <div id="modelSwitcherContent" class="${this.isCollapsed ? 'hidden' : ''}">
                    <div id="modelGroupsList" class="space-y-3 mb-3">
                        <!-- Model groups will be inserted here -->
                    </div>
                    
                    <div id="memoryInfo" class="hidden mb-3 p-3 bg-gray-50 border border-gray-200 rounded-md">
                        <div class="text-sm">
                            <div class="font-medium text-gray-700 mb-1">Memory Status</div>
                            <div id="memoryDetails" class="space-y-1 text-xs text-gray-600"></div>
                        </div>
                    </div>
                    
                    <div id="switchWarning" class="hidden mb-3 space-y-2">
                        <div class="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                            <div class="flex items-start">
                                <svg class="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                                </svg>
                                <div class="text-sm text-yellow-800">
                                    <strong>Warning:</strong> Switching models will affect all users. 
                                    <span id="switchTypeInfo"></span>
                                </div>
                            </div>
                        </div>
                        <div id="groupSwitchInfo" class="hidden p-3 bg-blue-50 border border-blue-200 rounded-md">
                            <div class="flex items-start">
                                <svg class="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                                </svg>
                                <div class="text-sm text-blue-800">
                                    <strong>Fast Switch:</strong> Models within the same group share a base model. 
                                    Switching within a group is much faster (~10s) than switching between groups (~1-2min).
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
        const preloadBtn = this.container.querySelector('#preloadGroupBtn');
        const header = this.container.querySelector('#modelSwitcherHeader');
        
        refreshBtn?.addEventListener('click', () => this.requestModelList());
        confirmBtn?.addEventListener('click', () => this.confirmSwitch());
        preloadBtn?.addEventListener('click', () => this.preloadCurrentGroup());
        header?.addEventListener('click', () => this.toggleCollapse());
    }
    
    // Toggle collapse state
    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
        
        const content = this.container.querySelector('#modelSwitcherContent');
        const status = this.container.querySelector('#modelStatus');
        const refreshBtn = this.container.querySelector('#refreshModelsBtn');
        const preloadBtn = this.container.querySelector('#preloadGroupBtn');
        const toggleIcon = this.container.querySelector('#toggleIcon');
        const headerDiv = this.container.querySelector('.flex.items-center.justify-between');
        const collapsedInfo = this.container.querySelector('#collapsedModelInfo');
        
        if (this.isCollapsed) {
            content?.classList.add('hidden');
            status?.classList.add('hidden');
            refreshBtn?.classList.add('hidden');
            preloadBtn?.classList.add('hidden');
            toggleIcon?.classList.add('-rotate-90');
            toggleIcon?.classList.remove('rotate-0');
            headerDiv?.classList.remove('mb-3');
            collapsedInfo?.classList.remove('hidden');
        } else {
            content?.classList.remove('hidden');
            status?.classList.remove('hidden');
            refreshBtn?.classList.remove('hidden');
            preloadBtn?.classList.remove('hidden');
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
        
        // Request appropriate endpoint based on API version
        this.ws.send(JSON.stringify({ 
            type: this.apiVersion === 'v2' ? 'list_model_groups' : 'list_models' 
        }));
    }
    
    // Handle grouped model list response
    handleModelGroupsList(data) {
        this.modelGroups = data.groups || [];
        this.currentGroup = data.current_group;
        
        // Build model_to_group mapping
        this.model_to_group = new Map();
        this.modelGroups.forEach(group => {
            group.models.forEach(model => {
                this.model_to_group.set(model.id, group.group_id);
            });
        });
        
        // Set current model if we have a current group but no current model
        if (this.currentGroup && !this.currentModel) {
            // Find the first loaded model in the current group
            const currentGroupData = this.modelGroups.find(g => g.group_id === this.currentGroup);
            if (currentGroupData && currentGroupData.models.length > 0) {
                // Prefer loaded models, otherwise use the first one
                const loadedModel = currentGroupData.models.find(m => m.is_loaded);
                this.currentModel = loadedModel ? loadedModel.id : currentGroupData.models[0].id;
                console.log('Auto-selected model:', this.currentModel);
                
                // Emit model-selected event
                const modelInfo = this.getModelInfo(this.currentModel);
                if (modelInfo) {
                    this.emit('model-selected', {
                        model_id: this.currentModel,
                        model_name: modelInfo.name,
                        is_initial: true
                    });
                }
            }
        }
        
        // Check if we're connecting during a switch
        const wasAlreadySwitching = data.is_switching && !this.isSwitching;
        this.isSwitching = data.is_switching || false;
        
        // Update cache info from model status
        if (data.model_status?.cache_info) {
            this.cachedGroups = data.model_status.cache_info.groups_loaded || [];
            this.cachedModels = data.model_status.cache_info.models_cached || [];
        }
        
        this.renderModelGroups();
        this.updateStatus();
        this.updateMemoryInfo(data.model_status);
        
        if (this.pendingWarning) {
            this.pendingWarning = false;
            this.showWarning();
        }
        
        // If we connected during a switch, emit the switch-started event
        if (wasAlreadySwitching) {
            console.log('Connected during model switch, emitting switch-started event');
            this.emit('switch-started', {
                model_id: data.switching_to || this.currentModel,
                status: 'in_progress'
            });
        }
        
        // Check if there's a queued group switch
        if (data.queued_switch) {
            console.log('Connected with queued group switch:', data.queued_switch);
            // Show the queued switch notification
            this.handleGroupSwitchQueued({
                ...data.queued_switch,
                message: `Group switch queued at position ${data.queued_switch.queue_position}. Will start after ${data.queued_switch.active_requests + data.queued_switch.queued_ahead} request(s) complete.`
            });
        }
    }
    
    // Render grouped model list
    renderModelGroups() {
        const groupsListEl = this.container.querySelector('#modelGroupsList');
        if (!groupsListEl) return;
        
        if (this.modelGroups.length === 0) {
            groupsListEl.innerHTML = '<div class="text-sm text-gray-500 italic">No models available</div>';
            return;
        }
        
        groupsListEl.innerHTML = this.modelGroups.map(group => {
            const isCurrentGroup = group.group_id === this.currentGroup;
            const groupCached = this.cachedGroups?.includes(group.group_id);
            
            return `
                <div class="model-group border rounded-lg overflow-hidden ${isCurrentGroup ? 'border-blue-500' : 'border-gray-300'}">
                    <div class="px-3 py-2 bg-gray-50 border-b border-gray-200">
                        <div class="flex items-center justify-between">
                            <h4 class="font-medium text-gray-900">
                                ${group.group_name}
                                ${groupCached ? '<span class="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">Cached</span>' : ''}
                            </h4>
                            <span class="text-xs text-gray-500">${group.base_model}</span>
                        </div>
                        ${group.description ? `<p class="text-xs text-gray-600 mt-1">${group.description}</p>` : ''}
                    </div>
                    <div class="divide-y divide-gray-100">
                        ${group.models.map(model => this.renderModelOption(model, group)).join('')}
                    </div>
                </div>
            `;
        }).join('');
        
        // Add change listeners
        const radios = groupsListEl.querySelectorAll('input[type="radio"]');
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                const [groupId, modelId] = e.target.value.split('::');
                this.handleModelSelect(groupId, modelId);
            });
        });
    }
    
    // Render individual model option
    renderModelOption(model, group) {
        const isSelected = model.id === this.currentModel;
        const isDisabled = this.isSwitching;
        const isCached = model.is_loaded || this.cachedModels?.includes(model.id);
        
        return `
            <label class="flex items-start p-3 cursor-pointer transition-colors hover:bg-gray-50 ${
                isSelected ? 'bg-blue-50' : ''
            } ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}">
                <input type="radio" name="model" value="${group.group_id}::${model.id}" 
                    ${isSelected ? 'checked' : ''} 
                    ${isDisabled ? 'disabled' : ''}
                    class="mt-1 mr-3 text-blue-600">
                <div class="flex-1">
                    <div class="flex items-center gap-2">
                        <span class="font-medium text-gray-900">${model.name}</span>
                        ${isCached ? '<span class="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">Loaded</span>' : ''}
                        ${model.is_current ? '<span class="text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">Current</span>' : ''}
                    </div>
                    ${model.description ? `<div class="text-sm text-gray-600">${model.description}</div>` : ''}
                    <div class="text-xs text-gray-500 mt-1">
                        Layer: ${model.layer}
                    </div>
                    ${model.checkpoint_filename ? `<div class="text-xs text-gray-400 mt-0.5" style="font-size: 0.7em;" ${model.checkpoint_full ? `title="${model.checkpoint_full}"` : ''}>${model.checkpoint_filename}</div>` : ''}
                </div>
            </label>
        `;
    }
    
    // Handle model selection
    handleModelSelect(groupId, modelId) {
        if (modelId === this.currentModel) {
            return;
        }
        
        // Check if this requires a group switch
        const requiresGroupSwitch = groupId !== this.currentGroup;
        
        if (requiresGroupSwitch) {
            // Group switch affects all users - show warning
            this.selectedGroup = groupId;
            this.selectedModel = modelId;
            this.showWarning(false);  // false = not within group
        } else {
            // Within-group switch is local only
            this.selectModelLocally(groupId, modelId);
        }
    }
    
    // Show warning for group switch
    showWarning(isWithinGroup = false) {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        const switchTypeInfo = this.container.querySelector('#switchTypeInfo');
        const groupSwitchInfo = this.container.querySelector('#groupSwitchInfo');
        
        // Group switches always show warning
        switchTypeInfo.textContent = 'This will switch the model group for ALL users and may take 1-2 minutes.';
        groupSwitchInfo?.classList.add('hidden');
        
        warningEl?.classList.remove('hidden');
        confirmBtn?.classList.remove('hidden');
        
        // Update button
        if (confirmBtn) {
            confirmBtn.textContent = 'Confirm Group Switch';
            confirmBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
            confirmBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
        }
        
        warningEl?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Hide warning
    hideWarning() {
        const warningEl = this.container.querySelector('#switchWarning');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        const groupSwitchInfo = this.container.querySelector('#groupSwitchInfo');
        
        warningEl?.classList.add('hidden');
        confirmBtn?.classList.add('hidden');
        groupSwitchInfo?.classList.add('hidden');
        
        // Reset button to normal state
        if (confirmBtn) {
            confirmBtn.textContent = 'Confirm Model Switch';
            confirmBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
            confirmBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
        }
        
        // Clear selected model and re-render
        this.selectedModel = null;
        this.selectedGroup = null;
        this.renderModelGroups();
    }
    
    // Get model info by ID
    getModelInfo(modelId) {
        for (const group of this.modelGroups) {
            const model = group.models.find(m => m.id === modelId);
            if (model) {
                return model;
            }
        }
        return null;
    }
    
    // Show notification
    showNotification(message, type = 'info') {
        // You can implement a proper notification system here
        // For now, just log it
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    // Select model locally (within same group)
    selectModelLocally(groupId, modelId) {
        console.log('Selecting model within group:', modelId);
        
        // Update local state
        const previousModel = this.currentModel;
        this.currentModel = modelId;
        
        // Get model info
        const modelInfo = this.getModelInfo(modelId);
        const displayName = modelInfo ? modelInfo.name : modelId;
        
        // Show notification
        this.showNotification(`Selected ${displayName} (fast switch)`, 'success');
        
        // Update UI
        this.renderModelGroups();
        this.updateStatus();
        
        // Emit event for local model selection
        this.emit('model-selected', {
            model_id: modelId,
            previous_model: previousModel,
            model_name: displayName,
            is_local: true
        });
    }
    
    // Confirm group switch (affects all users)
    confirmSwitch() {
        if (!this.selectedModel || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Show immediate notification
        this.showNotification('Switching group... This affects all users.', 'warning');
        
        // Send group switch request
        this.ws.send(JSON.stringify({
            type: this.apiVersion === 'v2' ? 'switch_model_grouped' : 'switch_model',
            model_id: this.selectedModel,
            group_id: this.selectedGroup
        }));
        
        this.hideWarning();
        this.showProgress();
    }
    
    // Preload current group
    preloadCurrentGroup() {
        if (!this.currentGroup || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        const confirmPreload = confirm(`Preload all models in the "${this.currentGroup}" group? This will speed up future switches within this group.`);
        if (!confirmPreload) return;
        
        this.ws.send(JSON.stringify({
            type: 'preload_group',
            group_id: this.currentGroup
        }));
        
        this.showStatus('Preloading group models...', 'info');
    }
    
    // Update memory info display
    updateMemoryInfo(modelStatus) {
        const memoryInfoEl = this.container.querySelector('#memoryInfo');
        const memoryDetailsEl = this.container.querySelector('#memoryDetails');
        
        if (!memoryInfoEl || !memoryDetailsEl || !modelStatus?.cache_info) {
            return;
        }
        
        const cacheInfo = modelStatus.cache_info;
        const memoryUsage = modelStatus.memory || {};
        
        memoryDetailsEl.innerHTML = `
            ${memoryUsage.allocated_memory_gb ? `<div>GPU Memory: ${memoryUsage.allocated_memory_gb.toFixed(1)}GB / ${memoryUsage.reserved_memory_gb?.toFixed(1) || '??'}GB</div>` : ''}
            <div>Groups on GPU: ${cacheInfo.groups_loaded?.filter(g => cacheInfo.base_locations?.[g] === 'cuda').length || 0}</div>
            <div>Groups on CPU: ${cacheInfo.groups_loaded?.filter(g => cacheInfo.base_locations?.[g] === 'cpu').length || 0}</div>
            <div>Models cached: ${cacheInfo.models_cached?.length || 0}</div>
        `;
        
        memoryInfoEl.classList.remove('hidden');
    }
    
    // Show/hide progress
    showProgress() {
        const progressEl = this.container.querySelector('#switchProgress');
        progressEl?.classList.remove('hidden');
        
        const inputs = this.container.querySelectorAll('input, button');
        inputs.forEach(input => input.disabled = true);
    }
    
    hideProgress() {
        const progressEl = this.container.querySelector('#switchProgress');
        progressEl?.classList.add('hidden');
        
        const inputs = this.container.querySelectorAll('input, button');
        inputs.forEach(input => input.disabled = false);
    }
    
    // Update status display
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
            // Find current model info
            let modelInfo = null;
            let groupInfo = null;
            
            for (const group of this.modelGroups) {
                const model = group.models.find(m => m.id === this.currentModel);
                if (model) {
                    modelInfo = model;
                    groupInfo = group;
                    break;
                }
            }
            
            if (modelInfo && groupInfo) {
                statusEl.innerHTML = `
                    <div class="text-sm">
                        <span class="text-gray-600">Current:</span>
                        <span class="font-medium text-gray-900 ml-1">${groupInfo.group_name} - ${modelInfo.name}</span>
                        <span class="text-gray-500 ml-2">(Layer ${modelInfo.layer})</span>
                    </div>
                `;
                if (collapsedInfo) {
                    collapsedInfo.textContent = `(${modelInfo.name})`;
                }
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
                this.selectedGroup = null;
                this.requestModelList();
                this.emit('switch-completed', data);
                break;
                
            case 'failed':
                this.showStatus(`Model switch failed: ${data.error}`, 'error');
                this.hideProgress();
                this.emit('switch-failed', data);
                break;
        }
    }
    
    // Handle group switch status updates
    handleGroupSwitchStatus(data) {
        switch (data.status) {
            case 'starting':
                this.isSwitching = true;
                this.showProgress();
                this.showStatus(`Switching to group ${data.group_id}... This affects all users.`, 'warning');
                this.updateStatus();
                this.emit('group-switch-started', data);
                break;
                
            case 'completed':
                this.isSwitching = false;
                this.hideProgress();
                this.currentGroup = data.group_id;
                // Update current model if we were switching to it
                if (this.selectedModel && this.model_to_group.get(this.selectedModel) === data.group_id) {
                    this.currentModel = this.selectedModel;
                    this.selectedModel = null;
                    this.selectedGroup = null;
                }
                this.showStatus('Group switch completed!', 'success');
                this.requestModelList();
                this.updateStatus();
                this.emit('group-switch-completed', data);
                break;
                
            case 'failed':
                this.isSwitching = false;
                this.hideProgress();
                this.showStatus(data.error || 'Group switch failed', 'error');
                this.updateStatus();
                this.emit('group-switch-failed', data);
                break;
        }
    }
    
    // Handle WebSocket messages
    handleMessage(data) {
        switch (data.type) {
            case 'model_groups_list':
                this.handleModelGroupsList(data);
                break;
                
            case 'models_list':
                // Handle legacy format if needed
                if (this.apiVersion === 'v1') {
                    this.handleLegacyModelsList(data);
                }
                break;
                
            case 'model_switch_status':
                this.handleSwitchStatus(data);
                break;
                
            case 'model_switch_complete':
                // Handle switch completion
                this.hideProgress();
                this.currentModel = data.model_id;
                this.isSwitching = false;
                
                // Check if this was a group switch
                const newGroup = this.model_to_group.get(data.model_id);
                if (newGroup && newGroup !== this.currentGroup) {
                    this.currentGroup = newGroup;
                    this.emit('group-switch-completed', {
                        model_id: data.model_id,
                        group_id: newGroup,
                        model_info: data.model_info
                    });
                } else {
                    this.emit('switch-completed', {
                        model_id: data.model_id,
                        model_info: data.model_info
                    });
                }
                
                this.showStatus(data.message || 'Model switch completed!', 'success');
                this.requestModelList(); // Refresh model list
                break;
                
            case 'model_switch_error':
                // Handle switch error
                this.hideProgress();
                this.isSwitching = false;
                this.showStatus(data.error || 'Model switch failed', 'error');
                this.emit('switch-failed', {
                    model_id: data.model_id,
                    error: data.error
                });
                break;
                
            case 'group_switch_status':
                this.handleGroupSwitchStatus(data);
                break;
                
            case 'group_preload_complete':
                this.showStatus(`Group "${data.group_id}" preloaded successfully!`, 'success');
                this.requestModelList(); // Refresh to show updated cache status
                break;
                
            case 'group_switch_queued':
                // Handle queued group switch
                this.handleGroupSwitchQueued(data);
                break;
                
            case 'group_switch_starting':
                // Handle group switch actually starting
                this.handleGroupSwitchStarting(data);
                break;
                
            case 'error':
                this.showStatus(data.error || 'An error occurred', 'error');
                break;
        }
    }
    
    // Handle group switch queued
    handleGroupSwitchQueued(data) {
        console.log('Group switch queued:', data);
        
        // Store the queued request ID
        this.queuedSwitchRequestId = data.request_id;
        this.queuedSwitchModelId = data.model_id;
        
        // Show queue notification with cancel button
        const queuedWarning = document.createElement('div');
        queuedWarning.id = 'groupSwitchQueuedWarning';
        queuedWarning.className = 'fixed top-4 right-4 z-50 max-w-md';
        queuedWarning.innerHTML = `
            <div class="bg-orange-100 border-l-4 border-orange-500 text-orange-700 p-4 rounded shadow-lg">
                <div class="flex items-start">
                    <svg class="w-6 h-6 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <div class="flex-1">
                        <p class="font-bold">Group Switch Queued</p>
                        <p class="text-sm mt-1">${data.message}</p>
                        <p class="text-xs mt-2 text-orange-600">This will affect all users once it starts.</p>
                    </div>
                    <button id="cancelGroupSwitch" class="ml-3 text-orange-600 hover:text-orange-800">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `;
        
        // Remove any existing warning
        const existingWarning = document.getElementById('groupSwitchQueuedWarning');
        if (existingWarning) {
            existingWarning.remove();
        }
        
        // Add to document body
        document.body.appendChild(queuedWarning);
        
        // Add cancel handler
        const cancelBtn = queuedWarning.querySelector('#cancelGroupSwitch');
        cancelBtn.addEventListener('click', () => this.cancelQueuedSwitch());
        
        // Update status in model switcher
        this.showStatus(`Group switch queued. Waiting for ${data.active_requests} active request(s)...`, 'warning');
    }
    
    // Cancel queued group switch
    cancelQueuedSwitch() {
        if (!this.queuedSwitchRequestId || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        console.log('Cancelling queued group switch:', this.queuedSwitchRequestId);
        
        this.ws.send(JSON.stringify({
            type: 'cancel_request',
            request_id: this.queuedSwitchRequestId
        }));
        
        // Remove the warning
        const warning = document.getElementById('groupSwitchQueuedWarning');
        if (warning) {
            warning.remove();
        }
        
        // Clear stored IDs
        this.queuedSwitchRequestId = null;
        this.queuedSwitchModelId = null;
        
        // Update status
        this.showStatus('Group switch cancelled', 'info');
    }
    
    // Handle group switch starting
    handleGroupSwitchStarting(data) {
        console.log('Group switch starting:', data);
        
        // Remove queued warning
        const queuedWarning = document.getElementById('groupSwitchQueuedWarning');
        if (queuedWarning) {
            queuedWarning.remove();
        }
        
        // Show progress
        this.isSwitching = true;
        this.showProgress();
        this.showStatus('Group switch in progress...', 'info');
        
        // Emit event for loading bar
        this.emit('group-switch-started', data);
    }
    
    // Handle legacy model list format (for backward compatibility)
    handleLegacyModelsList(data) {
        // Convert flat model list to grouped format
        const groups = {};
        
        for (const [modelId, info] of Object.entries(data.models)) {
            const family = info.model_family || 'Unknown';
            if (!groups[family]) {
                groups[family] = {
                    group_id: family.toLowerCase(),
                    group_name: family,
                    base_model: '',
                    models: []
                };
            }
            
            groups[family].models.push({
                id: modelId,
                name: info.display_name || modelId,
                description: info.description,
                layer: info.layer,
                is_loaded: this.cachedModels.includes(modelId),
                is_current: modelId === data.current_model
            });
        }
        
        this.modelGroups = Object.values(groups);
        this.currentModel = data.current_model;
        this.isSwitching = data.is_switching || false;
        
        this.renderModelGroups();
        this.updateStatus();
    }
}

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GroupedModelSwitcher;
}