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
        this.groupCollapseStates = {}; // Track collapse state for each group
        
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
            <style>
                .info-btn {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background-color: #e5e7eb;
                    color: #6b7280;
                    font-size: 10px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .info-btn:hover {
                    background-color: #3b82f6;
                    color: white;
                }
                .model-switcher-tooltip {
                    position: fixed;
                    z-index: 10000;
                    background-color: #1f2937;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    max-width: 300px;
                    pointer-events: none;
                    opacity: 0;
                    visibility: hidden;
                    transition: opacity 0.2s, visibility 0.2s;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .model-switcher-tooltip.show {
                    opacity: 1;
                    visibility: visible;
                }
                .model-switcher-tooltip::before {
                    content: '';
                    position: absolute;
                    top: -4px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 0;
                    height: 0;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-bottom: 4px solid #1f2937;
                }
            </style>
            <div class="model-switcher bg-white rounded-lg shadow-md p-4 mb-4">
                <div class="flex items-center justify-between ${this.isCollapsed ? '' : 'mb-3'}">
                    <h3 class="text-lg font-semibold text-gray-800 flex items-center gap-2 cursor-pointer" id="modelSwitcherHeader">
                        <svg class="w-4 h-4 transform transition-transform ${this.isCollapsed ? '-rotate-90' : ''}" id="toggleIcon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                        <span>Model/Layer Selection</span>
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
                    <div id="withinGroupInfo" class="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
                        <div class="flex items-start">
                            <svg class="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <div class="text-sm text-blue-800">
                                <strong>Quick Guide:</strong>
                                <ul class="mt-1 ml-4 list-disc space-y-1">
                                    <li><strong>Within-Group:</strong> Switch models instantly (same base model already in VRAM)</li>
                                    <li><strong>Between Groups:</strong> Moves new base model to VRAM. Takes 1-2 min, affects all users</li>
                                    <li><strong>Queue all analyses:</strong> Run one text through every model in a group</li>
                                    <li><strong>Unload:</strong> Free RAM (cached groups stay in system memory, not GPU)</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
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
        this.setupTooltips();
    }
    
    // Setup tooltip functionality
    setupTooltips() {
        // Check if tooltip already exists
        let tooltip = document.getElementById('model-switcher-tooltip');
        if (!tooltip) {
            // Create tooltip element only if it doesn't exist
            tooltip = document.createElement('div');
            tooltip.id = 'model-switcher-tooltip';
            tooltip.className = 'model-switcher-tooltip';
            document.body.appendChild(tooltip);
            
            // Add global event listeners only once
            document.addEventListener('mouseenter', (e) => {
                const infoBtn = e.target.closest('.info-btn');
                if (infoBtn && infoBtn.dataset.tooltip) {
                    const rect = infoBtn.getBoundingClientRect();
                    tooltip.textContent = infoBtn.dataset.tooltip;
                    
                    // Position tooltip
                    let top = rect.bottom + 8;
                    let left = rect.left + rect.width / 2;
                    
                    // Show tooltip temporarily to get dimensions
                    tooltip.style.visibility = 'hidden';
                    tooltip.classList.add('show');
                    
                    const tooltipRect = tooltip.getBoundingClientRect();
                    
                    // Check if tooltip would go off bottom of viewport
                    if (top + tooltipRect.height > window.innerHeight - 10) {
                        // Position above the button instead
                        top = rect.top - tooltipRect.height - 8;
                    }
                    
                    // Check horizontal bounds
                    if (left - tooltipRect.width / 2 < 10) {
                        left = tooltipRect.width / 2 + 10;
                    } else if (left + tooltipRect.width / 2 > window.innerWidth - 10) {
                        left = window.innerWidth - tooltipRect.width / 2 - 10;
                    }
                    
                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                    tooltip.style.transform = 'translateX(-50%)';
                    tooltip.style.visibility = 'visible';
                }
            }, true);
            
            document.addEventListener('mouseleave', (e) => {
                const infoBtn = e.target.closest('.info-btn');
                if (infoBtn) {
                    tooltip.classList.remove('show');
                }
            }, true);
        }
    }
    
    // Attach event listeners
    attachEventListeners() {
        const refreshBtn = this.container.querySelector('#refreshModelsBtn');
        const confirmBtn = this.container.querySelector('#confirmSwitchBtn');
        const preloadBtn = this.container.querySelector('#preloadGroupBtn');
        const header = this.container.querySelector('#modelSwitcherHeader');
        
        refreshBtn?.addEventListener('click', () => {
            this.showStatus('Refreshing model list...', 'info');
            this.requestModelList(true);
        });
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
    requestModelList(refresh = false) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showStatus('Not connected to server', 'error');
            return;
        }
        
        // Request appropriate endpoint based on API version
        this.ws.send(JSON.stringify({ 
            type: this.apiVersion === 'v2' ? 'list_model_groups' : 'list_models',
            refresh: refresh 
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
        
        // Update device states for multi-GPU support
        if (data.model_status?.device_states) {
            this.deviceStates = data.model_status.device_states;
        }
        
        this.renderModelGroups();
        this.updateStatus();
        this.updateMemoryInfo(data.model_status);
        
        if (this.pendingWarning) {
            this.pendingWarning = false;
            this.showWarning();
        }
        
        // If we connected during a switch, show the banner and emit event
        if (wasAlreadySwitching) {
            console.log('Connected during model switch, showing banner and emitting event');
            
            // Show the active switch banner for new users
            const switchingGroup = data.current_group || 'unknown';
            const switchingDevice = data.system_state?.devices?.[0]?.device || '';
            const deviceInfo = switchingDevice ? ` on ${switchingDevice.replace('cuda:', 'GPU ')}` : '';
            this.showActiveSwitchBanner(switchingGroup, deviceInfo);
            
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
        
        // Add device status display if we have device info
        let deviceStatusHtml = '';
        if (this.deviceStates && Object.keys(this.deviceStates).length > 1) {
            deviceStatusHtml = `
                <div class="mb-3 p-2 bg-gray-50 rounded-lg">
                    <div class="text-xs font-medium text-gray-600 mb-2">GPU Status:</div>
                    <div class="flex flex-wrap gap-2">
                        ${Object.entries(this.deviceStates).map(([device, state]) => {
                            const gpuNum = device.replace('cuda:', 'GPU ');
                            const groupName = state.current_group_id ? 
                                this.modelGroups.find(g => g.group_id === state.current_group_id)?.group_name || state.current_group_id : 
                                'Empty';
                            const isLoading = state.is_switching;
                            const statusColor = isLoading ? 'yellow' : (state.current_group_id ? 'green' : 'gray');
                            
                            return `
                                <div class="flex items-center gap-1 px-2 py-1 bg-white rounded border border-gray-200">
                                    <span class="text-xs font-medium">${gpuNum}:</span>
                                    <span class="text-xs ${isLoading ? 'text-yellow-600' : (state.current_group_id ? 'text-green-600' : 'text-gray-500')}">
                                        ${isLoading ? '⏳ Loading...' : groupName}
                                    </span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }
        
        groupsListEl.innerHTML = deviceStatusHtml + this.modelGroups.map(group => {
            const isCurrentGroup = group.group_id === this.currentGroup;
            const groupCached = this.cachedGroups?.includes(group.group_id);
            const isGroupCollapsed = Object.prototype.hasOwnProperty.call(this.groupCollapseStates, group.group_id)
                ? this.groupCollapseStates[group.group_id]
                : true;
            
            // Find which devices have this group loaded
            const devicesWithGroup = [];
            if (this.deviceStates) {
                Object.entries(this.deviceStates).forEach(([device, state]) => {
                    if (state.current_group_id === group.group_id && !state.is_switching) {
                        devicesWithGroup.push(device.replace('cuda:', 'GPU '));
                    }
                });
            }
            
            return `
                <div class="model-group border rounded-lg ${isCurrentGroup ? 'border-blue-500' : 'border-gray-300'}">
                    <div class="px-3 py-2 bg-gray-50 border-b border-gray-200">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <button class="group-toggle-btn p-0.5 hover:bg-gray-200 rounded" data-group-id="${group.group_id}" title="Click to expand/collapse this group">
                                    <svg class="w-4 h-4 transform transition-transform ${isGroupCollapsed ? '-rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                                    </svg>
                                </button>
                                <h4 class="font-medium text-gray-900">
                                    ${group.group_name}
                                    ${devicesWithGroup.length > 0 ? `<span class="ml-2 text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">${devicesWithGroup.join(', ')}</span>` : ''}
                                    ${groupCached && devicesWithGroup.length === 0 ? '<span class="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">Cached</span>' : ''}
                                </h4>
                                <div class="flex items-center gap-1">
                                    ${group.group_id ? `<button class="queue-all-btn text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed" data-group-id="${group.group_id}">
                                        Queue all analyses
                                    </button>` : ''}
                                    <span class="info-btn" data-tooltip="Runs your text through all models in this group. Creates one tab per model. Cancel via X on pending tabs.">i</span>
                                </div>
                                ${groupCached && !isCurrentGroup ? `<div class="flex items-center gap-1">
                                    <button class="unload-group-btn text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors" data-group-id="${group.group_id}">
                                        Unload
                                    </button>
                                    <span class="info-btn" data-tooltip="Frees RAM by removing this group. Only the active group uses GPU VRAM. Reloading takes ~30s.">i</span>
                                </div>` : ''}
                            </div>
                            <span class="text-xs text-gray-500">${group.base_model}</span>
                        </div>
                        ${group.description ? `<p class="text-xs text-gray-600 mt-1">${group.description}</p>` : ''}
                    </div>
                    <div class="divide-y divide-gray-100 ${isGroupCollapsed ? 'hidden' : ''}">
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
        
        // Add queue all button listeners
        const queueAllButtons = groupsListEl.querySelectorAll('.queue-all-btn');
        queueAllButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const groupId = e.target.getAttribute('data-group-id');
                this.queueAllAnalysesForGroup(groupId);
            });
        });
        
        // Add group toggle button listeners
        const toggleButtons = groupsListEl.querySelectorAll('.group-toggle-btn');
        toggleButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const groupId = e.currentTarget.getAttribute('data-group-id');
                this.toggleGroupCollapse(groupId);
            });
        });
        
        // Add unload group button listeners
        const unloadButtons = groupsListEl.querySelectorAll('.unload-group-btn');
        unloadButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const groupId = e.target.getAttribute('data-group-id');
                this.unloadGroup(groupId);
            });
        });
        
        // Re-setup tooltips for dynamically added elements
        this.setupTooltips();
    }
    
    // Render individual model option
    renderModelOption(model, group) {
        const isSelected = model.id === this.currentModel;
        const isDisabled = this.isSwitching;
        // Use new clearer flags
        const isGpuLoaded = model.is_gpu_loaded || model.is_loaded;  // Fallback to is_loaded for compatibility
        const isCpuCached = model.is_cpu_cached;
        
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
                        ${isGpuLoaded ? '<span class="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">GPU</span>' : ''}
                        ${isCpuCached && !isGpuLoaded ? '<span class="text-xs bg-yellow-100 text-yellow-800 px-1.5 py-0.5 rounded">Cached</span>' : ''}
                        ${model.is_current ? '<span class="text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">Current</span>' : ''}
                    </div>
                    ${model.description ? `<div class="text-sm text-gray-600">${model.description}</div>` : ''}
                    <div class="text-xs text-gray-500 mt-1">
                        Layer: ${model.layer} ${model.layer !== model.name.match(/L(\d+)/)?.[1] ? '(analyzes layer ' + model.layer + ' of the model)' : ''}
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
    
    // Toggle group collapse state
    toggleGroupCollapse(groupId) {
        // Toggle the collapse state
        this.groupCollapseStates[groupId] = !this.groupCollapseStates[groupId];
        
        // Re-render the groups to reflect the new state
        this.renderModelGroups();
    }
    
    // Unload a group from memory
    unloadGroup(groupId) {
        const group = this.modelGroups.find(g => g.group_id === groupId);
        if (!group) return;
        
        const confirmed = confirm(`Unload ${group.group_name} from RAM? This frees memory but takes ~30s to reload when needed again.`);
        if (!confirmed) return;
        
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showNotification('Not connected to server', 'error');
            return;
        }
        
        // Send unload request
        this.ws.send(JSON.stringify({
            type: 'unload_group',
            group_id: groupId
        }));
        
        // Show notification
        this.showNotification(`Unloading ${group.group_name} from memory...`, 'info');
    }
    
    // Queue analyses for all models in a group
    queueAllAnalysesForGroup(groupId) {
        // Check for null groupId
        if (!groupId) {
            console.error('No group ID provided for queue all');
            return;
        }
        
        // Find the group
        const group = this.modelGroups.find(g => g.group_id === groupId);
        if (!group) {
            console.error('Group not found:', groupId);
            return;
        }
        
        // Disable all queue buttons
        const allQueueButtons = this.container.querySelectorAll('.queue-all-btn');
        allQueueButtons.forEach(btn => {
            btn.disabled = true;
            if (btn.getAttribute('data-group-id') === groupId) {
                btn.textContent = 'Queueing...';
            }
        });
        
        // Re-enable buttons after a delay
        setTimeout(() => {
            allQueueButtons.forEach(btn => {
                btn.disabled = false;
                btn.textContent = 'Queue all analyses';
            });
        }, group.models.length * 100 + 1000); // Wait for all requests plus buffer
        
        // Emit event requesting to queue analyses for all models
        this.emit('queue-group-analyses', {
            group_id: groupId,
            group_name: group.group_name,
            models: group.models.map(m => ({
                id: m.id,
                name: m.name,
                layer: m.layer
            }))
        });
        
        // Show notification
        this.showNotification(`Queueing analyses for all ${group.models.length} models in ${group.group_name}`, 'info');
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
        const deviceInfo = data.device ? ` on ${data.device.replace('cuda:', 'GPU ')}` : '';
        
        switch (data.status) {
            case 'starting':
                this.isSwitching = true;
                this.showProgress();
                this.showStatus(`Switching to group ${data.group_id}${deviceInfo}... This affects all users.`, 'warning');
                this.updateStatus();
                
                // Show active switch banner
                this.showActiveSwitchBanner(data.group_id, deviceInfo);
                
                this.emit('group-switch-started', data);
                break;
                
            case 'progress':
                // Keep-alive progress update during long operations
                this.isSwitching = true;  // Ensure switching state is set
                this.showActiveSwitchBanner(data.group_id, deviceInfo);  // Show banner for users joining mid-switch
                
                if (data.message) {
                    // Show the progress message
                    this.showStatus(`${data.message}${deviceInfo}`, 'info');
                } else if (data.error) {
                    // The error field contains the progress message (backward compatibility)
                    this.showStatus(`${data.error}${deviceInfo}`, 'info');
                }
                // This keeps the WebSocket connection alive
                break;
                
            case 'completed':
                this.isSwitching = false;
                this.hideProgress();
                this.hideActiveSwitchBanner();  // Hide the banner
                this.currentGroup = data.group_id;
                // Update current model if we were switching to it
                if (this.selectedModel && this.model_to_group.get(this.selectedModel) === data.group_id) {
                    this.currentModel = this.selectedModel;
                    this.selectedModel = null;
                    this.selectedGroup = null;
                }
                this.showStatus(`Group switch completed${deviceInfo}!`, 'success');
                this.requestModelList();
                this.updateStatus();
                this.emit('group-switch-completed', data);
                break;
                
            case 'failed':
                this.isSwitching = false;
                this.hideProgress();
                this.hideActiveSwitchBanner();  // Hide the banner
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
                
            case 'group_unload_complete':
                // Handle group unload completion
                this.showNotification(`${data.group_name || data.group_id} unloaded from memory`, 'success');
                this.requestModelList(); // Refresh to show updated cache status
                break;
                
            case 'group_unload_error':
                // Handle group unload error
                this.showNotification(`Failed to unload group: ${data.error}`, 'error');
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
                        <div class="mt-3">
                            <button id="cancelGroupSwitch" class="px-3 py-1 text-sm bg-orange-600 text-white rounded hover:bg-orange-700 transition-colors">
                                Cancel Switch
                            </button>
                        </div>
                    </div>
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
    
    // Show active switch banner
    showActiveSwitchBanner(groupId, deviceInfo) {
        // Remove any existing banner
        const existingBanner = document.getElementById('groupSwitchActiveWarning');
        if (existingBanner) {
            existingBanner.remove();
        }
        
        // Create active switch banner
        const activeWarning = document.createElement('div');
        activeWarning.id = 'groupSwitchActiveWarning';
        activeWarning.className = 'fixed top-4 right-4 z-50 max-w-md';
        activeWarning.innerHTML = `
            <div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded shadow-lg">
                <div class="flex items-start">
                    <svg class="w-6 h-6 mr-3 flex-shrink-0 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    <div class="flex-1">
                        <p class="font-bold">⚠️ Group Switch in Progress</p>
                        <p class="text-sm mt-1">Switching to ${groupId}${deviceInfo}</p>
                        <p class="text-xs mt-2 text-yellow-700">This affects ALL users. The system may be temporarily unavailable.</p>
                        <p class="text-xs mt-1 text-yellow-600">Estimated time: 1-2 minutes</p>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(activeWarning);
        
        // Auto-remove after switch completes (will be removed by hideActiveSwitchBanner)
    }
    
    // Hide active switch banner
    hideActiveSwitchBanner() {
        const banner = document.getElementById('groupSwitchActiveWarning');
        if (banner) {
            banner.remove();
        }
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