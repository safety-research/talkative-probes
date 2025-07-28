/**
 * Visualization Core Module
 * Shared visualization functionality for both standalone and integrated versions
 */

// Define VisualizationCore as an IIFE
const VisualizationCore = (function() {
    'use strict';
    
    // Core configuration
    const config = {
        colors: {
            salience: {
                negative: 'rgb(0, 200, 0)',
                neutral: '#ffffff',
                positive: 'rgb(255, 0, 0)'
            }
        }
    };
    
    // Utility functions
    const formatNewlines = (str) => {
        if (typeof str !== 'string') return str;
        return str.replace(/\\n|\n/g, '↵ ');
    };
    
    const getSalienceColor = (value, min = -0.1, max = 0.3) => {
        if (typeof value !== 'number' || isNaN(value)) return 'transparent';
        
        const v = Math.max(-1, Math.min(max, value));
        
        if (v >= 0) {
            // Log-scaled red side
            const beta = 50;
            const norm = max > 0 ? Math.log10(1 + beta * v) / Math.log10(1 + beta * max) : 0;
            const g = Math.round(255 * (1 - norm));
            const b = g;
            return `rgb(255,${g},${b})`;
        } else {
            // Linear green side
            const norm = (-v) / 1;
            const r = Math.round(255 * (1 - norm));
            const b = r;
            return `rgb(${r},255,${b})`;
        }
    };
    
    const getTooltipTextColor = (salVal) => {
        return '#3E2723';
    };
    
    const copyToClipboard = (text, element) => {
        navigator.clipboard.writeText(text).then(() => {
            showCopyFeedback(element);
        });
    };
    
    const showCopyFeedback = (element) => {
        const rect = element.getBoundingClientRect();
        const feedback = document.createElement('div');
        feedback.className = 'copy-feedback';
        feedback.textContent = 'Copied!';
        feedback.style.position = 'absolute';
        feedback.style.left = `${rect.left + window.scrollX}px`;
        feedback.style.top = `${rect.top + window.scrollY - 25}px`;
        feedback.style.zIndex = '9999';
        
        document.body.appendChild(feedback);
        feedback.classList.add('show');
        
        setTimeout(() => {
            feedback.remove();
        }, 1500);
    };
    
    // Compute salience range (5-95 percentile)
    const computeSalienceRange = (rows) => {
        const raw = rows.flatMap(r => {
            if (Array.isArray(r.token_salience)) return r.token_salience;
            if (typeof r.token_salience === 'number') return [r.token_salience];
            return [];
        }).filter(v => typeof v === 'number' && !isNaN(v));
        
        if (raw.length === 0) return { min: -0.1, max: 0.1 };
        
        const sorted = [...raw].sort((a, b) => a - b);
        const low = sorted[Math.floor(sorted.length * 0.05)];
        const high = sorted[Math.floor(sorted.length * 0.95)];
        
        return { min: Math.min(-0.05, low, -1), max: high };
    };
    
    // Column visibility management
    const getDefaultVisibility = () => ({
        'position': false,
        'predicted': false,
        'target': false,
        'token': true,
        'rank': false,
        'loss': false,
        'mse': true,
        'ce': false,
        'kl': false,
        'kl_divergence': false,
        'relative_rmse': false,
        'max_predicted_prob': false,
        'predicted_prob': false,
        'explanation': true,
        'decoder_completions': false,
        'pred_token': false,
        'salience_plot': false,
        'token_salience': false,
        'explanation_concatenated': false,
        'explanation_structured': false,
        'tuned_lens_top': false,
        'logit_lens_top': false,
        'layer': false
    });
    
    // Create token box element with tooltip
    const createTokenBox = (token, explanation, tokenSalience, salienceColoringEnabled, row) => {
        const container = document.createElement('div');
        container.className = 'token-box-container';
        
        const tokenSpan = document.createElement('span');
        tokenSpan.className = 'token-box';
        tokenSpan.textContent = token;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        
        // Build color-coded explanation for tooltip
        if (row.explanation_structured && Array.isArray(row.explanation_structured)) {
            tooltip.innerHTML = '';
            row.explanation_structured.forEach((tok, idx) => {
                const span = document.createElement('span');
                span.textContent = formatNewlines(tok) + ' ';
                if (Array.isArray(row.token_salience) && idx < row.token_salience.length) {
                    const sv = row.token_salience[idx];
                    if (salienceColoringEnabled && typeof sv === 'number') {
                        span.style.backgroundColor = getSalienceColor(sv);
                        span.style.color = getTooltipTextColor(sv);
                    }
                    if (typeof sv === 'number') {
                        span.title = `Decoded: ${tok}\nSalience: ${sv.toFixed(3)}\nMSE: ${row.mse?.toFixed ? row.mse.toFixed(3) : row.mse}`;
                    }
                }
                tooltip.appendChild(span);
            });
        } else {
            tooltip.textContent = formatNewlines(explanation || 'No explanation available.');
        }
        
        container.appendChild(tokenSpan);
        container.appendChild(tooltip);
        
        return container;
    };
    
    // Rendering functions
    const renderMetadata = ({ metadata, container }) => {
        if (!metadata || Object.keys(metadata).length === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        container.innerHTML = '';
        
        // Create metadata section
        const metadataSection = document.createElement('div');
        metadataSection.className = 'bg-amber-50 border border-amber-200 rounded-lg p-4';
        
        const title = document.createElement('h3');
        title.className = 'text-sm font-semibold text-[#5D4037] mb-3';
        title.textContent = 'Analysis Metadata';
        metadataSection.appendChild(title);
        
        const list = document.createElement('div');
        list.className = 'space-y-1';
        
        // Define the order and formatting for metadata fields
        const metadataOrder = ['model_name', 'orig_model_name', 'checkpoint_path', 'layer', 'timestamp'];
        const formattedKeys = {
            'model_name': 'Decoder/Encoder Model',
            'orig_model_name': 'Subject Model',
            'checkpoint_path': 'Checkpoint',
            'layer': 'Layer',
            'timestamp': 'Timestamp'
        };
        
        // Sort metadata keys according to preferred order
        const sortedKeys = Object.keys(metadata).sort((a, b) => {
            const indexA = metadataOrder.indexOf(a);
            const indexB = metadataOrder.indexOf(b);
            if (indexA === -1 && indexB === -1) return 0;
            if (indexA === -1) return 1;
            if (indexB === -1) return -1;
            return indexA - indexB;
        });
        
        sortedKeys.forEach(key => {
            const value = metadata[key];
            if (value === null || value === undefined || value === '') return;
            
            const item = document.createElement('div');
            item.className = 'text-sm';
            
            const label = document.createElement('span');
            label.className = 'font-medium text-[#795548]';
            label.textContent = (formattedKeys[key] || key.replace(/_/g, ' ')) + ': ';
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'text-[#5D4037]';
            
            // Special formatting for certain fields
            if (key === 'checkpoint_path' && value.length > 50) {
                valueSpan.className += ' font-mono text-xs';
                valueSpan.title = value;
                valueSpan.textContent = '...' + value.slice(-47);
            } else {
                valueSpan.textContent = value;
            }
            
            item.appendChild(label);
            item.appendChild(valueSpan);
            list.appendChild(item);
        });
        
        metadataSection.appendChild(list);
        container.appendChild(metadataSection);
    };
    
    const createColumnToggleUI = ({ container, columnVisibility, onChange }) => {
        // Get all columns that actually have visibility settings
        const columns = Object.keys(columnVisibility).sort((a, b) => {
            // Use same column order as in renderTable
            const columnOrder = ['position', 'token', 'target', 'explanation', 'relative_rmse', 'mse', 'kl_divergence', 
                               'predicted', 'rank', 'loss', 'ce', 'kl', 'max_predicted_prob', 'predicted_prob',
                               'decoder_completions', 'pred_token', 'salience_plot', 'token_salience', 
                               'explanation_concatenated', 'explanation_structured', 'tuned_lens_top', 
                               'logit_lens_top', 'layer'];
            const indexA = columnOrder.indexOf(a);
            const indexB = columnOrder.indexOf(b);
            if (indexA === -1 && indexB === -1) return 0;
            if (indexA === -1) return 1;
            if (indexB === -1) return -1;
            return indexA - indexB;
        });
        
        const toggleHTML = `
            <div class="relative">
                <button id="column-toggle-btn" class="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded">
                    Toggle Columns
                </button>
                <div id="column-toggle-dropdown" class="hidden absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50">
                    <div class="py-1">
                        ${columns.map(col => `
                            <label class="flex items-center px-4 py-2 hover:bg-gray-100 cursor-pointer">
                                <input type="checkbox" class="column-toggle mr-2" data-column="${col}" 
                                       ${columnVisibility[col] ? 'checked' : ''}>
                                <span class="text-sm">${col.replace(/_/g, ' ')}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = toggleHTML;
        
        // Add event listeners
        const toggleBtn = container.querySelector('#column-toggle-btn');
        const dropdown = container.querySelector('#column-toggle-dropdown');
        
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.classList.toggle('hidden');
        });
        
        container.querySelectorAll('.column-toggle').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const column = e.target.dataset.column;
                columnVisibility[column] = e.target.checked;
                onChange(column, e.target.checked);
            });
        });
    };
    
    const renderFullTextBox = (config) => {
        const { data, container, salienceColoringEnabled = false } = config;
        
        // Clear container but preserve the placeholder if it exists
        const existingPlaceholder = container.querySelector('#full-text-placeholder');
        container.innerHTML = '';
        
        if (!data || data.length === 0) {
            if (existingPlaceholder) {
                existingPlaceholder.classList.remove('hidden');
                container.appendChild(existingPlaceholder);
            } else {
                const placeholder = document.createElement('p');
                placeholder.id = 'full-text-placeholder';
                placeholder.className = 'text-slate-500';
                placeholder.textContent = 'Full text will appear here after visualizing.';
                container.appendChild(placeholder);
            }
            return;
        }
        
        const salienceRange = computeSalienceRange(data);
        
        data.forEach((row) => {
            const tokenString = String(row.token || row.target || '');
            const parts = tokenString.split(/\\n|\n/);
            
            parts.forEach((part, index) => {
                if (part === '' && index === parts.length - 1 && parts.length > 1) return;
                
                const textContent = (index < parts.length - 1) ? part + ' ↵' : part;
                const tokenBox = createTokenBox(
                    textContent,
                    row.explanation,
                    row.token_salience,
                    salienceColoringEnabled,
                    row
                );
                
                container.appendChild(tokenBox);
                
                if (index < parts.length - 1) {
                    container.appendChild(document.createElement('br'));
                }
            });
        });
    };
    
    const renderTable = (config) => {
        const { data, tableHead, tableBody, columnVisibility, salienceColoringEnabled = false } = config;
        
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';
        
        if (!data || data.length === 0) return;
        
        // Get all unique keys from data and order them properly
        const allColumns = [...new Set(data.flatMap(row => Object.keys(row)))];
        
        // Define preferred column order
        const columnOrder = ['position', 'token', 'target', 'explanation', 'relative_rmse', 'mse', 'kl_divergence', 
                           'predicted', 'rank', 'loss', 'ce', 'kl', 'max_predicted_prob', 'predicted_prob',
                           'decoder_completions', 'pred_token', 'salience_plot', 'token_salience', 
                           'explanation_concatenated', 'explanation_structured', 'tuned_lens_top', 
                           'logit_lens_top', 'layer'];
        
        // Sort columns according to preferred order, with any unknown columns at the end
        const columnNames = allColumns.sort((a, b) => {
            const indexA = columnOrder.indexOf(a);
            const indexB = columnOrder.indexOf(b);
            if (indexA === -1 && indexB === -1) return 0;
            if (indexA === -1) return 1;
            if (indexB === -1) return -1;
            return indexA - indexB;
        });
        
        // Create header
        const headerRow = document.createElement('tr');
        headerRow.className = "bg-[#EFEBE9]";
        
        columnNames.forEach((colName, index) => {
            const th = document.createElement('th');
            th.className = "px-2 py-2 text-left text-sm font-semibold text-[#5D4037] uppercase tracking-wider";
            th.dataset.columnName = colName;
            th.dataset.columnIndex = index;
            th.textContent = colName.replace(/_/g, ' ');
            
            const resizeHandle = document.createElement('div');
            resizeHandle.className = 'resize-handle';
            th.appendChild(resizeHandle);
            
            headerRow.appendChild(th);
        });
        tableHead.appendChild(headerRow);
        
        // Calculate KL max for coloring
        const klValues = data.map(row => parseFloat(row.kl_divergence)).filter(v => !isNaN(v));
        const maxKl = Math.max(...klValues, 0);
        
        // Create rows
        data.forEach((row) => {
            const tr = document.createElement('tr');
            tr.className = 'hover:bg-[#EFEBE9]/80 border-b border-[#EFEBE9]';
            
            columnNames.forEach(colName => {
                const td = document.createElement('td');
                td.className = "px-2 py-2 align-top font-mono text-sm text-[#5D4037]";
                td.dataset.columnName = colName;
                td.title = 'Click to copy';
                
                let value = row[colName] || '';
                
                // Format numeric values
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
                
                // Handle structured explanations
                if ((colName === 'explanation' || colName === 'explanation_structured') && 
                    Array.isArray(row.explanation_structured)) {
                    td.textContent = '';
                    row.explanation_structured.forEach((w, idx) => {
                        const span = document.createElement('span');
                        span.className = 'explanation-word';
                        span.textContent = formatNewlines(w) + ' ';
                        
                        if (Array.isArray(row.token_salience) && idx < row.token_salience.length) {
                            const sv = row.token_salience[idx];
                            if (salienceColoringEnabled && typeof sv === 'number') {
                                span.style.backgroundColor = getSalienceColor(sv);
                                span.style.color = getTooltipTextColor(sv);
                            }
                            if (typeof sv === 'number') {
                                span.title = `Decoded: ${w}\nSalience: ${sv.toFixed(3)}\nMSE: ${row.mse?.toFixed ? row.mse.toFixed(3) : row.mse}`;
                            }
                        }
                        td.appendChild(span);
                    });
                } else {
                    td.textContent = formatNewlines(String(value));
                }
                
                // Special formatting
                if (['explanation', 'explanation_structured', 'tuned_lens_top', 'logit_lens_top'].includes(colName)) {
                    // Explanation columns - still monospace but different color
                    td.className = "px-2 py-2 align-top font-mono text-sm text-[#6D4C41]";
                    if (['tuned_lens_top', 'logit_lens_top'].includes(colName)) {
                        td.style.whiteSpace = 'nowrap';
                    }
                } else if (colName === 'token' || colName === 'target') {
                    // Token column with special color
                    td.className = "px-2 py-2 align-top font-mono text-sm font-semibold text-[#BF360C]";
                } else if (colName === 'position') {
                    // Position column with lighter color
                    td.className = "px-2 py-2 align-top font-mono text-sm text-slate-400";
                } else {
                    // All other columns use monospace font (already set as default)
                    // This includes: mse, relative_rmse, kl_divergence, etc.
                }
                
                // KL divergence coloring
                if (colName === 'kl_divergence') {
                    const klValue = parseFloat(row.kl_divergence);
                    const klIntensity = isNaN(klValue) || maxKl === 0 ? 0 : Math.min(klValue / (maxKl * 0.8), 1);
                    const r = 230 - (50 * klIntensity);
                    const g = 81 - (20 * klIntensity);
                    const b = 0;
                    td.style.color = `rgb(${r}, ${g}, ${b})`;
                    td.style.fontWeight = '500';
                }
                
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        // Update visibility
        for (const columnName in columnVisibility) {
            const visible = columnVisibility[columnName];
            document.querySelectorAll(`[data-column-name="${columnName}"]`).forEach(cell => {
                cell.style.display = visible ? '' : 'none';
            });
        }
    };
    
    const renderTransposedView = (config) => {
        const { data, container, colWidth = 80, salienceColoringEnabled = false } = config;
        
        container.innerHTML = '';
        const wrapper = document.createElement('div');
        wrapper.className = 'flex flex-row flex-wrap gap-x-2';
        
        const salienceRange = computeSalienceRange(data);
        
        data.forEach((row) => {
            const colDiv = document.createElement('div');
            colDiv.className = 'flex flex-col items-start font-mono text-sm transpose-col';
            colDiv.style.width = `${colWidth}px`;
            
            // Token header (rotated)
            const tokenWrapper = document.createElement('div');
            tokenWrapper.className = 'rotated-token-wrapper';
            
            const tokenDiv = document.createElement('div');
            tokenDiv.className = 'font-semibold text-base rotated-token text-[#BF360C]';
            const tokenText = row.token || row.target || '';
            tokenDiv.textContent = formatNewlines(tokenText);
            tokenDiv.title = `Token: ${formatNewlines(tokenText)}\nMSE: ${Number(row.mse).toFixed(3)}`;
            
            tokenWrapper.appendChild(tokenDiv);
            colDiv.appendChild(tokenWrapper);
            
            // Explanation words
            let explanationWords = [];
            if (row.explanation_structured && Array.isArray(row.explanation_structured)) {
                explanationWords = row.explanation_structured;
            } else if (row.explanation) {
                explanationWords = formatNewlines(row.explanation).split(/[\s↵]+/);
            }
            
            explanationWords.forEach((word, wordIndex) => {
                const wordDiv = document.createElement('div');
                wordDiv.className = 'text-[#6D4C41] font-mono w-full text-left explanation-word';
                wordDiv.textContent = formatNewlines(word);
                
                // Apply salience coloring
                if (salienceColoringEnabled && Array.isArray(row.token_salience) && 
                    wordIndex < row.token_salience.length) {
                    const salienceValue = row.token_salience[wordIndex];
                    if (typeof salienceValue === 'number' && !isNaN(salienceValue)) {
                        wordDiv.style.backgroundColor = getSalienceColor(salienceValue, salienceRange.min, salienceRange.max);
                        wordDiv.style.padding = '2px 4px';
                        wordDiv.style.borderRadius = '2px';
                        wordDiv.style.margin = '1px 0';
                        wordDiv.title = `Decoded: ${word}\nSalience: ${salienceValue.toFixed(3)}\nMSE: ${row.mse?.toFixed ? row.mse.toFixed(3) : row.mse}`;
                    }
                }
                
                colDiv.appendChild(wordDiv);
            });
            
            wrapper.appendChild(colDiv);
        });
        
        container.appendChild(wrapper);
    };
    
    // Public API
    return {
        config,
        formatNewlines,
        getSalienceColor,
        getTooltipTextColor,
        computeSalienceRange,
        createTokenBox,
        getDefaultVisibility,
        renderMetadata,
        createColumnToggleUI,
        renderFullTextBox,
        renderTable,
        renderTransposedView,
        copyToClipboard,
        showCopyFeedback
    };
})();

// Make it available globally for standalone usage
if (typeof window !== 'undefined') {
    window.VisualizationCore = VisualizationCore;
}
