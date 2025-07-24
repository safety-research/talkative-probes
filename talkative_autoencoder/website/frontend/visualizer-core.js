/**
 * Visualization Core Module
 * Shared visualization functionality for both standalone and integrated versions
 */

// For ES6 module compatibility, we'll export at the end
const VisualizationCore = (function() {
    'use strict';
    
    // Core configuration
    const columnNames = ["position", "token", "explanation", "kl_divergence", "mse", "relative_rmse", 
                        "tuned_lens_top", "explanation_structured", "token_salience", "logit_lens_top", "layer"];
    
    // Default column visibility
    const getDefaultVisibility = () => {
        const defaultVisibleCols = ['token', 'explanation', 'relative_rmse'];
        const visibility = {};
        columnNames.forEach(colName => {
            visibility[colName] = defaultVisibleCols.includes(colName);
        });
        return visibility;
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
    
    // Render full text view
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
            const tokenString = String(row.token || '');
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
    
    // Render metadata display
    const renderMetadata = (config) => {
        const { metadata, container } = config;
        
        if (metadata && Object.keys(metadata).length > 0) {
            container.classList.remove('hidden');
            
            // Clear existing content
            container.innerHTML = '';
            
            // Create list element
            const ul = document.createElement('ul');
            ul.className = 'list-disc list-inside space-y-1';
            
            if (metadata.model_name) {
                const li = document.createElement('li');
                const strong = document.createElement('strong');
                strong.textContent = 'Decoder/Encoder Model:';
                const span = document.createElement('span');
                span.className = 'font-semibold';
                span.textContent = metadata.model_name;
                li.appendChild(strong);
                li.appendChild(document.createTextNode(' '));
                li.appendChild(span);
                ul.appendChild(li);
            }
            
            if (metadata.orig_model_name) {
                const li = document.createElement('li');
                const strong = document.createElement('strong');
                strong.textContent = 'Subject Model:';
                const span = document.createElement('span');
                span.className = 'font-semibold';
                span.textContent = metadata.orig_model_name;
                li.appendChild(strong);
                li.appendChild(document.createTextNode(' '));
                li.appendChild(span);
                ul.appendChild(li);
            }
            
            if (metadata.checkpoint_path) {
                const li = document.createElement('li');
                const strong = document.createElement('strong');
                strong.textContent = 'Checkpoint:';
                const span = document.createElement('span');
                span.className = 'font-mono text-xs';
                span.textContent = metadata.checkpoint_path;
                li.appendChild(strong);
                li.appendChild(document.createTextNode(' '));
                li.appendChild(span);
                ul.appendChild(li);
            }
            
            container.appendChild(ul);
        } else {
            container.classList.add('hidden');
            container.innerHTML = '';
        }
    };
    
    // Render table view
    const renderTable = (config) => {
        const { data, tableHead, tableBody, columnVisibility, salienceColoringEnabled = false } = config;
        
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';
        
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
                if (['explanation', 'tuned_lens_top', 'logit_lens_top'].includes(colName)) {
                    td.className = "px-2 py-2 align-top font-sans text-sm text-[#6D4C41]";
                    td.style.whiteSpace = 'nowrap';
                }
                if (colName === 'token') {
                    td.className = "px-2 py-2 align-top font-mono text-sm font-semibold text-[#BF360C]";
                }
                if (colName === 'position') {
                    td.className = "px-2 py-2 align-top font-mono text-sm text-slate-400";
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
    
    // Render transposed view
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
            tokenDiv.textContent = formatNewlines(row.token);
            tokenDiv.title = `Token: ${formatNewlines(row.token)}\nMSE: ${Number(row.mse).toFixed(3)}`;
            
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
                wordDiv.className = 'text-[#6D4C41] font-sans w-full text-left explanation-word';
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
    
    // Create column toggle UI
    const createColumnToggleUI = (config) => {
        const { container, columnVisibility, onChange } = config;
        
        container.innerHTML = '';
        
        // Remove any existing dropdowns
        document.querySelectorAll('#column-toggle-dropdown').forEach(el => el.remove());
        
        const toggleButton = document.createElement('button');
        toggleButton.className = "px-4 py-2 text-sm secondary-btn font-semibold rounded-lg shadow-sm transition-all duration-200 transform hover:scale-105 flex items-center gap-2";
        toggleButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18"/></svg> Toggle Columns`;
        
        const dropdown = document.createElement('div');
        dropdown.id = 'column-toggle-dropdown';
        dropdown.className = "absolute right-0 mt-2 w-56 origin-top-right bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none hidden z-20 border border-slate-200";
        
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
                if (onChange) onChange(columnName, e.target.checked);
            }
        });
        
        container.appendChild(toggleButton);
        document.body.appendChild(dropdown);
        
        // Position dropdown relative to button
        toggleButton.addEventListener('click', () => {
            const rect = toggleButton.getBoundingClientRect();
            dropdown.style.left = `${rect.left + window.scrollX}px`;
            dropdown.style.top = `${rect.bottom + window.scrollY + 5}px`;
        });
        
        return dropdown;
    };
    
    // Copy to clipboard with feedback
    const copyToClipboard = (text, element) => {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        
        try {
            document.execCommand('copy');
            showCopyFeedback(element);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
        
        document.body.removeChild(textArea);
    };
    
    const showCopyFeedback = (element) => {
        const feedback = document.createElement('div');
        feedback.className = 'copy-feedback show';
        feedback.textContent = 'Copied!';
        element.style.position = 'relative';
        element.appendChild(feedback);
        
        setTimeout(() => {
            feedback.remove();
        }, 1000);
    };
    
    // Return public API
    return {
        columnNames,
        getDefaultVisibility,
        formatNewlines,
        getSalienceColor,
        getTooltipTextColor,
        computeSalienceRange,
        createTokenBox,
        renderFullTextBox,
        renderMetadata,
        renderTable,
        renderTransposedView,
        createColumnToggleUI,
        copyToClipboard,
        showCopyFeedback
    };
})();

// Export for ES6 module usage
export { VisualizationCore };

// Also make it available globally for standalone usage (when not loaded as module)
if (typeof window !== 'undefined' && !window.VisualizationCore) {
    window.VisualizationCore = VisualizationCore;
}