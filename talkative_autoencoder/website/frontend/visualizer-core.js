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
    const getSalienceColor = (salience) => {
        if (!salience || salience === 0) return '';
        
        const absValue = Math.abs(salience);
        const ratio = Math.min(absValue, 1);
        
        if (salience > 0) {
            const r = 255;
            const g = Math.round(255 * (1 - ratio));
            const b = Math.round(255 * (1 - ratio));
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            const r = Math.round(255 * (1 - ratio));
            const g = 200;
            const b = Math.round(255 * (1 - ratio));
            return `rgb(${r}, ${g}, ${b})`;
        }
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
    
    // Column visibility management
    const getDefaultVisibility = () => ({
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
    });
    
    // Rendering functions
    const renderMetadata = ({ metadata, container }) => {
        if (!metadata) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        container.innerHTML = `
            <div class="metadata-section">
                <h3 class="text-sm font-semibold text-gray-700 mb-2">Analysis Metadata</h3>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                    ${Object.entries(metadata).map(([key, value]) => `
                        <div>
                            <span class="font-medium text-gray-600">${key.replace(/_/g, ' ')}:</span>
                            <span class="text-gray-800">${value}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    };
    
    const createColumnToggleUI = ({ container, columnVisibility, onChange }) => {
        const columns = Object.keys(columnVisibility);
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
    
    const renderFullTextBox = ({ data, container, salienceColoringEnabled }) => {
        const fullText = data.map(row => {
            if (salienceColoringEnabled && row.token_salience) {
                const salience = parseFloat(row.token_salience);
                const color = getSalienceColor(salience);
                return `<span style="background-color: ${color}">${row.target || ''}</span>`;
            }
            return row.target || '';
        }).join('');
        
        const fullTextBox = container.querySelector('.full-text-box') || 
                           container.appendChild(Object.assign(document.createElement('div'), {
                               className: 'full-text-box'
                           }));
        
        fullTextBox.innerHTML = `
            <h3 class="font-semibold text-sm text-gray-700 mb-2">Full Text:</h3>
            <div class="p-3 bg-gray-50 rounded-lg text-sm leading-relaxed">
                ${fullText || '<span class="text-gray-500">No text available</span>'}
            </div>
        `;
    };
    
    const renderTable = ({ data, tableHead, tableBody, columnVisibility, salienceColoringEnabled }) => {
        if (!data || data.length === 0) return;
        
        // Get all unique keys from data
        const columns = [...new Set(data.flatMap(row => Object.keys(row)))];
        
        // Render header
        tableHead.innerHTML = `
            <tr>
                ${columns.map(col => `
                    <th data-column-name="${col}" 
                        style="${columnVisibility[col] === false ? 'display: none;' : ''}"
                        class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        ${col.replace(/_/g, ' ')}
                    </th>
                `).join('')}
            </tr>
        `;
        
        // Render rows
        tableBody.innerHTML = data.map((row, index) => `
            <tr class="${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}">
                ${columns.map(col => {
                    let value = row[col];
                    let style = columnVisibility[col] === false ? 'display: none;' : '';
                    let cellClass = 'px-4 py-2 text-sm';
                    
                    // Handle special columns
                    if (col === 'explanation' || col === 'explanation_concatenated') {
                        cellClass += ' explanation-cell';
                        // For explanation columns, render word-by-word with salience coloring
                        if (col === 'explanation' && salienceColoringEnabled && row.token_salience && Array.isArray(row.explanation_structured)) {
                            // Render each word with its own salience color
                            const words = row.explanation_structured;
                            const saliences = Array.isArray(row.token_salience) ? row.token_salience : [row.token_salience];
                            
                            // Create HTML for word-by-word rendering with individual salience colors
                            value = words.map((word, idx) => {
                                const salience = idx < saliences.length ? saliences[idx] : saliences[saliences.length - 1];
                                const wordColor = getSalienceColor(parseFloat(salience));
                                return `<span style="background-color: ${wordColor}; padding: 0 2px;">${word}</span>`;
                            }).join('');
                        } else if (salienceColoringEnabled && row.token_salience) {
                            const salience = parseFloat(row.token_salience);
                            style += `background-color: ${getSalienceColor(salience)};`;
                        }
                    } else if (col === 'token_salience') {
                        value = value ? parseFloat(value).toFixed(3) : '';
                    } else if (typeof value === 'number' && !Number.isInteger(value)) {
                        value = value.toFixed(4);
                    } else if (Array.isArray(value)) {
                        value = value.join(', ');
                    }
                    
                    return `
                        <td data-column-name="${col}" 
                            style="${style}"
                            class="${cellClass}"
                            title="Click to copy"
                            data-value="${value || ''}">
                            ${value || ''}
                        </td>
                    `;
                }).join('')}
            </tr>
        `).join('');
    };
    
    const renderTransposedView = ({ data, container, colWidth, salienceColoringEnabled }) => {
        if (!data || data.length === 0) return;
        
        const positions = data.map(row => ({
            position: row.position,
            target: row.target || '',
            predicted: row.predicted || '',
            mse: row.mse ? parseFloat(row.mse).toFixed(4) : '',
            explanation: row.explanation || '',
            explanation_structured: row.explanation_structured || null,
            salience: row.token_salience ? parseFloat(row.token_salience) : null
        }));
        
        const html = `
            <div class="transpose-container">
                ${positions.map(pos => {
                    const bgColor = salienceColoringEnabled && pos.salience !== null ? 
                                  getSalienceColor(pos.salience) : '';
                    
                    return `
                        <div class="transpose-col" style="width: ${colWidth}px;">
                            <div class="position-num">${pos.position}</div>
                            <div class="token-info">
                                <div class="target-token">${pos.target}</div>
                                ${pos.predicted ? `<div class="predicted-token">→ ${pos.predicted}</div>` : ''}
                            </div>
                            <div class="mse-value">MSE: ${pos.mse}</div>
                            <div class="explanation-section">
                                ${pos.explanation_structured ? 
                                    pos.explanation_structured.map((word, idx) => {
                                        let wordColor = '';
                                        if (salienceColoringEnabled && pos.salience !== null) {
                                            const saliences = Array.isArray(pos.salience) ? pos.salience : [pos.salience];
                                            const salience = idx < saliences.length ? saliences[idx] : saliences[saliences.length - 1];
                                            wordColor = getSalienceColor(parseFloat(salience));
                                        }
                                        return `<div class="explanation-word" style="background-color: ${wordColor}">${word}</div>`;
                                    }).join('') :
                                    pos.explanation.split(/(?=[A-Z])/).map(word => 
                                        `<div class="explanation-word" style="background-color: ${bgColor}">${word}</div>`
                                    ).join('')
                                }
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
        
        container.innerHTML = html;
    };
    
    // Public API
    return {
        config,
        getSalienceColor,
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