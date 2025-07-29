// Global Notifications System for Model Switching

class GlobalNotifications {
    constructor() {
        this.container = null;
        this.notifications = new Map();
        this.init();
    }
    
    init() {
        // Create container if it doesn't exist
        this.container = document.getElementById('global-notifications');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'global-notifications';
            this.container.className = 'fixed top-0 left-0 right-0 z-50 pointer-events-none';
            document.body.appendChild(this.container);
        }
    }
    
    show(id, message, type = 'info', persistent = false) {
        // Remove existing notification with same ID
        if (this.notifications.has(id)) {
            this.remove(id);
        }
        
        const notification = document.createElement('div');
        notification.className = `
            w-full p-4 text-center pointer-events-auto
            transform transition-all duration-300 ease-in-out
            ${this.getTypeClasses(type)}
        `;
        notification.style.transform = 'translateY(-100%)';
        notification.innerHTML = `
            <div class="container mx-auto px-4">
                <div class="flex items-center justify-center">
                    ${this.getIcon(type)}
                    <span class="font-medium">${message}</span>
                </div>
            </div>
        `;
        
        this.container.appendChild(notification);
        this.notifications.set(id, { element: notification, persistent });
        
        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateY(0)';
        });
        
        // Auto-remove non-persistent notifications after 10 seconds
        if (!persistent) {
            setTimeout(() => this.remove(id), 10000);
        }
    }
    
    remove(id) {
        const notification = this.notifications.get(id);
        if (!notification) return;
        
        // Animate out
        notification.element.style.transform = 'translateY(-100%)';
        
        setTimeout(() => {
            notification.element.remove();
            this.notifications.delete(id);
        }, 300);
    }
    
    getTypeClasses(type) {
        switch (type) {
            case 'warning':
                return 'bg-yellow-500 text-white';
            case 'error':
                return 'bg-red-500 text-white';
            case 'success':
                return 'bg-green-500 text-white';
            case 'info':
            default:
                return 'bg-blue-500 text-white';
        }
    }
    
    getIcon(type) {
        switch (type) {
            case 'warning':
                return `<svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                </svg>`;
            case 'error':
                return `<svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                </svg>`;
            case 'success':
                return `<svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                </svg>`;
            case 'info':
            default:
                return `<svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>
                </svg>`;
        }
    }
    
    // Handle model switch status updates
    handleModelSwitchStatus(data) {
        switch (data.status) {
            case 'starting':
                this.show(
                    'model-switch',
                    `Model is being switched to ${data.model_id}. All operations are temporarily queued.`,
                    'warning',
                    true
                );
                break;
                
            case 'completed':
                this.remove('model-switch');
                this.show(
                    'model-switch-complete',
                    'Model switch completed successfully!',
                    'success',
                    false
                );
                break;
                
            case 'failed':
                this.remove('model-switch');
                this.show(
                    'model-switch-error',
                    `Model switch failed: ${data.error}`,
                    'error',
                    false
                );
                break;
        }
    }
}

// Create global instance
const globalNotifications = new GlobalNotifications();

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = globalNotifications;
}