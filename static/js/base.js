/* Base JavaScript for Django RAG Chatbot */

(function() {
    'use strict';
    
    // Common utilities
    window.ChatbotUtils = {
        
        // CSRF token helper
        getCSRFToken: function() {
            const cookies = document.cookie.split(';');
            for (let cookie of cookies) {
                const [name, value] = cookie.trim().split('=');
                if (name === 'csrftoken') {
                    return value;
                }
            }
            
            // Try to get from meta tag
            const meta = document.querySelector('meta[name="csrf-token"]');
            return meta ? meta.content : null;
        },
        
        // Show notification
        showNotification: function(message, type = 'info', duration = 5000) {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.innerHTML = `
                <div class="notification-content">
                    <span class="notification-message">${message}</span>
                    <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                        ×
                    </button>
                </div>
            `;
            
            // Add styles if not already present
            if (!document.querySelector('#notification-styles')) {
                const styles = document.createElement('style');
                styles.id = 'notification-styles';
                styles.textContent = `
                    .notification {
                        position: fixed;
                        top: 2rem;
                        right: 2rem;
                        z-index: 1000;
                        min-width: 300px;
                        max-width: 500px;
                        padding: 1rem;
                        border-radius: 0.5rem;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                        animation: slideInRight 0.3s ease;
                    }
                    
                    .notification-info {
                        background: #dbeafe;
                        border: 1px solid #3b82f6;
                        color: #1e40af;
                    }
                    
                    .notification-success {
                        background: #dcfce7;
                        border: 1px solid #22c55e;
                        color: #15803d;
                    }
                    
                    .notification-error {
                        background: #fee2e2;
                        border: 1px solid #ef4444;
                        color: #dc2626;
                    }
                    
                    .notification-warning {
                        background: #fef3c7;
                        border: 1px solid #f59e0b;
                        color: #d97706;
                    }
                    
                    .notification-content {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 1rem;
                    }
                    
                    .notification-close {
                        background: none;
                        border: none;
                        font-size: 1.5rem;
                        cursor: pointer;
                        color: inherit;
                        opacity: 0.7;
                    }
                    
                    .notification-close:hover {
                        opacity: 1;
                    }
                    
                    @keyframes slideInRight {
                        from {
                            transform: translateX(100%);
                            opacity: 0;
                        }
                        to {
                            transform: translateX(0);
                            opacity: 1;
                        }
                    }
                    
                    @media (max-width: 768px) {
                        .notification {
                            top: 1rem;
                            right: 1rem;
                            left: 1rem;
                            min-width: auto;
                        }
                    }
                `;
                document.head.appendChild(styles);
            }
            
            document.body.appendChild(notification);
            
            // Auto remove after duration
            if (duration > 0) {
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.remove();
                    }
                }, duration);
            }
        },
        
        // Loading spinner helper
        showLoading: function(element) {
            const originalContent = element.innerHTML;
            element.innerHTML = '<i class="loading-spinner"></i> Loading...';
            element.disabled = true;
            
            return function hideLoading() {
                element.innerHTML = originalContent;
                element.disabled = false;
            };
        },
        
        // Animate counter
        animateCounter: function(element, target, duration = 2000) {
            const start = 0;
            const increment = target / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    element.textContent = target;
                    clearInterval(timer);
                } else {
                    element.textContent = Math.floor(current);
                }
            }, 16);
        },
        
        // Debounce function
        debounce: function(func, wait, immediate) {
            let timeout;
            return function executedFunction() {
                const context = this;
                const args = arguments;
                const later = function() {
                    timeout = null;
                    if (!immediate) func.apply(context, args);
                };
                const callNow = immediate && !timeout;
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
                if (callNow) func.apply(context, args);
            };
        },
        
        // Format bytes
        formatBytes: function(bytes, decimals = 2) {
            if (bytes === 0) return '0 Bytes';
            
            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
            
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            
            return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
        },
        
        // Copy to clipboard
        copyToClipboard: function(text) {
            if (navigator.clipboard && window.isSecureContext) {
                return navigator.clipboard.writeText(text)
                    .then(() => this.showNotification('Copied to clipboard!', 'success'))
                    .catch(() => this.showNotification('Failed to copy', 'error'));
            } else {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                try {
                    document.execCommand('copy');
                    this.showNotification('Copied to clipboard!', 'success');
                } catch (err) {
                    this.showNotification('Failed to copy', 'error');
                }
                
                textArea.remove();
            }
        }
    };
    
    // Initialize common functionality
    document.addEventListener('DOMContentLoaded', function() {
        
        // Add loading spinner CSS if not present
        if (!document.querySelector('#loading-spinner-styles')) {
            const styles = document.createElement('style');
            styles.id = 'loading-spinner-styles';
            styles.textContent = `
                .loading-spinner {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border: 2px solid currentColor;
                    border-radius: 50%;
                    border-top-color: transparent;
                    animation: spin 0.8s linear infinite;
                }
                
                @keyframes spin {
                    to {
                        transform: rotate(360deg);
                    }
                }
            `;
            document.head.appendChild(styles);
        }
        
        // Handle copy buttons
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('copy-btn')) {
                const text = e.target.dataset.copy || e.target.textContent;
                ChatbotUtils.copyToClipboard(text);
            }
        });
        
        // Handle external links
        document.addEventListener('click', function(e) {
            if (e.target.tagName === 'A' && e.target.hostname !== window.location.hostname) {
                e.target.setAttribute('rel', 'noopener noreferrer');
            }
        });
        
    });
    
})();
