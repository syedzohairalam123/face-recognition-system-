/**
 * Face Recognition Attendance System
 * Professional JavaScript Utilities
 */

document.addEventListener('DOMContentLoaded', function() {
    // ── Auto-dismiss alerts ─────────────────────────────────────
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });

    // ── Confirm dangerous actions ───────────────────────────────
    document.querySelectorAll('[data-confirm]').forEach(function(element) {
        element.addEventListener('click', function(e) {
            if (!confirm(this.dataset.confirm)) {
                e.preventDefault();
            }
        });
    });

    // ── Active nav link highlighting ────────────────────────────
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(function(link) {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // ── Tooltip initialization ──────────────────────────────────
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(function(tooltipTriggerEl) {
        new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // ── Form validation visual feedback ─────────────────────────
    document.querySelectorAll('form').forEach(function(form) {
        form.addEventListener('submit', function() {
            const requiredFields = form.querySelectorAll('[required]');
            requiredFields.forEach(function(field) {
                if (!field.value.trim()) {
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                    field.classList.add('is-valid');
                }
            });
        });
    });
});

// ── Utility Functions ──────────────────────────────────────────

/**
 * Show loading overlay
 */
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.querySelector('p').textContent = message;
        overlay.style.display = 'flex';
    }
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Bootstrap alert type (success, danger, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'success', duration = 5000) {
    const container = document.querySelector('.container-fluid');
    if (!container) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show animate-fadeIn`;
    
    const icons = {
        success: 'check-circle',
        danger: 'exclamation-triangle',
        warning: 'exclamation-circle',
        info: 'info-circle'
    };
    
    alertDiv.innerHTML = `
        <i class="bi bi-${icons[type] || 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(function() {
        const bsAlert = bootstrap.Alert.getOrCreateInstance(alertDiv);
        bsAlert.close();
    }, duration);
}

/**
 * Make an API call with error handling
 * @param {string} url - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise} Response data
 */
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'API request failed');
        }

        return data;
    } catch (error) {
        showNotification(error.message, 'danger');
        throw error;
    }
}

/**
 * Format a number as percentage
 * @param {number} value - Value to format (0-1)
 * @returns {string} Formatted percentage string
 */
function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Format a duration in minutes to human-readable string
 * @param {number} minutes - Duration in minutes
 * @returns {string} Formatted string like '2h 30m'
 */
function formatDuration(minutes) {
    if (!minutes) return '-';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
}

/**
 * Show a confirmation dialog
 * @param {string} message - Confirmation message
 * @returns {Promise<boolean>} User's choice
 */
function confirmAction(message) {
    return new Promise((resolve) => {
        resolve(confirm(message));
    });
}

/**
 * Set button loading state
 * @param {HTMLElement} button - Button element
 * @param {boolean} loading - Whether to show loading state
 * @param {string} text - Text to show while loading
 */
function setButtonLoading(button, loading, text = 'Loading...') {
    if (loading) {
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${text}`;
        button.disabled = true;
        button.classList.add('btn-loading');
    } else {
        button.innerHTML = button.dataset.originalText || button.innerHTML;
        button.disabled = false;
        button.classList.remove('btn-loading');
    }
}

// ── Table Sorting ───────────────────────────────────────────────

/**
 * Initialize sortable tables
 * Add data-sortable attribute to tables and data-sort-key to th elements
 */
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('table.table-hover').forEach(function(table) {
        const headers = table.querySelectorAll('thead th');
        headers.forEach(function(th, index) {
            // Skip action columns
            if (th.textContent.trim().toLowerCase() === 'actions') return;
            
            th.style.cursor = 'pointer';
            th.classList.add('sortable-header');
            
            // Add sort indicator
            if (!th.querySelector('.sort-indicator')) {
                th.innerHTML += ' <i class="bi bi-arrow-down-up sort-indicator text-muted" style="font-size: 0.75rem;"></i>';
            }
            
            th.addEventListener('click', function() {
                sortTable(table, index, this);
            });
        });
    });
});

/**
 * Sort a table by a column
 * @param {HTMLTableElement} table - Table element
 * @param {number} columnIndex - Column index to sort by
 * @param {HTMLTableCellElement} header - Header element clicked
 */
function sortTable(table, columnIndex, header) {
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isAsc = header.dataset.sortOrder !== 'asc';
    
    // Clear all sort indicators
    table.querySelectorAll('.sort-indicator').forEach(function(icon) {
        icon.className = 'bi bi-arrow-down-up sort-indicator text-muted';
    });
    
    // Update clicked header
    const indicator = header.querySelector('.sort-indicator');
    if (indicator) {
        indicator.className = `bi bi-arrow-${isAsc ? 'up' : 'down'} sort-indicator text-primary`;
    }
    header.dataset.sortOrder = isAsc ? 'asc' : 'desc';
    
    // Sort rows
    rows.sort(function(a, b) {
        const aText = a.cells[columnIndex] ? a.cells[columnIndex].textContent.trim() : '';
        const bText = b.cells[columnIndex] ? b.cells[columnIndex].textContent.trim() : '';
        
        // Try numeric sort
        const aNum = parseFloat(aText.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bText.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return isAsc ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string sort
        const comparison = aText.localeCompare(bText);
        return isAsc ? comparison : -comparison;
    });
    
    // Re-append rows
    rows.forEach(function(row) {
        tbody.appendChild(row);
    });
}
