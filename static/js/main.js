document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const reportForm = document.getElementById('report-form');
    const inputSection = document.getElementById('input-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    
    const companyNameInput = document.getElementById('company-name');
    const timePeriodInput = document.getElementById('time-period');
    const companyDisplay = document.getElementById('company-display');
    const resultCompanyDisplay = document.getElementById('result-company-display');
    
    const progressFill = document.getElementById('progress-fill');
    const progressPercentage = document.getElementById('progress-percentage');
    const statusMessage = document.getElementById('status-message');
    const logContent = document.getElementById('log-content');
    
    const downloadLink = document.getElementById('download-link');
    const previewContent = document.getElementById('preview-content');
    const errorMessage = document.getElementById('error-message');
    
    const cancelBtn = document.getElementById('cancel-btn');
    const newReportBtn = document.getElementById('new-report-btn');
    const retryBtn = document.getElementById('retry-btn');
    
    // State variables
    let isGenerating = false;
    let statusCheckInterval = null;
    let logHistory = [];
    
    // Submit form handler
    reportForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const companyName = companyNameInput.value.trim();
        const timePeriod = timePeriodInput.value.trim();
        
        if (!companyName || !timePeriod) {
            alert('Please enter both company name and time period.');
            return;
        }
        
        startReportGeneration(companyName, timePeriod);
    });
    
    // Button event listeners
    cancelBtn.addEventListener('click', cancelGeneration);
    newReportBtn.addEventListener('click', resetUI);
    retryBtn.addEventListener('click', resetUI);
    
    // Start report generation
    function startReportGeneration(companyName, timePeriod) {
        // Reset state
        isGenerating = true;
        logHistory = [];
        updateProgressUI(0, 'Initializing...');
        logContent.innerHTML = '';
        
        // Update UI sections
        inputSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        
        // Update company display
        companyDisplay.textContent = companyName ? ` for ${companyName}` : '';
        
        // Send request to backend
        const formData = new FormData();
        formData.append('company_name', companyName);
        formData.append('time_period', timePeriod);
        
        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Start checking status
                startStatusCheck();
            } else {
                showError(data.message || 'Failed to start report generation.');
            }
        })
        .catch(err => {
            showError('Network error: ' + err.message);
        });
    }
    
    // Start status check interval
    function startStatusCheck() {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        statusCheckInterval = setInterval(checkStatus, 1000);
    }
    
    // Check report generation status
    function checkStatus() {
        if (!isGenerating) return;
        
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update progress
                updateProgressUI(data.progress, data.message);
                
                // Add to log if message is new
                if (data.message && !logHistory.includes(data.message)) {
                    addLogEntry(data.message);
                    logHistory.push(data.message);
                }
                
                // Check if complete
                if (data.progress >= 1.0) {
                    reportComplete(data);
                } 
                // Check for error
                else if (data.error) {
                    showError(data.error);
                }
                
                // Update generating state
                isGenerating = data.is_generating;
                if (!isGenerating && statusCheckInterval) {
                    clearInterval(statusCheckInterval);
                }
            })
            .catch(err => {
                console.error('Status check error:', err);
            });
    }
    
    // Update progress UI
    function updateProgressUI(progress, message) {
        const percent = Math.min(Math.round(progress * 100), 100);
        progressFill.style.width = `${percent}%`;
        progressPercentage.textContent = `${percent}%`;
        
        if (message) {
            statusMessage.textContent = message;
        }
    }
    
    // Add log entry
    function addLogEntry(message) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.textContent = `[${timestamp}] ${message}`;
        logContent.appendChild(entry);
        
        // Scroll to bottom
        logContent.scrollTop = logContent.scrollHeight;
    }
    
    // Handle report completion
    function reportComplete(data) {
        isGenerating = false;
        clearInterval(statusCheckInterval);
        
        // Update UI sections
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Set company name
        resultCompanyDisplay.textContent = data.company_name ? ` for ${data.company_name}` : '';
        
        // Set download link
        if (data.report_file) {
            downloadLink.href = `/download/${data.report_file}`;
            downloadLink.setAttribute('download', data.report_file.split('/').pop());
            
            // Load report preview
            fetch(`/download/${data.report_file}`)
                .then(response => response.text())
                .then(content => {
                    // Limit preview length
                    const maxPreviewLength = 2000;
                    const preview = content.length > maxPreviewLength
                        ? content.substring(0, maxPreviewLength) + '...'
                        : content;
                    
                    previewContent.textContent = preview;
                })
                .catch(err => {
                    previewContent.textContent = 'Error loading preview: ' + err.message;
                });
        } else {
            downloadLink.classList.add('hidden');
            previewContent.textContent = 'Report file not available.';
        }
    }
    
    // Show error
    function showError(message) {
        isGenerating = false;
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        // Update UI sections
        inputSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        
        // Set error message
        errorMessage.textContent = message || 'An unknown error occurred.';
    }
    
    // Cancel report generation
    function cancelGeneration() {
        if (confirm('Are you sure you want to cancel the report generation?')) {
            isGenerating = false;
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            resetUI();
        }
    }
    
    // Reset UI to initial state
    function resetUI() {
        isGenerating = false;
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        // Reset form
        reportForm.reset();
        
        // Update UI sections
        inputSection.classList.remove('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
    }
}); 