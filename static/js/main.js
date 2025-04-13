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
    
    // Session ID elements
    const sessionIdDisplay = document.getElementById('session-id-display');
    const copySessionBtn = document.getElementById('copy-session-btn');
    const restoreSessionId = document.getElementById('restore-session-id');
    const restoreBtn = document.getElementById('restore-btn');
    
    // State variables
    let isGenerating = false;
    let statusCheckInterval = null;
    let logHistory = [];
    let sessionId = null;  // Store the session ID
    
    // Initialize: try to retrieve session ID from localStorage
    initializeSession();
    
    // Helper function to get session ID parameter for API calls
    function getSessionIdParam() {
        return sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    }
    
    // Helper function to append session ID to URLs
    function appendSessionId(url) {
        if (!sessionId) return url;
        
        const separator = url.includes('?') ? '&' : '?';
        return `${url}${separator}session_id=${encodeURIComponent(sessionId)}`;
    }
    
    function initializeSession() {
        // Try to get session ID from localStorage
        const savedSessionId = localStorage.getItem('deepResearcherSessionId');
        if (savedSessionId) {
            sessionId = savedSessionId;
            displaySessionId(sessionId);
            
            // Check if there's an active report for this session
            fetchSessionStatus(sessionId);
        } else {
            sessionIdDisplay.textContent = "No active session";
        }
    }
    
    function displaySessionId(id) {
        if (id) {
            sessionIdDisplay.textContent = id;
        } else {
            sessionIdDisplay.textContent = "No active session";
        }
    }
    
    // Copy session ID to clipboard
    copySessionBtn.addEventListener('click', function() {
        if (!sessionId) {
            alert("No active session to copy");
            return;
        }
        
        // Use clipboard API
        navigator.clipboard.writeText(sessionId).then(function() {
            // Change button text temporarily to indicate success
            const originalText = copySessionBtn.innerHTML;
            copySessionBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                copySessionBtn.innerHTML = originalText;
            }, 2000);
        }).catch(function(err) {
            alert('Could not copy session ID: ' + err);
        });
    });
    
    // Restore session
    restoreBtn.addEventListener('click', function() {
        const inputId = restoreSessionId.value.trim();
        if (!inputId) {
            showError("Please enter a session ID to restore", "EMPTY_SESSION_ID");
            return;
        }
        
        // Show loading state
        addLogEntry("Validating session...");
        statusMessage.textContent = "Validating session...";
        
        // First validate the session ID
        fetch(`/validate-session?session_id=${encodeURIComponent(inputId)}`)
            .then(response => response.json())
            .then(data => {
                if (data.valid) {
                    // Save to localStorage and update UI
                    sessionId = inputId;
                    localStorage.setItem('deepResearcherSessionId', sessionId);
                    displaySessionId(sessionId);
                    
                    addLogEntry(`Session restored successfully. Status: ${data.status}`);
                    
                    // Get session status
                    fetchSessionStatus(sessionId);
                    restoreSessionId.value = '';
                } else {
                    showError(data.message || "Invalid session ID", data.error_code);
                }
            })
            .catch(err => {
                showError("Error validating session: " + err.message, "NETWORK_ERROR");
            });
    });
    
    // Explicitly fetch and process session status
    function fetchSessionStatus(sessionId) {
        // Show loader
        statusMessage.textContent = "Checking session status...";
        
        // Explicitly include the sessionId as a query parameter
        fetch(`/status?session_id=${encodeURIComponent(sessionId)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Session status:", data);
                
                // Check for warnings
                if (data.warning) {
                    console.warn("Server warning:", data.warning);
                }
                
                // Process status
                processSessionStatus(data);
            })
            .catch(err => {
                console.error('Error fetching session status:', err);
                if (err.message.includes('404')) {
                    showError('Session not found or expired. Please try a different session ID.', 'INVALID_SESSION_ID');
                } else if (err.message.includes('400')) {
                    showError('Missing session ID. Please enter a valid session ID.', 'MISSING_SESSION_ID');
                } else {
                    showError('Error retrieving session status: ' + err.message, 'NETWORK_ERROR');
                }
            });
    }
    
    // Process session status and update UI accordingly
    function processSessionStatus(data) {
        // Check if there's an active report or completed report
        if (data.report_file) {
            // Report is complete, show results
            console.log("Found completed report:", data.report_file);
            reportComplete(data);
        } 
        else if (data.is_generating && data.progress > 0) {
            // Report is being generated, show progress
            console.log("Found report in progress:", data.progress);
            isGenerating = true;
            
            // Update UI sections
            inputSection.classList.add('hidden');
            progressSection.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            errorSection.classList.add('hidden');
            
            // Update company display and progress
            companyDisplay.textContent = data.company_name ? ` for ${data.company_name}` : '';
            updateProgressUI(data.progress, data.message);
            
            // Start status check
            startStatusCheck();
        }
        else if (data.error) {
            // Error encountered
            showError(data.error);
        }
        else {
            // No active report for this session
            console.log("No active report found for session");
            addLogEntry("No active report found for this session ID.");
            resetUI();
            
            // Show a message
            alert("No active report found for this session ID. You may start a new report.");
        }
    }
    
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
        
        // Add session ID to form data if available (for session continuity)
        if (sessionId) {
            formData.append('session_id', sessionId);
        }
        
        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Report generation failed: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Store session ID if provided
                if (data.session_id) {
                    sessionId = data.session_id;
                    // Save to localStorage for persistence
                    localStorage.setItem('deepResearcherSessionId', sessionId);
                    displaySessionId(sessionId);
                    console.log('Session ID:', sessionId);
                    
                    // Add to log
                    addLogEntry(`Session ID: ${sessionId}`);
                }
                // Start checking status
                startStatusCheck();
            } else {
                showError(data.message || 'Failed to start report generation.', data.error_code || 'GENERATION_FAILED');
            }
        })
        .catch(err => {
            showError('Network error: ' + err.message, 'NETWORK_ERROR');
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
        if (!sessionId) return;
        
        // Use helper function for session ID parameter
        fetch(`/status${getSessionIdParam()}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // If no active report, don't update UI
                if (!data.is_generating && data.progress === 0 && !data.report_file) {
                    return;
                }
                
                // If report is complete and we weren't generating, show the result
                if (data.progress >= 1.0 && !isGenerating) {
                    isGenerating = false;
                    reportComplete(data);
                    return;
                }
                
                // If we're not already tracking generation, update state
                if (!isGenerating && data.is_generating) {
                    isGenerating = true;
                    
                    // Update UI sections
                    inputSection.classList.add('hidden');
                    progressSection.classList.remove('hidden');
                    resultsSection.classList.add('hidden');
                    errorSection.classList.add('hidden');
                    
                    // Update company display
                    companyDisplay.textContent = data.company_name ? ` for ${data.company_name}` : '';
                }
                
                // Update progress
                if (isGenerating) {
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
                        showError(data.error, "REPORT_GENERATION_ERROR");
                    }
                }
                
                // Update generating state
                isGenerating = data.is_generating;
                if (!isGenerating && statusCheckInterval) {
                    clearInterval(statusCheckInterval);
                }
            })
            .catch(err => {
                console.error('Status check error:', err);
                // Don't show the error in UI during regular checks to avoid disrupting the UX
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
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        // Update UI sections
        inputSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        errorSection.classList.add('hidden');
        
        // Set company name
        resultCompanyDisplay.textContent = data.company_name ? ` for ${data.company_name}` : '';
        
        // Set download link
        if (data.report_file) {
            // Use helper function to append session ID
            downloadLink.href = appendSessionId(`/download/${data.report_file}`);
            downloadLink.setAttribute('download', data.report_file.split('/').pop());
            downloadLink.classList.remove('hidden');
            
            // Load report preview with session ID parameter
            fetch(appendSessionId(`/download/${data.report_file}`), {
                method: 'HEAD' // Use HEAD request to avoid downloading content multiple times
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Preview check failed: ${response.status}`);
                }
                
                // Now fetch the content for preview
                return fetch(appendSessionId(`/download/${data.report_file}`));
            })
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
                console.error('Error loading preview:', err);
                previewContent.textContent = 'Error loading preview: ' + err.message;
            });
        } else {
            downloadLink.classList.add('hidden');
            previewContent.textContent = 'Report file not available.';
        }
    }
    
    // Show error
    function showError(message, errorCode) {
        isGenerating = false;
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        // Update UI sections
        inputSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        
        // Base error message
        let fullErrorMessage = message || 'An unknown error occurred.';
        
        // Add guidance based on error code
        if (errorCode) {
            addLogEntry(`Error Code: ${errorCode}`);
            
            // Add specific guidance based on error code
            switch(errorCode) {
                case 'INVALID_SESSION_ID':
                    fullErrorMessage += '<br><br>The session ID may have expired or been deleted. Please check if you copied it correctly.';
                    break;
                case 'MISSING_SESSION_ID':
                    fullErrorMessage += '<br><br>Please provide a valid session ID or start a new report.';
                    break;
                case 'NETWORK_ERROR':
                    fullErrorMessage += '<br><br>Please check your internet connection and try again.';
                    break;
                case 'FILE_NOT_FOUND':
                    fullErrorMessage += '<br><br>The report file may have been deleted or moved. Try generating a new report.';
                    break;
                case 'EMPTY_SESSION_ID':
                    fullErrorMessage += '<br><br>Please enter a session ID in the restore field.';
                    break;
                case 'REPORT_GENERATION_ERROR':
                    fullErrorMessage += '<br><br>There was an error during report generation. Please try again with a different company name or time period.';
                    break;
            }
        }
        
        // Set error message with HTML for formatting
        errorMessage.innerHTML = fullErrorMessage;
    }
    
    // Cancel report generation
    function cancelGeneration() {
        if (!sessionId || !confirm('Are you sure you want to cancel the report generation?')) {
            return;
        }
        
        // Show cancellation in progress
        statusMessage.textContent = "Cancelling report generation...";
        
        // Send cancellation request to server with session ID
        fetch(appendSessionId('/cancel'), {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Cancellation failed: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                addLogEntry("Report generation cancelled by user");
                console.log("Report generation cancelled successfully");
            } else {
                console.error("Failed to cancel report generation:", data.message);
                addLogEntry(`Cancellation error: ${data.message}`);
            }
            
            // Reset UI regardless of server response
            isGenerating = false;
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            resetUI();
        })
        .catch(err => {
            console.error("Error cancelling report generation:", err);
            addLogEntry(`Cancellation error: ${err.message}`);
            
            // Reset UI even if there was an error
            isGenerating = false;
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            resetUI();
        });
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
        
        // Don't clear session ID - it should persist across UI resets
    }
}); 