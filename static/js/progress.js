/**
 * Progress Monitoring JavaScript
 * 
 * This script handles real-time progress updates for research tasks,
 * polling the progress API endpoints and updating the UI accordingly.
 */

// Configure polling interval (milliseconds)
const POLLING_INTERVAL = 2000;

// Store polling timeout ID to allow cancellation
let pollingTimeoutId = null;

// Track if polling is active
let isPolling = false;

/**
 * Initialize progress monitoring for a specific job
 * @param {string} jobId - The job ID to monitor
 */
function initProgressMonitoring(jobId) {
    // Get the progress container element
    const progressContainer = document.querySelector('.progress-container');
    
    // If no progress container found, exit
    if (!progressContainer) {
        console.error('Progress container not found');
        return;
    }
    
    // Start polling for updates
    startPolling(jobId);
    
    // Handle page visibility changes to pause/resume polling
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            if (!isPolling) {
                startPolling(jobId);
            }
        } else {
            stopPolling();
        }
    });
}

/**
 * Start polling for progress updates
 * @param {string} jobId - The job ID to monitor
 */
function startPolling(jobId) {
    if (isPolling) return;
    
    isPolling = true;
    
    // Immediately fetch initial state
    fetchProgressUpdate(jobId);
    
    // Set up recurring polling
    pollingTimeoutId = setTimeout(() => {
        pollJobStatus(jobId);
    }, POLLING_INTERVAL);
}

/**
 * Stop polling for progress updates
 */
function stopPolling() {
    if (pollingTimeoutId) {
        clearTimeout(pollingTimeoutId);
        pollingTimeoutId = null;
    }
    isPolling = false;
}

/**
 * Recursive function to poll job status
 * @param {string} jobId - The job ID to monitor
 */
function pollJobStatus(jobId) {
    fetchProgressUpdate(jobId)
        .then(status => {
            // If job is completed or encountered an error, stop polling
            if (status.is_completed || status.error) {
                stopPolling();
                
                // If completed successfully, show completion message
                if (status.is_completed && !status.error) {
                    showCompletionMessage(jobId);
                }
                
                return;
            }
            
            // Continue polling
            pollingTimeoutId = setTimeout(() => {
                pollJobStatus(jobId);
            }, POLLING_INTERVAL);
        })
        .catch(error => {
            console.error('Error polling job status:', error);
            
            // On error, try again after a longer delay
            pollingTimeoutId = setTimeout(() => {
                pollJobStatus(jobId);
            }, POLLING_INTERVAL * 2);
        });
}

/**
 * Fetch progress update from the API
 * @param {string} jobId - The job ID to fetch progress for
 * @returns {Promise<Object>} - Promise resolving to the job status
 */
function fetchProgressUpdate(jobId) {
    return fetch(`/api/progress/${jobId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(status => {
            updateProgressUI(status);
            return status;
        });
}

/**
 * Update the progress UI based on job status
 * @param {Object} status - The job status object
 */
function updateProgressUI(status) {
    // Update progress bar
    const progressBar = document.querySelector('.progress-fill');
    if (progressBar) {
        progressBar.style.width = `${status.progress_percentage}%`;
    }
    
    // Update progress percentage text
    const progressStatus = document.querySelector('.progress-status span:first-child');
    if (progressStatus) {
        progressStatus.textContent = `${Math.round(status.progress_percentage)}% Complete`;
    }
    
    // Update estimated time remaining
    const remainingTime = document.querySelector('.progress-status span:last-child');
    if (remainingTime) {
        remainingTime.textContent = `Est. completion: ${status.formatted_remaining}`;
    }
    
    // Update status message
    const statusIndicator = document.querySelector('.status-indicator:first-child span');
    if (statusIndicator) {
        statusIndicator.textContent = `Status: ${status.status_message}`;
    }
    
    // Update phase indicators
    updatePhaseIndicators(status.current_phase);
    
    // If there's an error, show error message
    if (status.error) {
        showErrorMessage(status.error);
    }
}

/**
 * Update the phase indicators based on current phase
 * @param {string} currentPhase - The current research phase
 */
function updatePhaseIndicators(currentPhase) {
    // Map phases to indices
    const phaseIndices = {
        'planning': 0,
        'searching': 1,
        'analyzing': 2,
        'synthesizing': 3,
        'report_generation': 4,
        'completed': 5
    };
    
    // Get all phase indicators except the first one (which is the status indicator)
    const phaseIndicators = document.querySelectorAll('.status-indicator:not(:first-child)');
    
    // Get the current phase index
    const currentIndex = phaseIndices[currentPhase] || 0;
    
    // Update classes for each phase indicator
    phaseIndicators.forEach((indicator, index) => {
        if (index <= currentIndex) {
            indicator.classList.add('active');
        } else {
            indicator.classList.remove('active');
        }
    });
}

/**
 * Show completion message and option to view results
 * @param {string} jobId - The completed job ID
 */
function showCompletionMessage(jobId) {
    const container = document.querySelector('.research-form');
    if (!container) return;
    
    // Create completion message element
    const completionMessage = document.createElement('div');
    completionMessage.className = 'completion-message';
    completionMessage.innerHTML = `
        <h3>Research Complete!</h3>
        <p>Your research has been successfully completed.</p>
        <div class="action-buttons">
            <a href="/results/${jobId}" class="button">View Results</a>
            <a href="/" class="button secondary">Back to Home</a>
        </div>
    `;
    
    // Add message to the container
    container.appendChild(completionMessage);
    
    // Scroll to the completion message
    completionMessage.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show error message
 * @param {string} errorMessage - The error message to display
 */
function showErrorMessage(errorMessage) {
    const container = document.querySelector('.research-form');
    if (!container) return;
    
    // Remove existing error message if any
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error message element
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.innerHTML = `
        <h3>Error Occurred</h3>
        <p>${errorMessage}</p>
        <div class="action-buttons">
            <a href="/" class="button">Back to Home</a>
        </div>
    `;
    
    // Add message to the container
    container.appendChild(errorElement);
    
    // Scroll to the error message
    errorElement.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Fetch and display event history for a job
 * @param {string} jobId - The job ID to fetch events for
 */
function loadEventHistory(jobId) {
    // Create event history container if it doesn't exist
    let historyContainer = document.querySelector('.event-history');
    
    if (!historyContainer) {
        const progressContainer = document.querySelector('.progress-container');
        if (!progressContainer) return;
        
        historyContainer = document.createElement('div');
        historyContainer.className = 'event-history';
        historyContainer.innerHTML = '<h3>Research Events</h3><ul class="event-list"></ul>';
        
        // Insert after progress container
        progressContainer.parentNode.insertBefore(historyContainer, progressContainer.nextSibling);
    }
    
    // Fetch event history
    fetch(`/api/progress/${jobId}/events`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(events => {
            const eventList = historyContainer.querySelector('.event-list');
            if (!eventList) return;
            
            // Clear existing events
            eventList.innerHTML = '';
            
            // Add events to the list
            events.forEach(event => {
                const eventTime = new Date(event.timestamp * 1000).toLocaleTimeString();
                const eventItem = document.createElement('li');
                eventItem.className = `event-item event-${event.phase}`;
                eventItem.innerHTML = `
                    <span class="event-time">${eventTime}</span>
                    <span class="event-phase">${event.phase}</span>
                    <span class="event-description">${event.description}</span>
                `;
                eventList.appendChild(eventItem);
            });
        })
        .catch(error => {
            console.error('Error loading event history:', error);
        });
}

// Initialize on page load if job ID is present
document.addEventListener('DOMContentLoaded', () => {
    const progressContainer = document.querySelector('.progress-container');
    if (progressContainer) {
        const jobId = progressContainer.dataset.jobId;
        if (jobId) {
            initProgressMonitoring(jobId);
            
            // Add a button to toggle event history
            const detailsButton = document.createElement('button');
            detailsButton.className = 'details-toggle';
            detailsButton.textContent = 'Show Details';
            detailsButton.addEventListener('click', function() {
                if (this.textContent === 'Show Details') {
                    loadEventHistory(jobId);
                    this.textContent = 'Hide Details';
                } else {
                    const historyContainer = document.querySelector('.event-history');
                    if (historyContainer) {
                        historyContainer.style.display = 'none';
                    }
                    this.textContent = 'Show Details';
                }
            });
            
            // Add button after progress container
            progressContainer.parentNode.insertBefore(detailsButton, progressContainer.nextSibling);
        }
    }
});
