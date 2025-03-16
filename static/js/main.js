/**
 * Research Assistant UI - Client-side JavaScript
 * 
 * This file handles client-side functionality for the Research Assistant UI,
 * including form submission, progress tracking, tab switching, and user interactions.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    setupTabNavigation();
    
    // Star rating
    setupStarRating();
    
    // Progress polling if on progress page
    if (document.querySelector('.progress-container')) {
        pollJobStatus();
    }
});

/**
 * Set up tab navigation
 */
function setupTabNavigation() {
    const tabs = document.querySelectorAll('.tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabTarget = this.dataset.target;
            if (tabTarget) {
                // If it's a local tab, handle it on the client
                if (tabTarget.startsWith('#')) {
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Hide all tab content
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.style.display = 'none';
                    });
                    
                    // Show selected tab content
                    const targetContent = document.querySelector(tabTarget);
                    if (targetContent) {
                        targetContent.style.display = 'block';
                    }
                } else {
                    // If it's a route, navigate to it
                    window.location.href = tabTarget;
                }
            }
        });
    });
}

/**
 * Set up star rating functionality
 */
function setupStarRating() {
    const stars = document.querySelectorAll('.star');
    const ratingInput = document.getElementById('rating-input');
    
    if (stars.length === 0) return;
    
    stars.forEach((star, index) => {
        star.addEventListener('click', function() {
            const rating = index + 1;
            
            // Update rating input if it exists
            if (ratingInput) {
                ratingInput.value = rating;
            }
            
            // Update star UI
            stars.forEach((s, i) => {
                if (i <= index) {
                    s.classList.add('filled');
                } else {
                    s.classList.remove('filled');
                }
            });
        });
        
        star.addEventListener('mouseover', function() {
            const hoveredIndex = index;
            stars.forEach((s, i) => {
                if (i <= hoveredIndex) {
                    s.classList.add('hovered');
                } else {
                    s.classList.remove('hovered');
                }
            });
        });
        
        star.addEventListener('mouseout', function() {
            stars.forEach(s => {
                s.classList.remove('hovered');
            });
        });
    });
}

/**
 * Poll job status for progress updates
 */
function pollJobStatus() {
    const progressElement = document.querySelector('.progress-container');
    if (!progressElement) return;
    
    const jobId = progressElement.dataset.jobId;
    if (!jobId) return;
    
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-status span:first-child');
    const remainingText = document.querySelector('.progress-status span:last-child');
    const statusIndicators = document.querySelectorAll('.status-indicator');
    
    function updateProgress(data) {
        // Update progress bar
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        
        // Update progress text
        if (progressText) {
            progressText.textContent = `${data.progress}% Complete`;
        }
        
        // Update remaining time
        if (remainingText && data.remaining_minutes !== null) {
            remainingText.textContent = `Est. completion: ${data.remaining_minutes} minutes`;
        }
        
        // Update phase indicators
        if (statusIndicators.length > 0) {
            const phases = ["Planning", "Searching", "Analyzing", "Synthesizing", "Report Generation"];
            const currentPhaseIndex = phases.indexOf(data.phase);
            
            statusIndicators.forEach((indicator, index) => {
                indicator.classList.remove('active');
                if (index === currentPhaseIndex) {
                    indicator.classList.add('active');
                }
            });
        }
        
        // Redirect to results if completed
        if (data.redirect_to_results) {
            window.location.href = `/research/${jobId}/results`;
        }
    }
    
    function checkStatus() {
        fetch(`/research/${jobId}/status`)
            .then(response => response.json())
            .then(data => {
                updateProgress(data);
                
                // Continue polling if not complete
                if (!data.redirect_to_results) {
                    setTimeout(checkStatus, 3000);
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                setTimeout(checkStatus, 5000);  // Retry with longer delay on error
            });
    }
    
    // Start polling
    checkStatus();
}

/**
 * Submit feedback form via AJAX
 */
function submitFeedback(formId) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        fetch('/report_feedback', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Show success message
                    const feedbackForm = document.querySelector('.feedback-form');
                    if (feedbackForm) {
                        feedbackForm.innerHTML = `
                            <h3>Feedback Submitted</h3>
                            <p>Thank you for your feedback! Your input helps us improve.</p>
                        `;
                    }
                }
            })
            .catch(error => {
                console.error('Error submitting feedback:', error);
                alert('Error submitting feedback. Please try again.');
            });
    });
}
