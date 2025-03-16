/**
 * Research Request JavaScript
 * 
 * This script handles interactions for the research request form,
 * including form validation and enhanced UI features.
 */

document.addEventListener('DOMContentLoaded', function() {
  // Get form elements
  const researchForm = document.querySelector('.research-form');
  const topicInput = document.getElementById('research-topic');
  const submitButton = document.querySelector('.submit-button');
  
  // Set up example questions functionality with a slight delay to ensure DOM is fully processed
  setTimeout(() => {
    setupExampleQuestions();
  }, 100);
  
  // Function to set up example questions
  function setupExampleQuestions() {
    const exampleQuestions = document.querySelectorAll('.example-questions li');
    console.log('Found example questions:', exampleQuestions.length);
    
    exampleQuestions.forEach(question => {
      question.style.cursor = 'pointer'; // Ensure cursor shows it's clickable
      
      question.addEventListener('click', function() {
        console.log('Example question clicked:', this.textContent);
        if (topicInput) {
          topicInput.value = this.textContent.trim();
          topicInput.focus();
          
          // Animate scroll to form top
          const formSection = document.querySelector('.topic-section');
          if (formSection) {
            formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        } else {
          console.error('Topic input field not found');
        }
      });
    });
  }
  
  // Form validation
  if (researchForm) {
    researchForm.addEventListener('submit', function(e) {
      // Check if research topic is provided
      if (!topicInput.value.trim()) {
        e.preventDefault();
        
        // Highlight the empty field
        topicInput.classList.add('error');
        
        // Show error message
        const errorMsg = document.createElement('div');
        errorMsg.className = 'error-message';
        errorMsg.textContent = 'Please enter a research topic or question.';
        
        // Insert error message after the textarea
        const inputGroup = topicInput.closest('.input-group');
        
        // Remove any existing error message first
        const existingError = inputGroup.querySelector('.error-message');
        if (existingError) {
          existingError.remove();
        }
        
        inputGroup.appendChild(errorMsg);
        
        // Scroll to the error
        topicInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        return false;
      }
      
      // Disable the submit button and show loading state
      submitButton.disabled = true;
      submitButton.textContent = 'Processing...';
      submitButton.classList.add('loading');
    });
    
    // Remove error styling when user starts typing
    topicInput.addEventListener('input', function() {
      this.classList.remove('error');
      const errorMsg = this.closest('.input-group').querySelector('.error-message');
      if (errorMsg) {
        errorMsg.remove();
      }
    });
  }
  
  // Dynamic tooltip positioning
  const tooltipContainers = document.querySelectorAll('.tooltip-container');
  tooltipContainers.forEach(container => {
    const tooltipContent = container.querySelector('.tooltip-content');
    
    container.addEventListener('mouseenter', function() {
      // Check if tooltip would go off-screen to the right
      const tooltipRect = tooltipContent.getBoundingClientRect();
      const rightEdge = tooltipRect.left + tooltipRect.width;
      
      if (rightEdge > window.innerWidth) {
        tooltipContent.style.left = 'auto';
        tooltipContent.style.right = '0';
        tooltipContent.style.transform = 'none';
      }
    });
  });
  
  // Select input changes update report preview
  const formatSelect = document.getElementById('report-format');
  const audienceSelect = document.getElementById('audience');
  
  if (formatSelect && audienceSelect) {
    // Array of sample structures for different formats
    const formatStructures = {
      'summary': `
        <li><strong>Executive Overview</strong>: Core findings and implications</li>
        <li><strong>Key Points</strong>: Main takeaways in bullet format</li>
        <li><strong>Context</strong>: Brief background information</li>
        <li><strong>Sources</strong>: List of information sources</li>
      `,
      'report': `
        <li><strong>Executive Summary</strong>: Key findings at a glance</li>
        <li><strong>Introduction</strong>: Topic overview and context</li>
        <li><strong>Methodology</strong>: How the research was conducted</li>
        <li><strong>Main Sections</strong>: Organized by subtopics with findings</li>
        <li><strong>Analysis</strong>: Interpretation of information</li>
        <li><strong>Conclusion</strong>: Summary of findings</li>
        <li><strong>References</strong>: All sources properly cited</li>
      `,
      'analysis': `
        <li><strong>Executive Summary</strong>: Overview of analysis</li>
        <li><strong>Problem Statement</strong>: Definition of research question</li>
        <li><strong>Data and Methods</strong>: Sources and analytical approach</li>
        <li><strong>Results</strong>: Detailed findings with data support</li>
        <li><strong>Critical Analysis</strong>: In-depth interpretation</li>
        <li><strong>Implications</strong>: Consequences of findings</li>
        <li><strong>References</strong>: All sources with analytical notes</li>
      `,
      'presentation': `
        <li><strong>Title and Overview</strong>: Main topic and key messages</li>
        <li><strong>Background Slides</strong>: Context and importance</li>
        <li><strong>Main Points</strong>: Key information in bullet format</li>
        <li><strong>Visual Elements</strong>: Data presented in charts/tables</li>
        <li><strong>Conclusions</strong>: Key takeaways from research</li>
        <li><strong>Sources</strong>: References for further reading</li>
      `
    };
    
    // Update description text based on audience
    const audienceDescriptions = {
      'general': 'Report will use everyday language accessible to a broad audience.',
      'academic': 'Report will use scholarly language with proper citations and academic structure.',
      'professional': 'Report will use business terminology and focus on practical applications.',
      'technical': 'Report will use specialized terminology and include technical details.'
    };
    
    // Function to update the preview based on selected format and audience
    function updatePreview() {
      const selectedFormat = formatSelect.value;
      const selectedAudience = audienceSelect.value;
      const reportPreview = document.querySelector('.report-preview ul');
      
      if (reportPreview && formatStructures[selectedFormat]) {
        reportPreview.innerHTML = formatStructures[selectedFormat];
      }
      
      // Update audience description in the tooltip
      const audienceTooltip = document.querySelector('select#audience').closest('.parameter-group').querySelector('.tooltip-content');
      if (audienceTooltip && audienceDescriptions[selectedAudience]) {
        audienceTooltip.textContent = audienceDescriptions[selectedAudience];
      }
    }
    
    // Update preview when format changes
    formatSelect.addEventListener('change', updatePreview);
    audienceSelect.addEventListener('change', updatePreview);
    
    // Initial update
    updatePreview();
  }
  
  // Depth and time constraint relationship
  const depthSelect = document.getElementById('depth');
  const timeSelect = document.getElementById('time-constraint');
  
  if (depthSelect && timeSelect) {
    // Update time when depth changes
    depthSelect.addEventListener('change', function() {
      const depthValue = this.value;
      
      // Suggest appropriate time based on depth
      switch(depthValue) {
        case 'brief':
          suggestOption(timeSelect, '15');
          break;
        case 'standard':
          suggestOption(timeSelect, '30');
          break;
        case 'comprehensive':
          suggestOption(timeSelect, '60');
          break;
        case 'expert':
          suggestOption(timeSelect, '120');
          break;
      }
    });
    
    // Helper to suggest a time option
    function suggestOption(selectElement, value) {
      // Only update if the current selection doesn't match the suggested one
      if (selectElement.value !== value) {
        // Flash the select to draw attention
        selectElement.classList.add('highlight');
        
        // Remove highlight after animation
        setTimeout(() => {
          selectElement.classList.remove('highlight');
        }, 1000);
      }
    }
  }
});

// Add highlight class styles
document.head.insertAdjacentHTML('beforeend', `
  <style>
    .submit-button.loading {
      position: relative;
      color: transparent;
    }
    
    .submit-button.loading::after {
      content: "";
      position: absolute;
      width: 20px;
      height: 20px;
      top: 50%;
      left: 50%;
      margin-top: -10px;
      margin-left: -10px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    .error {
      border-color: #dc3545 !important;
      box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.25) !important;
    }
    
    .error-message {
      color: #dc3545;
      font-size: 0.85em;
      margin-top: 5px;
    }
    
    .highlight {
      animation: pulse 1s ease-in-out;
    }
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(44, 123, 229, 0.5); }
      70% { box-shadow: 0 0 0 5px rgba(44, 123, 229, 0); }
      100% { box-shadow: 0 0 0 0 rgba(44, 123, 229, 0); }
    }
  </style>
`);
