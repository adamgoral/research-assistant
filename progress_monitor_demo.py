"""
Progress Monitor Demo for Research Assistant

This script demonstrates the functionality of the progress monitoring system
by simulating a research task with various phases and progress updates.
"""

import time
import uuid
import logging
import sys
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('progress_monitor_demo.log')
    ]
)

logger = logging.getLogger(__name__)

# Import the progress monitor
from progress_monitor import ProgressMonitor, ResearchPhase, ProgressMonitorRegistry

# Demo parameters
DEMO_PARAMETERS = {
    "audience": "professional",
    "depth": "comprehensive", 
    "format": "standard",
    "time_constraint": 15
}

def progress_update_callback(status):
    """Callback function for progress updates."""
    logger.info(f"Progress update: {status['progress_percentage']}% - {status['status_message']}")
    logger.info(f"Phase: {status['current_phase']}, Remaining: {status['formatted_remaining']}")


def simulate_research_task(topic="Artificial Intelligence in Healthcare"):
    """Simulate a research task with progress updates."""
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create monitor and register callback
    monitor = ProgressMonitor(job_id, topic, DEMO_PARAMETERS)
    monitor.register_observer(progress_update_callback)
    
    # Register with global registry
    ProgressMonitorRegistry.register(monitor)
    
    try:
        # Planning phase
        monitor.update_phase(ResearchPhase.PLANNING, "Creating research plan")
        logger.info(f"Starting research on '{topic}'")
        time.sleep(2)
        
        monitor.update_progress(10, "Analyzing research question")
        time.sleep(1)
        
        monitor.update_progress(15, "Identifying key aspects")
        time.sleep(1)
        
        subtopics = ["Clinical applications", "Data privacy", "Cost effectiveness", "Implementation challenges"]
        monitor.update_progress(20, "Finalized research plan", {"subtopics": subtopics})
        time.sleep(1)
        
        # Searching phase
        monitor.update_phase(ResearchPhase.SEARCHING, "Searching for sources")
        time.sleep(1)
        
        for i in range(5):
            progress = 25 + (i * 3)
            monitor.update_progress(progress, f"Found {i+1} sources")
            time.sleep(0.5)
            
        sources = [
            {"title": "AI Applications in Diagnostic Imaging", "url": "https://example.com/ai-imaging"},
            {"title": "Machine Learning in Clinical Decision Support", "url": "https://example.com/ml-decision"},
            {"title": "Healthcare Data Privacy Concerns", "url": "https://example.com/health-privacy"},
            {"title": "Cost-Benefit Analysis of AI in Healthcare", "url": "https://example.com/ai-cost"},
            {"title": "Implementation Barriers for AI Systems", "url": "https://example.com/ai-barriers"}
        ]
        monitor.update_progress(40, "Completed source search", {"sources": sources})
        time.sleep(1)
        
        # Analysis phase
        monitor.update_phase(ResearchPhase.ANALYZING, "Analyzing sources")
        time.sleep(1)
        
        for i in range(5):
            progress = 45 + (i * 3)
            monitor.update_progress(progress, f"Analyzing source {i+1}/{len(sources)}")
            time.sleep(1)
            
        key_findings = [
            "AI shows 15-30% improvement in diagnostic accuracy",
            "Privacy concerns include data ownership and consent management",
            "Implementation costs range from $100K to $1M depending on scale",
            "Staff training represents a significant adoption barrier"
        ]
        monitor.update_progress(60, "Completed source analysis", {"key_findings": key_findings})
        time.sleep(1)
        
        # Synthesis phase
        monitor.update_phase(ResearchPhase.SYNTHESIZING, "Synthesizing information")
        time.sleep(1)
        
        for i in range(4):
            progress = 65 + (i * 4)
            monitor.update_progress(progress, f"Synthesizing information for subtopic {i+1}/{len(subtopics)}")
            time.sleep(1)
            
        monitor.update_progress(80, "Completed information synthesis")
        time.sleep(1)
        
        # Report generation phase
        monitor.update_phase(ResearchPhase.GENERATING, "Generating report")
        time.sleep(1)
        
        sections = ["Introduction", "Methodology", "Findings", "Discussion", "Conclusion"]
        for i, section in enumerate(sections):
            progress = 82 + (i * 2)
            monitor.update_progress(progress, f"Generating {section} section")
            time.sleep(0.5)
            
        monitor.update_progress(95, "Formatting citations and references")
        time.sleep(1)
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # Save a sample report
        report_path = os.path.join("output", f"report_{job_id}.json")
        report_data = {
            "title": f"Research Report: {topic}",
            "job_id": job_id,
            "generated_at": datetime.now().isoformat(),
            "sections": {section: f"Content for {section}" for section in sections},
            "sources": sources,
            "key_findings": key_findings
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Save progress history
        progress_path = os.path.join("output", f"progress_{job_id}.json")
        monitor.save_to_file(progress_path)
        
        # Complete the task
        monitor.update_phase(ResearchPhase.COMPLETED, "Research completed successfully")
        logger.info(f"Research completed. Output saved to {report_path}")
        
        # Sleep to allow viewing the completed state
        time.sleep(2)
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error in research task: {e}")
        monitor.record_error(str(e))
    
    finally:
        # Unregister from registry when done
        ProgressMonitorRegistry.unregister(job_id)
        
    return job_id, monitor


def main():
    """Run the demo with sample topics."""
    topics = [
        "Artificial Intelligence in Healthcare",
        "Renewable Energy Technologies",
        "Cybersecurity Best Practices"
    ]
    
    for topic in topics:
        job_id, monitor = simulate_research_task(topic)
        print(f"\nCompleted research on '{topic}'")
        print(f"Final progress: {monitor.progress_percentage}%")
        print(f"Total events recorded: {len(monitor.events)}")
        print(f"Elapsed time: {monitor.get_status()['formatted_elapsed']}")
        print("-" * 50)


if __name__ == "__main__":
    print("Starting Progress Monitor Demo")
    main()
    print("Demo completed")
