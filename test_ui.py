"""
UI Test for Progress Monitoring System

This script tests the Flask UI integration of the progress monitoring system
by starting a test server, creating a research job, and updating its progress.
"""

import time
import threading
import webbrowser
import uuid
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import progress monitoring components
from progress_monitor import ProgressMonitor, ResearchPhase, ProgressMonitorRegistry
from flask import Flask

def test_process():
    """Run a test process that demonstrates the progress monitoring UI."""
    # First create a job with progress monitor
    job_id = str(uuid.uuid4())
    logger.info(f"Created test job with ID: {job_id}")
    
    # Create parameters
    parameters = {
        "audience": "professional",
        "depth": "comprehensive",
        "format": "standard",
        "time_constraint": 5
    }
    
    # Create monitor
    monitor = ProgressMonitor(job_id, "UI Test Research Topic", parameters)
    ProgressMonitorRegistry.register(monitor)
    logger.info("Registered progress monitor")
    
    # Wait for a moment to allow web browser to open
    time.sleep(3)
    
    # Simulate progress
    phases = [
        (ResearchPhase.PLANNING, "Planning research"),
        (ResearchPhase.SEARCHING, "Searching for sources"),
        (ResearchPhase.ANALYZING, "Analyzing sources"),
        (ResearchPhase.SYNTHESIZING, "Synthesizing information"),
        (ResearchPhase.GENERATING, "Generating report"),
        (ResearchPhase.COMPLETED, "Research completed successfully")
    ]
    
    # Open browser to view progress
    url = f"http://localhost:5000/progress/{job_id}"
    logger.info(f"Opening browser to view progress at {url}")
    webbrowser.open(url)
    
    # Run through phases
    for phase, message in phases:
        if phase != ResearchPhase.COMPLETED:
            monitor.update_phase(phase, message)
            logger.info(f"Updated phase to {phase.value}: {message}")
            
            # Add some progress updates within the phase
            base_progress = 20 * (phases.index((phase, message)))
            for i in range(4):
                progress = base_progress + (i * 5)
                monitor.update_progress(progress, f"Progress step {i+1}")
                logger.info(f"Updated progress to {progress}%")
                time.sleep(1)
        else:
            # Short pause before completing
            time.sleep(2)
            monitor.update_phase(phase, message)
            logger.info("Research completed successfully")


def run_app_server():
    """Run the Flask application server."""
    from app import app
    app.run(debug=True, use_reloader=False)


def run_test():
    """Run the complete UI test."""
    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_app_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Run the test process
    test_process()
    
    # Wait for a moment to view the final state
    time.sleep(10)
    
    logger.info("Test completed")


if __name__ == "__main__":
    print("Starting UI test for progress monitoring system...")
    print("This will launch a browser to view the progress page.")
    print("Press Ctrl+C to quit the test when finished.")
    
    try:
        run_test()
        # Keep the main thread alive to allow viewing the final state
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Test stopped by user")
