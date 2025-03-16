"""
Complete Progress Monitoring System Demo for Research Assistant

This script demonstrates the entire progress monitoring system, including:
1. Creating progress monitors
2. Updating progress through different research phases
3. Recording events and details
4. Simulating a complete research workflow

Run this script to test the progress monitoring system in isolation.
"""

import asyncio
import logging
import sys
import time
import uuid
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('progress_system_demo.log')
    ]
)

logger = logging.getLogger(__name__)

# Import progress monitoring components
from progress_monitor import ProgressMonitor, ResearchPhase, ProgressMonitorRegistry


def print_section(title):
    """Print a section title for better readability."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")


async def simulate_research_task(topic="Artificial Intelligence Applications", parameters=None):
    """Simulate a research task with real-time progress updates."""
    if parameters is None:
        parameters = {
            "audience": "professional",
            "depth": "comprehensive",
            "format": "standard",
            "time_constraint": 15
        }
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    print(f"Created research job with ID: {job_id}")
    
    # Create monitor and register with global registry
    monitor = ProgressMonitor(job_id, topic, parameters)
    ProgressMonitorRegistry.register(monitor)
    print(f"Registered progress monitor for '{topic}'")
    
    # Add a progress update callback for demonstration
    def progress_callback(status):
        print(f"Progress: {status['progress_percentage']}% - {status['status_message']} (Phase: {status['current_phase']})")
    
    monitor.register_observer(progress_callback)
    
    try:
        print_section("Starting Research")
        
        # Planning phase
        print("Entering PLANNING phase...")
        monitor.update_phase(ResearchPhase.PLANNING, f"Planning research for '{topic}'")
        await asyncio.sleep(1)
        
        subtopics = ["Healthcare", "Transportation", "Education", "Finance", "Environmental Science"]
        
        # Planning steps
        print("Analyzing research question...")
        monitor.update_progress(10, "Analyzing research question")
        await asyncio.sleep(1)
        
        print("Identifying key aspects...")
        monitor.update_progress(15, "Identifying key aspects")
        await asyncio.sleep(1)
        
        print("Finalizing research plan...")
        monitor.update_progress(20, "Finalized research plan", {"subtopics": subtopics})
        await asyncio.sleep(1)
        
        # Searching phase
        print("\nEntering SEARCHING phase...")
        monitor.update_phase(ResearchPhase.SEARCHING, "Searching for sources")
        await asyncio.sleep(1)
        
        sources = []
        # Search steps
        for i in range(5):
            source = {
                "title": f"AI in {subtopics[i]}: A Comprehensive Review",
                "url": f"https://example.com/ai-{subtopics[i].lower().replace(' ', '-')}",
                "credibility": 0.85 + (i * 0.01)
            }
            sources.append(source)
            
            progress = 25 + (i * 3)
            print(f"Found source: {source['title']}")
            monitor.update_progress(progress, f"Found source {i+1}", {"source": source})
            await asyncio.sleep(0.5)
        
        print("Completed source search...")
        monitor.update_progress(40, "Completed source search", {"sources": sources})
        await asyncio.sleep(1)
        
        # Analyzing phase
        print("\nEntering ANALYZING phase...")
        monitor.update_phase(ResearchPhase.ANALYZING, "Analyzing sources")
        await asyncio.sleep(1)
        
        key_findings = []
        # Analysis steps
        for i, source in enumerate(sources):
            finding = f"Key finding from {source['title']}: AI improves efficiency by 30-40% in {subtopics[i]}"
            key_findings.append(finding)
            
            progress = 45 + (i * 3)
            print(f"Analyzing: {finding}")
            monitor.update_progress(progress, f"Analyzing source {i+1}/{len(sources)}")
            await asyncio.sleep(1)
        
        print("Completed source analysis...")
        monitor.update_progress(60, "Completed source analysis", {"key_findings": key_findings})
        await asyncio.sleep(1)
        
        # Synthesizing phase
        print("\nEntering SYNTHESIZING phase...")
        monitor.update_phase(ResearchPhase.SYNTHESIZING, "Synthesizing information")
        await asyncio.sleep(1)
        
        # Synthesis steps
        for i, subtopic in enumerate(subtopics[:4]):
            progress = 65 + (i * 4)
            print(f"Synthesizing information for {subtopic}...")
            monitor.update_progress(
                progress, 
                f"Synthesizing information for {subtopic}",
                {"subtopic": subtopic, "sources_used": i + 2}
            )
            await asyncio.sleep(1)
        
        print("Completed information synthesis...")
        monitor.update_progress(80, "Completed information synthesis")
        await asyncio.sleep(1)
        
        # Report generation phase
        print("\nEntering REPORT GENERATION phase...")
        monitor.update_phase(ResearchPhase.GENERATING, "Generating report")
        await asyncio.sleep(1)
        
        # Report generation steps
        sections = ["Introduction", "Methodology", "Findings", "Discussion", "Conclusion"]
        for i, section in enumerate(sections):
            progress = 82 + (i * 3)
            print(f"Generating {section} section...")
            monitor.update_progress(progress, f"Generating {section} section")
            await asyncio.sleep(0.5)
        
        print("Formatting citations and references...")
        monitor.update_progress(95, "Formatting citations and references")
        await asyncio.sleep(1)
        
        # Save progress data to a file
        print("\nSaving progress data...")
        os.makedirs("output", exist_ok=True)
        progress_path = os.path.join("output", f"progress_system_{job_id}.json")
        monitor.save_to_file(progress_path)
        print(f"Progress data saved to {progress_path}")
        
        # Create a sample report JSON
        report_path = os.path.join("output", f"report_system_{job_id}.json")
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
        print(f"Sample report saved to {report_path}")
        
        # Complete the task
        print("\nCompleting research task...")
        monitor.update_phase(ResearchPhase.COMPLETED, "Research completed successfully")
        print("Research completed successfully!")
        
        # Print the final events recorded
        print_section("Final Event History")
        for i, event in enumerate(monitor.events):
            print(f"{i+1}. [{event.phase.value}] {event.description}")
            
        await asyncio.sleep(1)
        
    except Exception as e:
        logger.error(f"Error in research task: {e}", exc_info=True)
        monitor.record_error(str(e))
        print(f"ERROR: {str(e)}")
    
    finally:
        # For demo purposes, we'll keep the monitor registered
        # In production, we would unregister when appropriate
        # ProgressMonitorRegistry.unregister(job_id)
        pass
        
    return job_id, monitor


def active_jobs_display():
    """Display all active jobs in the registry."""
    active_jobs = ProgressMonitorRegistry.list_active_jobs()
    
    print_section("Active Research Jobs")
    if active_jobs:
        for job in active_jobs:
            print(f"Job ID: {job['job_id']}")
            print(f"Topic: {job['topic']}")
            print(f"Progress: {job['progress']}%")
            print(f"Phase: {job['phase']}")
            print(f"Started: {datetime.fromtimestamp(job['start_time']).strftime('%H:%M:%S')}")
            print("-" * 30)
    else:
        print("No active research jobs.")


async def main():
    """Run multiple research tasks in parallel for demonstration."""
    print_section("Progress Monitoring System Demo")
    print("Starting multiple research tasks...")
    
    # Create a few research tasks with different parameters
    tasks = [
        simulate_research_task("Artificial Intelligence in Healthcare", {
            "audience": "professional",
            "depth": "comprehensive",
            "format": "standard",
            "time_constraint": 15
        }),
        simulate_research_task("Climate Change Mitigation Strategies", {
            "audience": "academic",
            "depth": "expert",
            "format": "analytical",
            "time_constraint": 20
        }),
        simulate_research_task("Blockchain Applications in Finance", {
            "audience": "professional", 
            "depth": "standard",
            "format": "report",
            "time_constraint": 10
        })
    ]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)
    
    # Display active jobs after all tasks complete
    active_jobs_display()
    
    print_section("Demo Completed")
    print("Progress monitoring system demo completed successfully.")
    print("Explore the files in the 'output' directory for saved progress data and sample reports.")
    print("The Progress Monitor Registry still has all jobs registered - in a real application,")
    print("completed jobs would either be unregistered or kept for a limited time.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
