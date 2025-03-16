"""
Research Assistant Web UI

This module implements a Flask web application for the Research Assistant system,
providing a user interface for submitting research requests, tracking progress,
and viewing results.
"""

import asyncio
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Import progress monitoring system
from progress_monitor import ProgressMonitor, ResearchPhase, ProgressMonitorRegistry
from progress_api import progress_bp, create_progress_monitor, update_research_phase, update_research_progress

# Import from research pipeline
from research_pipeline import (
    ResearchPipeline,
    ResearchTopic,
    SearchQuery
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_research_assistant')

# Register the progress_bp blueprint
app.register_blueprint(progress_bp)

# In-memory storage for research jobs and results
# In a production app, this would be a database
RESEARCH_JOBS = {}
RESEARCH_RESULTS = {}
RECENT_REPORTS = []

# Custom Jinja2 filters
@app.template_filter('datetime')
def format_datetime(value, format='%b %d, %Y at %H:%M'):
    """Format a datetime to a pretty string."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime(format)

@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime(timestamp, format='%b %d, %Y at %H:%M'):
    """Convert a Unix timestamp to a formatted datetime string."""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime(format)
    return "Unknown"


# ----- Helper Functions -----

def _get_pipeline() -> ResearchPipeline:
    """Create and return a research pipeline instance."""
    # Get API key from environment (if available)
    api_key = os.environ.get("SERPAPI_API_KEY")
    use_real_search = api_key is not None
    
    return ResearchPipeline(use_real_search=use_real_search, api_key=api_key)


def _generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())


def _create_topic(form_data: Dict) -> ResearchTopic:
    """Create a ResearchTopic from form data."""
    # Extract keywords from the topic
    topic_text = form_data.get('research_topic', '')
    keywords = [kw.strip() for kw in topic_text.split(',') if kw.strip()]
    
    # If no comma-separated keywords, use the first few words
    if not keywords and topic_text:
        words = topic_text.split()
        keywords = [w for w in words if len(w) > 3][:5]
    
    return ResearchTopic(
        id=_generate_job_id(),
        title=topic_text,
        description=topic_text,
        keywords=keywords
    )


def run_async_task(coro):
    """Run an async task in a new thread with its own event loop."""
    import threading
    
    def _run_in_thread(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
    
    thread = threading.Thread(target=_run_in_thread, args=(coro,))
    thread.daemon = True
    thread.start()
    return thread


async def _run_research(job_id: str, topic: ResearchTopic, max_depth: int):
    """Run the research in a background task."""
    try:
        # Create a pipeline instance
        pipeline = _get_pipeline()
        
        # Update job status
        RESEARCH_JOBS[job_id]['status'] = 'running'
        RESEARCH_JOBS[job_id]['start_time'] = datetime.now()
        
        # Update progress - Planning phase
        update_research_phase(
            job_id,
            ResearchPhase.PLANNING,
            f"Planning research for '{topic.title}'"
        )
        
        # Planning phase
        update_research_progress(job_id, 10, "Analyzing research question")
        await asyncio.sleep(1)  # Simulate work
        
        update_research_progress(job_id, 15, "Identifying key aspects")
        await asyncio.sleep(1)  # Simulate work
        
        # Extract subtopics from keywords
        subtopics = topic.keywords[:5] if topic.keywords else ["General research"]
        update_research_progress(
            job_id, 
            20, 
            "Finalized research plan", 
            {"subtopics": subtopics}
        )
        
        # Update progress - Searching phase
        update_research_phase(
            job_id,
            ResearchPhase.SEARCHING,
            "Searching for sources"
        )
        
        # Run the research
        stats = await pipeline.research_topic(
            topic,
            max_queries=max_depth,
            max_results_per_query=max_depth,
            progress_callback=lambda progress, message, details=None: 
                update_research_progress(job_id, 25 + (progress * 0.15), message, details)
        )
        
        # Update progress - Analyzing phase
        update_research_phase(
            job_id,
            ResearchPhase.ANALYZING,
            "Analyzing sources"
        )
        
        # Get the research results
        results = await pipeline.get_research_results(
            topic.id,
            progress_callback=lambda progress, message, details=None: 
                update_research_progress(job_id, 45 + (progress * 0.15), message, details)
        )
        
        # Update progress - Synthesizing phase
        update_research_phase(
            job_id,
            ResearchPhase.SYNTHESIZING,
            "Synthesizing information"
        )
        
        # Synthesis phase
        for i, subtopic in enumerate(subtopics):
            progress = 65 + (i * 4)
            update_research_progress(job_id, progress, f"Synthesizing information for {subtopic}")
            await asyncio.sleep(0.5)  # Simulate work
            
        update_research_progress(job_id, 80, "Completed information synthesis")
        
        # Update progress - Report Generation phase
        update_research_phase(
            job_id,
            ResearchPhase.GENERATING,
            "Generating report"
        )
        
        # Report generation phase
        sections = ["Introduction", "Methodology", "Findings", "Discussion", "Conclusion"]
        for i, section in enumerate(sections):
            progress = 82 + (i * 2)
            update_research_progress(job_id, progress, f"Generating {section} section")
            await asyncio.sleep(0.5)  # Simulate work
            
        update_research_progress(job_id, 95, "Formatting citations and references")
        
        # Store results
        RESEARCH_RESULTS[job_id] = {
            'stats': stats,
            'results': results,
            'topic': topic.dict(),
            'completion_time': datetime.now()
        }
        
        # Save progress data to a file
        monitor = ProgressMonitorRegistry.get(job_id)
        if monitor:
            os.makedirs("output", exist_ok=True)
            progress_path = os.path.join("output", f"progress_{job_id}.json")
            monitor.save_to_file(progress_path)
            
        # Update progress - Completed phase
        update_research_phase(
            job_id,
            ResearchPhase.COMPLETED,
            "Research completed successfully"
        )
        
        # Update job status
        RESEARCH_JOBS[job_id]['status'] = 'completed'
        RESEARCH_JOBS[job_id]['completion_time'] = datetime.now()
        
        # Add to recent reports
        RECENT_REPORTS.insert(0, {
            'job_id': job_id,
            'title': topic.title,
            'timestamp': datetime.now(),
            'report_type': 'Standard Report'
        })
        
        # Keep only the 10 most recent reports
        while len(RECENT_REPORTS) > 10:
            RECENT_REPORTS.pop()
            
    except Exception as e:
        logger.error(f"Error in research task: {e}", exc_info=True)
        
        # Update progress monitor with error
        monitor = ProgressMonitorRegistry.get(job_id)
        if monitor:
            monitor.record_error(str(e))
            
        # Update job status on error
        RESEARCH_JOBS[job_id]['status'] = 'failed'
        RESEARCH_JOBS[job_id]['error'] = str(e)


# ----- Routes -----

@app.route('/')
def index():
    """Render the main page."""
    return render_template(
        'index.html',
        recent_reports=RECENT_REPORTS[:4],  # Show only 4 most recent
        active_tab='new_research'
    )


@app.route('/submit_research', methods=['POST'])
def submit_research():
    """Handle research submission."""
    # Get form data
    form_data = request.form.to_dict()
    
    # Create topic
    topic = _create_topic(form_data)
    job_id = topic.id
    
    # Get research parameters
    audience = form_data.get('audience', 'general')
    depth = form_data.get('depth', 'standard')
    report_format = form_data.get('format', 'report')
    time_constraint = int(form_data.get('time', 30))
    
    # Map depth to query counts
    depth_mapping = {
        'brief': 1,
        'standard': 2,
        'comprehensive': 3,
        'expert': 5
    }
    max_depth = depth_mapping.get(depth, 2)
    
    # Set up parameters for progress monitor
    parameters = {
        'audience': audience,
        'depth': depth,
        'format': report_format,
        'time_constraint': time_constraint
    }
    
    # Create job
    RESEARCH_JOBS[job_id] = {
        'topic': topic.dict(),
        'status': 'queued',
        'submit_time': datetime.now(),
        'parameters': parameters
    }
    
    # Create progress monitor
    create_progress_monitor(topic.title, parameters)
    
    # Start research in background
    run_async_task(_run_research(job_id, topic, max_depth))
    
    # Redirect to progress page using the new route
    return redirect(url_for('progress_page', job_id=job_id))


# Route redirecting from old progress URL to new one
@app.route('/research/<job_id>/progress')
def view_progress(job_id):
    """Redirect to the new progress page."""
    return redirect(url_for('progress_page', job_id=job_id))


# Redirect old status API to new progress API
@app.route('/research/<job_id>/status', methods=['GET'])
def job_status(job_id):
    """Redirects to the new progress API endpoint."""
    return redirect(url_for('get_progress', job_id=job_id))


@app.route('/research/<job_id>/results')
def view_results(job_id):
    """Show results for a completed research job."""
    if job_id not in RESEARCH_JOBS:
        return render_template('error.html', message="Research job not found"), 404
    
    if job_id not in RESEARCH_RESULTS:
        if RESEARCH_JOBS[job_id]['status'] == 'failed':
            return render_template(
                'error.html', 
                message=f"Research failed: {RESEARCH_JOBS[job_id].get('error', 'Unknown error')}"
            ), 500
        return redirect(url_for('view_progress', job_id=job_id))
    
    job = RESEARCH_JOBS[job_id]
    results = RESEARCH_RESULTS[job_id]
    
    return render_template(
        'results.html',
        job=job,
        results=results,
        active_tab='completed_reports'
    )


@app.route('/active_research')
def active_research():
    """Show all active research jobs."""
    # Get active jobs from both the legacy system and progress monitoring system
    legacy_active_jobs = {
        job_id: job for job_id, job in RESEARCH_JOBS.items()
        if job['status'] in ['queued', 'running']
    }
    
    # Get active jobs from ProgressMonitorRegistry
    progress_active_jobs = ProgressMonitorRegistry.list_active_jobs()
    
    # If we have progress monitoring jobs, use them instead
    if progress_active_jobs:
        return render_template(
            'active_research_new.html',
            active_jobs=progress_active_jobs,
            active_tab='active_research'
        )
    else:
        # Fall back to legacy template if no progress monitors
        return render_template(
            'active_research.html',
            jobs=legacy_active_jobs,
            active_tab='active_research'
        )


@app.route('/completed_reports')
def completed_reports():
    """Show all completed research reports."""
    completed_jobs = {
        job_id: job for job_id, job in RESEARCH_JOBS.items()
        if job['status'] == 'completed'
    }
    
    return render_template(
        'completed_reports.html',
        jobs=completed_jobs,
        reports=RECENT_REPORTS,
        active_tab='completed_reports'
    )


@app.route('/report_feedback', methods=['POST'])
def report_feedback():
    """Handle report feedback submission."""
    # In a real app, this would store feedback in a database
    job_id = request.form.get('job_id')
    rating = request.form.get('rating')
    comments = request.form.get('comments')
    
    # Just acknowledge receipt in this demo
    return jsonify({
        'status': 'success',
        'message': 'Feedback received. Thank you!'
    })


# ----- Main -----

def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
