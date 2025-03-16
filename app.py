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
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Import from research pipeline
from research_pipeline import (
    ResearchPipeline,
    ResearchTopic,
    SearchQuery
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_research_assistant')

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
        
        # Run the research
        stats = await pipeline.research_topic(
            topic,
            max_queries=max_depth,
            max_results_per_query=max_depth
        )
        
        # Get the research results
        results = await pipeline.get_research_results(topic.id)
        
        # Store results
        RESEARCH_RESULTS[job_id] = {
            'stats': stats,
            'results': results,
            'topic': topic.dict(),
            'completion_time': datetime.now()
        }
        
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
    
    # Create job
    job_id = topic.id
    RESEARCH_JOBS[job_id] = {
        'topic': topic.dict(),
        'status': 'queued',
        'submit_time': datetime.now(),
        'parameters': {
            'audience': audience,
            'depth': depth,
            'format': report_format,
            'time_constraint': time_constraint
        }
    }
    
    # Start research in background
    run_async_task(_run_research(job_id, topic, max_depth))
    
    # Redirect to progress page
    return redirect(url_for('view_progress', job_id=job_id))


@app.route('/research/<job_id>/progress')
def view_progress(job_id):
    """Show progress for a research job."""
    if job_id not in RESEARCH_JOBS:
        return render_template('error.html', message="Research job not found"), 404
    
    job = RESEARCH_JOBS[job_id]
    
    # If completed, redirect to results
    if job['status'] == 'completed':
        return redirect(url_for('view_results', job_id=job_id))
    
    # Calculate progress percentage based on status
    progress_mapping = {
        'queued': 5,
        'running': 50,
        'completed': 100,
        'failed': 100
    }
    progress = progress_mapping.get(job['status'], 0)
    
    # Calculate remaining time (mock calculation)
    remaining = "Unknown"
    if 'start_time' in job and job['status'] == 'running':
        elapsed = (datetime.now() - job['start_time']).total_seconds() / 60
        if elapsed < job['parameters']['time_constraint']:
            remaining = f"{int(job['parameters']['time_constraint'] - elapsed)} minutes"
    
    return render_template(
        'progress.html',
        job=job,
        progress=progress,
        remaining=remaining,
        active_tab='active_research'
    )


@app.route('/research/<job_id>/status', methods=['GET'])
def job_status(job_id):
    """API endpoint to get job status."""
    if job_id not in RESEARCH_JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = RESEARCH_JOBS[job_id]
    
    # Calculate progress
    progress_mapping = {
        'queued': 5,
        'running': 50,
        'completed': 100,
        'failed': 100
    }
    progress = progress_mapping.get(job['status'], 0)
    
    # Add detailed progress info for running jobs
    phase = "Planning"
    if job['status'] == 'running':
        phases = ["Planning", "Searching", "Analyzing", "Synthesizing", "Report Generation"]
        if progress <= 20:
            phase = phases[0]
        elif progress <= 40:
            phase = phases[1]
        elif progress <= 60:
            phase = phases[2]
        elif progress <= 80:
            phase = phases[3]
        else:
            phase = phases[4]
    
    # Calculate remaining time
    remaining = None
    if 'start_time' in job and job['status'] == 'running':
        elapsed = (datetime.now() - job['start_time']).total_seconds() / 60
        if elapsed < job['parameters']['time_constraint']:
            remaining = int(job['parameters']['time_constraint'] - elapsed)
    
    return jsonify({
        'status': job['status'],
        'progress': progress,
        'phase': phase,
        'remaining_minutes': remaining,
        'redirect_to_results': job['status'] == 'completed'
    })


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
    active_jobs = {
        job_id: job for job_id, job in RESEARCH_JOBS.items()
        if job['status'] in ['queued', 'running']
    }
    
    return render_template(
        'active_research.html',
        jobs=active_jobs,
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
