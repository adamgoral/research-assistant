"""
Progress API Module for Research Assistant

This module provides Flask routes for progress monitoring and status updates,
integrating the progress monitoring system with the web application.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List
from flask import Blueprint, jsonify, request, render_template, abort

from progress_monitor import ProgressMonitor, ResearchPhase, ProgressMonitorRegistry

# Configure logger
logger = logging.getLogger(__name__)

# Create Blueprint for progress API routes
progress_bp = Blueprint('progress', __name__)


@progress_bp.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id: str):
    """
    Get the current progress of a research job.
    
    Args:
        job_id: The ID of the research job
        
    Returns:
        JSON response with the current progress status
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(monitor.get_status())


@progress_bp.route('/api/progress/<job_id>/events', methods=['GET'])
def get_progress_events(job_id: str):
    """
    Get the event history for a research job.
    
    Args:
        job_id: The ID of the research job
        
    Returns:
        JSON response with the event history
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(monitor.get_event_history())


@progress_bp.route('/api/progress/active', methods=['GET'])
def list_active_jobs():
    """
    List all active research jobs.
    
    Returns:
        JSON response with active job information
    """
    active_jobs = ProgressMonitorRegistry.list_active_jobs()
    return jsonify({"active_jobs": active_jobs})


@progress_bp.route('/progress/<job_id>', methods=['GET'])
def progress_page(job_id: str):
    """
    Render the progress tracking page for a specific research job.
    
    Args:
        job_id: The ID of the research job
        
    Returns:
        Rendered progress template
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        # If job not found in memory, try to load from file (for completed jobs)
        try:
            import json
            import os
            progress_path = os.path.join("output", f"progress_{job_id}.json")
            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    job_data = json.load(f)
                return render_template(
                    'progress.html',
                    job_id=job_id,
                    topic=job_data.get('topic', "Unknown"),
                    progress=job_data.get('progress_percentage', 100),
                    phase=job_data.get('current_phase', 'completed'),
                    status_message=job_data.get('status_message', 'Completed'),
                    parameters=job_data.get('parameters', {}),
                    remaining="0 seconds",
                    is_completed=True
                )
        except Exception as e:
            logger.error(f"Error loading job from file: {e}")
            abort(404)
    
    status = monitor.get_status()
    return render_template(
        'progress.html',
        job_id=job_id,
        topic=status['topic'],
        progress=status['progress_percentage'],
        phase=status['current_phase'],
        status_message=status['status_message'],
        parameters=status['parameters'],
        remaining=status['formatted_remaining'],
        is_completed=status['is_completed']
    )


def create_progress_monitor(topic: str, parameters: Dict[str, Any]) -> Tuple[str, ProgressMonitor]:
    """
    Create a new progress monitor for a research job.
    
    Args:
        topic: The research topic
        parameters: Research parameters (audience, depth, format, time_constraint)
        
    Returns:
        Tuple of job_id and the created ProgressMonitor instance
    """
    job_id = str(uuid.uuid4())
    monitor = ProgressMonitor(job_id, topic, parameters)
    ProgressMonitorRegistry.register(monitor)
    logger.info(f"Created progress monitor for '{topic}' with job ID: {job_id}")
    return job_id, monitor


def get_progress_for_template(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get progress information formatted for template rendering.
    
    Args:
        job_id: The ID of the research job
        
    Returns:
        Dictionary with progress information or None if job not found
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        return None
    
    status = monitor.get_status()
    return {
        "job_id": job_id,
        "topic": status['topic'],
        "progress": status['progress_percentage'],
        "phase": status['current_phase'],
        "status_message": status['status_message'],
        "parameters": status['parameters'],
        "remaining": status['formatted_remaining'],
        "is_completed": status['is_completed'],
        "elapsed": status['formatted_elapsed']
    }


def update_research_phase(job_id: str, phase: ResearchPhase, message: str, 
                        details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update the phase of a research job.
    
    Args:
        job_id: The ID of the research job
        phase: The new research phase
        message: Status message for the phase update
        details: Optional details for the update
        
    Returns:
        True if update was successful, False if job not found
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        logger.warning(f"Attempted to update phase for unknown job: {job_id}")
        return False
    
    monitor.update_phase(phase, message, details)
    return True


def update_research_progress(job_id: str, percentage: float, message: str,
                           details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update the progress of a research job.
    
    Args:
        job_id: The ID of the research job
        percentage: New progress percentage (0-100)
        message: Status message for the progress update
        details: Optional details for the update
        
    Returns:
        True if update was successful, False if job not found
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        logger.warning(f"Attempted to update progress for unknown job: {job_id}")
        return False
    
    monitor.update_progress(percentage, message, details)
    return True


def record_research_error(job_id: str, error_message: str, 
                        details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Record an error for a research job.
    
    Args:
        job_id: The ID of the research job
        error_message: Error message to record
        details: Optional error details
        
    Returns:
        True if error was recorded, False if job not found
    """
    monitor = ProgressMonitorRegistry.get(job_id)
    if not monitor:
        logger.warning(f"Attempted to record error for unknown job: {job_id}")
        return False
    
    monitor.record_error(error_message, details)
    return True
