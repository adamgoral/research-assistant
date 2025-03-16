"""
Progress Monitoring Module for Research Assistant

This module provides tracking and reporting capabilities for research tasks,
following the Observer pattern to monitor state changes across components.
"""

import time
import logging
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ResearchPhase(Enum):
    """Enum representing the different phases of the research process."""
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    GENERATING = "report_generation"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressEvent:
    """Represents a significant event in the research process."""
    
    def __init__(self, 
                 phase: ResearchPhase, 
                 description: str, 
                 timestamp: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.phase = phase
        self.description = description
        self.timestamp = timestamp or time.time()
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "description": self.description,
            "timestamp": self.timestamp,
            "details": self.details
        }


class ProgressMonitor:
    """
    Monitors and tracks progress of research tasks.
    
    Implements the Observer pattern to monitor research pipeline components
    and generate user-facing updates.
    """
    
    def __init__(self, job_id: str, topic: str, parameters: Dict[str, Any]):
        self.job_id = job_id
        self.topic = topic
        self.parameters = parameters
        self.start_time = time.time()
        self.estimated_completion_time = self._calculate_estimated_time()
        
        # Current state
        self.current_phase = ResearchPhase.PLANNING
        self.progress_percentage = 0
        self.status_message = "Initializing research task"
        self.is_completed = False
        self.error = None
        
        # History tracking
        self.events: List[ProgressEvent] = []
        self.phase_timestamps: Dict[ResearchPhase, float] = {}
        
        # Observers (callbacks to notify of progress updates)
        self.observers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Record start event
        self.record_event(
            ResearchPhase.PLANNING, 
            f"Started research on '{topic}'",
            {"parameters": parameters}
        )
    
    def _calculate_estimated_time(self) -> float:
        """Calculate estimated completion time based on parameters."""
        # Default to 15 minutes if not specified
        base_time = self.parameters.get("time_constraint", 15) * 60
        
        # Adjust based on depth
        depth_multipliers = {
            "brief": 0.7,
            "standard": 1.0,
            "comprehensive": 1.3,
            "expert": 1.5
        }
        depth = self.parameters.get("depth", "standard")
        depth_multiplier = depth_multipliers.get(depth, 1.0)
        
        return self.start_time + (base_time * depth_multiplier)
    
    def register_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register an observer to be notified of progress updates."""
        self.observers.append(callback)
    
    def unregister_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister an observer."""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def notify_observers(self) -> None:
        """Notify all observers of the current progress state."""
        state = self.get_status()
        for observer in self.observers:
            try:
                observer(state)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")
    
    def update_phase(self, phase: ResearchPhase, message: str, 
                    details: Optional[Dict[str, Any]] = None) -> None:
        """Update the current phase of research."""
        self.current_phase = phase
        self.status_message = message
        self.phase_timestamps[phase] = time.time()
        
        # Calculate progress percentage based on phase
        phase_weights = {
            ResearchPhase.PLANNING: 20,
            ResearchPhase.SEARCHING: 40,
            ResearchPhase.ANALYZING: 60,
            ResearchPhase.SYNTHESIZING: 80,
            ResearchPhase.GENERATING: 90,
            ResearchPhase.COMPLETED: 100,
            ResearchPhase.ERROR: self.progress_percentage  # Maintain current percentage on error
        }
        
        if phase in phase_weights:
            self.progress_percentage = phase_weights[phase]
        
        # Record event
        self.record_event(phase, message, details)
        
        # Mark as completed if applicable
        if phase == ResearchPhase.COMPLETED:
            self.is_completed = True
        
        # Notify observers
        self.notify_observers()
    
    def update_progress(self, percentage: float, message: str,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Update progress within the current phase."""
        # Ensure percentage is between 0 and 100
        self.progress_percentage = max(0, min(100, percentage))
        self.status_message = message
        
        # Record event if details provided or significant progress change
        if details:
            self.record_event(self.current_phase, message, details)
            
        # Notify observers
        self.notify_observers()
    
    def record_event(self, phase: ResearchPhase, description: str,
                    details: Optional[Dict[str, Any]] = None) -> None:
        """Record a significant event in the research process."""
        event = ProgressEvent(phase, description, time.time(), details)
        self.events.append(event)
        logger.info(f"Research event: {event.description} (Phase: {event.phase.value})")
    
    def record_error(self, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record an error in the research process."""
        self.error = error_message
        self.update_phase(ResearchPhase.ERROR, f"Error: {error_message}", details)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the research task."""
        current_time = time.time()
        elapsed_seconds = current_time - self.start_time
        
        remaining_seconds = max(0, self.estimated_completion_time - current_time)
        
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "parameters": self.parameters,
            "current_phase": self.current_phase.value,
            "progress_percentage": self.progress_percentage,
            "status_message": self.status_message,
            "is_completed": self.is_completed,
            "error": self.error,
            "start_time": self.start_time,
            "elapsed_time": elapsed_seconds,
            "estimated_completion_time": self.estimated_completion_time,
            "remaining_time": remaining_seconds,
            "formatted_elapsed": self._format_time(elapsed_seconds),
            "formatted_remaining": self._format_time(remaining_seconds)
        }
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get the history of events for this research task."""
        return [event.to_dict() for event in self.events]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the progress monitor to a dictionary for serialization."""
        status = self.get_status()
        status["events"] = self.get_event_history()
        return status
    
    def save_to_file(self, filepath: str) -> None:
        """Save the current state to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to a human-readable string."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        
        minutes, seconds = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {seconds}s"


class ProgressMonitorRegistry:
    """
    Registry to manage and access all active progress monitors.
    
    This allows global access to progress monitors by job ID.
    """
    
    _monitors: Dict[str, ProgressMonitor] = {}
    
    @classmethod
    def register(cls, monitor: ProgressMonitor) -> None:
        """Register a new progress monitor."""
        cls._monitors[monitor.job_id] = monitor
    
    @classmethod
    def get(cls, job_id: str) -> Optional[ProgressMonitor]:
        """Get a progress monitor by job ID."""
        return cls._monitors.get(job_id)
    
    @classmethod
    def unregister(cls, job_id: str) -> None:
        """Remove a progress monitor from the registry."""
        if job_id in cls._monitors:
            del cls._monitors[job_id]
    
    @classmethod
    def list_active_jobs(cls) -> List[Dict[str, Any]]:
        """List all active research jobs."""
        return [
            {
                "job_id": job_id,
                "topic": monitor.topic,
                "progress": monitor.progress_percentage,
                "phase": monitor.current_phase.value,
                "start_time": monitor.start_time
            }
            for job_id, monitor in cls._monitors.items()
        ]
