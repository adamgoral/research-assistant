"""
Planning Module for Web Research Assistant

This module implements the planning capabilities of the Web Research Assistant as
described in the system architecture. It provides functionality for creating research
plans, breaking down topics into tasks, reasoning about approaches, and coordinating
the research process.

The planning module consists of:
1. PlanningModule: Main coordinator class for planning operations
2. ResearchPlan: Data model for research plans
3. ResearchTask: Data model for individual research tasks
4. PlanReasoner: Component for reasoning about research approaches
5. TaskManager: Component for managing and prioritizing tasks

The planning module interfaces with:
- Memory System: For storing and retrieving plans, topics, etc.
- Research Pipeline: For executing the research tasks
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from memory_system import (
    MemoryItem,
    MemoryItemType,
    MemoryQuery,
    MemoryManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("planning_module")


# ----- JSON Serialization Helpers -----

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_for_json(obj: Any) -> Any:
    """
    Serialize objects for JSON, handling special cases like enums and datetimes.
    
    Args:
        obj: The object to serialize
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
        # Handle Pydantic models or anything with a dict() method
        return serialize_for_json(obj.dict())
    else:
        return obj


# ----- Data Models -----

class ResearchDepth(Enum):
    """Enum representing research depth levels."""
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"


class AudienceLevel(Enum):
    """Enum representing audience knowledge levels."""
    GENERAL = "general"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    TECHNICAL = "technical"


class TaskStatus(Enum):
    """Enum representing the status of a research task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Enum representing the priority of a research task."""
    CRITICAL = "critical"  # Must be done first
    HIGH = "high"          # Important for core research
    MEDIUM = "medium"      # Standard priority
    LOW = "low"            # Nice to have if time permits
    BACKGROUND = "background"  # Can be done in parallel


class TaskType(Enum):
    """Types of research tasks."""
    SEARCH = "search"  # Web search tasks
    CONTENT = "content"  # Content retrieval tasks
    EXTRACT = "extract"  # Information extraction tasks
    ANALYZE = "analyze"  # Analysis and synthesis tasks
    VERIFY = "verify"  # Fact-checking and verification tasks
    META = "meta"  # Meta-tasks like planning or evaluation


class ResearchParameters(BaseModel):
    """Model representing parameters for the research process."""
    depth: ResearchDepth = ResearchDepth.STANDARD
    audience: AudienceLevel = AudienceLevel.GENERAL
    time_limit_minutes: int = 15
    max_sources: int = 10
    min_sources: int = 3


class ResearchTask(BaseModel):
    """Model representing a single research task."""
    id: str
    type: TaskType
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)  # IDs of dependent tasks
    estimated_time_seconds: int = 60
    actual_time_seconds: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None


class ResearchPlan(BaseModel):
    """Model representing a research plan."""
    id: str
    topic_id: str
    topic_title: str
    topic_description: str
    parameters: ResearchParameters
    tasks: List[ResearchTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def pending_tasks(self) -> List[ResearchTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]
    
    @property
    def in_progress_tasks(self) -> List[ResearchTask]:
        """Get all in-progress tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS]
    
    @property
    def completed_tasks(self) -> List[ResearchTask]:
        """Get all completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    @property
    def failed_tasks(self) -> List[ResearchTask]:
        """Get all failed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]
    
    @property
    def blocked_tasks(self) -> List[ResearchTask]:
        """Get all blocked tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.BLOCKED]
    
    @property
    def completion_percentage(self) -> float:
        """Calculate the completion percentage of the plan."""
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.tasks) * 100
    
    @property
    def estimated_remaining_time_seconds(self) -> int:
        """Estimate the remaining time to complete the plan in seconds."""
        remaining = 0
        for task in self.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]:
                remaining += task.estimated_time_seconds
            elif task.status == TaskStatus.IN_PROGRESS and task.started_at:
                # Estimate remaining time based on elapsed time and total estimated time
                elapsed = (datetime.now() - task.started_at).total_seconds()
                remaining += max(0, task.estimated_time_seconds - elapsed)
        return remaining
    
    def get_next_tasks(self) -> List[ResearchTask]:
        """
        Get the next tasks that can be executed based on dependencies.
        
        Returns:
            A list of tasks that have all dependencies satisfied.
        """
        completed_ids = {task.id for task in self.completed_tasks}
        next_tasks = []
        
        for task in self.pending_tasks:
            # Check if all dependencies are completed
            if all(dep_id in completed_ids for dep_id in task.dependencies):
                next_tasks.append(task)
        
        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4
        }
        
        next_tasks.sort(key=lambda t: priority_order[t.priority])
        return next_tasks


# ----- Planning Components -----

class PlanReasoner:
    """
    Component responsible for reasoning about research approaches.
    
    In a production implementation, this would use GPT-4o to analyze
    the research topic and determine the best approach.
    """
    
    async def analyze_topic(
        self, 
        topic_title: str, 
        topic_description: str, 
        parameters: ResearchParameters
    ) -> Dict[str, Any]:
        """
        Analyze a research topic to determine the best approach.
        
        Args:
            topic_title: The title of the research topic
            topic_description: The description of the research topic
            parameters: The research parameters
            
        Returns:
            A dictionary containing analysis results
        """
        logger.info(f"Analyzing topic: {topic_title}")
        
        # In a real implementation, this would call GPT-4o
        # For this PoC, we'll generate a simple analysis
        
        # Extract potential subtopics (would be done by LLM in production)
        keywords = topic_description.lower().split()
        subtopics = []
        
        # Generate mock subtopics based on keywords
        if len(keywords) >= 3:
            for i in range(min(5, len(keywords) - 2)):
                subtopic = f"{keywords[i]} {keywords[i+1]} {keywords[i+2]}"
                subtopics.append(subtopic)
        
        # If we couldn't generate subtopics, create some defaults
        if not subtopics:
            subtopics = [
                f"Background of {topic_title}",
                f"Current status of {topic_title}",
                f"Future implications of {topic_title}",
                f"Controversies related to {topic_title}",
                f"Analysis of {topic_title}"
            ]
        
        # Adjust number of subtopics based on research depth
        if parameters.depth == ResearchDepth.BRIEF:
            subtopics = subtopics[:2]
        elif parameters.depth == ResearchDepth.STANDARD:
            subtopics = subtopics[:3]
        elif parameters.depth == ResearchDepth.COMPREHENSIVE:
            subtopics = subtopics[:4]
        
        # Generate potential search queries (would be done by LLM in production)
        search_queries = [
            topic_title,
            f"{topic_title} overview",
            f"{topic_title} analysis",
        ]
        
        for subtopic in subtopics:
            search_queries.append(subtopic)
        
        # Add more specific queries based on audience level
        if parameters.audience == AudienceLevel.ACADEMIC or parameters.audience == AudienceLevel.TECHNICAL:
            search_queries.append(f"{topic_title} research paper")
            search_queries.append(f"{topic_title} technical analysis")
        
        # Determine approach based on parameters
        approach = {
            "search_strategy": "broad_first" if parameters.depth == ResearchDepth.BRIEF else "depth_first",
            "verification_level": "basic" if parameters.depth == ResearchDepth.BRIEF else "thorough",
            "source_diversity": "low" if parameters.depth == ResearchDepth.BRIEF else "high",
        }
        
        # Estimate time allocation based on research depth and time limit
        total_minutes = parameters.time_limit_minutes
        time_allocation = {
            "search": int(total_minutes * 0.2),
            "content_retrieval": int(total_minutes * 0.3),
            "information_extraction": int(total_minutes * 0.3),
            "synthesis": int(total_minutes * 0.2),
        }
        
        analysis = {
            "subtopics": subtopics,
            "search_queries": search_queries,
            "approach": approach,
            "time_allocation": time_allocation,
            "estimated_sources_needed": max(parameters.min_sources, min(parameters.max_sources, 
                                        3 if parameters.depth == ResearchDepth.BRIEF else
                                        5 if parameters.depth == ResearchDepth.STANDARD else
                                        8 if parameters.depth == ResearchDepth.COMPREHENSIVE else 10))
        }
        
        logger.info(f"Completed analysis for topic: {topic_title}")
        return analysis


class TaskManager:
    """
    Component responsible for managing and prioritizing research tasks.
    
    The TaskManager:
    - Creates tasks based on the research plan
    - Prioritizes tasks for execution
    - Tracks task dependencies and statuses
    - Updates task metadata as execution progresses
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the task manager.
        
        Args:
            memory_manager: Memory manager for storing task data
        """
        self.memory_manager = memory_manager
    
    async def create_tasks_from_analysis(
        self, 
        topic_id: str,
        topic_title: str,
        analysis: Dict[str, Any],
        parameters: ResearchParameters
    ) -> List[ResearchTask]:
        """
        Create research tasks based on topic analysis.
        
        Args:
            topic_id: The ID of the research topic
            topic_title: The title of the research topic
            analysis: The analysis results from PlanReasoner
            parameters: The research parameters
            
        Returns:
            A list of ResearchTask objects
        """
        logger.info(f"Creating tasks for topic: {topic_title}")
        
        tasks = []
        task_ids = {}
        
        # Create planning meta-task
        planning_task = ResearchTask(
            id=self._generate_task_id("planning"),
            type=TaskType.META,
            description=f"Create research plan for: {topic_title}",
            priority=TaskPriority.CRITICAL,
            status=TaskStatus.COMPLETED,  # This task is completed by creating the plan
            estimated_time_seconds=60,
            actual_time_seconds=60,
            metadata={
                "topic_id": topic_id,
                "analysis": analysis
            },
            created_at=datetime.now(),
            started_at=datetime.now() - timedelta(seconds=60),
            completed_at=datetime.now()
        )
        
        tasks.append(planning_task)
        task_ids["planning"] = planning_task.id
        
        # Create search tasks based on search queries
        search_tasks = []
        for i, query in enumerate(analysis["search_queries"]):
            # Determine priority based on position
            priority = TaskPriority.HIGH if i < 3 else TaskPriority.MEDIUM
            
            search_task = ResearchTask(
                id=self._generate_task_id("search"),
                type=TaskType.SEARCH,
                description=f"Search for: {query}",
                priority=priority,
                dependencies=[task_ids["planning"]],
                estimated_time_seconds=30,
                metadata={
                    "topic_id": topic_id,
                    "search_query": query
                }
            )
            
            search_tasks.append(search_task)
            tasks.append(search_task)
            task_ids[f"search_{i}"] = search_task.id
        
        # Create content retrieval tasks (these depend on search tasks)
        # We don't know the exact URLs yet, so we'll create placeholder tasks
        content_tasks = []
        
        # Create a content task for each major subtopic
        for i, subtopic in enumerate(analysis["subtopics"]):
            content_task = ResearchTask(
                id=self._generate_task_id("content"),
                type=TaskType.CONTENT,
                description=f"Retrieve content about: {subtopic}",
                priority=TaskPriority.MEDIUM,
                dependencies=[task_ids[f"search_{min(i, len(search_tasks)-1)}"]],
                estimated_time_seconds=45,
                metadata={
                    "topic_id": topic_id,
                    "subtopic": subtopic,
                    "max_sources": 3
                }
            )
            
            content_tasks.append(content_task)
            tasks.append(content_task)
            task_ids[f"content_{i}"] = content_task.id
        
        # Create extraction tasks (these depend on content tasks)
        extraction_tasks = []
        for i, subtopic in enumerate(analysis["subtopics"]):
            extract_task = ResearchTask(
                id=self._generate_task_id("extract"),
                type=TaskType.EXTRACT,
                description=f"Extract information about: {subtopic}",
                priority=TaskPriority.MEDIUM,
                dependencies=[task_ids[f"content_{i}"]],
                estimated_time_seconds=60,
                metadata={
                    "topic_id": topic_id,
                    "subtopic": subtopic
                }
            )
            
            extraction_tasks.append(extract_task)
            tasks.append(extract_task)
            task_ids[f"extract_{i}"] = extract_task.id
        
        # Create analysis task (depends on all extraction tasks)
        analysis_task = ResearchTask(
            id=self._generate_task_id("analyze"),
            type=TaskType.ANALYZE,
            description=f"Synthesize information about: {topic_title}",
            priority=TaskPriority.HIGH,
            dependencies=[task.id for task in extraction_tasks],
            estimated_time_seconds=120,
            metadata={
                "topic_id": topic_id,
                "audience_level": parameters.audience.value
            }
        )
        
        tasks.append(analysis_task)
        task_ids["analysis"] = analysis_task.id
        
        # Create verification task if needed (depends on analysis task)
        if parameters.depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT]:
            verify_task = ResearchTask(
                id=self._generate_task_id("verify"),
                type=TaskType.VERIFY,
                description=f"Verify facts about: {topic_title}",
                priority=TaskPriority.HIGH,
                dependencies=[task_ids["analysis"]],
                estimated_time_seconds=90,
                metadata={
                    "topic_id": topic_id,
                    "verification_level": "thorough"
                }
            )
            
            tasks.append(verify_task)
            task_ids["verify"] = verify_task.id
        
        logger.info(f"Created {len(tasks)} tasks for topic: {topic_title}")
        return tasks
    
    async def update_task_status(
        self,
        plan: ResearchPlan,
        task_id: str,
        new_status: TaskStatus,
        result: Optional[Dict[str, Any]] = None
    ) -> ResearchPlan:
        """
        Update the status of a task within a research plan.
        
        Args:
            plan: The research plan containing the task
            task_id: The ID of the task to update
            new_status: The new status to set
            result: Optional result data from the task
            
        Returns:
            The updated research plan
        """
        # Find the task in the plan
        task_index = None
        for i, task in enumerate(plan.tasks):
            if task.id == task_id:
                task_index = i
                break
        
        if task_index is None:
            logger.warning(f"Task {task_id} not found in plan {plan.id}")
            return plan
        
        # Update the task
        task = plan.tasks[task_index]
        old_status = task.status
        task.status = new_status
        
        # Update timestamps based on status
        now = datetime.now()
        if new_status == TaskStatus.IN_PROGRESS and not task.started_at:
            task.started_at = now
        elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.completed_at = now
            if task.started_at:
                task.actual_time_seconds = int((now - task.started_at).total_seconds())
        
        # Update result if provided
        if result:
            task.result = result
        
        # Update the task in the plan
        plan.tasks[task_index] = task
        plan.updated_at = now
        
        # Update plan status based on task statuses
        if all(t.status == TaskStatus.COMPLETED for t in plan.tasks):
            plan.status = TaskStatus.COMPLETED
        elif any(t.status == TaskStatus.IN_PROGRESS for t in plan.tasks):
            plan.status = TaskStatus.IN_PROGRESS
        elif any(t.status == TaskStatus.FAILED for t in plan.tasks):
            # Keep plan in progress if some tasks failed but others are still in progress or pending
            if any(t.status in [TaskStatus.IN_PROGRESS, TaskStatus.PENDING] for t in plan.tasks):
                plan.status = TaskStatus.IN_PROGRESS
        
        logger.info(f"Updated task {task_id} status from {old_status.value} to {new_status.value}")
        
        # If using memory manager, save the updated plan
        if self.memory_manager:
            # Use our serialization helper to convert the plan to a JSON-serializable dict
            serialized_plan = serialize_for_json(plan)
            
            # Convert plan to memory item and save
            memory_item = MemoryItem(
                id=f"plan_{plan.id}",
                item_type=MemoryItemType.RESEARCH_TOPIC,
                content=serialized_plan,
                importance_score=0.9
            )
            
            await self.memory_manager.store(memory_item, ephemeral=True, persistent=True)
            logger.info(f"Saved updated plan {plan.id} to memory")
        
        return plan
    
    def prioritize_tasks(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """
        Prioritize a list of tasks for execution.
        
        Args:
            tasks: The list of tasks to prioritize
            
        Returns:
            The prioritized list of tasks
        """
        # Define priority order
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4
        }
        
        # Sort by priority first, then by estimated time (shorter tasks first)
        return sorted(tasks, key=lambda t: (priority_order[t.priority], t.estimated_time_seconds))
    
    def _generate_task_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID with a prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ----- Planning Module -----

class PlanningModule:
    """
    Main module for research planning functionality.
    
    The PlanningModule:
    - Creates research plans for topics
    - Coordinates the planning process
    - Interfaces with memory system for plan storage/retrieval
    - Provides plans to the research pipeline for execution
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the planning module.
        
        Args:
            memory_manager: Memory manager for storing planning data
        """
        self.memory_manager = memory_manager
        self.plan_reasoner = PlanReasoner()
        self.task_manager = TaskManager(memory_manager)
        logger.info("Initialized PlanningModule")
    
    async def create_research_plan(
        self,
        topic_id: str,
        topic_title: str,
        topic_description: str,
        parameters: Optional[ResearchParameters] = None
    ) -> ResearchPlan:
        """
        Create a research plan for a given topic.
        
        Args:
            topic_id: The ID of the research topic
            topic_title: The title of the research topic
            topic_description: The description of the research topic
            parameters: Optional research parameters (uses defaults if not provided)
            
        Returns:
            A ResearchPlan object
        """
        logger.info(f"Creating research plan for topic: {topic_title}")
        
        # Use default parameters if not provided
        if not parameters:
            parameters = ResearchParameters()
        
        # Analyze the topic to determine the best research approach
        analysis = await self.plan_reasoner.analyze_topic(
            topic_title,
            topic_description,
            parameters
        )
        
        # Create tasks based on the analysis
        tasks = await self.task_manager.create_tasks_from_analysis(
            topic_id,
            topic_title,
            analysis,
            parameters
        )
        
        # Create the research plan
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        plan = ResearchPlan(
            id=plan_id,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_description=topic_description,
            parameters=parameters,
            tasks=tasks,
            status=TaskStatus.PENDING,
            metadata={
                "analysis": analysis,
                "creation_time": datetime.now().isoformat(),
                "estimated_completion_time": (
                    datetime.now() + timedelta(minutes=parameters.time_limit_minutes)
                ).isoformat()
            }
        )
        
        # Store the plan in memory if memory manager is available
        if self.memory_manager:
            # Use our serialization helper to convert the plan to a JSON-serializable dict
            serialized_plan = serialize_for_json(plan)
            
            memory_item = MemoryItem(
                id=f"plan_{plan_id}",
                item_type=MemoryItemType.RESEARCH_TOPIC,
                content=serialized_plan,
                importance_score=0.9
            )
            
            await self.memory_manager.store(memory_item, ephemeral=True, persistent=True)
            logger.info(f"Stored research plan {plan_id} in memory")
        
        logger.info(f"Created research plan {plan_id} with {len(tasks)} tasks")
        return plan
    
    async def get_research_plan(self, plan_id: str) -> Optional[ResearchPlan]:
        """
        Retrieve a research plan from memory.
        
        Args:
            plan_id: The ID of the plan to retrieve
            
        Returns:
            The ResearchPlan if found, None otherwise
        """
        if not self.memory_manager:
            logger.warning("Cannot retrieve plan: no memory manager available")
            return None
        
        memory_id = f"plan_{plan_id}"
        memory_item = await self.memory_manager.retrieve(memory_id)
        
        if not memory_item:
            logger.warning(f"Research plan {plan_id} not found in memory")
            return None
        
        plan_dict = memory_item.content
        return ResearchPlan(**plan_dict)
    
    async def update_research_plan(self, plan: ResearchPlan) -> bool:
        """
        Update a research plan in memory.
        
        Args:
            plan: The updated research plan
            
        Returns:
            True if successful, False otherwise
        """
        if not self.memory_manager:
            logger.warning("Cannot update plan: no memory manager available")
            return False
        
        memory_id = f"plan_{plan.id}"
        plan.updated_at = datetime.now()
        
        # Use our serialization helper to convert the plan to a JSON-serializable dict
        serialized_plan = serialize_for_json(plan)
        
        # Create memory item
        memory_item = MemoryItem(
            id=memory_id,
            item_type=MemoryItemType.RESEARCH_TOPIC,
            content=serialized_plan,
            importance_score=0.9
        )
        
        # Update in memory
        await self.memory_manager.store(memory_item, ephemeral=True, persistent=True)
        logger.info(f"Updated research plan {plan.id} in memory")
        return True
    
    async def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        new_status: TaskStatus,
        result: Optional[Dict[str, Any]] = None
    ) -> Optional[ResearchPlan]:
        """
        Update the status of a task within a research plan.
        
        Args:
            plan_id: The ID of the research plan
            task_id: The ID of the task to update
            new_status: The new status to set
            result: Optional result data from the task
            
        Returns:
            The updated ResearchPlan if successful, None otherwise
        """
        # Retrieve the plan
        plan = await self.get_research_plan(plan_id)
        if not plan:
            logger.warning(f"Cannot update task: plan {plan_id} not found")
            return None
        
        # Update the task status
        updated_plan = await self.task_manager.update_task_status(
            plan, task_id, new_status, result
        )
        
        # Save the updated plan
        success = await self.update_research_plan(updated_plan)
        if not success:
            logger.warning(f"Failed to save updated plan {plan_id}")
            return None
        
        return updated_plan
    
    async def get_next_tasks(self, plan_id: str, max_tasks: int = 3) -> List[ResearchTask]:
        """
        Get the next tasks to execute for a research plan.
        
        Args:
            plan_id: The ID of the research plan
            max_tasks: Maximum number of tasks to return
            
        Returns:
            A list of tasks that can be executed next
        """
        # Retrieve the plan
        plan = await self.get_research_plan(plan_id)
        if not plan:
            logger.warning(f"Cannot get next tasks: plan {plan_id} not found")
            return []
        
        # Get the next executable tasks
        next_tasks = plan.get_next_tasks()
        
        # Prioritize tasks
        prioritized_tasks = self.task_manager.prioritize_tasks(next_tasks)
        
        # Limit to max_tasks
        return prioritized_tasks[:max_tasks]
    
    async def adapt_plan(
        self,
        plan_id: str,
        feedback: Dict[str, Any]
    ) -> Optional[ResearchPlan]:
        """
        Adapt a research plan based on feedback.
        
        Args:
            plan_id: The ID of the research plan
            feedback: Feedback data to guide adaptation
            
        Returns:
            The updated ResearchPlan if successful, None otherwise
        """
        # Retrieve the plan
        plan = await self.get_research_plan(plan_id)
        if not plan:
            logger.warning(f"Cannot adapt plan: plan {plan_id} not found")
            return None
        
        # In a real implementation, this would analyze the feedback and adapt the plan
        # For this PoC, we'll just add the feedback to the plan metadata
        
        logger.info(f"Adapting plan {plan_id} based on feedback")
        
        # Update plan metadata
        if "feedback_history" not in plan.metadata:
            plan.metadata["feedback_history"] = []
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        
        plan.metadata["feedback_history"].append(feedback_entry)
        plan.metadata["last_feedback"] = feedback
        
        # Example adaptation: if feedback mentions time constraint, adjust task estimations
        if "time_constraint" in feedback:
            time_factor = feedback.get("time_factor", 0.8)  # Default to 20% reduction
            
            # Adjust remaining tasks
            for task in plan.tasks:
                if task.status == TaskStatus.PENDING:
                    task.estimated_time_seconds = int(task.estimated_time_seconds * time_factor)
        
        # Save the updated plan
        success = await self.update_research_plan(plan)
        if not success:
            logger.warning(f"Failed to save adapted plan {plan_id}")
            return None
        
        return plan


# ----- Demo Function -----

async def demo_planning_module():
    """Demonstrate the planning module functionality."""
    from memory_system import MemoryManager, EphemeralMemory, FileBasedPersistentMemory
    
    print("\n=== Planning Module Demo ===\n")
    
    # Create memory manager for the demo
    memory_manager = MemoryManager(
        ephemeral_memory=EphemeralMemory(capacity=100),
        persistent_memory=FileBasedPersistentMemory(storage_dir="data/planning_module_test"),
        auto_persist_threshold=0.8
    )
    
    # Initialize planning module
    planning_module = PlanningModule(memory_manager)
    
    # Create a sample research topic
    topic_id = "topic_test"
    topic_title = "Artificial Intelligence in Healthcare"
    topic_description = ("Research the applications of artificial intelligence in healthcare, "
                        "including diagnostics, treatment planning, drug discovery, and "
                        "administrative workflow optimization.")
    
    # Create research parameters
    parameters = ResearchParameters(
        depth=ResearchDepth.STANDARD,
        audience=AudienceLevel.PROFESSIONAL,
        time_limit_minutes=10,
        max_sources=5,
        min_sources=2
    )
    
    print(f"Creating research plan for topic: {topic_title}")
    print(f"Research depth: {parameters.depth.value}")
    print(f"Audience level: {parameters.audience.value}")
    print(f"Time limit: {parameters.time_limit_minutes} minutes")
    print("-" * 50)
    
    # Create a research plan
    plan = await planning_module.create_research_plan(
        topic_id=topic_id,
        topic_title=topic_title,
        topic_description=topic_description,
        parameters=parameters
    )
    
    print(f"\nCreated research plan: {plan.id}")
    print(f"Total tasks: {len(plan.tasks)}")
    
    # Display task breakdown
    print("\nTask breakdown by type:")
    task_types = {}
    for task in plan.tasks:
        task_type = task.type.value
        if task_type not in task_types:
            task_types[task_type] = 0
        task_types[task_type] += 1
    
    for task_type, count in task_types.items():
        print(f"- {task_type}: {count} tasks")
    
    # Display some tasks
    print("\nSample tasks:")
    for i, task in enumerate(plan.tasks[:5]):  # Show first 5 tasks
        print(f"{i+1}. [{task.priority.value}] {task.type.value}: {task.description}")
    
    print("\nDemo completed!")


async def main():
    """Run the planning module demo."""
    print("=" * 50)
    print("PLANNING MODULE DEMO")
    print("=" * 50)
    
    await demo_planning_module()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
