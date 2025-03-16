"""
Planning Module Demo

This script demonstrates the functionality of the planning module for the
Web Research Assistant. It shows how to:
1. Initialize the planning module
2. Create research plans
3. Update task statuses
4. Adapt plans based on feedback
"""

import asyncio
import json
from datetime import datetime

from memory_system import (
    EphemeralMemory,
    FileBasedPersistentMemory,
    MemoryManager
)

from planning_module import (
    PlanningModule,
    ResearchDepth,
    AudienceLevel,
    TaskStatus,
    ResearchParameters
)


async def demo_planning_module():
    """
    Run a demonstration of the planning module functionality.
    """
    print("\n=== Planning Module Demo ===\n")
    
    # Create a memory system for storing plans
    memory_manager = MemoryManager(
        ephemeral_memory=EphemeralMemory(capacity=100),
        persistent_memory=FileBasedPersistentMemory(storage_dir="data/planning_demo"),
        auto_persist_threshold=0.8
    )
    
    # Initialize the planning module
    planning_module = PlanningModule(memory_manager)
    
    # Define a research topic
    topic_id = "climate_agriculture"
    topic_title = "Climate change impacts on agriculture"
    topic_description = ("Research the effects of climate change on agricultural systems, "
                         "including crop yields, farming practices, adaptation strategies, "
                         "and food security implications.")
    
    # Define research parameters
    parameters = ResearchParameters(
        depth=ResearchDepth.COMPREHENSIVE,
        audience=AudienceLevel.PROFESSIONAL,
        time_limit_minutes=15,
        max_sources=8,
        min_sources=4
    )
    
    # Display plan creation parameters
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
    
    # Display plan details
    print(f"\nCreated research plan: {plan.id}")
    print(f"Total tasks: {len(plan.tasks)}")
    print(f"Estimated completion time: {plan.metadata['estimated_completion_time']}")
    
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
    
    # Display some sample tasks
    print("\nSample tasks:")
    for i, task in enumerate(plan.tasks[:5]):  # Show first 5 tasks
        print(f"{i+1}. [{task.priority.value}] {task.type.value}: {task.description}")
    print(f"... and {len(plan.tasks) - 5} more tasks")
    
    # Get the next tasks to execute
    next_tasks = await planning_module.get_next_tasks(plan.id, max_tasks=7)
    print("\nNext executable tasks:")
    for i, task in enumerate(next_tasks):
        print(f"{i+1}. [{task.priority.value}] {task.description}")
    
    # Simulate task execution
    print("\nSimulating task execution...")
    search_task = next_tasks[0]  # Get the first search task
    print(f"Executing task: {search_task.description}")
    
    # Update the task status to in-progress
    updated_plan = await planning_module.update_task_status(
        plan.id, 
        search_task.id, 
        TaskStatus.IN_PROGRESS
    )
    
    # Simulate some work (1 second)
    await asyncio.sleep(1)
    
    # Complete the task
    updated_plan = await planning_module.update_task_status(
        plan.id,
        search_task.id,
        TaskStatus.COMPLETED,
        result={
            "search_results": [
                {"title": "Effects of Climate Change on Agriculture", "url": "https://example.com/climate1"},
                {"title": "Agricultural Adaptation to Climate Change", "url": "https://example.com/climate2"}
            ],
            "success": True
        }
    )
    
    print("Task completed successfully")
    
    # Display plan progress
    print(f"\nPlan progress: {updated_plan.completion_percentage:.1f}%")
    print(f"Completed tasks: {len(updated_plan.completed_tasks)}")
    print(f"Pending tasks: {len(updated_plan.pending_tasks)}")
    print(f"In-progress tasks: {len(updated_plan.in_progress_tasks)}")
    
    # Get the next tasks again after the first task is completed
    next_tasks = await planning_module.get_next_tasks(plan.id, max_tasks=3)
    print("\nNext executable tasks after first completion:")
    for i, task in enumerate(next_tasks):
        print(f"{i+1}. [{task.priority.value}] {task.description}")
    
    # Simulate user feedback
    print("\nSimulating user feedback and plan adaptation...")
    adapted_plan = await planning_module.adapt_plan(
        plan.id,
        feedback={
            "message": "Need to complete faster and focus more on practical adaptation strategies",
            "time_constraint": True,
            "time_factor": 0.7,
            "focus_areas": ["adaptation_strategies", "practical_examples"]
        }
    )
    
    print("Plan adapted successfully based on feedback")
    print(f"Feedback message: 'Need to complete faster and focus more on practical adaptation strategies'")
    print(f"Original estimated completion time: {plan.metadata['estimated_completion_time']}")
    print(f"Remaining time after adaptation: {(adapted_plan.estimated_remaining_time_seconds / 60):.1f} minutes")
    
    print("\nDemo completed!")


async def main():
    """Run the demonstration."""
    print("=" * 50)
    print("PLANNING MODULE DEMO")
    print("=" * 50)
    
    await demo_planning_module()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
