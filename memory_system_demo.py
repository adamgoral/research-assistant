"""
Memory System Demo

This script demonstrates the usage of the memory system for the Web Research Assistant.
It shows how to create, store, retrieve, and query memory items using both
ephemeral and persistent memory systems.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta

from memory_system import (
    MemoryItem,
    MemoryItemType,
    MemoryQuery,
    EphemeralMemory,
    FileBasedPersistentMemory,
    MemoryManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memory_demo")


def generate_id(prefix: str = "item") -> str:
    """Generate a unique ID with a prefix"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def demo_ephemeral_memory():
    """Demonstrate EphemeralMemory functionality"""
    print("\n=== Ephemeral Memory Demo ===\n")
    
    # Create ephemeral memory with small capacity for demo purposes
    memory = EphemeralMemory(capacity=10, decay_factor=0.1)
    
    # Store some items
    print("Storing items in ephemeral memory...")
    items = []
    for i in range(5):
        item = MemoryItem(
            id=generate_id("ephemeral"),
            item_type=MemoryItemType.EXTRACTED_INFO,
            content={
                "text": f"Important information #{i+1}",
                "relevance": 0.9 - (i * 0.1)
            },
            importance_score=0.9 - (i * 0.1)
        )
        items.append(item)
        await memory.store(item)
    
    # Show memory stats
    stats = await memory.get_stats()
    print(f"Memory contains {stats['count']} items")
    print(f"Memory utilization: {stats['utilization']:.1%}")
    
    # Query items
    print("\nQuerying for all items...")
    query = MemoryQuery(
        item_type=MemoryItemType.EXTRACTED_INFO,
        limit=10
    )
    results = await memory.query(query)
    
    print(f"Found {len(results)} items:")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['text']}")
    
    # Demonstrate temporal decay
    print("\nApplying temporal decay...")
    # First, update creation time of an item to make it appear older
    old_item = items[0]
    old_time = datetime.now() - timedelta(hours=10)
    await memory.update(old_item.id, {"created_at": old_time})
    
    # Apply decay to all items
    await memory.apply_temporal_decay()
    
    # Show items after decay
    print("\nItems after decay:")
    results = await memory.query(query)
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['text']}")
    
    # Demonstrate capacity limits by adding more items
    print("\nAdding more items to demonstrate capacity limits...")
    for i in range(5, 15):
        item = MemoryItem(
            id=generate_id("ephemeral"),
            item_type=MemoryItemType.EXTRACTED_INFO,
            content={
                "text": f"Additional information #{i+1}",
                "relevance": 0.5 - (i * 0.02)
            },
            importance_score=0.5 - (i * 0.02)  # Lower importance
        )
        await memory.store(item)
    
    # Check stats again
    stats = await memory.get_stats()
    print(f"Memory now contains {stats['count']} items (capacity: {stats['capacity']})")
    
    # Query again to see which items remain (should be the most important ones)
    results = await memory.query(query)
    print("\nRemaining items (after eviction):")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['text']}")


async def demo_persistent_memory():
    """Demonstrate FileBasedPersistentMemory functionality"""
    print("\n=== Persistent Memory Demo ===\n")
    
    # Create persistent memory
    memory = FileBasedPersistentMemory(storage_dir="data/memory_demo")
    
    # Clear any existing data for demo purposes
    await memory.clear()
    
    # Store some items
    print("Storing items in persistent memory...")
    stored_ids = []
    for i in range(5):
        item = MemoryItem(
            id=generate_id("persistent"),
            item_type=MemoryItemType.RESEARCH_FINDING,
            content={
                "finding": f"Research finding #{i+1}",
                "confidence": 0.9 - (i * 0.1),
                "keywords": ["research", "finding", f"topic{i}"]
            },
            importance_score=0.9 - (i * 0.1)
        )
        item_id = await memory.store(item)
        stored_ids.append(item_id)
        print(f"Stored item {item_id}")
    
    # Get memory stats
    stats = await memory.get_stats()
    print(f"Persistent memory contains {stats['count']} items")
    
    # Retrieve a specific item
    if stored_ids:
        print(f"\nRetrieving item {stored_ids[0]}...")
        item = await memory.retrieve(stored_ids[0])
        if item:
            print(f"Retrieved: {item.content['finding']} (Importance: {item.importance_score:.2f})")
        else:
            print("Item not found")
    
    # Query items
    print("\nQuerying for all research findings...")
    query = MemoryQuery(
        item_type=MemoryItemType.RESEARCH_FINDING,
        limit=10
    )
    results = await memory.query(query)
    
    print(f"Found {len(results)} items:")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['finding']}")
    
    # Update an item
    if stored_ids:
        print(f"\nUpdating item {stored_ids[0]}...")
        updated = await memory.update(
            stored_ids[0], 
            {
                "importance_score": 1.0,
                "content": {
                    "finding": "UPDATED: Very important research finding",
                    "confidence": 1.0,
                    "keywords": ["research", "finding", "important"]
                }
            }
        )
        
        if updated:
            item = await memory.retrieve(stored_ids[0])
            print(f"Updated item: {item.content['finding']} (Importance: {item.importance_score:.2f})")
        else:
            print("Update failed")
    
    # Query with semantic search (simple keyword matching in this implementation)
    print("\nPerforming semantic search for 'important'...")
    query = MemoryQuery(
        item_type=MemoryItemType.RESEARCH_FINDING,
        semantic_query="important",
        limit=10
    )
    results = await memory.query(query)
    
    print(f"Found {len(results)} items:")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['finding']}")


async def demo_memory_manager():
    """Demonstrate MemoryManager functionality"""
    print("\n=== Memory Manager Demo ===\n")
    
    # Create memory manager
    memory_manager = MemoryManager(
        ephemeral_memory=EphemeralMemory(capacity=20),
        persistent_memory=FileBasedPersistentMemory(storage_dir="data/memory_manager_demo"),
        auto_persist_threshold=0.8
    )
    
    # Clear existing data for demo purposes
    await memory_manager.clear(ephemeral=True, persistent=True)
    
    # Create and store items with varying importance
    print("Storing items with varying importance levels...")
    
    # Create 20 items with descending importance
    items = []
    for i in range(20):
        importance = max(0.1, 1.0 - (i * 0.05))
        item = MemoryItem(
            id=generate_id("manager"),
            item_type=MemoryItemType.RESEARCH_FINDING,
            content={
                "finding": f"Research finding with importance {importance:.2f}",
                "keywords": ["research", "importance", f"level{i}"]
            },
            importance_score=importance
        )
        items.append(item)
        
        # Only store in ephemeral memory initially
        await memory_manager.store(item, ephemeral=True, persistent=False)
        
        if importance >= memory_manager.auto_persist_threshold:
            print(f"Item {i+1}: importance {importance:.2f} (should be auto-persisted)")
        else:
            print(f"Item {i+1}: importance {importance:.2f}")
    
    # Get memory stats
    stats = await memory_manager.get_stats()
    print(f"\nEphemeral memory contains {stats['ephemeral']['count']} items")
    print(f"Persistent memory contains {stats['persistent']['count']} items")
    print(f"Auto-persist threshold: {stats['auto_persist_threshold']}")
    
    # Perform maintenance (which should auto-persist important items)
    print("\nPerforming memory maintenance...")
    maintenance_stats = await memory_manager.maintain()
    print(f"Decayed {maintenance_stats['decayed_items']} items in ephemeral memory")
    print(f"Auto-persisted {maintenance_stats['persisted_items']} important items")
    
    # Get memory stats again
    stats = await memory_manager.get_stats()
    print(f"\nAfter maintenance:")
    print(f"Ephemeral memory contains {stats['ephemeral']['count']} items")
    print(f"Persistent memory contains {stats['persistent']['count']} items")
    
    # Query both memories using the manager
    print("\nQuerying for all research findings across both memories...")
    query = MemoryQuery(
        item_type=MemoryItemType.RESEARCH_FINDING,
        limit=10,
        sort_by="importance"
    )
    results = await memory_manager.query(query)
    
    print(f"Found {len(results)} items (top 10 by importance):")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['finding']}")
    
    # Query from persistent memory only
    print("\nQuerying from persistent memory only...")
    results = await memory_manager.query(query, ephemeral=False, persistent=True)
    
    print(f"Found {len(results)} items in persistent memory:")
    for item in results:
        print(f"- [{item.importance_score:.2f}] {item.content['finding']}")


async def main():
    """Run all demos"""
    print("=" * 50)
    print("MEMORY SYSTEM DEMO")
    print("=" * 50)
    
    # Demo each component
    await demo_ephemeral_memory()
    await demo_persistent_memory()
    await demo_memory_manager()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
