"""
Memory System for Web Research Assistant

This module implements the memory system architecture as described in the system design.
It provides both ephemeral (short-term) and persistent (long-term) memory capabilities
for the research assistant.

The memory system consists of:
1. MemoryManager: Orchestrates memory operations across different memory types
2. EphemeralMemory: Short-term storage for the current research task
3. PersistentMemory: Long-term storage for knowledge that can be reused
4. Memory: Base class for memory implementations

The memory system supports:
- Storage of structured data with metadata
- Vector-based semantic search
- Temporal decay for ephemeral memory
- Persistence across sessions for long-term memory
- Efficient query capabilities for different use cases
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memory_system")


# ----- Memory Item Models -----

class MemoryItemType(Enum):
    """Types of memory items"""
    RESEARCH_TOPIC = "research_topic"
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    WEB_CONTENT = "web_content"
    EXTRACTED_INFO = "extracted_info"
    SOURCE_EVALUATION = "source_evaluation"
    RESEARCH_FINDING = "research_finding"
    USER_FEEDBACK = "user_feedback"


class MemoryItem(BaseModel):
    """Base model for all memory items"""
    id: str
    item_type: MemoryItemType
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.0


class MemoryQuery(BaseModel):
    """Model for querying memory systems"""
    item_type: Optional[MemoryItemType] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    vector_query: Optional[List[float]] = None
    semantic_query: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    limit: int = 10
    offset: int = 0
    sort_by: str = "relevance"  # Options: relevance, recency, importance, access_count


# ----- Memory Interface -----

class Memory(ABC):
    """Abstract base class for all memory implementations"""
    
    @abstractmethod
    async def store(self, item: MemoryItem) -> str:
        """
        Store an item in memory
        
        Args:
            item: The memory item to store
            
        Returns:
            The ID of the stored item
        """
        pass
    
    @abstractmethod
    async def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve an item from memory by ID
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def query(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Query items from memory based on query parameters
        
        Args:
            query: The query parameters
            
        Returns:
            A list of memory items matching the query
        """
        pass
    
    @abstractmethod
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an item in memory
        
        Args:
            item_id: The ID of the item to update
            updates: The fields to update
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """
        Delete an item from memory
        
        Args:
            item_id: The ID of the item to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all items from memory
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            A dictionary of statistics
        """
        pass


# ----- Ephemeral Memory Implementation -----

class EphemeralMemory(Memory):
    """
    In-memory storage for short-term memory items
    
    Characteristics:
    - Fast access and modification
    - Items decay over time (temporal decay)
    - Limited capacity with least important items evicted first
    - Not persisted across sessions
    """
    
    def __init__(self, capacity: int = 1000, decay_factor: float = 0.05):
        """
        Initialize ephemeral memory
        
        Args:
            capacity: Maximum number of items to store
            decay_factor: Factor by which importance decays over time
        """
        self.items: Dict[str, MemoryItem] = {}
        self.capacity = capacity
        self.decay_factor = decay_factor
        logger.info(f"Initialized EphemeralMemory with capacity {capacity}")
    
    async def store(self, item: MemoryItem) -> str:
        """Store an item in ephemeral memory"""
        # Check if at capacity and evict if necessary
        if len(self.items) >= self.capacity and item.id not in self.items:
            await self._evict_items(1)
        
        # Store the item
        self.items[item.id] = item
        logger.info(f"Stored item {item.id} of type {item.item_type.value} in ephemeral memory")
        return item.id
    
    async def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from ephemeral memory"""
        item = self.items.get(item_id)
        if item:
            # Update access metadata
            now = datetime.now()
            item.last_accessed = now
            item.access_count += 1
            logger.info(f"Retrieved item {item_id} from ephemeral memory")
            return item
        logger.warning(f"Item {item_id} not found in ephemeral memory")
        return None
    
    async def query(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query items from ephemeral memory"""
        results = []
        
        # Filter items based on query parameters
        for item in self.items.values():
            # Filter by item type if specified
            if query.item_type and item.item_type != query.item_type:
                continue
            
            # Filter by custom filters
            if query.filters:
                # Check if all filters match
                match = True
                for key, value in query.filters.items():
                    # Handle nested paths (e.g., "content.title")
                    parts = key.split(".")
                    item_value = item.dict()
                    for part in parts:
                        if isinstance(item_value, dict) and part in item_value:
                            item_value = item_value[part]
                        else:
                            match = False
                            break
                    
                    # Compare values
                    if item_value != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Filter by time range if specified
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= item.created_at <= end_time):
                    continue
            
            # TODO: Vector and semantic query implementation
            # This would use vector similarity search in a production implementation
            # For now, we'll use a placeholder for semantic search
            if query.semantic_query:
                # Simple keyword matching as placeholder
                if not any(
                    word.lower() in str(item.content).lower() 
                    for word in query.semantic_query.lower().split()
                ):
                    continue
            
            # Update access metadata
            now = datetime.now()
            item.last_accessed = now
            item.access_count += 1
            
            results.append(item)
        
        # Apply temporal decay before sorting
        for item in results:
            time_decay = self._calculate_temporal_decay(item.created_at)
            item.importance_score *= time_decay
        
        # Sort results based on sort_by parameter
        if query.sort_by == "relevance":
            # For relevance sorting, we would use vector similarity 
            # For now, just use importance score as proxy
            results.sort(key=lambda x: x.importance_score, reverse=True)
        elif query.sort_by == "recency":
            results.sort(key=lambda x: x.created_at, reverse=True)
        elif query.sort_by == "importance":
            results.sort(key=lambda x: x.importance_score, reverse=True)
        elif query.sort_by == "access_count":
            results.sort(key=lambda x: x.access_count, reverse=True)
        
        # Apply limit and offset
        results = results[query.offset:query.offset + query.limit]
        
        logger.info(f"Found {len(results)} items in ephemeral memory matching query")
        return results
    
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update an item in ephemeral memory"""
        item = self.items.get(item_id)
        if not item:
            logger.warning(f"Item {item_id} not found in ephemeral memory")
            return False
        
        # Update the fields
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        logger.info(f"Updated item {item_id} in ephemeral memory")
        return True
    
    async def delete(self, item_id: str) -> bool:
        """Delete an item from ephemeral memory"""
        if item_id in self.items:
            del self.items[item_id]
            logger.info(f"Deleted item {item_id} from ephemeral memory")
            return True
        
        logger.warning(f"Item {item_id} not found in ephemeral memory")
        return False
    
    async def clear(self) -> bool:
        """Clear all items from ephemeral memory"""
        count = len(self.items)
        self.items = {}
        logger.info(f"Cleared {count} items from ephemeral memory")
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ephemeral memory statistics"""
        if not self.items:
            return {
                "count": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "item_types": {},
                "average_importance": 0.0,
            }
        
        # Count items by type
        type_counts = {}
        for item in self.items.values():
            type_name = item.item_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "count": len(self.items),
            "capacity": self.capacity,
            "utilization": len(self.items) / self.capacity,
            "item_types": type_counts,
            "average_importance": sum(item.importance_score for item in self.items.values()) / len(self.items),
        }
    
    async def apply_temporal_decay(self) -> int:
        """
        Apply temporal decay to all items
        
        Returns:
            Number of items decayed
        """
        decayed_count = 0
        for item_id, item in self.items.items():
            time_decay = self._calculate_temporal_decay(item.created_at)
            importance_before = item.importance_score
            item.importance_score *= time_decay
            if item.importance_score < importance_before:
                decayed_count += 1
        
        logger.info(f"Applied temporal decay to {decayed_count} items in ephemeral memory")
        return decayed_count
    
    async def _evict_items(self, count: int) -> int:
        """
        Evict least important items
        
        Args:
            count: Number of items to evict
            
        Returns:
            Number of items actually evicted
        """
        if not self.items:
            return 0
        
        # Apply temporal decay to ensure current importance scores
        await self.apply_temporal_decay()
        
        # Sort items by importance (ascending)
        sorted_items = sorted(
            self.items.items(), 
            key=lambda x: (x[1].importance_score, x[1].last_accessed)
        )
        
        # Evict the least important items
        evict_count = min(count, len(sorted_items))
        for i in range(evict_count):
            item_id, _ = sorted_items[i]
            del self.items[item_id]
        
        logger.info(f"Evicted {evict_count} items from ephemeral memory")
        return evict_count
    
    def _calculate_temporal_decay(self, created_at: datetime) -> float:
        """
        Calculate temporal decay factor based on item age
        
        Args:
            created_at: When the item was created
            
        Returns:
            Decay factor (0.0-1.0)
        """
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        decay = max(0.1, 1.0 - (self.decay_factor * age_hours))
        return decay


# ----- Persistent Memory Implementation -----

class FileBasedPersistentMemory(Memory):
    """
    File-based implementation of persistent memory
    
    Characteristics:
    - Persisted across sessions
    - Slower access than ephemeral memory
    - No temporal decay or capacity limits
    - Basic vector search capability
    
    Note: This is a simplified implementation for the POC.
    In production, this would use ChromaDB for vector storage and MongoDB for document storage.
    """
    
    def __init__(self, storage_dir: str = "data/memory"):
        """
        Initialize persistent memory
        
        Args:
            storage_dir: Directory to store memory files
        """
        self.storage_dir = Path(storage_dir)
        self.items_dir = self.storage_dir / "items"
        self.index_path = self.storage_dir / "index.json"
        
        # Create directories if they don't exist
        self.items_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.index = {}
        
        logger.info(f"Initialized FileBasedPersistentMemory at {storage_dir}")
    
    async def store(self, item: MemoryItem) -> str:
        """Store an item in persistent memory"""
        # Save item to file
        item_path = self.items_dir / f"{item.id}.json"
        
        # Convert to dict for serialization, ensuring enum and datetime objects are properly serialized
        item_dict = item.dict(exclude={"vector_embedding"})
        # Convert enum to string for JSON serialization
        item_dict["item_type"] = item.item_type.value
        # Convert datetime objects to ISO format strings
        item_dict["created_at"] = item_dict["created_at"].isoformat()
        item_dict["last_accessed"] = item_dict["last_accessed"].isoformat()
        
        # Store basic metadata in index for efficient queries
        self.index[item.id] = {
            "id": item.id,
            "item_type": item.item_type.value,
            "created_at": item.created_at.isoformat(),
            "last_accessed": item.last_accessed.isoformat(),
            "access_count": item.access_count,
            "importance_score": item.importance_score,
        }
        
        try:
            # Save item to file
            with open(item_path, "w") as f:
                json.dump(item_dict, f)
            
            # Update index file
            with open(self.index_path, "w") as f:
                json.dump(self.index, f)
            
            logger.info(f"Stored item {item.id} of type {item.item_type.value} in persistent memory")
            return item.id
            
        except Exception as e:
            logger.error(f"Error storing item {item.id}: {str(e)}")
            # Clean up index if file write failed
            if item.id in self.index:
                del self.index[item.id]
            return ""
    
    async def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve an item from persistent memory"""
        item_path = self.items_dir / f"{item_id}.json"
        
        if not item_path.exists():
            logger.warning(f"Item {item_id} not found in persistent memory")
            return None
        
        try:
            # Load item from file
            with open(item_path, "r") as f:
                item_dict = json.load(f)
            
            # Convert datetime strings back to datetime objects
            item_dict["created_at"] = datetime.fromisoformat(item_dict["created_at"])
            item_dict["last_accessed"] = datetime.fromisoformat(item_dict["last_accessed"])
            
            # Convert item_type string back to enum
            item_dict["item_type"] = MemoryItemType(item_dict["item_type"])
            
            item = MemoryItem(**item_dict)
            
            # Update access metadata
            now = datetime.now()
            item.last_accessed = now
            item.access_count += 1
            
            # Update metadata on disk
            self.index[item_id]["last_accessed"] = now.isoformat()
            self.index[item_id]["access_count"] = item.access_count
            
            with open(self.index_path, "w") as f:
                json.dump(self.index, f)
            
            logger.info(f"Retrieved item {item_id} from persistent memory")
            return item
            
        except Exception as e:
            logger.error(f"Error retrieving item {item_id}: {str(e)}")
            return None
    
    async def query(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query items from persistent memory"""
        # First filter by index metadata to avoid loading all files
        filtered_ids = []
        
        for item_id, metadata in self.index.items():
            # Filter by item type if specified
            if query.item_type and metadata["item_type"] != query.item_type.value:
                continue
            
            # Filter by time range if specified
            if query.time_range:
                start_time, end_time = query.time_range
                created_at = datetime.fromisoformat(metadata["created_at"])
                if not (start_time <= created_at <= end_time):
                    continue
            
            filtered_ids.append(item_id)
        
        # Load items for more detailed filtering
        results = []
        for item_id in filtered_ids:
            item = await self.retrieve(item_id)
            if not item:
                continue
            
            # Filter by custom filters
            if query.filters:
                # Check if all filters match
                match = True
                for key, value in query.filters.items():
                    # Handle nested paths (e.g., "content.title")
                    parts = key.split(".")
                    item_value = item.dict()
                    for part in parts:
                        if isinstance(item_value, dict) and part in item_value:
                            item_value = item_value[part]
                        else:
                            match = False
                            break
                    
                    # Compare values
                    if item_value != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # TODO: Vector and semantic query implementation
            # This would use vector similarity search in a production implementation
            # For now, we'll use a placeholder for semantic search
            if query.semantic_query:
                # Simple keyword matching as placeholder
                if not any(
                    word.lower() in str(item.content).lower() 
                    for word in query.semantic_query.lower().split()
                ):
                    continue
            
            results.append(item)
        
        # Sort results based on sort_by parameter
        if query.sort_by == "relevance":
            # For relevance sorting, we would use vector similarity 
            # For now, just use importance score as proxy
            results.sort(key=lambda x: x.importance_score, reverse=True)
        elif query.sort_by == "recency":
            results.sort(key=lambda x: x.created_at, reverse=True)
        elif query.sort_by == "importance":
            results.sort(key=lambda x: x.importance_score, reverse=True)
        elif query.sort_by == "access_count":
            results.sort(key=lambda x: x.access_count, reverse=True)
        
        # Apply limit and offset
        results = results[query.offset:query.offset + query.limit]
        
        logger.info(f"Found {len(results)} items in persistent memory matching query")
        return results
    
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update an item in persistent memory"""
        # First load the item
        item = await self.retrieve(item_id)
        if not item:
            logger.warning(f"Item {item_id} not found in persistent memory")
            return False
        
        # Update the fields
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        # Save updated item back to disk
        try:
            # Convert to dict for serialization, ensuring enum and datetime objects are properly serialized
            item_dict = item.dict(exclude={"vector_embedding"})
            # Convert enum to string for JSON serialization
            item_dict["item_type"] = item.item_type.value
            # Convert datetime objects to ISO format strings
            item_dict["created_at"] = item_dict["created_at"].isoformat()
            item_dict["last_accessed"] = item_dict["last_accessed"].isoformat()
            
            # Save item to file
            item_path = self.items_dir / f"{item_id}.json"
            with open(item_path, "w") as f:
                json.dump(item_dict, f)
            
            # Update index metadata
            self.index[item_id] = {
                "id": item.id,
                "item_type": item.item_type.value,
                "created_at": item.created_at.isoformat(),
                "last_accessed": item.last_accessed.isoformat(),
                "access_count": item.access_count,
                "importance_score": item.importance_score,
            }
            
            # Update index file
            with open(self.index_path, "w") as f:
                json.dump(self.index, f)
            
            logger.info(f"Updated item {item_id} in persistent memory")
            return True
            
        except Exception as e:
            logger.error(f"Error updating item {item_id}: {str(e)}")
            return False
    
    async def delete(self, item_id: str) -> bool:
        """Delete an item from persistent memory"""
        item_path = self.items_dir / f"{item_id}.json"
        
        if not item_path.exists():
            logger.warning(f"Item {item_id} not found in persistent memory")
            return False
        
        try:
            # Delete item file
            item_path.unlink()
            
            # Update index
            if item_id in self.index:
                del self.index[item_id]
                
            # Save updated index
            with open(self.index_path, "w") as f:
                json.dump(self.index, f)
            
            logger.info(f"Deleted item {item_id} from persistent memory")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all items from persistent memory"""
        try:
            count = len(self.index)
            
            # Delete all item files
            for item_id in self.index:
                item_path = self.items_dir / f"{item_id}.json"
                if item_path.exists():
                    item_path.unlink()
            
            # Clear index
            self.index = {}
            
            # Save empty index
            with open(self.index_path, "w") as f:
                json.dump(self.index, f)
            
            logger.info(f"Cleared {count} items from persistent memory")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing persistent memory: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get persistent memory statistics"""
        if not self.index:
            return {
                "count": 0,
                "item_types": {},
                "average_importance": 0.0,
                "disk_usage_mb": 0.0,
            }
        
        # Count items by type
        type_counts = {}
        for item in self.index.values():
            type_name = item["item_type"]
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Calculate disk usage
        disk_usage = 0
        for item_id in self.index:
            item_path = self.items_dir / f"{item_id}.json"
            if item_path.exists():
                disk_usage += item_path.stat().st_size
        
        return {
            "count": len(self.index),
            "item_types": type_counts,
            "average_importance": sum(item["importance_score"] for item in self.index.values()) / len(self.index),
            "disk_usage_mb": disk_usage / (1024 * 1024),
        }


# ----- Memory Manager -----

class MemoryManager:
    """
    Orchestrates operations across different memory systems
    
    The MemoryManager:
    - Provides a unified interface to both ephemeral and persistent memory
    - Handles the routing and synchronization between memory systems
    - Manages the memory lifecycle (creation, access, modification, deletion)
    - Controls importance scoring and temporal decay
    """
    
    def __init__(
        self,
        ephemeral_memory: Optional[EphemeralMemory] = None,
        persistent_memory: Optional[Memory] = None,
        auto_persist_threshold: float = 0.8,
    ):
        """
        Initialize the memory manager
        
        Args:
            ephemeral_memory: EphemeralMemory instance (created if None)
            persistent_memory: PersistentMemory instance (created if None)
            auto_persist_threshold: Importance threshold for auto-persisting items
        """
        self.ephemeral_memory = ephemeral_memory or EphemeralMemory()
        self.persistent_memory = persistent_memory or FileBasedPersistentMemory()
        self.auto_persist_threshold = auto_persist_threshold
        logger.info("Initialized MemoryManager")
    
    async def store(
        self, 
        item: MemoryItem, 
        ephemeral: bool = True, 
        persistent: bool = False
    ) -> str:
        """
        Store an item in memory
        
        Args:
            item: The memory item to store
            ephemeral: Whether to store in ephemeral memory
            persistent: Whether to store in persistent memory
            
        Returns:
            The ID of the stored item
        """
        result = ""
        
        if ephemeral:
            result = await self.ephemeral_memory.store(item)
        
        if persistent:
            result = await self.persistent_memory.store(item)
        elif item.importance_score >= self.auto_persist_threshold:
            # Auto-persist important items
            logger.info(f"Auto-persisting important item {item.id} (score: {item.importance_score})")
            result = await self.persistent_memory.store(item)
        
        return result
    
    async def retrieve(self, item_id: str, prefer_ephemeral: bool = True) -> Optional[MemoryItem]:
        """
        Retrieve an item from memory
        
        Args:
            item_id: The ID of the item to retrieve
            prefer_ephemeral: Whether to check ephemeral memory first
            
        Returns:
            The memory item if found, None otherwise
        """
        item = None
        
        if prefer_ephemeral:
            # Try ephemeral first, then persistent
            item = await self.ephemeral_memory.retrieve(item_id)
            if not item:
                item = await self.persistent_memory.retrieve(item_id)
                # If found in persistent, also cache in ephemeral
                if item:
                    await self.ephemeral_memory.store(item)
        else:
            # Try persistent first, then ephemeral
            item = await self.persistent_memory.retrieve(item_id)
            if not item:
                item = await self.ephemeral_memory.retrieve(item_id)
        
        return item
    
    async def query(
        self, 
        query: MemoryQuery,
        ephemeral: bool = True,
        persistent: bool = True,
        merge_results: bool = True
    ) -> List[MemoryItem]:
        """
        Query items from memory
        
        Args:
            query: The query parameters
            ephemeral: Whether to query ephemeral memory
            persistent: Whether to query persistent memory
            merge_results: Whether to merge results from both memories
            
        Returns:
            A list of memory items matching the query
        """
        results = []
        
        # Query ephemeral memory if requested
        if ephemeral:
            ephemeral_results = await self.ephemeral_memory.query(query)
            if ephemeral_results:
                results.extend(ephemeral_results)
        
        # Query persistent memory if requested
        if persistent:
            # Adjust limit if we already have results from ephemeral
            if results and merge_results:
                if query.limit > 0:
                    # Only get enough items to fill the limit
                    remaining = max(0, query.limit - len(results))
                    if remaining > 0:
                        persistent_query = MemoryQuery(**query.dict())
                        persistent_query.limit = remaining
                        persistent_results = await self.persistent_memory.query(persistent_query)
                        results.extend(persistent_results)
            else:
                persistent_results = await self.persistent_memory.query(query)
                if persistent_results:
                    if merge_results:
                        results.extend(persistent_results)
                    else:
                        # Return only persistent results
                        return persistent_results
        
        # Sort combined results if we merged
        if merge_results and results:
            if query.sort_by == "relevance":
                results.sort(key=lambda x: x.importance_score, reverse=True)
            elif query.sort_by == "recency":
                results.sort(key=lambda x: x.created_at, reverse=True)
            elif query.sort_by == "importance":
                results.sort(key=lambda x: x.importance_score, reverse=True)
            elif query.sort_by == "access_count":
                results.sort(key=lambda x: x.access_count, reverse=True)
            
            # Apply limit if needed
            if query.limit > 0:
                results = results[:query.limit]
        
        return results
    
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an item in memory
        
        Args:
            item_id: The ID of the item to update
            updates: The fields to update
            
        Returns:
            True if successful, False otherwise
        """
        # Try to update in ephemeral first
        ephemeral_success = await self.ephemeral_memory.update(item_id, updates)
        
        # Try to update in persistent regardless (item might be in both)
        persistent_success = await self.persistent_memory.update(item_id, updates)
        
        # Return true if updated in either memory
        return ephemeral_success or persistent_success
    
    async def delete(self, item_id: str) -> bool:
        """
        Delete an item from memory
        
        Args:
            item_id: The ID of the item to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from both memories
        ephemeral_success = await self.ephemeral_memory.delete(item_id)
        persistent_success = await self.persistent_memory.delete(item_id)
        
        # Return true if deleted from either memory
        return ephemeral_success or persistent_success
    
    async def clear(self, ephemeral: bool = True, persistent: bool = False) -> bool:
        """
        Clear memory
        
        Args:
            ephemeral: Whether to clear ephemeral memory
            persistent: Whether to clear persistent memory
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        if ephemeral:
            success = success and await self.ephemeral_memory.clear()
        
        if persistent:
            success = success and await self.persistent_memory.clear()
        
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            A dictionary of statistics for both memories
        """
        ephemeral_stats = await self.ephemeral_memory.get_stats()
        persistent_stats = await self.persistent_memory.get_stats()
        
        return {
            "ephemeral": ephemeral_stats,
            "persistent": persistent_stats,
            "auto_persist_threshold": self.auto_persist_threshold,
        }
    
    async def maintain(self) -> Dict[str, Any]:
        """
        Perform maintenance tasks on memory systems
        
        Returns:
            A dictionary of maintenance statistics
        """
        # Apply temporal decay to ephemeral memory
        decayed = await self.ephemeral_memory.apply_temporal_decay()
        
        # Auto-persist important items
        persisted_count = 0
        query = MemoryQuery(
            limit=100,
            sort_by="importance",
            filters={
                "importance_score": {"$gte": self.auto_persist_threshold}
            }
        )
        
        important_items = await self.ephemeral_memory.query(query)
        for item in important_items:
            # Check if already in persistent memory
            existing = await self.persistent_memory.retrieve(item.id)
            if not existing:
                # Persist important item
                await self.persistent_memory.store(item)
                persisted_count += 1
        
        return {
            "decayed_items": decayed,
            "persisted_items": persisted_count,
        }


# ----- Example Usage -----

async def main():
    """Example usage of the memory system."""
    print("Memory System Demo")
    print("-" * 50)
    
    # Create memory manager
    memory_manager = MemoryManager(
        ephemeral_memory=EphemeralMemory(capacity=100),
        persistent_memory=FileBasedPersistentMemory(storage_dir="data/memory"),
        auto_persist_threshold=0.8
    )
    
    # Create sample memory items
    items = []
    
    # Research topic item
    topic_item = MemoryItem(
        id="topic_001",
        item_type=MemoryItemType.RESEARCH_TOPIC,
        content={
            "title": "Climate change impacts on agriculture",
            "description": "Research the effects of climate change on agricultural productivity and food security",
            "keywords": ["crop yields", "temperature rise", "food security", "adaptation strategies", "drought"]
        },
        importance_score=0.9
    )
    items.append(topic_item)
    
    # Search query items
    query_items = []
    for i, query in enumerate(["climate change agriculture", "crop yields temperature rise", "food security climate"]):
        query_item = MemoryItem(
            id=f"query_{i+1}",
            item_type=MemoryItemType.SEARCH_QUERY,
            content={
                "query": query,
                "topic_id": "topic_001"
            },
            importance_score=0.7 - (i * 0.1)
        )
        query_items.append(query_item)
        items.append(query_item)
    
    # Extracted information items
    info_items = []
    for i in range(5):
        info_item = MemoryItem(
            id=f"info_{i+1}",
            item_type=MemoryItemType.EXTRACTED_INFO,
            content={
                "text": f"Example extracted information about climate change and agriculture #{i+1}",
                "source_url": f"https://example.com/article-{i+1}",
                "topic_id": "topic_001",
                "relevance_score": 0.9 - (i * 0.1)
            },
            importance_score=0.8 - (i * 0.1)
        )
        info_items.append(info_item)
        items.append(info_item)
    
    # Store items in memory
    print("\nStoring items in memory...")
    for item in items:
        # Store high importance items in persistent memory
        persistent = item.importance_score >= memory_manager.auto_persist_threshold
        await memory_manager.store(item, ephemeral=True, persistent=persistent)
    
    # Get memory stats
    stats = await memory_manager.get_stats()
    print("\nMemory Statistics:")
    print(f"- Ephemeral Memory: {stats['ephemeral']['count']} items")
    print(f"- Ephemeral Utilization: {stats['ephemeral']['utilization']:.1%}")
    print(f"- Persistent Memory: {stats['persistent']['count']} items")
    
    # Query for research topic
    print("\nQuerying for research topic...")
    topic_query = MemoryQuery(
        item_type=MemoryItemType.RESEARCH_TOPIC,
        limit=10
    )
    
    topic_results = await memory_manager.query(topic_query)
    print(f"Found {len(topic_results)} research topics:")
    for topic in topic_results:
        print(f"- {topic.content['title']}")
    
    # Query for information related to a specific keyword
    print("\nQuerying for information about 'crop yields'...")
    info_query = MemoryQuery(
        item_type=MemoryItemType.EXTRACTED_INFO,
        semantic_query="crop yields",
        limit=10
    )
    
    info_results = await memory_manager.query(info_query)
    print(f"Found {len(info_results)} information items:")
    for info in info_results:
        print(f"- [{info.importance_score:.2f}] {info.content['text'][:50]}...")
    
    # Perform memory maintenance
    print("\nPerforming memory maintenance...")
    maintenance_stats = await memory_manager.maintain()
    print(f"- Decayed items: {maintenance_stats['decayed_items']}")
    print(f"- Persisted items: {maintenance_stats['persisted_items']}")
    
    # Get updated memory stats
    stats = await memory_manager.get_stats()
    print("\nUpdated Memory Statistics:")
    print(f"- Ephemeral Memory: {stats['ephemeral']['count']} items")
    print(f"- Ephemeral Utilization: {stats['ephemeral']['utilization']:.1%}")
    print(f"- Persistent Memory: {stats['persistent']['count']} items")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
