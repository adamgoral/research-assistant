"""
Memory System Implementation for Web Research Assistant
"""

from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

# ---- Memory Data Models ----

class MemoryEntry(BaseModel):
    """Base model for all memory entries"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    memory_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    
    def age(self) -> float:
        """Return age of memory in seconds"""
        return time.time() - self.timestamp


class ResearchQuery(MemoryEntry):
    """Memory entry for a research query"""
    memory_type: str = "research_query"
    content: Dict[str, Any] = {
        "query": "",
        "parameters": {},
    }


class SearchResult(MemoryEntry):
    """Memory entry for a search result"""
    memory_type: str = "search_result"
    content: Dict[str, Any] = {
        "query": "",
        "url": "",
        "title": "",
        "snippet": "",
        "relevance_score": 0.0,
    }


class WebContent(MemoryEntry):
    """Memory entry for retrieved web content"""
    memory_type: str = "web_content"
    content: Dict[str, Any] = {
        "url": "",
        "title": "",
        "text": "",
        "source_evaluation": {},
    }


class ExtractedFact(MemoryEntry):
    """Memory entry for an extracted fact"""
    memory_type: str = "extracted_fact"
    content: Dict[str, Any] = {
        "fact": "",
        "source_url": "",
        "confidence": 0.0,
        "subtopic": "",
    }


class ResearchConclusion(MemoryEntry):
    """Memory entry for a research conclusion"""
    memory_type: str = "research_conclusion"
    content: Dict[str, Any] = {
        "conclusion": "",
        "supporting_facts": [],
        "confidence": 0.0,
        "subtopic": "",
    }


class UserInteraction(MemoryEntry):
    """Memory entry for user interaction"""
    memory_type: str = "user_interaction"
    content: Dict[str, Any] = {
        "user_input": "",
        "agent_response": "",
        "interaction_type": "",
    }


# ---- Memory Storage ----

class MemoryStore:
    """Base class for memory storage"""
    
    def add(self, entry: MemoryEntry) -> str:
        """Add an entry to memory, return ID"""
        raise NotImplementedError
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID"""
        raise NotImplementedError
    
    def query(self, filters: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Query entries by filters"""
        raise NotImplementedError
    
    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entry"""
        raise NotImplementedError
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry"""
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """Simple in-memory implementation of MemoryStore"""
    
    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
    
    def add(self, entry: MemoryEntry) -> str:
        self.entries[entry.id] = entry
        return entry.id
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)
    
    def query(self, filters: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        results = []
        
        for entry in self.entries.values():
            match = True
            
            for key, value in filters.items():
                if key == "memory_type" and entry.memory_type != value:
                    match = False
                    break
                
                if key == "content" and isinstance(value, dict):
                    for content_key, content_value in value.items():
                        if content_key not in entry.content or entry.content[content_key] != content_value:
                            match = False
                            break
                
                if key == "metadata" and isinstance(value, dict):
                    for meta_key, meta_value in value.items():
                        if meta_key not in entry.metadata or entry.metadata[meta_key] != meta_value:
                            match = False
                            break
            
            if match:
                results.append(entry)
                if len(results) >= limit:
                    break
        
        return results
    
    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        
        if "content" in updates and isinstance(updates["content"], dict):
            entry.content.update(updates["content"])
        
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            entry.metadata.update(updates["metadata"])
        
        return True
    
    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False


class VectorMemoryStore(MemoryStore):
    """Vector-based memory store for semantic retrieval using Chroma"""
    
    def __init__(self, collection_name: str = "research_memory"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")
        
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.entries: Dict[str, MemoryEntry] = {}
    
    def add(self, entry: MemoryEntry) -> str:
        # Store the full entry
        self.entries[entry.id] = entry
        
        # Extract text for embedding
        if entry.memory_type == "research_query":
            text = entry.content.get("query", "")
        elif entry.memory_type == "search_result":
            text = f"{entry.content.get('title', '')} {entry.content.get('snippet', '')}"
        elif entry.memory_type == "web_content":
            text = f"{entry.content.get('title', '')} {entry.content.get('text', '')[:1000]}"
        elif entry.memory_type == "extracted_fact":
            text = entry.content.get("fact", "")
        elif entry.memory_type == "research_conclusion":
            text = entry.content.get("conclusion", "")
        else:
            text = str(entry.content)
        
        # Add to vector store
        self.collection.add(
            documents=[text],
            metadatas=[{
                "id": entry.id,
                "memory_type": entry.memory_type,
                "timestamp": entry.timestamp,
                **{f"metadata_{k}": str(v) for k, v in entry.metadata.items()}
            }],
            ids=[entry.id]
        )
        
        return entry.id
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)
    
    def query(self, filters: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        # Convert filters to chromadb format
        where_clause = {}
        
        if "memory_type" in filters:
            where_clause["memory_type"] = filters["memory_type"]
        
        if "metadata" in filters and isinstance(filters["metadata"], dict):
            for k, v in filters["metadata"].items():
                where_clause[f"metadata_{k}"] = str(v)
        
        # If we have a text query, do a similarity search
        if "text_query" in filters:
            results = self.collection.query(
                query_texts=[filters["text_query"]],
                where=where_clause,
                n_results=limit
            )
            
            # Get the document IDs
            ids = results.get("ids", [[]])[0]
            
            # Return the full entries
            return [self.entries[id] for id in ids if id in self.entries]
        
        # Otherwise, do a metadata-only search
        else:
            results = self.collection.get(
                where=where_clause,
                limit=limit
            )
            
            # Get the document IDs
            ids = results.get("ids", [])
            
            # Return the full entries
            return [self.entries[id] for id in ids if id in self.entries]
    
    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        
        if "content" in updates and isinstance(updates["content"], dict):
            entry.content.update(updates["content"])
        
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            entry.metadata.update(updates["metadata"])
        
        # Note: We're not updating the vector store here for simplicity
        # In a full implementation, you'd want to delete and re-add the entry
        
        return True
    
    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            # Remove from dictionary
            del self.entries[entry_id]
            
            # Remove from vector store
            try:
                self.collection.delete(ids=[entry_id])
            except:
                pass  # Ignore errors
                
            return True
        return False


# ---- Memory Manager ----

class MemoryManager:
    """Manager for working with memory stores"""
    
    def __init__(self):
        self.ephemeral_store = InMemoryStore()  # Short-term memory
        self.persistent_store = VectorMemoryStore()  # Long-term memory
    
    def add_research_query(self, query: str, parameters: Dict[str, Any]) -> str:
        """Add a research query to memory"""
        entry = ResearchQuery(content={
            "query": query,
            "parameters": parameters
        })
        
        # Add to both stores
        self.ephemeral_store.add(entry)
        entry_id = self.persistent_store.add(entry)
        
        return entry_id
    
    def add_search_result(self, query: str, url: str, title: str, snippet: str, relevance_score: float) -> str:
        """Add a search result to memory"""
        entry = SearchResult(content={
            "query": query,
            "url": url,
            "title": title,
            "snippet": snippet,
            "relevance_score": relevance_score
        })
        
        # Add to both stores
        self.ephemeral_store.add(entry)
        entry_id = self.persistent_store.add(entry)
        
        return entry_id
    
    def add_web_content(self, url: str, title: str, text: str, source_evaluation: Dict[str, Any]) -> str:
        """Add web content to memory"""
        entry = WebContent(content={
            "url": url,
            "title": title,
            "text": text,
            "source_evaluation": source_evaluation
        })
        
        # Add to both stores
        self.ephemeral_store.add(entry)
        entry_id = self.persistent_store.add(entry)
        
        return entry_id
    
    def add_extracted_fact(self, fact: str, source_url: str, confidence: float, subtopic: str) -> str:
        """Add an extracted fact to memory"""
        entry = ExtractedFact(content={
            "fact": fact,
            "source_url": source_url,
            "confidence": confidence,
            "subtopic": subtopic
        })
        
        # Add to both stores
        self.ephemeral_store.add(entry)
        entry_id = self.persistent_store.add(entry)
        
        return entry_id
    
    def add_research_conclusion(self, conclusion: str, supporting_facts: List[str], confidence: float, subtopic: str) -> str:
        """Add a research conclusion to memory"""
        entry = ResearchConclusion(content={
            "conclusion": conclusion,
            "supporting_facts": supporting_facts,
            "confidence": confidence,
            "subtopic": subtopic
        })
        
        # Add to both stores
        self.ephemeral_store.add(entry)
        entry_id = self.persistent_store.add(entry)
        
        return entry_id
    
    def add_user_interaction(self, user_input: str, agent_response: str, interaction_type: str) -> str:
        """Add a user interaction to memory"""
        entry = UserInteraction(content={
            "user_input": user_input,
            "agent_response": agent_response,
            "interaction_type": interaction_type
        })
        
        # Add only to ephemeral store unless it's an important interaction
        if interaction_type in ["research_request", "feedback", "clarification"]:
            self.ephemeral_store.add(entry)
            entry_id = self.persistent_store.add(entry)
        else:
            entry_id = self.ephemeral_store.add(entry)
        
        return entry_id
    
    def get_recent_facts(self, limit: int = 20) -> List[ExtractedFact]:
        """Get recent facts from memory"""
        entries = self.ephemeral_store.query({"memory_type": "extracted_fact"}, limit=limit)
        return [entry for entry in entries if isinstance(entry, ExtractedFact)]
    
    def get_facts_by_subtopic(self, subtopic: str, limit: int = 20) -> List[ExtractedFact]:
        """Get facts by subtopic"""
        entries = self.ephemeral_store.query({
            "memory_type": "extracted_fact",
            "content": {"subtopic": subtopic}
        }, limit=limit)
        return [entry for entry in entries if isinstance(entry, ExtractedFact)]
    
    def search_facts(self, query: str, limit: int = 10) -> List[ExtractedFact]:
        """Search facts by semantic similarity"""
        entries = self.persistent_store.query({
            "memory_type": "extracted_fact",
            "text_query": query
        }, limit=limit)
        return [entry for entry in entries if isinstance(entry, ExtractedFact)]
    
    def get_all_conclusions(self) -> List[ResearchConclusion]:
        """Get all research conclusions"""
        entries = self.ephemeral_store.query({"memory_type": "research_conclusion"}, limit=100)
        return [entry for entry in entries if isinstance(entry, ResearchConclusion)]
    
    def get_research_context(self) -> Dict[str, Any]:
        """Get comprehensive context for current research"""
        # Get the original query
        queries = self.ephemeral_store.query({"memory_type": "research_query"}, limit=1)
        query = queries[0] if queries else None
        
        # Get recent conclusions
        conclusions = self.get_all_conclusions()
        
        # Get recent user interactions
        interactions = self.ephemeral_store.query({"memory_type": "user_interaction"}, limit=5)
        
        return {
            "query": query.content if query else None,
            "conclusions": [c.content for c in conclusions],
            "recent_interactions": [i.content for i in interactions]
        }
