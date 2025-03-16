"""
Research Pipeline Proof of Concept

This module implements a basic proof-of-concept for the Web Research Assistant's
research pipeline, following the architecture defined in the system design.

The pipeline consists of the following components:
1. SearchQueryGenerator: Generates optimized search queries based on research topics
2. WebSearchTool: Performs web searches using SerpAPI
3. ContentRetriever: Extracts content from web pages using Playwright
4. InformationExtractor: Extracts relevant information from retrieved content
5. SourceEvaluator: Evaluates source credibility and relevance
6. KnowledgeBase: Stores extracted information and source metadata

This PoC demonstrates the flow of information through the pipeline and the
interfaces between components.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import the real WebSearchTool implementation
# This is imported conditionally to avoid issues if dependencies are not installed
try:
    from web_search import WebSearchTool as RealWebSearchTool
    REAL_SEARCH_AVAILABLE = True
except ImportError:
    REAL_SEARCH_AVAILABLE = False
    logging.getLogger("research_pipeline").warning(
        "Real web search functionality not available. Using mock implementation instead."
    )

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("research_pipeline")


# ----- Data Models -----

class SourceCredibility(Enum):
    """Enum representing source credibility levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SearchQuery(BaseModel):
    """Model representing a search query."""
    query: str
    priority: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Model representing a search result."""
    title: str
    url: str
    snippet: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebContent(BaseModel):
    """Model representing retrieved web content."""
    url: str
    title: str
    content: str
    html: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedInformation(BaseModel):
    """Model representing extracted information."""
    text: str
    source_url: str
    source_title: str
    relevance_score: float
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceEvaluation(BaseModel):
    """Model representing source evaluation results."""
    url: str
    title: str
    credibility: SourceCredibility = SourceCredibility.UNKNOWN
    relevance_score: float = 0.0
    domain_authority: Optional[float] = None
    citation_count: Optional[int] = None
    publication_date: Optional[datetime] = None
    author_credentials: Optional[str] = None
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_factors: Dict[str, Any] = Field(default_factory=dict)


class ResearchTopic(BaseModel):
    """Model representing a research topic."""
    id: str
    title: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    search_queries: List[SearchQuery] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ----- Pipeline Components -----

class SearchQueryGenerator:
    """
    Generates optimized search queries based on research topics.
    
    In a production implementation, this would use GPT-3.5 Turbo to generate
    effective search queries based on the research topic.
    """
    
    async def generate_queries(
        self, topic: ResearchTopic, num_queries: int = 3
    ) -> List[SearchQuery]:
        """
        Generate search queries for a given research topic.
        
        Args:
            topic: The research topic
            num_queries: Number of queries to generate
            
        Returns:
            A list of generated SearchQuery objects
        """
        logger.info(f"Generating {num_queries} search queries for topic: {topic.title}")
        
        # In a real implementation, this would call GPT-3.5 Turbo
        # For this PoC, we'll generate queries based on the topic and keywords
        
        queries = []
        
        # Add the main topic as a query
        queries.append(
            SearchQuery(
                query=topic.title,
                priority=1,
                metadata={"type": "primary", "source": "topic_title"}
            )
        )
        
        # Add queries based on keywords
        for i, keyword in enumerate(topic.keywords[:num_queries-1]):
            queries.append(
                SearchQuery(
                    query=f"{topic.title} {keyword}",
                    priority=i+2,
                    metadata={"type": "keyword", "keyword": keyword}
                )
            )
            
        logger.info(f"Generated {len(queries)} search queries")
        return queries


class WebSearchTool:
    """
    Performs web searches using a search API.
    
    In a production implementation, this would use SerpAPI to perform
    actual web searches.
    """
    
    async def search(
        self, query: SearchQuery, num_results: int = 5
    ) -> List[SearchResult]:
        """
        Execute a web search for the given query.
        
        Args:
            query: The search query to execute
            num_results: Maximum number of results to return
            
        Returns:
            A list of SearchResult objects
        """
        logger.info(f"Executing search for query: {query.query}")
        
        # In a real implementation, this would call SerpAPI
        # For this PoC, we'll return mock search results
        
        mock_domains = [
            "wikipedia.org", 
            "nytimes.com", 
            "academia.edu", 
            "arxiv.org",
            "nature.com", 
            "sciencedirect.com", 
            "mit.edu", 
            "stanford.edu",
            "harvard.edu", 
            "bbc.com", 
            "reuters.com"
        ]
        
        results = []
        
        # Generate mock search results
        for i in range(min(num_results, 10)):
            domain = mock_domains[i % len(mock_domains)]
            title = f"Result {i+1} for {query.query}"
            url = f"https://www.{domain}/article-{i+1}"
            snippet = f"This is a snippet of information about {query.query}..."
            
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    position=i+1,
                    metadata={
                        "domain": domain,
                        "query": query.query,
                    }
                )
            )
            
        logger.info(f"Found {len(results)} results for query: {query.query}")
        return results


class ContentRetriever:
    """
    Extracts content from web pages.
    
    In a production implementation, this would use Playwright to
    navigate to web pages and extract their content.
    """
    
    async def retrieve_content(self, result: SearchResult) -> Optional[WebContent]:
        """
        Retrieve and extract content from a search result URL.
        
        Args:
            result: The search result to retrieve content from
            
        Returns:
            A WebContent object with the extracted content, or None if retrieval failed
        """
        logger.info(f"Retrieving content from URL: {result.url}")
        
        # In a real implementation, this would use Playwright to visit the URL and extract content
        # For this PoC, we'll generate mock content
        
        # Simulate occasional retrieval failures
        if result.position % 7 == 0:
            logger.warning(f"Failed to retrieve content from URL: {result.url}")
            return None
            
        content = f"""
        <h1>{result.title}</h1>
        <p>This is a detailed article about {result.snippet}</p>
        <p>It contains information relevant to the search query.</p>
        <p>There are facts, figures, and analysis in this article.</p>
        <p>The article cites multiple sources and provides comprehensive coverage.</p>
        <p>Key points include important aspects of the topic.</p>
        """
        
        extracted_text = f"""
        {result.title}
        
        This is a detailed article about {result.snippet}
        
        It contains information relevant to the search query.
        
        There are facts, figures, and analysis in this article.
        
        The article cites multiple sources and provides comprehensive coverage.
        
        Key points include important aspects of the topic.
        """
        
        web_content = WebContent(
            url=result.url,
            title=result.title,
            content=extracted_text.strip(),
            html=content.strip(),
            metadata={
                "domain": result.metadata["domain"],
                "query": result.metadata["query"],
                "word_count": len(extracted_text.split()),
                "retrieval_timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info(f"Successfully retrieved content from URL: {result.url}")
        return web_content


# Import real InformationExtractor implementation
try:
    from information_extraction import InformationExtractor as RealInformationExtractor
    from information_extraction import InformationSynthesizer
    REAL_EXTRACTION_AVAILABLE = True
except ImportError:
    REAL_EXTRACTION_AVAILABLE = False
    logging.getLogger("research_pipeline").warning(
        "Real information extraction functionality not available. Using mock implementation instead."
    )


class InformationExtractor:
    """
    Extracts relevant information from retrieved content.
    
    This class provides a fallback mock implementation when the real
    LLM-based extractor is not available.
    """
    
    def __init__(self):
        """Initialize the mock extractor."""
        # Try to use the real implementation if available
        self.real_extractor = None
        if REAL_EXTRACTION_AVAILABLE:
            try:
                self.real_extractor = RealInformationExtractor()
                logger.info("Using real LLM-based information extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize real extractor: {str(e)}")
    
    async def extract_information(
        self, content: WebContent, topic: ResearchTopic
    ) -> List[ExtractedInformation]:
        """
        Extract relevant information from web content.
        
        Args:
            content: The web content to extract information from
            topic: The research topic for relevance matching
            
        Returns:
            A list of ExtractedInformation objects
        """
        # Use real extractor if available
        if self.real_extractor:
            try:
                return await self.real_extractor.extract_information(
                    content, topic, extraction_types=["facts", "claims", "summary"]
                )
            except Exception as e:
                logger.error(f"Real extraction failed, falling back to mock: {str(e)}")
                # Fall through to mock implementation
        
        logger.info(f"Using mock extraction for content: {content.url}")
        
        # Mock implementation begins here
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.content.split("\n\n") if p.strip()]
        
        extracted_info = []
        
        for i, paragraph in enumerate(paragraphs):
            # Skip very short paragraphs
            if len(paragraph.split()) < 5:
                continue
                
            # Calculate a mock relevance score (would be done by LLM in production)
            relevance_score = 0.5 + (0.5 / (i + 1))
            if any(kw.lower() in paragraph.lower() for kw in topic.keywords):
                relevance_score += 0.3
            if topic.title.lower() in paragraph.lower():
                relevance_score += 0.2
                
            # Clip to 0-1 range
            relevance_score = min(1.0, max(0.0, relevance_score))
            
            extracted_info.append(
                ExtractedInformation(
                    text=paragraph,
                    source_url=content.url,
                    source_title=content.title,
                    relevance_score=relevance_score,
                    metadata={
                        "type": "mock_extraction",
                        "paragraph_index": i,
                        "word_count": len(paragraph.split()),
                        "domain": content.metadata["domain"],
                    }
                )
            )
            
        logger.info(f"Extracted {len(extracted_info)} information items from content: {content.url}")
        return extracted_info


class SourceEvaluator:
    """
    Evaluates source credibility and relevance.
    
    In a production implementation, this would use a combination of
    heuristics and LLM analysis to evaluate sources.
    """
    
    async def evaluate_source(
        self, 
        content: WebContent, 
        extracted_info: List[ExtractedInformation]
    ) -> SourceEvaluation:
        """
        Evaluate the credibility and relevance of a source.
        
        Args:
            content: The web content to evaluate
            extracted_info: The information extracted from the content
            
        Returns:
            A SourceEvaluation object
        """
        logger.info(f"Evaluating source: {content.url}")
        
        # In a real implementation, this would use various metrics and possibly GPT-4o
        # For this PoC, we'll use simple heuristics
        
        domain = content.metadata["domain"]
        
        # Simple credibility heuristics based on domain
        credibility_map = {
            "wikipedia.org": SourceCredibility.MEDIUM,
            "nytimes.com": SourceCredibility.HIGH,
            "academia.edu": SourceCredibility.HIGH,
            "arxiv.org": SourceCredibility.HIGH,
            "nature.com": SourceCredibility.HIGH,
            "sciencedirect.com": SourceCredibility.HIGH,
            "mit.edu": SourceCredibility.HIGH,
            "stanford.edu": SourceCredibility.HIGH,
            "harvard.edu": SourceCredibility.HIGH,
            "bbc.com": SourceCredibility.HIGH,
            "reuters.com": SourceCredibility.HIGH,
        }
        
        credibility = credibility_map.get(domain, SourceCredibility.MEDIUM)
        
        # Calculate average relevance score from extracted information
        avg_relevance = 0.0
        if extracted_info:
            avg_relevance = sum(info.relevance_score for info in extracted_info) / len(extracted_info)
        
        evaluation = SourceEvaluation(
            url=content.url,
            title=content.title,
            credibility=credibility,
            relevance_score=avg_relevance,
            domain_authority=0.8 if "edu" in domain else 0.7,
            citation_count=None,  # Would require actual analysis
            publication_date=None,  # Would require actual extraction
            evaluation_factors={
                "domain_type": "academic" if "edu" in domain else "news" if "news" in domain else "general",
                "content_length": content.metadata.get("word_count", 0),
                "has_references": "references" in content.content.lower(),
            }
        )
        
        logger.info(f"Evaluated source {content.url}: credibility={credibility.value}, relevance={avg_relevance:.2f}")
        return evaluation


class KnowledgeBase:
    """
    Stores extracted information and source metadata.
    
    In a production implementation, this would use ChromaDB for vector storage
    and MongoDB for document storage.
    """
    
    def __init__(self):
        """Initialize the knowledge base."""
        self.information_items = []
        self.source_evaluations = {}
        self.topic_information_map = {}
        
    async def store_information(
        self, 
        topic: ResearchTopic,
        information: ExtractedInformation, 
        evaluation: SourceEvaluation
    ) -> str:
        """
        Store extracted information and source evaluation.
        
        Args:
            topic: The research topic
            information: The extracted information
            evaluation: The source evaluation
            
        Returns:
            The ID of the stored information
        """
        # Generate a simple ID
        info_id = f"info_{len(self.information_items) + 1}"
        
        # Store the information
        self.information_items.append({
            "id": info_id,
            "topic_id": topic.id,
            "information": information.dict(),
            "evaluation": evaluation.dict()
        })
        
        # Update source evaluations
        self.source_evaluations[information.source_url] = evaluation
        
        # Update topic->information mapping
        if topic.id not in self.topic_information_map:
            self.topic_information_map[topic.id] = []
        self.topic_information_map[topic.id].append(info_id)
        
        logger.info(f"Stored information {info_id} in knowledge base for topic {topic.id}")
        return info_id
        
    async def get_information_for_topic(
        self, topic_id: str, min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve information for a given topic.
        
        Args:
            topic_id: The ID of the topic
            min_relevance: Minimum relevance score threshold
            
        Returns:
            A list of information items
        """
        if topic_id not in self.topic_information_map:
            return []
            
        info_ids = self.topic_information_map[topic_id]
        result = []
        
        for item in self.information_items:
            if item["id"] in info_ids:
                relevance = item["information"]["relevance_score"]
                if relevance >= min_relevance:
                    result.append(item)
                    
        # Sort by relevance (descending)
        result.sort(key=lambda x: x["information"]["relevance_score"], reverse=True)
        return result
        
    async def get_source_evaluation(self, url: str) -> Optional[SourceEvaluation]:
        """
        Retrieve evaluation for a given source.
        
        Args:
            url: The URL of the source
            
        Returns:
            The SourceEvaluation or None if not found
        """
        return self.source_evaluations.get(url)
        
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            A dictionary of statistics
        """
        topics = set(item["topic_id"] for item in self.information_items)
        sources = set(item["information"]["source_url"] for item in self.information_items)
        
        return {
            "total_information_items": len(self.information_items),
            "total_topics": len(topics),
            "total_sources": len(sources),
            "average_relevance": sum(item["information"]["relevance_score"] for item in self.information_items) / len(self.information_items) if self.information_items else 0,
        }


# ----- Research Pipeline -----

class ResearchPipeline:
    """
    Main research pipeline that orchestrates the research process.
    """
    
    def __init__(self, use_real_search: bool = True, api_key: Optional[str] = None):
        """
        Initialize the research pipeline components.
        
        Args:
            use_real_search: Whether to use real web search when available
            api_key: SerpAPI key for real web search (if None, will try to get from environment)
        """
        self.query_generator = SearchQueryGenerator()
        
        # Use real web search implementation if available and requested
        if use_real_search and REAL_SEARCH_AVAILABLE:
            logger.info("Using real web search implementation with SerpAPI")
            self.web_search = RealWebSearchTool(api_key=api_key)
        else:
            if use_real_search and not REAL_SEARCH_AVAILABLE:
                logger.warning("Real web search requested but not available. Using mock implementation.")
            else:
                logger.info("Using mock web search implementation")
            self.web_search = WebSearchTool()
            
        self.content_retriever = ContentRetriever()
        self.information_extractor = InformationExtractor()
        self.source_evaluator = SourceEvaluator()
        self.knowledge_base = KnowledgeBase()
        
    async def research_topic(
        self, 
        topic: ResearchTopic, 
        max_queries: int = 3,
        max_results_per_query: int = 5,
        relevance_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Execute the research pipeline for a given topic.
        
        Args:
            topic: The research topic to research
            max_queries: Maximum number of search queries to generate
            max_results_per_query: Maximum number of search results per query
            relevance_threshold: Minimum relevance score for information to be stored
            
        Returns:
            A dictionary with research statistics
        """
        logger.info(f"Starting research for topic: {topic.title}")
        
        start_time = datetime.now()
        processed_urls = set()
        stats = {
            "queries_generated": 0,
            "search_results": 0,
            "pages_retrieved": 0,
            "information_items_extracted": 0,
            "information_items_stored": 0,
            "errors": 0,
        }
        
        # Step 1: Generate search queries
        if not topic.search_queries:
            topic.search_queries = await self.query_generator.generate_queries(
                topic, num_queries=max_queries
            )
        stats["queries_generated"] = len(topic.search_queries)
        
        # Step 2: Execute search queries
        for query in topic.search_queries:
            try:
                search_results = await self.web_search.search(
                    query, num_results=max_results_per_query
                )
                stats["search_results"] += len(search_results)
                
                # Step 3-6: Process search results
                for result in search_results:
                    # Skip already processed URLs
                    if result.url in processed_urls:
                        continue
                    processed_urls.add(result.url)
                    
                    try:
                        # Step 3: Retrieve content
                        content = await self.content_retriever.retrieve_content(result)
                        if not content:
                            continue
                        stats["pages_retrieved"] += 1
                        
                        # Step 4: Extract information
                        extracted_info = await self.information_extractor.extract_information(
                            content, topic
                        )
                        stats["information_items_extracted"] += len(extracted_info)
                        
                        # Step 5: Evaluate source
                        evaluation = await self.source_evaluator.evaluate_source(
                            content, extracted_info
                        )
                        
                        # Step 6: Store in knowledge base
                        for info in extracted_info:
                            if info.relevance_score >= relevance_threshold:
                                await self.knowledge_base.store_information(
                                    topic, info, evaluation
                                )
                                stats["information_items_stored"] += 1
                                
                    except Exception as e:
                        logger.error(f"Error processing search result {result.url}: {str(e)}")
                        stats["errors"] += 1
                        
            except Exception as e:
                logger.error(f"Error executing search query {query.query}: {str(e)}")
                stats["errors"] += 1
                
        # Calculate research duration
        duration = (datetime.now() - start_time).total_seconds()
        stats["duration_seconds"] = duration
        stats["kb_stats"] = await self.knowledge_base.get_stats()
        
        logger.info(f"Completed research for topic: {topic.title}")
        logger.info(f"Research stats: {json.dumps(stats, default=str)}")
        
        return stats
        
    async def get_research_results(
        self, topic_id: str, min_relevance: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get research results for a topic.
        
        Args:
            topic_id: The ID of the topic
            min_relevance: Minimum relevance score for included information
            
        Returns:
            A dictionary with research results
        """
        information = await self.knowledge_base.get_information_for_topic(
            topic_id, min_relevance=min_relevance
        )
        
        # Group by source
        sources = {}
        for item in information:
            url = item["information"]["source_url"]
            if url not in sources:
                sources[url] = {
                    "url": url,
                    "title": item["information"]["source_title"],
                    "evaluation": item["evaluation"],
                    "information": []
                }
            sources[url]["information"].append(item["information"])
            
        return {
            "topic_id": topic_id,
            "total_information_items": len(information),
            "sources": list(sources.values()),
            "min_relevance": min_relevance,
        }


# ----- Example Usage -----

async def main():
    """Example usage of the research pipeline."""
    print("Research Pipeline PoC Demo")
    print("-" * 50)
    
    # Get API key from environment (if available)
    api_key = os.environ.get("SERPAPI_API_KEY")
    
    # Create a research topic
    topic = ResearchTopic(
        id="topic_001",
        title="Climate change impacts on agriculture",
        description="Research the effects of climate change on agricultural productivity and food security",
        keywords=["crop yields", "temperature rise", "food security", "adaptation strategies", "drought"]
    )
    
    print(f"Researching topic: {topic.title}")
    print(f"Keywords: {', '.join(topic.keywords)}")
    print("-" * 50)
    
    # Determine search type
    use_real_search = REAL_SEARCH_AVAILABLE and api_key is not None
    search_type = "real" if use_real_search else "mock"
    
    print(f"Using {search_type} web search implementation")
    if search_type == "real":
        print("SerpAPI key found in environment")
    else:
        if REAL_SEARCH_AVAILABLE and api_key is None:
            print("SerpAPI key not found. Set SERPAPI_API_KEY environment variable to use real search.")
        elif not REAL_SEARCH_AVAILABLE:
            print("Real search dependencies not available. Install required packages to use real search.")
    
    print("-" * 50)
    
    # Create and execute the research pipeline
    pipeline = ResearchPipeline(use_real_search=use_real_search, api_key=api_key)
    
    print("Executing research pipeline...")
    stats = await pipeline.research_topic(topic)
    
    print("\nResearch completed!")
    print(f"Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"Queries generated: {stats['queries_generated']}")
    print(f"Search results found: {stats['search_results']}")
    print(f"Pages successfully retrieved: {stats['pages_retrieved']}")
    print(f"Information items extracted: {stats['information_items_extracted']}")
    print(f"Information items stored (relevance threshold): {stats['information_items_stored']}")
    print(f"Errors encountered: {stats['errors']}")
    print("-" * 50)
    
    print("\nRetrieving research results...")
    results = await pipeline.get_research_results(topic.id)
    
    print(f"\nFound {results['total_information_items']} relevant information items from {len(results['sources'])} sources")
    
    print("\nSample of top sources and information:")
    for i, source in enumerate(results['sources'][:3]):  # Show top 3 sources
        credibility = source['evaluation']['credibility']
        relevance = source['evaluation']['relevance_score']
        print(f"\nSource {i+1}: {source['title']} ({source['url']})")
        print(f"Credibility: {credibility}, Relevance: {relevance:.2f}")
        print("Sample information:")
        
        # Show top 2 information items per source
        for j, info in enumerate(source['information'][:2]):
            print(f"  [{info['relevance_score']:.2f}] {info['text'][:100]}...")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
