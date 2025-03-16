"""
Research Pipeline Demo with Content Extraction

This script demonstrates the integrated research pipeline with real content
extraction capabilities using Playwright.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional, List

from research_pipeline import (
    ResearchPipeline, 
    ResearchTopic, 
    SearchQuery, 
    WebContent,
    SearchResult
)
from content_retriever import ContentRetriever
from content_extraction import ContentExtractor

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_pipeline_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("research_pipeline_demo")


class EnhancedResearchPipeline(ResearchPipeline):
    """
    Enhanced research pipeline that uses the real ContentRetriever
    implementation instead of the mock version.
    """
    
    def __init__(self, use_real_search: bool = True, api_key: Optional[str] = None):
        """
        Initialize the research pipeline with the real ContentRetriever.
        
        Args:
            use_real_search: Whether to use real web search when available
            api_key: SerpAPI key for real web search (if None, will try to get from environment)
        """
        # Call parent constructor first
        super().__init__(use_real_search=use_real_search, api_key=api_key)
        
        # Replace the mock ContentRetriever with our real implementation
        self.content_retriever = ContentRetriever(
            headless=True,
            timeout=30000,
            max_retries=2,
            parallel_extractions=3
        )
        
        logger.info("Enhanced research pipeline initialized with real content retriever")
        
    async def close(self):
        """Close resources used by the pipeline."""
        if hasattr(self, 'content_retriever'):
            await self.content_retriever.close()


async def run_demo(topic_title: str, topic_description: str, keywords: List[str]):
    """
    Run the research pipeline demo with the specified topic.
    
    Args:
        topic_title: The title of the research topic
        topic_description: The description of the research topic
        keywords: List of keywords related to the topic
    """
    print(f"\n{'=' * 80}")
    print(f"Research Pipeline Demo with Content Extraction")
    print(f"{'=' * 80}")
    print(f"Topic: {topic_title}")
    print(f"Description: {topic_description}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"{'=' * 80}\n")
    
    # Check if SerpAPI key is available
    api_key = os.environ.get("SERPAPI_API_KEY")
    use_real_search = api_key is not None
    
    if use_real_search:
        print("Using real web search with SerpAPI")
    else:
        print("SerpAPI key not found. Using mock search implementation.")
        print("Set SERPAPI_API_KEY environment variable to use real search.")
    
    print("\nInitializing research pipeline...")
    
    # Create the research topic
    topic = ResearchTopic(
        id=f"topic_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        title=topic_title,
        description=topic_description,
        keywords=keywords
    )
    
    # Create and execute the enhanced research pipeline
    pipeline = EnhancedResearchPipeline(use_real_search=use_real_search, api_key=api_key)
    
    try:
        # Phase 1: Research the topic
        print("\nPhase 1: Researching topic...")
        stats = await pipeline.research_topic(
            topic, 
            max_queries=2,           # Limit to 2 queries for demo
            max_results_per_query=3  # Limit to 3 results per query for demo
        )
        
        print("\nResearch completed!")
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Queries generated: {stats['queries_generated']}")
        print(f"Search results found: {stats['search_results']}")
        print(f"Pages successfully retrieved: {stats['pages_retrieved']}")
        print(f"Information items extracted: {stats['information_items_extracted']}")
        print(f"Information items stored: {stats['information_items_stored']}")
        
        # Phase 2: Get research results
        print("\nPhase 2: Retrieving research results...")
        results = await pipeline.get_research_results(topic.id)
        
        print(f"\nFound {results['total_information_items']} relevant information items from {len(results['sources'])} sources")
        
        # Phase 3: Display sample of research results
        print("\nPhase 3: Sample of top sources and information:")
        for i, source in enumerate(results['sources'][:3]):  # Show top 3 sources
            credibility = source['evaluation']['credibility']
            relevance = source['evaluation']['relevance_score']
            print(f"\nSource {i+1}: {source['title']}")
            print(f"URL: {source['url']}")
            print(f"Credibility: {credibility}, Relevance: {relevance:.2f}")
            
            if 'domain' in source['evaluation']['evaluation_factors']:
                print(f"Domain type: {source['evaluation']['evaluation_factors']['domain_type']}")
                
            if 'word_count' in source['information'][0]['metadata']:
                print(f"Word count: {source['information'][0]['metadata']['word_count']}")
                
            print("\nSample information:")
            # Show top 2 information items per source
            for j, info in enumerate(source['information'][:2]):
                print(f"\n  [{info['relevance_score']:.2f}] {info['text'][:200]}...")
    
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        print(f"\nError: {str(e)}")
    
    finally:
        # Cleanup
        await pipeline.close()
        print("\nResearch pipeline demo completed!")


async def main():
    """Main demo function."""
    # Default research topic
    default_topic = {
        "title": "Climate Change Mitigation Strategies",
        "description": "Research on effective strategies to mitigate climate change, including renewable energy, carbon capture, and policy approaches.",
        "keywords": ["renewable energy", "carbon capture", "climate policy", "emissions reduction", "sustainable development"]
    }
    
    # Alternative topics
    topics = {
        "ai": {
            "title": "Artificial Intelligence Ethics",
            "description": "Research on ethical considerations in AI development and deployment, including bias, transparency, and governance.",
            "keywords": ["AI ethics", "algorithmic bias", "AI governance", "transparent AI", "responsible AI"]
        },
        "space": {
            "title": "Space Exploration Technologies",
            "description": "Research on recent advancements in space exploration technologies and future missions.",
            "keywords": ["space exploration", "Mars missions", "rocket technology", "satellite systems", "space habitats"]
        },
        "health": {
            "title": "Emerging Infectious Diseases",
            "description": "Research on emerging infectious diseases, prevention strategies, and global health responses.",
            "keywords": ["infectious diseases", "epidemic prevention", "global health", "disease surveillance", "outbreak response"]
        }
    }
    
    # Check if a topic was specified
    selected_topic = default_topic
    if len(sys.argv) > 1 and sys.argv[1] in topics:
        selected_topic = topics[sys.argv[1]]
        
    # Run the demo with the selected topic
    await run_demo(
        selected_topic["title"],
        selected_topic["description"],
        selected_topic["keywords"]
    )


if __name__ == "__main__":
    asyncio.run(main())
