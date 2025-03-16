"""
Research Pipeline Demo with Content Extraction and Information Synthesis

This script demonstrates the integrated research pipeline with real content
extraction capabilities using Playwright and information extraction and synthesis
using LLM-based analysis.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

from research_pipeline import (
    ResearchPipeline, 
    ResearchTopic, 
    SearchQuery, 
    WebContent,
    SearchResult,
    ExtractedInformation
)
from content_retriever import ContentRetriever
from content_extraction import ContentExtractor

# Import information extraction and synthesis
try:
    from information_extraction import InformationExtractor, InformationSynthesizer
    SYNTHESIS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AVAILABLE = False
    logging.getLogger("research_pipeline_demo").warning(
        "Information synthesis functionality not available. Install required packages."
    )

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
    implementation and adds information synthesis capabilities.
    """
    
    def __init__(self, use_real_search: bool = True, api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the research pipeline with the real implementations.
        
        Args:
            use_real_search: Whether to use real web search when available
            api_key: SerpAPI key for real web search (if None, will try to get from environment)
            openai_api_key: OpenAI API key for LLM-based extraction and synthesis
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
        
        # Add information synthesizer if available
        self.synthesizer = None
        if SYNTHESIS_AVAILABLE:
            try:
                self.synthesizer = InformationSynthesizer(openai_api_key=openai_api_key)
                logger.info("Information synthesizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize information synthesizer: {str(e)}")
        
        logger.info("Enhanced research pipeline initialized with real components")
        
    async def close(self):
        """Close resources used by the pipeline."""
        if hasattr(self, 'content_retriever'):
            await self.content_retriever.close()
            
    async def synthesize_research(
        self, topic: ResearchTopic, synthesis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Synthesize research results into a comprehensive analysis.
        
        Args:
            topic: The research topic
            synthesis_type: Type of synthesis (summary, comparison, analysis)
            
        Returns:
            Dictionary with synthesis results
        """
        if not self.synthesizer:
            logger.warning("Information synthesizer not available")
            return {
                "synthesis_text": "Information synthesis not available",
                "synthesis_type": synthesis_type,
                "error": "Synthesizer not initialized"
            }
            
        # Get information items for the topic
        information = await self.knowledge_base.get_information_for_topic(topic.id)
        
        if not information:
            return {
                "synthesis_text": f"No information available for {topic.title}",
                "synthesis_type": synthesis_type
            }
            
        # Convert to ExtractedInformation objects
        extracted_info = []
        for item in information:
            info_dict = item["information"]
            extracted_info.append(
                ExtractedInformation(
                    text=info_dict["text"],
                    source_url=info_dict["source_url"],
                    source_title=info_dict["source_title"],
                    relevance_score=info_dict["relevance_score"],
                    extraction_timestamp=datetime.fromisoformat(info_dict["extraction_timestamp"]),
                    metadata=info_dict["metadata"]
                )
            )
            
        # Perform synthesis
        logger.info(f"Synthesizing {len(extracted_info)} information items for {topic.title}")
        synthesis_result = await self.synthesizer.synthesize_information(
            topic, extracted_info, synthesis_type=synthesis_type
        )
        
        return synthesis_result


async def run_demo(topic_title: str, topic_description: str, keywords: List[str]):
    """
    Run the research pipeline demo with the specified topic.
    
    Args:
        topic_title: The title of the research topic
        topic_description: The description of the research topic
        keywords: List of keywords related to the topic
    """
    print(f"\n{'=' * 80}")
    print(f"Research Pipeline Demo with Content Extraction and Information Synthesis")
    print(f"{'=' * 80}")
    print(f"Topic: {topic_title}")
    print(f"Description: {topic_description}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"{'=' * 80}\n")
    
    # Check if SerpAPI key is available
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    use_real_search = serpapi_key is not None
    
    # Check if OpenAI API key is available
    openai_key = os.environ.get("OPENAI_API_KEY")
    use_real_synthesis = openai_key is not None
    
    if use_real_search:
        print("Using real web search with SerpAPI")
    else:
        print("SerpAPI key not found. Using mock search implementation.")
        print("Set SERPAPI_API_KEY environment variable to use real search.")
        
    if use_real_synthesis:
        print("Using real information extraction and synthesis with OpenAI")
    else:
        print("OpenAI API key not found. Using fallback extraction and synthesis methods.")
        print("Set OPENAI_API_KEY environment variable to use real LLM-based analysis.")
    
    print("\nInitializing research pipeline...")
    
    # Create the research topic
    topic = ResearchTopic(
        id=f"topic_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        title=topic_title,
        description=topic_description,
        keywords=keywords
    )
    
    # Create and execute the enhanced research pipeline
    pipeline = EnhancedResearchPipeline(
        use_real_search=use_real_search, 
        api_key=serpapi_key,
        openai_api_key=openai_key
    )
    
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
        print(f"Information items stored (relevance threshold): {stats['information_items_stored']}")
        
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
                
            if source['information'] and 'word_count' in source['information'][0].get('metadata', {}):
                print(f"Word count: {source['information'][0]['metadata']['word_count']}")
                
            print("\nSample information:")
            # Show top 2 information items per source
            for j, info in enumerate(source['information'][:2]):
                type_label = ""
                if 'type' in info.get('metadata', {}):
                    type_label = f" ({info['metadata']['type']})"
                print(f"\n  [{info['relevance_score']:.2f}]{type_label} {info['text'][:200]}...")
                
        # Phase 4: Information Synthesis (if available)
        if hasattr(pipeline, 'synthesizer') and pipeline.synthesizer:
            print(f"\n{'=' * 80}")
            print("\nPhase 4: Information Synthesis")
            print(f"{'=' * 80}")
            
            # Perform different types of synthesis
            synthesis_types = ["summary", "comparison", "analysis"]
            
            for synthesis_type in synthesis_types:
                print(f"\n{synthesis_type.upper()} SYNTHESIS")
                print("-" * 60)
                
                try:
                    synthesis_result = await pipeline.synthesize_research(
                        topic, synthesis_type=synthesis_type
                    )
                    
                    if synthesis_type == "summary":
                        print(f"\n{synthesis_result.get('summary', 'No summary generated')}")
                        
                        if "key_findings" in synthesis_result and synthesis_result["key_findings"]:
                            print("\nKey Findings:")
                            for finding in synthesis_result["key_findings"][:5]:  # Limit to 5 findings
                                print(f"• {finding}")
                                
                    elif synthesis_type == "comparison":
                        print(f"\n{synthesis_result.get('comparison_summary', 'No comparison generated')}")
                        
                        if "consensus_points" in synthesis_result and synthesis_result["consensus_points"]:
                            print("\nConsensus Points:")
                            for point in synthesis_result["consensus_points"][:3]:  # Limit to 3 points
                                print(f"• {point}")
                                
                        if "disagreements" in synthesis_result and synthesis_result["disagreements"]:
                            print("\nDisagreements or Contradictions:")
                            for disagree in synthesis_result["disagreements"][:3]:  # Limit to 3 points
                                print(f"• {disagree}")
                                
                    elif synthesis_type == "analysis":
                        print(f"\n{synthesis_result.get('analysis', 'No analysis generated')}")
                        
                        if "key_insights" in synthesis_result and synthesis_result["key_insights"]:
                            print("\nKey Insights:")
                            for insight in synthesis_result["key_insights"][:3]:  # Limit to 3 insights
                                print(f"• {insight}")
                                
                        if "implications" in synthesis_result and synthesis_result["implications"]:
                            print("\nImplications:")
                            for implication in synthesis_result["implications"][:3]:  # Limit to 3 implications
                                print(f"• {implication}")
                
                except Exception as e:
                    logger.error(f"Error in {synthesis_type} synthesis: {str(e)}")
                    print(f"\nError in {synthesis_type} synthesis: {str(e)}")
        else:
            print("\nPhase 4: Information Synthesis - Not Available")
            print("To enable synthesis capabilities, install the required packages and set OPENAI_API_KEY.")
    
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
        },
        "energy": {
            "title": "Renewable Energy Transition",
            "description": "Research on the global transition to renewable energy sources, including economic impacts, technological challenges, and policy frameworks",
            "keywords": ["solar power", "wind energy", "energy storage", "grid integration", "climate policy", "economic impact"]
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
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher.")
        sys.exit(1)
        
    # Run the demo
    asyncio.run(main())
