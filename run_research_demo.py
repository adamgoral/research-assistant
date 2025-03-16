#!/usr/bin/env python3
"""
Research Pipeline Demo Runner

This script provides a user-friendly way to run the research pipeline proof-of-concept
with custom research topics and parameters.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

from research_pipeline import ResearchPipeline, ResearchTopic


async def run_demo(
    topic_title: str,
    topic_description: str,
    keywords: list[str],
    max_queries: int = 3,
    max_results_per_query: int = 5,
    relevance_threshold: float = 0.5,
    min_display_relevance: float = 0.6,
    verbose: bool = False,
):
    """
    Run the research pipeline demo with the specified parameters.
    
    Args:
        topic_title: The title of the research topic
        topic_description: A description of the research topic
        keywords: A list of keywords related to the topic
        max_queries: Maximum number of search queries to generate
        max_results_per_query: Maximum number of search results per query
        relevance_threshold: Minimum relevance score for information to be stored
        min_display_relevance: Minimum relevance score for displaying research results
        verbose: Whether to display detailed information
    """
    print("=" * 80)
    print(f"Research Pipeline PoC Demo - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create research topic
    topic_id = "topic_" + str(int(datetime.now().timestamp()))
    topic = ResearchTopic(
        id=topic_id,
        title=topic_title,
        description=topic_description,
        keywords=keywords
    )
    
    print(f"Researching Topic: {topic.title}")
    print(f"Description: {topic.description}")
    print(f"Keywords: {', '.join(topic.keywords)}")
    print("-" * 80)
    
    print("Configuration:")
    print(f"- Maximum queries: {max_queries}")
    print(f"- Maximum results per query: {max_results_per_query}")
    print(f"- Relevance threshold for storage: {relevance_threshold}")
    print(f"- Minimum display relevance: {min_display_relevance}")
    print("-" * 80)
    
    # Create and execute the research pipeline
    pipeline = ResearchPipeline()
    
    print("Executing research pipeline...")
    start_time = datetime.now()
    stats = await pipeline.research_topic(
        topic, 
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        relevance_threshold=relevance_threshold
    )
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\nResearch completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Queries generated: {stats['queries_generated']}")
    print(f"Search results found: {stats['search_results']}")
    print(f"Pages successfully retrieved: {stats['pages_retrieved']}")
    print(f"Information items extracted: {stats['information_items_extracted']}")
    print(f"Information items stored (relevance threshold): {stats['information_items_stored']}")
    print(f"Errors encountered: {stats['errors']}")
    
    if verbose:
        print("\nDetailed Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    print("-" * 80)
    
    print("\nRetrieving research results...")
    results = await pipeline.get_research_results(topic.id, min_relevance=min_display_relevance)
    
    print(f"\nFound {results['total_information_items']} relevant information items from {len(results['sources'])} sources")
    
    print("\nTop Sources and Information:")
    for i, source in enumerate(results['sources']):
        credibility = source['evaluation']['credibility']
        relevance = source['evaluation']['relevance_score']
        domain_type = source['evaluation']['evaluation_factors'].get('domain_type', 'unknown')
        
        print(f"\nSource {i+1}: {source['title']}")
        print(f"URL: {source['url']}")
        print(f"Credibility: {credibility}")
        print(f"Relevance: {relevance:.2f}")
        print(f"Domain Type: {domain_type}")
        
        print("Key Information:")
        # Sort information by relevance score (descending)
        sorted_info = sorted(source['information'], key=lambda x: x['relevance_score'], reverse=True)
        
        for j, info in enumerate(sorted_info):
            if j >= 3 and not verbose:  # Show only top 3 unless verbose
                break
            print(f"  [{info['relevance_score']:.2f}] {info['text'][:150]}...")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def main():
    """Parse command line arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Run the Research Pipeline PoC demo")
    parser.add_argument(
        "--topic", 
        type=str, 
        default="Climate change impacts on agriculture",
        help="Research topic title"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="Research the effects of climate change on agricultural productivity and food security",
        help="Research topic description"
    )
    parser.add_argument(
        "--keywords", 
        type=str, 
        default="crop yields,temperature rise,food security,adaptation strategies,drought",
        help="Comma-separated list of keywords"
    )
    parser.add_argument(
        "--max-queries", 
        type=int, 
        default=3,
        help="Maximum number of search queries to generate"
    )
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=5,
        help="Maximum number of search results per query"
    )
    parser.add_argument(
        "--relevance-threshold", 
        type=float, 
        default=0.5,
        help="Minimum relevance score for information to be stored"
    )
    parser.add_argument(
        "--min-display-relevance", 
        type=float, 
        default=0.6,
        help="Minimum relevance score for displaying research results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Display detailed information"
    )
    
    args = parser.parse_args()
    
    # Convert keywords string to list
    keywords = [kw.strip() for kw in args.keywords.split(",")]
    
    try:
        asyncio.run(run_demo(
            args.topic,
            args.description,
            keywords,
            args.max_queries,
            args.max_results,
            args.relevance_threshold,
            args.min_display_relevance,
            args.verbose
        ))
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running demo: {str(e)}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
