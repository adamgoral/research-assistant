"""
Web Search Demo Script

This script demonstrates the web search functionality of the Research Assistant,
using both mock and real search implementations based on available dependencies.
"""

import os
import asyncio
from datetime import datetime
import json

from research_pipeline import ResearchTopic, SearchQuery
from web_search import WebSearchTool  # Real search implementation


async def demo_web_search():
    """Run a demonstration of web search functionality."""
    print("Web Search Functionality Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        print("Note: SERPAPI_API_KEY environment variable not found.")
        print("Running in demo mode with limited functionality.")
        print("To use real search, set the SERPAPI_API_KEY environment variable.")
        print("=" * 50)
        demo_mode = True
    else:
        print("SerpAPI key found in environment variables.")
        demo_mode = False
    
    # Create the search tool
    search_tool = WebSearchTool(api_key=api_key)
    
    # Sample research topics
    topics = [
        ResearchTopic(
            id="topic_001",
            title="Climate change impacts on agriculture",
            description="Effects of climate change on agricultural productivity",
            keywords=["crop yields", "temperature rise", "food security"]
        ),
        ResearchTopic(
            id="topic_002",
            title="Artificial intelligence in healthcare",
            description="Applications of AI in medical diagnosis and treatment",
            keywords=["medical diagnosis", "treatment", "patient care"]
        )
    ]
    
    for topic in topics:
        print(f"\nResearching: {topic.title}")
        print("-" * 50)
        
        # Generate search queries
        queries = [
            SearchQuery(
                query=topic.title,
                priority=1,
                metadata={"type": "primary"}
            )
        ]
        
        # Add keyword-based queries
        for i, keyword in enumerate(topic.keywords):
            queries.append(
                SearchQuery(
                    query=f"{topic.title} {keyword}",
                    priority=i+2,
                    metadata={"type": "keyword", "keyword": keyword}
                )
            )
        
        # Execute searches
        for query in queries[:2]:  # Limit to 2 queries for demo
            print(f"\nExecuting search for: {query.query}")
            start_time = datetime.now()
            
            try:
                results = await search_tool.search(query, num_results=3)
                
                duration = (datetime.now() - start_time).total_seconds()
                print(f"Found {len(results)} results in {duration:.2f} seconds")
                
                # Display results
                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Title: {result.title}")
                    print(f"URL: {result.url}")
                    print(f"Snippet: {result.snippet[:100]}...")
                    print(f"Domain: {result.metadata.get('domain', 'unknown')}")
            
            except Exception as e:
                if demo_mode:
                    print(f"Error: {str(e)}")
                    print("This error may be due to missing API key. Set SERPAPI_API_KEY environment variable.")
                else:
                    print(f"Error executing search: {str(e)}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(demo_web_search())
