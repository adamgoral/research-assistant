"""
Web Search Integration Module

This module implements the WebSearchTool class that integrates with SerpAPI
to provide real web search capabilities for the Research Assistant.

It replaces the mock implementation in the research_pipeline.py with
actual web search functionality.
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import aiohttp
from serpapi import GoogleSearch

# Import data models from research_pipeline
from research_pipeline import SearchQuery, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("web_search")


class WebSearchTool:
    """
    Performs web searches using SerpAPI.
    
    This is the production implementation of the WebSearchTool class
    from the research_pipeline, replacing the mock implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WebSearchTool with the SerpAPI key.
        
        Args:
            api_key: SerpAPI key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        if not self.api_key:
            logger.warning(
                "SerpAPI key not provided. Please set SERPAPI_API_KEY environment variable "
                "or pass api_key to WebSearchTool constructor."
            )
        
    async def search(
        self, query: SearchQuery, num_results: int = 5
    ) -> List[SearchResult]:
        """
        Execute a web search for the given query using SerpAPI.
        
        Args:
            query: The search query to execute
            num_results: Maximum number of results to return
            
        Returns:
            A list of SearchResult objects
        """
        if not self.api_key:
            raise ValueError(
                "SerpAPI key not provided. Please set SERPAPI_API_KEY environment variable "
                "or pass api_key to WebSearchTool constructor."
            )
            
        logger.info(f"Executing search for query: {query.query}")
        
        try:
            # SerpAPI parameters
            params = {
                "q": query.query,
                "num": num_results,
                "api_key": self.api_key,
                "engine": "google",
            }
            
            # Execute search
            search = GoogleSearch(params)
            search_results = search.get_dict()
            
            # Check for errors
            if "error" in search_results:
                logger.error(f"SerpAPI error: {search_results['error']}")
                return []
                
            # Extract organic results
            organic_results = search_results.get("organic_results", [])
            
            # Convert to SearchResult objects
            results = []
            for i, result in enumerate(organic_results[:num_results]):
                title = result.get("title", "Untitled")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                
                # Extract domain from link
                domain = link.split("//")[-1].split("/")[0] if link else ""
                
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        position=i + 1,
                        metadata={
                            "domain": domain,
                            "query": query.query,
                            "search_timestamp": datetime.now().isoformat(),
                        }
                    )
                )
                
            logger.info(f"Found {len(results)} results for query: {query.query}")
            return results
            
        except Exception as e:
            logger.error(f"Error executing search query {query.query}: {str(e)}")
            return []
            
    async def search_async(
        self, query: SearchQuery, num_results: int = 5
    ) -> List[SearchResult]:
        """
        Execute a web search asynchronously using aiohttp.
        
        This method uses aiohttp to make async HTTP requests to SerpAPI,
        which can be more efficient when making multiple search requests.
        
        Args:
            query: The search query to execute
            num_results: Maximum number of results to return
            
        Returns:
            A list of SearchResult objects
        """
        if not self.api_key:
            raise ValueError(
                "SerpAPI key not provided. Please set SERPAPI_API_KEY environment variable "
                "or pass api_key to WebSearchTool constructor."
            )
            
        logger.info(f"Executing async search for query: {query.query}")
        
        try:
            # SerpAPI parameters
            params = {
                "q": query.query,
                "num": num_results,
                "api_key": self.api_key,
                "engine": "google",
            }
            
            serpapi_url = "https://serpapi.com/search"
            
            # Make async HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.get(serpapi_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"SerpAPI error ({response.status}): {error_text}")
                        return []
                        
                    search_results = await response.json()
                    
            # Check for errors
            if "error" in search_results:
                logger.error(f"SerpAPI error: {search_results['error']}")
                return []
                
            # Extract organic results
            organic_results = search_results.get("organic_results", [])
            
            # Convert to SearchResult objects
            results = []
            for i, result in enumerate(organic_results[:num_results]):
                title = result.get("title", "Untitled")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                
                # Extract domain from link
                domain = link.split("//")[-1].split("/")[0] if link else ""
                
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        position=i + 1,
                        metadata={
                            "domain": domain,
                            "query": query.query,
                            "search_timestamp": datetime.now().isoformat(),
                        }
                    )
                )
                
            logger.info(f"Found {len(results)} results for query: {query.query}")
            return results
            
        except Exception as e:
            logger.error(f"Error executing async search query {query.query}: {str(e)}")
            return []


async def test_web_search():
    """Test the WebSearchTool with a sample query."""
    # Get API key from environment
    api_key = os.environ.get("SERPAPI_API_KEY")
    
    if not api_key:
        print("Error: SERPAPI_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        return
        
    # Create a sample query
    query = SearchQuery(
        query="climate change effects on agriculture",
        priority=1,
        metadata={"type": "test"}
    )
    
    # Create the WebSearchTool
    search_tool = WebSearchTool(api_key=api_key)
    
    # Execute search
    print(f"Searching for: {query.query}")
    results = await search_tool.search(query, num_results=5)
    
    # Display results
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Snippet: {result.snippet}")
        print(f"Domain: {result.metadata.get('domain', 'unknown')}")


if __name__ == "__main__":
    # Run test function
    asyncio.run(test_web_search())
