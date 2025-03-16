"""
Content Retriever Module

This module implements the ContentRetriever class that integrates with
the ContentExtractor to provide web content retrieval for the Research Assistant.

It replaces the mock ContentRetriever in the research_pipeline.py with
actual content extraction functionality.
"""

import asyncio
import logging
from typing import Optional

from research_pipeline import WebContent, SearchResult
from content_extraction import ContentExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("content_retriever")


class ContentRetriever:
    """
    Retrieves and extracts content from web pages.
    
    This is a production implementation that replaces the mock ContentRetriever
    in the research_pipeline with actual extraction using Playwright.
    """
    
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        max_retries: int = 2,
        parallel_extractions: int = 3
    ):
        """
        Initialize the ContentRetriever.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Navigation timeout in milliseconds
            max_retries: Maximum number of retry attempts for failed requests
            parallel_extractions: Maximum number of parallel extractions
        """
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.parallel_extractions = parallel_extractions
        self._extractor = None
        
    async def initialize(self):
        """Initialize the content extractor."""
        if self._extractor is None:
            self._extractor = ContentExtractor(
                headless=self.headless,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
            
    async def close(self):
        """Close the content extractor."""
        if self._extractor:
            await self._extractor.close()
            self._extractor = None
            
    async def retrieve_content(self, result: SearchResult) -> Optional[WebContent]:
        """
        Retrieve and extract content from a search result URL.
        
        Args:
            result: The search result to retrieve content from
            
        Returns:
            A WebContent object with the extracted content, or None if retrieval failed
        """
        await self.initialize()
        
        logger.info(f"Retrieving content from URL: {result.url}")
        
        try:
            content = await self._extractor.extract_content(result)
            
            if content:
                logger.info(f"Successfully retrieved content from URL: {result.url}")
                return content
            else:
                logger.warning(f"Failed to retrieve content from URL: {result.url}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving content from URL: {result.url} - {str(e)}")
            return None
            
    async def retrieve_multiple(self, results: list[SearchResult]) -> list[WebContent]:
        """
        Retrieve content from multiple search results in parallel.
        
        Args:
            results: List of search results to retrieve content from
            
        Returns:
            List of WebContent objects for successfully retrieved content
        """
        logger.info(f"Retrieving content from {len(results)} URLs")
        
        # Initialize the content extractor
        await self.initialize()
        
        # Process search results in batches for parallel extraction
        contents = []
        semaphore = asyncio.Semaphore(self.parallel_extractions)
        
        async def process_result(result: SearchResult):
            async with semaphore:
                return await self.retrieve_content(result)
                
        # Create tasks for all results
        tasks = [process_result(result) for result in results]
        
        # Execute all tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and failed retrievals
        for content in results:
            if content is not None and not isinstance(content, Exception):
                contents.append(content)
                
        logger.info(f"Successfully retrieved content from {len(contents)}/{len(results)} URLs")
        
        return contents


async def test_content_retriever():
    """Test the ContentRetriever with a sample URL."""
    print("ContentRetriever Test")
    print("-" * 50)
    
    # Create a sample search result
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    result = SearchResult(
        title="Artificial Intelligence - Wikipedia",
        url=test_url,
        snippet="Artificial intelligence (AI) is intelligence demonstrated by machines...",
        position=1,
        metadata={"query": "artificial intelligence", "domain": "wikipedia.org"}
    )
    
    # Create the ContentRetriever
    retriever = ContentRetriever(headless=True)
    
    try:
        print(f"Retrieving content from: {test_url}")
        content = await retriever.retrieve_content(result)
        
        if content:
            print("\nRetrieval successful:")
            print(f"Title: {content.title}")
            print(f"Content length: {len(content.content)} characters")
            print(f"Word count: {content.metadata.get('word_count', 'unknown')}")
            print(f"\nFirst 300 characters of content:")
            print(content.content[:300] + "...")
        else:
            print("Retrieval failed")
    finally:
        await retriever.close()
        print("\nTest completed")


async def test_parallel_retrieval():
    """Test parallel content retrieval with multiple URLs."""
    print("Parallel Content Retrieval Test")
    print("-" * 50)
    
    # Create sample search results
    results = [
        SearchResult(
            title="Artificial Intelligence - Wikipedia",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            snippet="Artificial intelligence (AI) is intelligence demonstrated by machines...",
            position=1,
            metadata={"query": "artificial intelligence", "domain": "wikipedia.org"}
        ),
        SearchResult(
            title="Python (programming language) - Wikipedia",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            snippet="Python is a high-level, general-purpose programming language...",
            position=2,
            metadata={"query": "python programming", "domain": "wikipedia.org"}
        ),
        SearchResult(
            title="NASA - National Aeronautics and Space Administration",
            url="https://www.nasa.gov/",
            snippet="NASA's website for information on space exploration...",
            position=3,
            metadata={"query": "nasa", "domain": "nasa.gov"}
        )
    ]
    
    # Create the ContentRetriever with parallel processing
    retriever = ContentRetriever(headless=True, parallel_extractions=3)
    
    try:
        print(f"Retrieving content from {len(results)} URLs in parallel...")
        
        start_time = asyncio.get_event_loop().time()
        contents = await retriever.retrieve_multiple(results)
        end_time = asyncio.get_event_loop().time()
        
        print(f"\nRetrieval complete. Time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully retrieved {len(contents)}/{len(results)} pages")
        
        for i, content in enumerate(contents):
            print(f"\nContent {i+1}:")
            print(f"Title: {content.title}")
            print(f"URL: {content.url}")
            print(f"Content length: {len(content.content)} characters")
            print(f"Word count: {content.metadata.get('word_count', 'unknown')}")
    finally:
        await retriever.close()
        print("\nTest completed")


if __name__ == "__main__":
    # Run test functions
    asyncio.run(test_content_retriever())
    # Uncomment to test parallel retrieval
    # asyncio.run(test_parallel_retrieval())
