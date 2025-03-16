"""
Content Extraction Demo

This script demonstrates the use of the ContentExtractor class
to extract content from web pages using Playwright.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

from content_extraction import ContentExtractor
from research_pipeline import SearchResult


# Sample search results with different types of websites
SAMPLE_SEARCH_RESULTS = [
    SearchResult(
        title="Artificial Intelligence - Wikipedia",
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        snippet="Artificial intelligence (AI) is intelligence demonstrated by machines...",
        position=1,
        metadata={"query": "artificial intelligence", "domain": "wikipedia.org"}
    ),
    SearchResult(
        title="NASA - National Aeronautics and Space Administration",
        url="https://www.nasa.gov/",
        snippet="NASA's website for information on space exploration, science, and technology...",
        position=2,
        metadata={"query": "nasa space exploration", "domain": "nasa.gov"}
    ),
    SearchResult(
        title="Climate Change: Vital Signs of the Planet - NASA",
        url="https://climate.nasa.gov/",
        snippet="Information about climate change, global warming, and NASA's climate research...",
        position=3,
        metadata={"query": "climate change data", "domain": "climate.nasa.gov"}
    )
]


async def extract_and_display(url_index: int = 0):
    """
    Extract content from a sample URL and display the results.
    
    Args:
        url_index: Index of the sample URL to use
    """
    if url_index < 0 or url_index >= len(SAMPLE_SEARCH_RESULTS):
        print(f"Error: URL index {url_index} is out of range. Must be between 0 and {len(SAMPLE_SEARCH_RESULTS) - 1}.")
        return
        
    result = SAMPLE_SEARCH_RESULTS[url_index]
    
    print(f"\n{'=' * 80}")
    print(f"Content Extraction Demo: {result.title}")
    print(f"{'=' * 80}")
    print(f"URL: {result.url}")
    print(f"Query: {result.metadata.get('query', 'N/A')}")
    print(f"{'=' * 80}\n")
    
    # Create the content extractor
    extractor = ContentExtractor(headless=True)
    
    try:
        print(f"Extracting content from: {result.url}")
        print("This may take a few moments...\n")
        
        start_time = datetime.now()
        content = await extractor.extract_content(result)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if content:
            print(f"Extraction successful! (Took {elapsed_time:.2f} seconds)")
            print(f"\nTitle: {content.title}")
            print(f"Content length: {len(content.content)} characters")
            print(f"Word count: {content.metadata.get('word_count', 'unknown')} words")
            if "estimated_reading_time_minutes" in content.metadata:
                print(f"Estimated reading time: {content.metadata['estimated_reading_time_minutes']} minutes")
                
            # Print content preview
            max_preview_len = 500
            content_preview = content.content[:max_preview_len]
            if len(content.content) > max_preview_len:
                content_preview += "..."
                
            print(f"\n{'-' * 40}")
            print("Content Preview:")
            print(f"{'-' * 40}")
            print(content_preview)
            print(f"{'-' * 40}")
            
            # Print metadata
            print("\nMetadata:")
            metadata_str = json.dumps(content.metadata, indent=2, default=str)
            print(metadata_str)
            
        else:
            print("Extraction failed.")
            
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        
    finally:
        # Close the browser
        await extractor.close()
        print("\nDemo completed")


async def extract_multiple():
    """Extract content from multiple sample URLs."""
    print("\nExtracting content from multiple URLs...")
    
    # Create a single extractor instance for all extractions
    extractor = ContentExtractor(headless=True)
    
    try:
        results = []
        
        # Process each sample URL
        for i, result in enumerate(SAMPLE_SEARCH_RESULTS):
            print(f"\nProcessing URL {i+1}/{len(SAMPLE_SEARCH_RESULTS)}: {result.url}")
            
            start_time = datetime.now()
            content = await extractor.extract_content(result)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            if content:
                print(f"✅ Extraction successful! (Took {elapsed_time:.2f} seconds)")
                print(f"   Title: {content.title}")
                print(f"   Content length: {len(content.content)} characters")
                print(f"   Word count: {content.metadata.get('word_count', 'unknown')} words")
                
                # Store results for summary
                results.append({
                    "url": result.url,
                    "title": content.title,
                    "word_count": content.metadata.get("word_count", 0),
                    "extraction_time": elapsed_time
                })
            else:
                print(f"❌ Extraction failed for: {result.url}")
                
        # Print summary
        if results:
            print("\n" + "=" * 60)
            print("Extraction Summary")
            print("=" * 60)
            
            total_time = sum(r["extraction_time"] for r in results)
            total_words = sum(r["word_count"] for r in results)
            
            print(f"Total URLs processed: {len(results)}/{len(SAMPLE_SEARCH_RESULTS)}")
            print(f"Total extraction time: {total_time:.2f} seconds")
            print(f"Total word count: {total_words} words")
            print(f"Average extraction time: {total_time/len(results):.2f} seconds per URL")
            
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        
    finally:
        # Close the browser
        await extractor.close()
        print("\nMulti-extraction demo completed")


async def main():
    """Main demo function."""
    print("Content Extraction Demo")
    print("=" * 60)
    print("This demo shows how the ContentExtractor extracts content from web pages.")
    print("It will demonstrate extraction from several sample URLs.")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "multiple":
            await extract_multiple()
        else:
            try:
                url_index = int(sys.argv[1])
                await extract_and_display(url_index)
            except ValueError:
                print(f"Error: Invalid URL index. Must be a number between 0 and {len(SAMPLE_SEARCH_RESULTS) - 1}.")
    else:
        # Default to extracting from the first URL
        await extract_and_display(0)


if __name__ == "__main__":
    asyncio.run(main())
