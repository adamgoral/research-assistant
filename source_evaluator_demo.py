"""
Source Evaluator Demo

This script demonstrates the usage of the SourceEvaluator class
with both mock data and real content when available.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Optional

from research_pipeline import (
    SearchQuery,
    SearchResult,
    WebContent,
    ExtractedInformation,
    SourceEvaluation,
    SourceCredibility
)
from source_evaluator import SourceEvaluator

# Optional imports for real content testing
try:
    from web_search import WebSearchTool
    from content_retriever import ContentRetriever
    REAL_COMPONENTS_AVAILABLE = True
except ImportError:
    REAL_COMPONENTS_AVAILABLE = False
    print("Web search or content retriever modules not available. Using mock data only.")


# Create mock data for testing
def create_mock_content() -> List[tuple]:
    """Create mock web content and extracted information for testing."""
    
    # Sample domains to test
    domains = [
        "wikipedia.org",
        "nytimes.com", 
        "example.com",
        "harvard.edu",
        "blogspot.com"
    ]
    
    mock_data = []
    
    for i, domain in enumerate(domains):
        # Create mock web content
        url = f"https://www.{domain}/article-{i+1}"
        title = f"Sample Article {i+1} from {domain}"
        
        # Create different content lengths and characteristics
        if "edu" in domain:
            content = """
            This is a detailed academic article with extensive research and citations.
            The study methodology includes statistical analysis and peer review.
            Several references are cited throughout the text to support the findings.
            Tables and figures present the data in structured format.
            The conclusion summarizes the implications and suggests areas for future research.
            
            References:
            1. Smith, J. (2022). Related research in the field.
            2. Johnson, A. et al. (2021). Supporting evidence from previous studies.
            3. Williams, R. (2023). Theoretical framework and methodology.
            """
        elif "news" in domain:
            content = """
            Breaking news article reporting on recent developments.
            The article includes quotes from experts and eyewitnesses.
            Facts and figures are presented to provide context.
            The story covers multiple perspectives on the issue.
            
            By: Jane Reporter
            Published: January 15, 2023
            """
        else:
            content = f"""
            This is a basic web article about a topic.
            It contains some information that may be relevant.
            The article is from {domain} which has varying levels of credibility.
            """
        
        # Create mock HTML that reflects the content characteristics
        html = f"""
        <html>
            <head>
                <title>{title}</title>
                <meta name="author" content="Sample Author {i+1}">
                <meta name="publication_date" content="2023-01-{i+1:02d}">
            </head>
            <body>
                <h1>{title}</h1>
                <div class="content">
                    {content.replace('\n', '<br>')}
                </div>
                {'<div class="references">References section with citations</div>' if 'edu' in domain else ''}
                {'<table><tr><td>Data point 1</td><td>Value 1</td></tr></table>' if i % 2 == 0 else ''}
            </body>
        </html>
        """
        
        web_content = WebContent(
            url=url,
            title=title,
            content=content.strip(),
            html=html.strip(),
            metadata={
                "domain": domain,
                "word_count": len(content.split()),
                "author": f"Sample Author {i+1}" if i % 2 == 0 else None,
                "published_date": f"2023-01-{i+1:02d}" if i % 3 == 0 else None
            }
        )
        
        # Create mock extracted information
        extracted_info = []
        num_paragraphs = min(5, max(2, len(content.strip().split('\n\n'))))
        
        for j, paragraph in enumerate(content.strip().split('\n')[:num_paragraphs]):
            paragraph = paragraph.strip()
            if paragraph:
                # Different relevance scores based on domain type
                base_relevance = 0.9 if 'edu' in domain else 0.7 if 'news' in domain else 0.5
                relevance = base_relevance - (j * 0.1)  # Decreasing relevance for later paragraphs
                
                extracted_info.append(
                    ExtractedInformation(
                        text=paragraph,
                        source_url=url,
                        source_title=title,
                        relevance_score=max(0.1, min(1.0, relevance)),
                        metadata={
                            "paragraph_index": j,
                            "word_count": len(paragraph.split()),
                            "domain": domain
                        }
                    )
                )
        
        mock_data.append((web_content, extracted_info))
    
    return mock_data


async def test_with_mock_data():
    """Test the SourceEvaluator with mock data."""
    print("\n=== Testing Source Evaluator with Mock Data ===\n")
    
    # Create mock data
    mock_data = create_mock_content()
    
    # Create the source evaluator
    evaluator = SourceEvaluator(
        use_llm=False,
        enable_external_checks=False
    )
    
    print(f"Evaluating {len(mock_data)} mock sources...\n")
    
    # Evaluate each source
    for i, (content, info) in enumerate(mock_data):
        print(f"--- Source {i+1}: {content.url} ---")
        
        evaluation = await evaluator.evaluate_source(content, info)
        
        print(f"Title: {evaluation.title}")
        print(f"Domain: {content.metadata['domain']}")
        print(f"Credibility: {evaluation.credibility.value}")
        print(f"Relevance score: {evaluation.relevance_score:.2f}")
        print(f"Domain authority: {evaluation.domain_authority:.2f}")
        
        if evaluation.author_credentials:
            print(f"Author: {evaluation.author_credentials}")
            
        if evaluation.publication_date:
            print(f"Publication date: {evaluation.publication_date.isoformat()[:10]}")
            
        # Print key evaluation factors
        print("\nKey evaluation factors:")
        for key, value in evaluation.evaluation_factors.items():
            if key in ["content_length", "domain_type", "link_count", "references_present"]:
                print(f"- {key}: {value}")
                
        print("\n")
    
    # Now test the parallel evaluation
    print("--- Testing parallel evaluation of all sources ---")
    
    start_time = datetime.now()
    evaluations = await evaluator.evaluate_multiple_sources(mock_data)
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Completed parallel evaluation of {len(evaluations)} sources in {duration:.2f} seconds")
    
    # Show a summary of results
    print("\nSummary of evaluations:")
    credibility_counts = {cred.value: 0 for cred in SourceCredibility}
    
    for eval in evaluations:
        credibility_counts[eval.credibility.value] += 1
        
    print(f"Credibility distribution:")
    for cred, count in credibility_counts.items():
        if count > 0:
            print(f"- {cred}: {count} sources")
    
    print(f"Average relevance: {sum(e.relevance_score for e in evaluations) / len(evaluations):.2f}")


async def test_with_real_data():
    """Test the SourceEvaluator with real data if available."""
    if not REAL_COMPONENTS_AVAILABLE:
        print("\nSkipping real data test - required components not available")
        return
        
    print("\n=== Testing Source Evaluator with Real Data ===\n")
    
    # Create a sample query and search
    query = SearchQuery(
        query="climate change effects on agriculture",
        priority=1,
        metadata={"type": "test"}
    )
    
    try:
        # Search for real content
        search_tool = WebSearchTool()
        results = await search_tool.search(query, num_results=2)
        
        if not results:
            print("No search results found. Skipping real data test.")
            return
            
        # Retrieve content
        content_retriever = ContentRetriever(headless=True)
        try:
            # Process each result
            for i, result in enumerate(results):
                print(f"\nProcessing search result {i+1}: {result.title}")
                
                # Retrieve content
                content = await content_retriever.retrieve_content(result)
                
                if not content:
                    print(f"Failed to retrieve content from {result.url}")
                    continue
                    
                # Create mock extracted information for demonstration
                extracted_info = []
                paragraphs = [p.strip() for p in content.content.split('\n\n') if p.strip()]
                
                for j, paragraph in enumerate(paragraphs[:3]):  # Just use first 3 paragraphs
                    extracted_info.append(
                        ExtractedInformation(
                            text=paragraph,
                            source_url=content.url,
                            source_title=content.title,
                            relevance_score=0.8 - (j * 0.1),  # Simple relevance score
                            metadata={
                                "paragraph_index": j,
                                "word_count": len(paragraph.split())
                            }
                        )
                    )
                
                # Create the source evaluator
                evaluator = SourceEvaluator(
                    use_llm=False,
                    enable_external_checks=False
                )
                
                # Evaluate the source
                evaluation = await evaluator.evaluate_source(content, extracted_info)
                
                print(f"Title: {evaluation.title}")
                print(f"URL: {evaluation.url}")
                print(f"Credibility: {evaluation.credibility.value}")
                print(f"Relevance score: {evaluation.relevance_score:.2f}")
                print(f"Domain authority: {evaluation.domain_authority:.2f}")
                
                if evaluation.author_credentials:
                    print(f"Author: {evaluation.author_credentials}")
                    
                if evaluation.publication_date:
                    print(f"Publication date: {evaluation.publication_date.isoformat()[:10]}")
                    
                # Print key evaluation factors
                print("\nKey evaluation factors:")
                for key, value in evaluation.evaluation_factors.items():
                    if key in ["content_length", "domain_type", "link_count", "references_present"]:
                        print(f"- {key}: {value}")
        
        finally:
            # Close the content retriever
            await content_retriever.close()
            
    except Exception as e:
        print(f"Error in real data test: {str(e)}")


async def test_llm_evaluation():
    """
    Test LLM-based evaluation (placeholder for now).
    In a real implementation, this would use OpenAI or similar to evaluate content.
    """
    print("\n=== LLM-Based Evaluation (Placeholder) ===\n")
    
    # Create mock data
    mock_data = create_mock_content()[0]  # Just use the first mock item
    content, _ = mock_data
    
    print(f"This would evaluate the content from {content.url} using an LLM.")
    print("In a production implementation, this would:")
    print("1. Send content to GPT-4o for analysis")
    print("2. Evaluate objectivity, factuality, bias, etc.")
    print("3. Assess information quality and authority")
    print("4. Return detailed credibility metrics")
    
    # Mock LLM result (placeholder)
    mock_llm_result = {
        "objectivity_score": 0.85,
        "fact_based_score": 0.78,
        "bias_indicators": ["minimal bias detected", "balanced presentation"],
        "authority_indicators": ["cites primary sources", "includes expert quotes"],
        "depth_of_analysis": "comprehensive",
        "llm_confidence": 0.92
    }
    
    print("\nExample LLM evaluation output:")
    print(json.dumps(mock_llm_result, indent=2))


async def main():
    """Run all tests."""
    print("Source Evaluator Demo")
    print("=" * 50)
    
    # Test with mock data
    await test_with_mock_data()
    
    # Test with real data if available
    await test_with_real_data()
    
    # Test LLM evaluation
    await test_llm_evaluation()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
