"""
Information Extraction and Synthesis Demo

This script demonstrates the use of the InformationExtractor and InformationSynthesizer
classes to extract and synthesize information from web content for research topics.

Example usage:
    python information_extraction_demo.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from research_pipeline import ResearchTopic, WebContent, ExtractedInformation
from information_extraction import InformationExtractor, InformationSynthesizer


# Sample research topic and content
def create_sample_data():
    """Create sample research topic and web content for demonstration."""
    
    # Create a research topic on renewable energy
    topic = ResearchTopic(
        id="topic_002",
        title="Renewable Energy Transition",
        description="Research on the global transition to renewable energy sources, including economic impacts, technological challenges, and policy frameworks",
        keywords=["solar power", "wind energy", "energy storage", "grid integration", "climate policy", "economic impact"]
    )
    
    # Create sample web contents from different sources
    contents = [
        WebContent(
            url="https://example.com/renewable-economics",
            title="Economic Implications of Renewable Energy Transition",
            content=(
                "The global transition to renewable energy represents both economic challenges and opportunities. "
                "Initial investment costs for renewable infrastructure are high, but long-term operational costs are significantly lower than fossil fuels. "
                "Studies indicate that renewable energy projects create more jobs per unit of energy generated than traditional fossil fuel industries. "
                "For example, solar power creates approximately 7.7 jobs per megawatt compared to 1.1 jobs in coal power. "
                "Government subsidies and carbon pricing mechanisms remain important policy tools to accelerate adoption. "
                "The levelized cost of electricity (LCOE) for solar and wind has decreased by over 70% in the past decade, making renewables increasingly competitive. "
                "However, transitioning economies dependent on fossil fuel exports face significant economic disruption and require supportive policies for diversification. "
                "Investment in renewable energy reached $282.2 billion globally in 2019, with China and the United States leading in total investment. "
                "Market analysts project the renewable energy sector to grow at a CAGR of 8.4% between 2020 and 2025."
            ),
            timestamp=datetime.now(),
            metadata={"domain": "example.com", "word_count": 150, "publication_date": "2023-04-15"}
        ),
        WebContent(
            url="https://example.org/energy-technology",
            title="Technological Challenges in Renewable Energy Systems",
            content=(
                "Despite significant advances, renewable energy systems continue to face several technological challenges. "
                "Energy storage remains a critical limitation for intermittent sources like solar and wind. "
                "Current battery technologies are inadequate for long-duration grid storage needs, though pumped hydro storage offers some solutions where geographically feasible. "
                "Grid integration requires substantial modernization of existing power infrastructure, including smart grid technologies and improved transmission capabilities. "
                "The efficiency of commercial solar panels typically ranges from 15-22%, though laboratory prototypes have achieved over 47% efficiency. "
                "Wind turbines face design limitations related to the Betz limit, which theoretically caps efficiency at 59.3%. "
                "Materials science innovations are needed for next-generation solar cells, particularly to replace rare earth elements with more abundant materials. "
                "Hydrogen production through electrolysis is becoming more viable for energy storage, but efficiency losses in conversion remain problematic. "
                "The intermittency of renewable sources requires advanced forecasting systems and complementary generation sources."
            ),
            timestamp=datetime.now(),
            metadata={"domain": "example.org", "word_count": 145, "publication_date": "2023-05-20"}
        ),
        WebContent(
            url="https://example.net/climate-policy",
            title="Policy Frameworks for Renewable Energy Adoption",
            content=(
                "Effective policy frameworks are essential for accelerating renewable energy adoption. "
                "Feed-in tariffs (FITs) have proven successful in many European countries, guaranteeing fixed payments for renewable electricity generation. "
                "Renewable Portfolio Standards (RPS) require utilities to source a percentage of energy from renewables, driving market growth in the United States. "
                "Carbon pricing mechanisms, including carbon taxes and cap-and-trade systems, help internalize the external costs of fossil fuels. "
                "The European Union's Emissions Trading System (EU ETS) is the world's largest carbon market and has contributed to emissions reductions. "
                "Many countries have established net-zero emissions targets, with varying timelines from 2030 to 2060. "
                "Permitting and regulatory barriers often slow renewable project development, with approval processes taking 2-5 years in many jurisdictions. "
                "International climate agreements, particularly the Paris Agreement, have established frameworks for national commitments to emissions reductions. "
                "Public opinion increasingly favors stronger climate policies, with 64% of people globally considering climate change a global emergency."
            ),
            timestamp=datetime.now(),
            metadata={"domain": "example.net", "word_count": 140, "publication_date": "2023-06-10"}
        ),
    ]
    
    return topic, contents


async def extract_and_synthesize():
    """Demonstrate information extraction and synthesis."""
    
    print("Information Extraction and Synthesis Demo")
    print("=" * 80)
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: No OpenAI API key found in environment variable OPENAI_API_KEY.")
        print("   The demo will use fallback methods with limited capabilities.")
        print("   Set the OPENAI_API_KEY environment variable for full functionality.\n")
    
    # Create sample data
    topic, contents = create_sample_data()
    
    print(f"Research Topic: {topic.title}")
    print(f"Description: {topic.description}")
    print(f"Keywords: {', '.join(topic.keywords)}")
    print("-" * 80)
    
    # Initialize the extractor and synthesizer
    extractor = InformationExtractor(openai_api_key=api_key)
    synthesizer = InformationSynthesizer(openai_api_key=api_key)
    
    print(f"Processing {len(contents)} web content sources...")
    
    # Extract information from each content source
    all_extracted_info = []
    
    for i, content in enumerate(contents):
        print(f"\nSource {i+1}: {content.title}")
        print(f"URL: {content.url}")
        
        # Extract information using all extraction types
        start_time = datetime.now()
        extracted_info = await extractor.extract_information(
            content, topic, extraction_types=["facts", "claims", "summary"]
        )
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Extracted {len(extracted_info)} information items in {extraction_time:.2f} seconds")
        
        # Print sample of extracted information
        if extracted_info:
            print("\nSample extracted information:")
            # Show top 3 most relevant items
            top_items = sorted(extracted_info, key=lambda x: x.relevance_score, reverse=True)[:3]
            for j, info in enumerate(top_items):
                print(f"  [{info.relevance_score:.2f}] {info.text[:100]}..." if len(info.text) > 100 else f"  [{info.relevance_score:.2f}] {info.text}")
        
        # Add to collection
        all_extracted_info.extend(extracted_info)
    
    print("\n" + "=" * 80)
    print(f"Total extracted information items: {len(all_extracted_info)}")
    
    # Synthesize the information
    print("\nSynthesizing information...\n")
    start_time = datetime.now()
    
    # Perform different types of synthesis
    synthesis_types = ["summary", "comparison", "analysis"]
    
    for synthesis_type in synthesis_types:
        print(f"\n{synthesis_type.upper()} SYNTHESIS")
        print("-" * 80)
        
        synthesis_result = await synthesizer.synthesize_information(
            topic, all_extracted_info, synthesis_type=synthesis_type
        )
        
        if synthesis_type == "summary":
            print(f"\n{synthesis_result.get('summary', 'No summary generated')}")
            
            if "key_findings" in synthesis_result and synthesis_result["key_findings"]:
                print("\nKey Findings:")
                for finding in synthesis_result["key_findings"]:
                    print(f"• {finding}")
                    
            if "knowledge_gaps" in synthesis_result and synthesis_result["knowledge_gaps"]:
                print("\nKnowledge Gaps:")
                for gap in synthesis_result["knowledge_gaps"]:
                    print(f"• {gap}")
                    
        elif synthesis_type == "comparison":
            print(f"\n{synthesis_result.get('comparison_summary', 'No comparison generated')}")
            
            if "consensus_points" in synthesis_result and synthesis_result["consensus_points"]:
                print("\nConsensus Points:")
                for point in synthesis_result["consensus_points"]:
                    print(f"• {point}")
                    
            if "disagreements" in synthesis_result and synthesis_result["disagreements"]:
                print("\nDisagreements or Contradictions:")
                for disagree in synthesis_result["disagreements"]:
                    print(f"• {disagree}")
                    
        elif synthesis_type == "analysis":
            print(f"\n{synthesis_result.get('analysis', 'No analysis generated')}")
            
            if "key_insights" in synthesis_result and synthesis_result["key_insights"]:
                print("\nKey Insights:")
                for insight in synthesis_result["key_insights"]:
                    print(f"• {insight}")
                    
            if "implications" in synthesis_result and synthesis_result["implications"]:
                print("\nImplications:")
                for implication in synthesis_result["implications"]:
                    print(f"• {implication}")
    
    synthesis_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"Synthesis completed in {synthesis_time:.2f} seconds")
    print("\nDemo completed!")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher")
        sys.exit(1)
        
    # Run the demonstration
    asyncio.run(extract_and_synthesize())
