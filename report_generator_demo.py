"""
Report Generator Module Demo

This script demonstrates the functionality of the Report Generator module,
showing how to generate structured, well-cited reports from research results.

The demo showcases:
1. Creating different report configurations
2. Generating reports with various formats and styles
3. Using different citation styles
4. Customizing report structure based on report type
5. Saving reports in multiple output formats
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from report_generator import (
    AudienceLevel,
    CitationStyle,
    ReportConfiguration,
    ReportFormat,
    ReportGenerator,
    ReportType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("report_generator_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("report_generator_demo")


async def load_sample_research_results(file_path=None):
    """
    Load or generate sample research results for the demo.
    
    Args:
        file_path: Optional path to a JSON file with research results
        
    Returns:
        Dictionary containing research results
    """
    # If a file path is provided and exists, load from file
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading research results from {file_path}: {str(e)}")
            # Fall through to generate sample data
    
    # Generate sample research results
    logger.info("Generating sample research results")
    
    # Sample research results similar to what would come from the research pipeline
    return {
        "topic_id": "topic_climate_agriculture",
        "total_information_items": 25,
        "sources": [
            {
                "url": "https://example.com/article1",
                "title": "Understanding Climate Change Impacts on Agriculture",
                "evaluation": {
                    "credibility": "high",
                    "relevance_score": 0.95,
                    "domain_authority": 0.8,
                    "author_credentials": "Dr. Jane Smith",
                    "publication_date": "2023-01-15"
                },
                "information": [
                    {
                        "text": "Climate change is affecting agricultural productivity worldwide, with some regions experiencing greater impacts than others.",
                        "relevance_score": 0.95
                    },
                    {
                        "text": "Rising temperatures and changing precipitation patterns are the primary climate factors impacting crop yields.",
                        "relevance_score": 0.92
                    }
                ]
            },
            {
                "url": "https://example.org/research/agriculture",
                "title": "Crop Yields and Temperature Increases: A Global Study",
                "evaluation": {
                    "credibility": "high",
                    "relevance_score": 0.88,
                    "domain_authority": 0.75,
                    "author_credentials": "James Thompson et al.",
                    "publication_date": "2022-11-22"
                },
                "information": [
                    {
                        "text": "Studies show a 10% decrease in crop yields for each degree Celsius increase in global temperature.",
                        "relevance_score": 0.88
                    },
                    {
                        "text": "Regions near the equator are expected to experience the most significant crop yield reductions.",
                        "relevance_score": 0.85
                    }
                ]
            },
            {
                "url": "https://agri-research.edu/climate",
                "title": "Adaptation Strategies for Food Security in a Changing Climate",
                "evaluation": {
                    "credibility": "high",
                    "relevance_score": 0.82,
                    "domain_authority": 0.9,
                    "author_credentials": "Agricultural Research Institute",
                    "publication_date": "2023-03-10"
                },
                "information": [
                    {
                        "text": "Adaptation strategies including drought-resistant crops and improved water management are critical for future food security.",
                        "relevance_score": 0.82
                    },
                    {
                        "text": "Early adoption of climate-smart agricultural practices can mitigate up to 40% of climate-related productivity losses.",
                        "relevance_score": 0.79
                    }
                ]
            },
            {
                "url": "https://university.edu/research/climate-food",
                "title": "Food Security Challenges Under Climate Change Scenarios",
                "evaluation": {
                    "credibility": "high",
                    "relevance_score": 0.78,
                    "domain_authority": 0.85,
                    "author_credentials": "Prof. Robert Chen, PhD",
                    "publication_date": "2023-02-05"
                },
                "information": [
                    {
                        "text": "Global food security faces unprecedented challenges due to climate change, population growth, and changing consumption patterns.",
                        "relevance_score": 0.78
                    },
                    {
                        "text": "Climate change may reduce global food production by up to 30% by 2050 if current trends continue without intervention.",
                        "relevance_score": 0.76
                    }
                ]
            },
            {
                "url": "https://climate-institute.org/agriculture-report",
                "title": "Economic Impacts of Climate Change on Agricultural Systems",
                "evaluation": {
                    "credibility": "medium",
                    "relevance_score": 0.75,
                    "domain_authority": 0.7,
                    "author_credentials": "Climate Institute",
                    "publication_date": "2022-09-18"
                },
                "information": [
                    {
                        "text": "Annual economic losses in the agricultural sector attributed to climate change could reach $500 billion by 2050.",
                        "relevance_score": 0.75
                    },
                    {
                        "text": "Developing countries with agriculture-dependent economies are projected to experience the most severe economic impacts.",
                        "relevance_score": 0.72
                    }
                ]
            }
        ]
    }


async def generate_standard_report(research_results):
    """Generate a standard report with APA citations."""
    logger.info("Generating standard report with APA citations")
    
    # Create a report configuration
    config = ReportConfiguration(
        title="Climate Change Impacts on Agriculture",
        report_type=ReportType.STANDARD_REPORT,
        audience_level=AudienceLevel.PROFESSIONAL,
        citation_style=CitationStyle.APA,
        output_formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        include_executive_summary=True,
        include_table_of_contents=True,
        include_references=True
    )
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate and save the report
    output_dir = "output/reports/standard"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = await generator.generate_and_save_report(
        research_results, config, output_dir
    )
    
    return output_files


async def generate_analytical_report(research_results):
    """Generate a detailed analytical report with IEEE citations."""
    logger.info("Generating analytical report with IEEE citations")
    
    # Create a report configuration
    config = ReportConfiguration(
        title="Comprehensive Analysis of Climate Change Effects on Agricultural Productivity",
        report_type=ReportType.ANALYTICAL_REPORT,
        audience_level=AudienceLevel.ACADEMIC,
        citation_style=CitationStyle.IEEE,
        output_formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        include_executive_summary=True,
        include_table_of_contents=True,
        include_references=True
    )
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate and save the report
    output_dir = "output/reports/analytical"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = await generator.generate_and_save_report(
        research_results, config, output_dir
    )
    
    return output_files


async def generate_executive_summary(research_results):
    """Generate a brief executive summary with minimal citations."""
    logger.info("Generating executive summary report")
    
    # Create a report configuration
    config = ReportConfiguration(
        title="Climate Change and Agriculture: Executive Briefing",
        report_type=ReportType.EXECUTIVE_SUMMARY,
        audience_level=AudienceLevel.PROFESSIONAL,
        citation_style=CitationStyle.APA,
        output_formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        include_executive_summary=False,  # Not needed for executive summary type
        include_table_of_contents=False,  # Too brief to need TOC
        include_references=True
    )
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate and save the report
    output_dir = "output/reports/summary"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = await generator.generate_and_save_report(
        research_results, config, output_dir
    )
    
    return output_files


async def generate_presentation_report(research_results):
    """Generate a presentation-style report with MLA citations."""
    logger.info("Generating presentation-style report with MLA citations")
    
    # Create a report configuration
    config = ReportConfiguration(
        title="Climate Change & Agriculture: Key Findings for Stakeholders",
        report_type=ReportType.PRESENTATION,
        audience_level=AudienceLevel.GENERAL,
        citation_style=CitationStyle.MLA,
        output_formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        include_executive_summary=True,
        include_table_of_contents=True,
        include_references=True
    )
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate and save the report
    output_dir = "output/reports/presentation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = await generator.generate_and_save_report(
        research_results, config, output_dir
    )
    
    return output_files


async def main():
    """Run the demo."""
    print("\n" + "="*50)
    print("REPORT GENERATOR MODULE DEMONSTRATION")
    print("="*50 + "\n")
    
    # Create root output directory
    os.makedirs("output/reports", exist_ok=True)
    
    # Load or generate sample research results
    research_results = await load_sample_research_results()
    print(f"Loaded research data about '{research_results['topic_id']}' with {len(research_results['sources'])} sources")
    print(f"Total information items: {research_results['total_information_items']}")
    print("-"*50)
    
    # Run different report generation examples
    
    # 1. Standard Report with APA citations
    print("\n1. Generating Standard Report (APA citations)")
    standard_files = await generate_standard_report(research_results)
    print(f"Standard report generated successfully in {len(standard_files)} formats:")
    for fmt, path in standard_files.items():
        print(f"  - {fmt}: {path}")
    
    # 2. Analytical Report with IEEE citations
    print("\n2. Generating Analytical Report (IEEE citations)")
    analytical_files = await generate_analytical_report(research_results)
    print(f"Analytical report generated successfully in {len(analytical_files)} formats:")
    for fmt, path in analytical_files.items():
        print(f"  - {fmt}: {path}")
    
    # 3. Executive Summary
    print("\n3. Generating Executive Summary")
    summary_files = await generate_executive_summary(research_results)
    print(f"Executive summary generated successfully in {len(summary_files)} formats:")
    for fmt, path in summary_files.items():
        print(f"  - {fmt}: {path}")
    
    # 4. Presentation Report with MLA citations
    print("\n4. Generating Presentation Report (MLA citations)")
    presentation_files = await generate_presentation_report(research_results)
    print(f"Presentation report generated successfully in {len(presentation_files)} formats:")
    for fmt, path in presentation_files.items():
        print(f"  - {fmt}: {path}")
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)
    print(f"\nAll reports have been saved to the 'output/reports' directory")
    
    # Return statistics about all the reports generated
    total_reports = len(standard_files) + len(analytical_files) + len(summary_files) + len(presentation_files)
    return {
        "total_reports": total_reports,
        "report_types": 4,
        "citation_styles": 3,  # APA, IEEE, MLA
        "output_formats": list(set([fmt for files in [standard_files, analytical_files, summary_files, presentation_files] for fmt in files.keys()]))
    }


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nGenerated {result['total_reports']} reports across {result['report_types']} report types")
    print(f"Citation styles used: {', '.join([style.value for style in [CitationStyle.APA, CitationStyle.IEEE, CitationStyle.MLA]])}")
    print(f"Output formats: {', '.join(result['output_formats'])}")
