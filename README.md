# Research Pipeline Proof of Concept

This repository contains a proof-of-concept implementation of the research pipeline for the Web Research Assistant & Report Writer project.

## Overview

The Research Pipeline is responsible for automating the end-to-end process of conducting web research on user-specified topics. It handles search query generation, web searching, content retrieval, information extraction, source evaluation, and knowledge storage.

This proof-of-concept demonstrates the flow of information through the pipeline and the interfaces between components. It includes mock implementations that simulate the behavior of real components without requiring actual web access or LLM calls.

## Pipeline Components

1. **SearchQueryGenerator**: Generates optimized search queries based on research topics
   - In production: Uses GPT-3.5 Turbo to generate effective search queries
   - In PoC: Generates simple queries based on the topic and keywords

2. **WebSearchTool**: Performs web searches using search APIs
   - In production: Uses SerpAPI to perform actual web searches
   - In PoC: Returns mock search results from simulated domains

3. **ContentRetriever**: Extracts content from web pages
   - In production: Uses Playwright to navigate to web pages and extract content
   - In PoC: Generates mock content with occasional simulated retrieval failures

4. **InformationExtractor**: Extracts relevant information from retrieved content
   - In production: Uses GPT-4o to analyze content and extract relevant information
   - In PoC: Generates mock extracted information with simulated relevance scoring

5. **SourceEvaluator**: Evaluates source credibility and relevance
   - In production: Uses a combination of heuristics and LLM analysis
   - In PoC: Uses simple domain-based heuristics for credibility scoring

6. **KnowledgeBase**: Stores extracted information and source metadata
   - In production: Uses ChromaDB for vector storage and MongoDB for document storage
   - In PoC: Uses in-memory storage with similar interface

## Data Models

The pipeline uses Pydantic models for data validation and serialization:

- **ResearchTopic**: Represents a research topic with title, description, and keywords
- **SearchQuery**: Represents a search query with priority and metadata
- **SearchResult**: Represents a search result with title, URL, and snippet
- **WebContent**: Represents retrieved web content with extracted text and HTML
- **ExtractedInformation**: Represents information extracted from web content
- **SourceEvaluation**: Represents the credibility and relevance evaluation of a source

## Running the Demo

To run the demo, you need Python 3.10+ and the following dependencies:
- pydantic

Install the dependencies:

```bash
pip install pydantic
```

Run the demo:

```bash
python run_research_demo.py
```

Or directly run the research pipeline:

```bash
python research_pipeline.py
```

## Expected Output

The demo will:
1. Create a sample research topic ("Climate change impacts on agriculture")
2. Execute the research pipeline on the topic
3. Print statistics about the research process
4. Retrieve and display sample research results

## Next Steps

This proof-of-concept demonstrates the basic architecture and flow of the research pipeline. Next steps include:

1. Implementing actual search API integration (SerpAPI)
2. Adding Playwright for real web content retrieval
3. Integrating GPT-4o for information extraction and analysis
4. Implementing ChromaDB for vector storage of extracted information
5. Adding MongoDB for structured data storage
6. Developing proper error handling and recovery mechanisms
7. Creating metrics collection for pipeline performance

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Research Topic │────▶│ Query Generator │────▶│   Web Search    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Knowledge Base │◀────│ Source Evaluator│◀────│Content Retriever│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                        ▲                       │
        │                        │                       ▼
        │                ┌─────────────────┐            
        └───────────────│    Information   │◀───────────┘
                        │    Extractor     │
                        └─────────────────┘
