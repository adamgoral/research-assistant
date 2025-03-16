# LLM Selection and Configuration

## Requirements Analysis

| Requirement | Description | Priority |
|-------------|-------------|----------|
| Reasoning | Complex multi-step planning and research synthesis | High |
| Knowledge | Broad general knowledge for contextualizing research | Medium |
| Accuracy | High factual precision in information processing | High |
| Tool Use | Sophisticated tool usage for web interaction | High |
| Verbosity | Detailed output for comprehensive reports | Medium |
| Latency | Acceptable response time for interactive sessions | Medium |
| Cost | Cost-effectiveness for production deployment | Medium |

## Model Comparison

| Model | Reasoning | Knowledge | Tool Use | Accuracy | Cost | Latency |
|-------|-----------|-----------|----------|----------|------|---------|
| GPT-4 Turbo | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★☆ | $$$$ | ★★★☆☆ |
| Claude 3 Opus | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | $$$$ | ★★★☆☆ |
| Claude 3 Sonnet | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | $$$ | ★★★★☆ |
| GPT-4o | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★☆ | $$$ | ★★★★☆ |
| Mistral Large | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | $$ | ★★★★☆ |
| Llama 3 70B | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | $$ | ★★★★☆ |

## Selected Model: GPT-4o
Based on the requirements analysis and comparative evaluation, GPT-4o provides the optimal balance of capabilities for this application:
- Strong reasoning for complex research tasks
- Excellent tool use capabilities for web interaction
- Good cost-to-performance ratio compared to GPT-4 Turbo and Claude Opus
- Acceptable latency for interactive research sessions

## Model Configuration

```python
# Primary Model Configuration
PRIMARY_MODEL = {
    "model": "gpt-4o",
    "temperature": 0.2,  # Low temperature for factual outputs
    "max_tokens": 4096,  # Sufficient for detailed research synthesis
    "top_p": 0.95,
    "system_message": """You are a professional research assistant. Your task is to conduct thorough web research on topics provided by users and compile comprehensive, fact-based reports with proper citations. Focus on credible sources, maintain objectivity, and verify information across multiple sources when possible. Structure information clearly and prioritize accuracy over speculation."""
}

# Task-Specific Configurations

SEARCH_QUERY_MODEL = {
    "model": "gpt-3.5-turbo",  # More cost-effective for simpler tasks
    "temperature": 0.3,
    "max_tokens": 256,
    "system_message": """Generate effective search queries based on research questions. Create queries that will yield relevant, authoritative results. Generate multiple query variations to ensure comprehensive coverage."""
}

CONTENT_EVALUATION_MODEL = {
    "model": "gpt-4o",
    "temperature": 0.1,  # Very low temperature for critical evaluation
    "max_tokens": 2048,
    "system_message": """Critically evaluate content for relevance, credibility, and information value. Assess source authority, identify potential biases, and rate factual reliability. Extract key information and provide a confidence score for extracted facts."""
}

REPORT_GENERATION_MODEL = {
    "model": "gpt-4o",
    "temperature": 0.4,  # Slightly higher temperature for narrative flow
    "max_tokens": 8192,  # Extended for comprehensive reports
    "system_message": """Generate comprehensive research reports based on provided information. Structure content logically, create clear narratives, include proper citations, and maintain a professional, objective tone. Synthesize information rather than simply aggregating it."""
}
```

## Prompt Engineering Strategy

### System Message Components
1. **Role definition**: Professional researcher with focus on accuracy
2. **Task description**: Web research and report writing
3. **Value alignment**: Credibility, objectivity, verification
4. **Output guidelines**: Structure, citations, fact-based

### Input Formatting
```
TOPIC: {research_topic}
PARAMETERS:
- Depth: {depth_level}
- Audience: {audience_type}
- Format: {report_format}
- Length: {report_length}

EXISTING KNOWLEDGE:
{known_information}

SEARCH CONTEXT:
{previous_search_results}

INSTRUCTION:
{specific_task_instruction}
```

### Chain-of-Thought Prompting
For complex research synthesis, implement chain-of-thought prompting:
```
To create an effective research plan for "{research_topic}":

1. First, let's identify the key aspects of this topic that require investigation.
2. For each aspect, let's determine the most authoritative sources likely to have relevant information.
3. Let's formulate specific questions that will help uncover comprehensive information.
4. Finally, let's create a structured outline for our research approach.

Research plan:
```

## Fallback and Redundancy Strategy
- Implement parallel queries to multiple models for critical evaluations
- Use GPT-3.5 Turbo as fallback for non-critical tasks during high load or API issues
- Maintain Claude 3 Sonnet as secondary model for verification of controversial facts
