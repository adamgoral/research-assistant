"""
Information Extraction Module

This module implements the InformationExtractor class that uses GPT-4o
to analyze web content and extract relevant information for the Research Assistant.

It provides a real implementation to replace the mock InformationExtractor
in the research_pipeline.py.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import openai
from pydantic import BaseModel, Field
from research_pipeline import ResearchTopic, WebContent, ExtractedInformation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("information_extraction")


class ExtractionPromptTemplate(BaseModel):
    """Model for storing prompt templates for information extraction."""
    
    extract_facts: str = Field(
        default=(
            "You are an expert researcher analyzing web content to extract relevant information "
            "for a research topic. Extract factual, objective information from the provided content "
            "that is relevant to the topic. Focus on extracting standalone facts, data points, and "
            "important information. Extract each relevant piece of information as a separate item.\n\n"
            "Research Topic: {topic_title}\n"
            "Topic Description: {topic_description}\n"
            "Keywords: {keywords}\n\n"
            "Content to analyze:\n{content}\n\n"
            "Provide your output as a JSON array of objects with the following structure:\n"
            "[\n"
            "  {{\n"
            "    \"text\": \"The extracted factual information\",\n"
            "    \"relevance_score\": A number between 0 and 1 indicating relevance to the topic,\n"
            "    \"reasoning\": \"Brief explanation of why this information is relevant\",\n"
            "    \"key_entities\": [\"Array\", \"of\", \"key\", \"entities\", \"mentioned\"]\n"
            "  }}\n"
            "]\n\n"
            "Ensure each extracted item:\n"
            "1. Is factual and objective (not opinions unless clearly attributed)\n"
            "2. Is relevant to the research topic\n"
            "3. Is substantial and meaningful (not trivial facts)\n"
            "4. Is a complete thought or data point that can stand on its own\n"
            "5. Has an accurate relevance score (higher for direct relevance, lower for tangential relevance)\n\n"
            "If no relevant information is found, return an empty array."
        )
    )
    
    extract_claims: str = Field(
        default=(
            "You are an expert researcher analyzing a web page to identify claims related "
            "to the research topic. Focus on extracting assertions, statements, arguments, "
            "or positions from the content that are relevant to the topic.\n\n"
            "Research Topic: {topic_title}\n"
            "Topic Description: {topic_description}\n"
            "Keywords: {keywords}\n\n"
            "Content to analyze:\n{content}\n\n"
            "Provide your output as a JSON array of objects with the following structure:\n"
            "[\n"
            "  {{\n"
            "    \"claim\": \"The statement or assertion being made\",\n"
            "    \"relevance_score\": A number between 0 and 1 indicating relevance to the topic,\n"
            "    \"claim_type\": \"One of: factual, interpretive, opinion, prediction, recommendation\",\n"
            "    \"attribution\": \"Source of the claim if mentioned (e.g., 'According to researchers...')\",\n"
            "    \"evidence_provided\": true/false - whether the claim is supported by evidence\n"
            "  }}\n"
            "]\n\n"
            "Only include claims that are relevant to the research topic. If no relevant claims are found, return an empty array."
        )
    )
    
    extract_summary: str = Field(
        default=(
            "You are an expert researcher analyzing a web page to create a concise summary "
            "of its content as it relates to a specific research topic.\n\n"
            "Research Topic: {topic_title}\n"
            "Topic Description: {topic_description}\n"
            "Keywords: {keywords}\n\n"
            "Content to analyze:\n{content}\n\n"
            "Provide your output as a JSON object with the following structure:\n"
            "{{\n"
            "  \"summary\": \"A concise summary of the content as it relates to the research topic\",\n"
            "  \"relevance_score\": A number between 0 and 1 indicating overall relevance to the topic,\n"
            "  \"key_points\": [\"Array\", \"of\", \"key\", \"points\", \"from\", \"the\", \"content\"],\n"
            "  \"main_topic\": \"The main topic of the content\",\n"
            "  \"perspective\": \"The general perspective or stance taken in the content if any\"\n"
            "}}\n\n"
            "If the content is not relevant to the research topic, provide a low relevance score and a brief explanation."
        )
    )


class InformationExtractor:
    """
    Extracts relevant information from retrieved web content using LLM analysis.
    
    This class provides a production implementation to replace the mock
    InformationExtractor in the research pipeline.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        prompt_templates: Optional[ExtractionPromptTemplate] = None,
        batch_size: int = 3,
    ):
        """
        Initialize the InformationExtractor.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use
            temperature: Temperature setting for OpenAI API
            max_tokens: Maximum tokens for OpenAI API response
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Delay between retries in seconds
            prompt_templates: Custom prompt templates for extraction
            batch_size: Number of concurrent API calls
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM-based extraction will not work.")
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prompt_templates = prompt_templates or ExtractionPromptTemplate()
        self.batch_size = batch_size
        
        # Set up OpenAI client if API key is available
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized InformationExtractor with model: {self.model}")
        else:
            self.client = None
            
    async def extract_information(
        self, 
        content: WebContent, 
        topic: ResearchTopic,
        extraction_types: List[str] = ["facts"]
    ) -> List[ExtractedInformation]:
        """
        Extract relevant information from web content.
        
        Args:
            content: The web content to extract information from
            topic: The research topic for relevance matching
            extraction_types: Types of extractions to perform (facts, claims, summary)
            
        Returns:
            A list of ExtractedInformation objects
        """
        logger.info(f"Extracting information from: {content.url}")
        
        if not self.client:
            logger.warning("No OpenAI client available. Using fallback extraction method.")
            return await self._fallback_extraction(content, topic)
            
        # Prepare content text (truncate if too long to fit in context)
        content_text = self._prepare_content_text(content.content)
        
        # Set up extraction tasks based on requested types
        extraction_tasks = []
        
        if "facts" in extraction_types:
            extraction_tasks.append(self._extract_facts(content_text, topic))
            
        if "claims" in extraction_types:
            extraction_tasks.append(self._extract_claims(content_text, topic))
            
        if "summary" in extraction_types:
            extraction_tasks.append(self._extract_summary(content_text, topic))
            
        # Execute extraction tasks concurrently
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Process results
        all_extracted_info = []
        for result in extraction_results:
            if isinstance(result, Exception):
                logger.error(f"Extraction error: {str(result)}")
                continue
                
            if isinstance(result, list):
                all_extracted_info.extend(result)
            elif result:  # Handle single item case (e.g., summary)
                all_extracted_info.append(result)
                
        # Sort by relevance score (descending)
        all_extracted_info.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Extracted {len(all_extracted_info)} information items from: {content.url}")
        return all_extracted_info
        
    async def extract_information_batch(
        self,
        contents: List[Tuple[WebContent, ResearchTopic]],
        extraction_types: List[str] = ["facts"]
    ) -> Dict[str, List[ExtractedInformation]]:
        """
        Extract information from multiple web contents in batches.
        
        Args:
            contents: List of (WebContent, ResearchTopic) tuples
            extraction_types: Types of extractions to perform
            
        Returns:
            Dictionary mapping content URLs to lists of ExtractedInformation
        """
        logger.info(f"Batch extracting information from {len(contents)} web contents")
        
        results = {}
        batch_size = self.batch_size
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            
            # Create tasks for the batch
            tasks = [
                self.extract_information(content, topic, extraction_types)
                for content, topic in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, (content, _) in enumerate(batch):
                result = batch_results[j]
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction error for {content.url}: {str(result)}")
                    results[content.url] = []
                else:
                    results[content.url] = result
                    
        return results
        
    async def _extract_facts(
        self, content_text: str, topic: ResearchTopic
    ) -> List[ExtractedInformation]:
        """
        Extract factual information from content.
        
        Args:
            content_text: Prepared content text
            topic: Research topic
            
        Returns:
            List of ExtractedInformation objects
        """
        # Format prompt with topic information
        prompt = self.prompt_templates.extract_facts.format(
            topic_title=topic.title,
            topic_description=topic.description,
            keywords=", ".join(topic.keywords),
            content=content_text
        )
        
        # Call OpenAI API to extract facts
        try:
            extracted_facts = await self._call_openai_with_retry(prompt)
            
            # Parse the response
            try:
                facts_data = json.loads(extracted_facts)
                
                # Ensure it's a list
                if not isinstance(facts_data, list):
                    logger.warning(f"Expected list but got {type(facts_data)} in facts extraction")
                    return []
                    
                # Convert to ExtractedInformation objects
                result = []
                timestamp = datetime.now()
                
                for fact in facts_data:
                    if not isinstance(fact, dict) or "text" not in fact:
                        continue
                        
                    # Get values with defaults
                    text = fact.get("text", "")
                    relevance_score = fact.get("relevance_score", 0.5)
                    reasoning = fact.get("reasoning", "")
                    entities = fact.get("key_entities", [])
                    
                    # Ensure relevance_score is a float between 0 and 1
                    try:
                        relevance_score = float(relevance_score)
                        relevance_score = max(0.0, min(1.0, relevance_score))
                    except (ValueError, TypeError):
                        relevance_score = 0.5
                        
                    if text:
                        extracted_info = ExtractedInformation(
                            text=text,
                            source_url=topic.id,  # Will be set correctly in main extract_information
                            source_title="",  # Will be set correctly in main extract_information
                            relevance_score=relevance_score,
                            extraction_timestamp=timestamp,
                            metadata={
                                "type": "fact",
                                "reasoning": reasoning,
                                "key_entities": entities,
                                "topic_id": topic.id,
                                "topic_title": topic.title,
                            }
                        )
                        result.append(extracted_info)
                        
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse facts extraction response: {str(e)}")
                logger.debug(f"Response: {extracted_facts}")
                return []
                
        except Exception as e:
            logger.error(f"Error in facts extraction: {str(e)}")
            return []
            
    async def _extract_claims(
        self, content_text: str, topic: ResearchTopic
    ) -> List[ExtractedInformation]:
        """
        Extract claims and assertions from content.
        
        Args:
            content_text: Prepared content text
            topic: Research topic
            
        Returns:
            List of ExtractedInformation objects
        """
        # Format prompt with topic information
        prompt = self.prompt_templates.extract_claims.format(
            topic_title=topic.title,
            topic_description=topic.description,
            keywords=", ".join(topic.keywords),
            content=content_text
        )
        
        # Call OpenAI API to extract claims
        try:
            extracted_claims = await self._call_openai_with_retry(prompt)
            
            # Parse the response
            try:
                claims_data = json.loads(extracted_claims)
                
                # Ensure it's a list
                if not isinstance(claims_data, list):
                    logger.warning(f"Expected list but got {type(claims_data)} in claims extraction")
                    return []
                    
                # Convert to ExtractedInformation objects
                result = []
                timestamp = datetime.now()
                
                for claim in claims_data:
                    if not isinstance(claim, dict) or "claim" not in claim:
                        continue
                        
                    # Get values with defaults
                    text = claim.get("claim", "")
                    relevance_score = claim.get("relevance_score", 0.5)
                    claim_type = claim.get("claim_type", "unknown")
                    attribution = claim.get("attribution", "")
                    evidence_provided = claim.get("evidence_provided", False)
                    
                    # Ensure relevance_score is a float between 0 and 1
                    try:
                        relevance_score = float(relevance_score)
                        relevance_score = max(0.0, min(1.0, relevance_score))
                    except (ValueError, TypeError):
                        relevance_score = 0.5
                        
                    if text:
                        extracted_info = ExtractedInformation(
                            text=text,
                            source_url=topic.id,  # Will be set correctly in main extract_information
                            source_title="",  # Will be set correctly in main extract_information
                            relevance_score=relevance_score,
                            extraction_timestamp=timestamp,
                            metadata={
                                "type": "claim",
                                "claim_type": claim_type,
                                "attribution": attribution,
                                "evidence_provided": evidence_provided,
                                "topic_id": topic.id,
                                "topic_title": topic.title,
                            }
                        )
                        result.append(extracted_info)
                        
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse claims extraction response: {str(e)}")
                logger.debug(f"Response: {extracted_claims}")
                return []
                
        except Exception as e:
            logger.error(f"Error in claims extraction: {str(e)}")
            return []
            
    async def _extract_summary(
        self, content_text: str, topic: ResearchTopic
    ) -> Optional[ExtractedInformation]:
        """
        Extract a summary of the content.
        
        Args:
            content_text: Prepared content text
            topic: Research topic
            
        Returns:
            ExtractedInformation object containing the summary
        """
        # Format prompt with topic information
        prompt = self.prompt_templates.extract_summary.format(
            topic_title=topic.title,
            topic_description=topic.description,
            keywords=", ".join(topic.keywords),
            content=content_text
        )
        
        # Call OpenAI API to extract summary
        try:
            extracted_summary = await self._call_openai_with_retry(prompt)
            
            # Parse the response
            try:
                summary_data = json.loads(extracted_summary)
                
                # Ensure it's a dictionary
                if not isinstance(summary_data, dict):
                    logger.warning(f"Expected dict but got {type(summary_data)} in summary extraction")
                    return None
                    
                # Get values with defaults
                summary_text = summary_data.get("summary", "")
                relevance_score = summary_data.get("relevance_score", 0.5)
                key_points = summary_data.get("key_points", [])
                main_topic = summary_data.get("main_topic", "")
                perspective = summary_data.get("perspective", "")
                
                # Ensure relevance_score is a float between 0 and 1
                try:
                    relevance_score = float(relevance_score)
                    relevance_score = max(0.0, min(1.0, relevance_score))
                except (ValueError, TypeError):
                    relevance_score = 0.5
                    
                if summary_text:
                    extracted_info = ExtractedInformation(
                        text=summary_text,
                        source_url=topic.id,  # Will be set correctly in main extract_information
                        source_title="",  # Will be set correctly in main extract_information
                        relevance_score=relevance_score,
                        extraction_timestamp=datetime.now(),
                        metadata={
                            "type": "summary",
                            "key_points": key_points,
                            "main_topic": main_topic,
                            "perspective": perspective,
                            "topic_id": topic.id,
                            "topic_title": topic.title,
                        }
                    )
                    return extracted_info
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse summary extraction response: {str(e)}")
                logger.debug(f"Response: {extracted_summary}")
                return None
                
        except Exception as e:
            logger.error(f"Error in summary extraction: {str(e)}")
            return None
            
    async def _call_openai_with_retry(self, prompt: str) -> str:
        """
        Call OpenAI API with retry logic.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The API response text
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant that extracts relevant information from web content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract and return the content
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"OpenAI API call failed: {str(e)}. Retrying in {delay:.2f}s... ({attempt+1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Failed to call OpenAI API after {self.max_retries} retries: {str(e)}")
                    
    def _prepare_content_text(self, content: str, max_length: int = 12000) -> str:
        """
        Prepare content text for analysis, truncating if necessary.
        
        Args:
            content: Raw content text
            max_length: Maximum character length
            
        Returns:
            Prepared content text
        """
        # Clean up text
        content = content.strip()
        
        # Truncate if too long
        if len(content) > max_length:
            logger.info(f"Truncating content from {len(content)} to {max_length} characters")
            # Try to truncate at paragraph boundary
            truncated = content[:max_length]
            last_paragraph = truncated.rfind("\n\n")
            if last_paragraph > max_length * 0.8:  # If we can find a paragraph break in the last 20%
                truncated = content[:last_paragraph]
            return truncated + "\n\n[Content truncated due to length...]"
        
        return content
        
    async def _fallback_extraction(
        self, content: WebContent, topic: ResearchTopic
    ) -> List[ExtractedInformation]:
        """
        Fallback method for extraction when LLM is not available.
        Uses keyword matching and heuristics.
        
        Args:
            content: The web content to extract from
            topic: The research topic
            
        Returns:
            List of ExtractedInformation objects
        """
        logger.info("Using fallback extraction method (keyword-based)")
        
        # Extract paragraphs from content
        paragraphs = [p.strip() for p in content.content.split("\n\n") if p.strip()]
        
        # Skip very short paragraphs
        paragraphs = [p for p in paragraphs if len(p.split()) >= 10]
        
        # Create a set of topic keywords (case insensitive)
        keywords = set([topic.title.lower()] + [k.lower() for k in topic.keywords])
        
        # Extract information from paragraphs
        extracted_info = []
        timestamp = datetime.now()
        
        for i, paragraph in enumerate(paragraphs):
            # Calculate relevance based on keyword matches
            paragraph_lower = paragraph.lower()
            
            # Count keyword occurrences
            keyword_count = sum(1 for kw in keywords if kw in paragraph_lower)
            
            # Calculate relevance score (simple heuristic)
            relevance_score = min(1.0, 0.3 + (0.2 * keyword_count))
            
            # Boost score for paragraphs with topic title
            if topic.title.lower() in paragraph_lower:
                relevance_score = min(1.0, relevance_score + 0.2)
                
            # Only include sufficiently relevant paragraphs
            if relevance_score >= 0.4:
                extracted_info.append(
                    ExtractedInformation(
                        text=paragraph,
                        source_url=content.url,
                        source_title=content.title,
                        relevance_score=relevance_score,
                        extraction_timestamp=timestamp,
                        metadata={
                            "type": "fallback_extraction",
                            "paragraph_index": i,
                            "keyword_matches": keyword_count,
                            "word_count": len(paragraph.split()),
                            "topic_id": topic.id,
                            "topic_title": topic.title,
                        }
                    )
                )
                
        # Sort by relevance (descending)
        extracted_info.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Fallback extraction found {len(extracted_info)} relevant paragraphs")
        return extracted_info


class InformationSynthesizer:
    """
    Synthesizes extracted information into cohesive knowledge.
    
    This class combines information from multiple sources, identifies
    relationships, and creates a unified view of the research topic.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        max_retries: int = 3,
    ):
        """
        Initialize the InformationSynthesizer.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use
            temperature: Temperature setting for OpenAI API
            max_tokens: Maximum tokens for OpenAI API response
            max_retries: Maximum number of retry attempts for failed API calls
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM-based synthesis will not work.")
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        # Set up OpenAI client if API key is available
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized InformationSynthesizer with model: {self.model}")
        else:
            self.client = None
            
    async def synthesize_information(
        self,
        topic: ResearchTopic,
        information_items: List[ExtractedInformation],
        synthesis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Synthesize extracted information into cohesive knowledge.
        
        Args:
            topic: The research topic
            information_items: List of extracted information items
            synthesis_type: Type of synthesis to perform (summary, comparison, analysis)
            
        Returns:
            Dictionary containing the synthesized information
        """
        logger.info(f"Synthesizing {len(information_items)} information items for topic: {topic.title}")
        
        if not self.client:
            logger.warning("No OpenAI client available. Using fallback synthesis method.")
            return self._fallback_synthesis(topic, information_items, synthesis_type)
            
        if not information_items:
            logger.warning("No information items to synthesize")
            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "synthesis_type": synthesis_type,
                "synthesis_text": f"No information available for {topic.title}",
                "metadata": {
                    "information_count": 0,
                    "synthesis_timestamp": datetime.now().isoformat(),
                }
            }
            
        # Prepare information items for synthesis
        formatted_items = self._format_information_items(information_items)
        
        # Choose synthesis method based on type
        if synthesis_type == "summary":
            result = await self._synthesize_summary(topic, formatted_items)
        elif synthesis_type == "comparison":
            result = await self._synthesize_comparison(topic, formatted_items)
        elif synthesis_type == "analysis":
            result = await self._synthesize_analysis(topic, formatted_items)
        else:
            logger.warning(f"Unknown synthesis type: {synthesis_type}, using summary instead")
            result = await self._synthesize_summary(topic, formatted_items)
            
        # Add metadata
        result.update({
            "topic_id": topic.id,
            "topic_title": topic.title,
            "synthesis_type": synthesis_type,
            "metadata": {
                "information_count": len(information_items),
                "synthesis_timestamp": datetime.now().isoformat(),
            }
        })
        
        logger.info(f"Completed {synthesis_type} synthesis for topic: {topic.title}")
        return result
        
    def _format_information_items(
        self, information_items: List[ExtractedInformation]
    ) -> str:
        """
        Format information items for synthesis input.
        
        Args:
            information_items: List of extracted information items
            
        Returns:
            Formatted information text
        """
        formatted_text = []
        
        # Group by source
        sources = {}
        for item in information_items:
            if item.source_url not in sources:
                sources[item.source_url] = {
                    "title": item.source_title,
                    "items": []
                }
            sources[item.source_url]["items"].append(item)
            
        # Format each source's information
        for url, source_data in sources.items():
            formatted_text.append(f"Source: {source_data['title']} ({url})")
            
            # Sort items by relevance
            items = sorted(source_data["items"], key=lambda x: x.relevance_score, reverse=True)
            
            # Add each item with its relevance score
            for item in items:
                formatted_text.append(f"[Relevance: {item.relevance_score:.2f}] {item.text}")
                
            formatted_text.append("")  # Empty line between sources
            
        return "\n".join(formatted_text)
        
    async def _synthesize_summary(
        self, topic: ResearchTopic, formatted_items: str
    ) -> Dict[str, Any]:
        """
        Synthesize information into a comprehensive summary.
        
        Args:
            topic: Research topic
            formatted_items: Formatted information items
            
        Returns:
            Dictionary with synthesis results
        """
        prompt = (
            f"You are an expert researcher synthesizing information for a research report. "
            f"Create a comprehensive summary of the following information related to the research topic.\n\n"
            f"Research Topic: {topic.title}\n"
            f"Topic Description: {topic.description}\n"
            f"Keywords: {', '.join(topic.keywords)}\n\n"
            f"Information Items:\n{formatted_items}\n\n"
            f"Synthesize this information into a coherent, well-organized summary that provides "
            f"a comprehensive overview of the research topic. Focus on key findings, patterns, "
            f"and significant information. Maintain objectivity and factual accuracy.\n\n"
            f"Format your response as a JSON object with the following structure:\n"
            f"{{\n"
            f"  \"summary\": \"The comprehensive summary text\",\n"
            f"  \"key_findings\": [\"Array\", \"of\", \"key\", \"findings\"],\n"
            f"  \"knowledge_gaps\": [\"Array\", \"of\", \"identified\", \"knowledge\", \"gaps\"],\n"
            f"  \"source_diversity\": \"Assessment of the diversity of sources (low, medium, high)\"\n"
            f"}}"
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse synthesis response: {str(e)}")
                # Fallback to using the raw text
                return {
                    "summary": response_text,
                    "key_findings": [],
                    "knowledge_gaps": [],
                    "source_diversity": "unknown"
                }
                
        except Exception as e:
            logger.error(f"Error in summary synthesis: {str(e)}")
            return {
                "summary": f"Error synthesizing information: {str(e)}",
                "key_findings": [],
                "knowledge_gaps": [],
                "source_diversity": "unknown"
            }
            
    async def _synthesize_comparison(
        self, topic: ResearchTopic, formatted_items: str
    ) -> Dict[str, Any]:
        """
        Synthesize information with a focus on comparing different perspectives.
        
        Args:
            topic: Research topic
            formatted_items: Formatted information items
            
        Returns:
            Dictionary with synthesis results
        """
        prompt = (
            f"You are an expert researcher comparing information from different sources. "
            f"Analyze the following information related to the research topic and identify "
            f"similarities, differences, and varying perspectives.\n\n"
            f"Research Topic: {topic.title}\n"
            f"Topic Description: {topic.description}\n"
            f"Keywords: {', '.join(topic.keywords)}\n\n"
            f"Information Items:\n{formatted_items}\n\n"
            f"Compare the information from different sources, identifying areas of consensus, "
            f"contradictions, and different perspectives. Highlight key points of agreement and "
            f"disagreement. Maintain objectivity and avoid favoring any particular perspective.\n\n"
            f"Format your response as a JSON object with the following structure:\n"
            f"{{\n"
            f"  \"comparison_summary\": \"Overall comparison of the information\",\n"
            f"  \"consensus_points\": [\"Array\", \"of\", \"points\", \"with\", \"consensus\"],\n"
            f"  \"disagreements\": [\"Array\", \"of\", \"points\", \"with\", \"disagreement\"],\n"
            f"  \"perspectives\": [\"Array\", \"of\", \"different\", \"perspectives\", \"identified\"]\n"
            f"}}"
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that compares information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse comparison response: {str(e)}")
                # Fallback to using the raw text
                return {
                    "comparison_summary": response_text,
                    "consensus_points": [],
                    "disagreements": [],
                    "perspectives": []
                }
                
        except Exception as e:
            logger.error(f"Error in comparison synthesis: {str(e)}")
            return {
                "comparison_summary": f"Error synthesizing comparison: {str(e)}",
                "consensus_points": [],
                "disagreements": [],
                "perspectives": []
            }
            
    async def _synthesize_analysis(
        self, topic: ResearchTopic, formatted_items: str
    ) -> Dict[str, Any]:
        """
        Synthesize information with a focus on critical analysis.
        
        Args:
            topic: Research topic
            formatted_items: Formatted information items
            
        Returns:
            Dictionary with synthesis results
        """
        prompt = (
            f"You are an expert researcher analyzing information for a research topic. "
            f"Critically analyze the following information and identify key insights, "
            f"trends, implications, and potential biases.\n\n"
            f"Research Topic: {topic.title}\n"
            f"Topic Description: {topic.description}\n"
            f"Keywords: {', '.join(topic.keywords)}\n\n"
            f"Information Items:\n{formatted_items}\n\n"
            f"Analyze this information critically, identifying patterns, relationships, and implications. "
            f"Assess the quality and reliability of the information, and note any gaps or limitations. "
            f"Maintain analytical rigor and objectivity in your assessment.\n\n"
            f"Format your response as a JSON object with the following structure:\n"
            f"{{\n"
            f"  \"analysis\": \"Critical analysis of the information\",\n"
            f"  \"key_insights\": [\"Array\", \"of\", \"key\", \"insights\"],\n"
            f"  \"trends\": [\"Array\", \"of\", \"identified\", \"trends\"],\n"
            f"  \"limitations\": [\"Array\", \"of\", \"identified\", \"limitations\", \"or\", \"biases\"],\n"
            f"  \"implications\": [\"Array\", \"of\", \"implications\", \"or\", \"applications\"]\n"
            f"}}"
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that analyzes information critically."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis response: {str(e)}")
                # Fallback to using the raw text
                return {
                    "analysis": response_text,
                    "key_insights": [],
                    "trends": [],
                    "limitations": [],
                    "implications": []
                }
                
        except Exception as e:
            logger.error(f"Error in analysis synthesis: {str(e)}")
            return {
                "analysis": f"Error synthesizing analysis: {str(e)}",
                "key_insights": [],
                "trends": [],
                "limitations": [],
                "implications": []
            }
            
    def _fallback_synthesis(
        self, 
        topic: ResearchTopic, 
        information_items: List[ExtractedInformation],
        synthesis_type: str
    ) -> Dict[str, Any]:
        """
        Fallback method for synthesis when LLM is not available.
        Uses simple aggregation and filtering.
        
        Args:
            topic: Research topic
            information_items: List of extracted information items
            synthesis_type: Type of synthesis requested
            
        Returns:
            Dictionary with synthesized information
        """
        logger.info("Using fallback synthesis method (aggregation-based)")
        
        if not information_items:
            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "synthesis_type": synthesis_type,
                "synthesis_text": f"No information available for {topic.title}",
                "metadata": {
                    "information_count": 0,
                    "synthesis_timestamp": datetime.now().isoformat(),
                }
            }
            
        # Sort by relevance score (descending)
        sorted_items = sorted(information_items, key=lambda x: x.relevance_score, reverse=True)
        
        # Get top items (most relevant)
        top_items = sorted_items[:min(10, len(sorted_items))]
        
        # Group by source
        sources = {}
        for item in information_items:
            if item.source_url not in sources:
                sources[item.source_url] = {
                    "title": item.source_title,
                    "items": []
                }
            sources[item.source_url]["items"].append(item)
            
        # Count sources and items
        source_count = len(sources)
        item_count = len(information_items)
        
        # Create a simple synthesis based on type
        synthesis_text = f"Information synthesis for: {topic.title}\n\n"
        
        if synthesis_type == "summary":
            synthesis_text += f"Summary of {item_count} information items from {source_count} sources:\n\n"
            
            # Add top items
            for i, item in enumerate(top_items):
                synthesis_text += f"{i+1}. {item.text}\n"
                
            # Add basic stats
            synthesis_text += f"\nInformation was gathered from {source_count} different sources."
            
            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "synthesis_type": "summary",
                "synthesis_text": synthesis_text,
                "summary": synthesis_text,
                "key_findings": [item.text[:100] + "..." for item in top_items[:5]],
                "knowledge_gaps": ["Full analysis not available without LLM"],
                "source_diversity": "low" if source_count < 3 else "medium" if source_count < 6 else "high",
                "metadata": {
                    "information_count": item_count,
                    "source_count": source_count,
                    "synthesis_timestamp": datetime.now().isoformat(),
                    "fallback_method": True
                }
            }
            
        elif synthesis_type == "comparison":
            synthesis_text += f"Comparison of information from {source_count} sources:\n\n"
            
            # Add sources and their top items
            for url, source_data in sources.items():
                source_items = sorted(source_data["items"], key=lambda x: x.relevance_score, reverse=True)
                top_source_items = source_items[:min(3, len(source_items))]
                
                synthesis_text += f"Source: {source_data['title']}\n"
                for item in top_source_items:
                    synthesis_text += f"- {item.text}\n"
                synthesis_text += "\n"
                
            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "synthesis_type": "comparison",
                "synthesis_text": synthesis_text,
                "comparison_summary": synthesis_text,
                "consensus_points": ["Full comparison not available without LLM"],
                "disagreements": ["Full comparison not available without LLM"],
                "perspectives": [f"Information from {source_data['title']}" for _, source_data in sources.items()],
                "metadata": {
                    "information_count": item_count,
                    "source_count": source_count,
                    "synthesis_timestamp": datetime.now().isoformat(),
                    "fallback_method": True
                }
            }
            
        else:  # analysis or fallback
            synthesis_text += f"Analysis of {item_count} information items from {source_count} sources:\n\n"
            
            # Add top items with their relevance scores
            for i, item in enumerate(top_items):
                synthesis_text += f"{i+1}. [Relevance: {item.relevance_score:.2f}] {item.text}\n"
                
            # Add some basic analysis
            avg_relevance = sum(item.relevance_score for item in information_items) / len(information_items)
            synthesis_text += f"\nAverage relevance score: {avg_relevance:.2f}"
            synthesis_text += f"\nInformation was gathered from {source_count} different sources."
            
            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "synthesis_type": "analysis",
                "synthesis_text": synthesis_text,
                "analysis": synthesis_text,
                "key_insights": [item.text[:100] + "..." for item in top_items[:3]],
                "trends": ["Full trend analysis not available without LLM"],
                "limitations": [
                    "Limited synthesis capability without LLM",
                    f"Analysis based on {source_count} sources only"
                ],
                "implications": ["Full implications analysis not available without LLM"],
                "metadata": {
                    "information_count": item_count,
                    "source_count": source_count,
                    "synthesis_timestamp": datetime.now().isoformat(),
                    "fallback_method": True
                }
            }


async def test_information_extraction():
    """Test the InformationExtractor with sample content."""
    print("InformationExtractor Test")
    print("-" * 50)
    
    # Create a sample topic
    topic = ResearchTopic(
        id="topic_001",
        title="Climate change impacts on agriculture",
        description="Research the effects of climate change on agricultural productivity and food security",
        keywords=["crop yields", "temperature rise", "food security", "adaptation strategies", "drought"]
    )
    
    # Create a sample web content
    content = WebContent(
        url="https://example.com/climate-agriculture",
        title="Climate Change and Global Agriculture",
        content=(
            "Climate change poses significant threats to global agriculture. "
            "Rising temperatures are affecting crop yields worldwide. "
            "Studies show that for each degree Celsius of warming, wheat yields may decrease by 6%. "
            "Farmers in developing countries are particularly vulnerable to these changes. "
            "Adaptation strategies include developing drought-resistant crops and improving irrigation systems. "
            "Food security remains a major concern as climate patterns become more unpredictable. "
            "Some regions may actually see temporary increases in yields due to longer growing seasons. "
            "However, the overall global impact is expected to be negative for most major crops. "
            "Extreme weather events like floods and droughts are becoming more frequent, disrupting agricultural production. "
            "Agricultural researchers are working to develop crops that can withstand these changing conditions."
        ),
        metadata={"domain": "example.com", "word_count": 120}
    )
    
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key found in environment. Using fallback extraction method.")
        
    # Create the extractor
    extractor = InformationExtractor(openai_api_key=api_key)
    
    # Extract information
    print(f"Extracting information from: {content.title}")
    extracted_info = await extractor.extract_information(
        content, topic, extraction_types=["facts", "claims", "summary"]
    )
    
    # Print results
    print(f"\nExtracted {len(extracted_info)} information items:")
    for i, info in enumerate(extracted_info[:5]):  # Show top 5 items
        print(f"\nItem {i+1} [Relevance: {info.relevance_score:.2f}]")
        print(f"Type: {info.metadata.get('type', 'unknown')}")
        print(f"Text: {info.text}")
        
    # Test synthesis if API key is available
    if api_key:
        print("\n" + "-" * 50)
        print("Testing InformationSynthesizer")
        
        # Create the synthesizer
        synthesizer = InformationSynthesizer(openai_api_key=api_key)
        
        # Synthesize information
        print(f"Synthesizing information for topic: {topic.title}")
        synthesis = await synthesizer.synthesize_information(
            topic, extracted_info, synthesis_type="summary"
        )
        
        # Print results
        print("\nSynthesis Results:")
        print(f"Type: {synthesis.get('synthesis_type', 'unknown')}")
        
        if "summary" in synthesis:
            print(f"\nSummary:\n{synthesis['summary']}")
            
        if "key_findings" in synthesis:
            print("\nKey Findings:")
            for finding in synthesis.get("key_findings", [])[:3]:
                print(f"- {finding}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    # Run test function
    asyncio.run(test_information_extraction())
