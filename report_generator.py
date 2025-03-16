"""
Report Generation Module for Web Research Assistant

This module implements the report generation component of the Web Research Assistant.
It takes research results from the knowledge base and organizes them into
well-structured, properly cited reports in various formats.

The report generation system consists of:
1. ReportGenerator: Main class for generating research reports
2. CitationManager: Handles citation formatting and reference management
3. OutputFormatter: Converts report data to different output formats (Markdown, PDF, Word)

Key features:
- Hierarchical report structure with sections and subsections
- Automatic citation generation and management
- Support for multiple output formats
- Customizable report templates
- Reading level and tone adjustment
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("report_generator")


# ----- Data Models -----

class AudienceLevel(Enum):
    """Enum representing the target audience level for a report."""
    GENERAL = "general"  # General public, minimal technical jargon
    PROFESSIONAL = "professional"  # Professionals in the field, moderate technical terms
    ACADEMIC = "academic"  # Academic audience, technical and formal
    TECHNICAL = "technical"  # Technical experts, highly specialized language


class ReportFormat(Enum):
    """Enum representing available report formats."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    HTML = "html"


class ReportType(Enum):
    """Enum representing report types with different structures."""
    EXECUTIVE_SUMMARY = "executive_summary"  # Brief overview focused on key findings
    STANDARD_REPORT = "standard_report"  # Standard report with balanced sections
    ANALYTICAL_REPORT = "analytical_report"  # Detailed report with in-depth analysis
    PRESENTATION = "presentation"  # Streamlined for presentation slides


class CitationStyle(Enum):
    """Enum representing citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"


class ReportConfiguration(BaseModel):
    """Model for configuring report generation parameters."""
    title: str
    report_type: ReportType = ReportType.STANDARD_REPORT
    audience_level: AudienceLevel = AudienceLevel.PROFESSIONAL
    citation_style: CitationStyle = CitationStyle.APA
    output_formats: List[ReportFormat] = Field(default_factory=lambda: [ReportFormat.MARKDOWN])
    include_executive_summary: bool = True
    include_table_of_contents: bool = True
    include_references: bool = True
    max_length: Optional[int] = None  # Maximum word count (None for no limit)
    min_sources: int = 5
    tone: str = "informative"  # informative, persuasive, analytical, objective
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    """Model representing a source used in the report."""
    url: str
    title: str
    authors: Optional[List[str]] = None
    publication_date: Optional[datetime] = None
    publication: Optional[str] = None
    credibility: Optional[str] = None
    access_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Model representing a citation within the report text."""
    source_id: str
    text: str
    location: str  # Section ID or location in the document
    context: Optional[str] = None  # Text excerpt where citation appears


class ReportSection(BaseModel):
    """Model representing a section in the report."""
    id: str
    title: str
    content: str
    level: int = 1  # Section depth/hierarchy level
    order: int = 0  # Order within the parent section
    parent_id: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('content')
    def content_not_empty(cls, v):
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError('Section content cannot be empty')
        return v


class Report(BaseModel):
    """Model representing a complete report."""
    id: str
    title: str
    configuration: ReportConfiguration
    sections: List[ReportSection] = Field(default_factory=list)
    sources: Dict[str, Source] = Field(default_factory=dict)
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ----- Citation Manager -----

class CitationManager:
    """
    Manages citations and references in the report.
    
    The CitationManager:
    - Formats citations according to selected style
    - Maintains a list of all sources used in the report
    - Generates formatted references section
    - Tracks in-text citations and their locations
    """
    
    def __init__(self, style: CitationStyle = CitationStyle.APA):
        """
        Initialize the citation manager with the specified style.
        
        Args:
            style: The citation style to use
        """
        self.style = style
        self.sources: Dict[str, Source] = {}
        self.citations: List[Citation] = []
        
    def add_source(self, source: Source) -> str:
        """
        Add a source to the citation manager.
        
        Args:
            source: The source to add
            
        Returns:
            The ID of the added source
        """
        # Generate a unique ID for the source if it doesn't have one
        source_id = source.metadata.get("id")
        if not source_id:
            source_id = str(uuid.uuid4())[:8]
            
        # Check if the source already exists (by URL)
        for existing_id, existing_source in self.sources.items():
            if existing_source.url == source.url:
                logger.info(f"Source already exists: {source.url}")
                return existing_id
                
        # Add the source
        self.sources[source_id] = source
        logger.info(f"Added source: {source.title} ({source_id})")
        return source_id
        
    def add_citation(self, source_id: str, text: str, location: str, context: Optional[str] = None) -> Citation:
        """
        Add a citation to the report.
        
        Args:
            source_id: The ID of the source being cited
            text: The citation text
            location: Where in the report the citation appears
            context: Surrounding text for the citation
            
        Returns:
            The created Citation object
        """
        if source_id not in self.sources:
            raise ValueError(f"Source with ID {source_id} not found")
            
        citation = Citation(
            source_id=source_id,
            text=text,
            location=location,
            context=context
        )
        
        self.citations.append(citation)
        logger.info(f"Added citation from source {source_id} at location {location}")
        return citation
        
    def format_in_text_citation(self, source_id: str) -> str:
        """
        Generate an in-text citation string based on the citation style.
        
        Args:
            source_id: The ID of the source to cite
            
        Returns:
            Formatted in-text citation string
        """
        if source_id not in self.sources:
            raise ValueError(f"Source with ID {source_id} not found")
            
        source = self.sources[source_id]
        
        # Format based on citation style
        if self.style == CitationStyle.APA:
            # APA: (Author, Year)
            authors = source.authors[0] if source.authors and len(source.authors) > 0 else "n.a."
            if source.authors and len(source.authors) > 1:
                authors = f"{source.authors[0]} et al."
            year = source.publication_date.year if source.publication_date else "n.d."
            return f"({authors}, {year})"
            
        elif self.style == CitationStyle.MLA:
            # MLA: (Author page)
            authors = source.authors[0] if source.authors and len(source.authors) > 0 else "n.a."
            return f"({authors})"
            
        elif self.style == CitationStyle.CHICAGO:
            # Chicago: (Author Year, Page)
            authors = source.authors[0] if source.authors and len(source.authors) > 0 else "n.a."
            year = source.publication_date.year if source.publication_date else "n.d."
            return f"({authors} {year})"
            
        elif self.style == CitationStyle.IEEE:
            # IEEE: [number]
            # Find the index of this source in our sources dict
            index = list(self.sources.keys()).index(source_id) + 1
            return f"[{index}]"
            
        elif self.style == CitationStyle.HARVARD:
            # Harvard: (Author, Year)
            authors = source.authors[0] if source.authors and len(source.authors) > 0 else "n.a."
            year = source.publication_date.year if source.publication_date else "n.d."
            return f"({authors}, {year})"
            
        # Default
        return f"({source_id})"
    
    def generate_references_list(self) -> str:
        """
        Generate a references section formatted according to the selected citation style.
        
        Returns:
            Formatted references section as a string
        """
        if not self.sources:
            return "No references cited."
            
        # Sort sources appropriately based on citation style
        if self.style in [CitationStyle.APA, CitationStyle.MLA, CitationStyle.CHICAGO, CitationStyle.HARVARD]:
            # Sort by author name
            sorted_sources = sorted(
                self.sources.items(),
                key=lambda x: (
                    x[1].authors[0] if x[1].authors and len(x[1].authors) > 0 else "zzz",
                    x[1].publication_date.year if x[1].publication_date else 9999
                )
            )
        else:  # IEEE: sorted by order of appearance
            # For IEEE, we'd ideally sort by first appearance in the text
            # But as a simplification, we'll use the current order
            sorted_sources = list(self.sources.items())
            
        references = []
        
        for i, (source_id, source) in enumerate(sorted_sources):
            # Format reference based on citation style
            if self.style == CitationStyle.APA:
                reference = self._format_apa_reference(source)
            elif self.style == CitationStyle.MLA:
                reference = self._format_mla_reference(source)
            elif self.style == CitationStyle.CHICAGO:
                reference = self._format_chicago_reference(source)
            elif self.style == CitationStyle.IEEE:
                reference = self._format_ieee_reference(source, i+1)
            elif self.style == CitationStyle.HARVARD:
                reference = self._format_harvard_reference(source)
            else:
                # Default generic format
                reference = self._format_generic_reference(source)
                
            references.append(reference)
            
        # Join with appropriate separators
        return "\n\n".join(references)
    
    def _format_apa_reference(self, source: Source) -> str:
        """Format a reference in APA style."""
        authors_text = "n.a."
        if source.authors:
            if len(source.authors) == 1:
                authors_text = f"{source.authors[0]}."
            elif len(source.authors) == 2:
                authors_text = f"{source.authors[0]} & {source.authors[1]}."
            else:
                authors_text = f"{source.authors[0]} et al."
                
        year = source.publication_date.year if source.publication_date else "n.d."
        title = source.title
        publication = source.publication if source.publication else "Web"
        
        return f"{authors_text} ({year}). {title}. {publication}. Retrieved from {source.url}"
    
    def _format_mla_reference(self, source: Source) -> str:
        """Format a reference in MLA style."""
        authors_text = "n.a."
        if source.authors:
            if len(source.authors) == 1:
                authors_text = f"{source.authors[0]}."
            elif len(source.authors) == 2:
                authors_text = f"{source.authors[0]} and {source.authors[1]}."
            else:
                authors_text = f"{source.authors[0]} et al."
                
        title = f'"{source.title}".'
        publication = source.publication if source.publication else "Web"
        date = source.publication_date.strftime("%d %b. %Y") if source.publication_date else "n.d."
        access_date = source.access_date.strftime("%d %b. %Y")
        
        return f"{authors_text} {title} {publication}, {date}, {source.url}. Accessed {access_date}."
    
    def _format_chicago_reference(self, source: Source) -> str:
        """Format a reference in Chicago style."""
        authors_text = "n.a."
        if source.authors:
            if len(source.authors) == 1:
                authors_text = f"{source.authors[0]}."
            elif len(source.authors) < 4:
                authors_text = ", ".join(source.authors[:-1]) + f", and {source.authors[-1]}."
            else:
                authors_text = f"{source.authors[0]} et al."
                
        title = f'"{source.title}".'
        publication = source.publication if source.publication else "Web"
        date = source.publication_date.strftime("%B %d, %Y") if source.publication_date else "n.d."
        access_date = source.access_date.strftime("%B %d, %Y")
        
        return f"{authors_text} {title} {publication}, {date}. {source.url}. Accessed {access_date}."
    
    def _format_ieee_reference(self, source: Source, index: int) -> str:
        """Format a reference in IEEE style."""
        authors_text = "n.a."
        if source.authors:
            if len(source.authors) == 1:
                authors_text = f"{source.authors[0]}"
            elif len(source.authors) == 2:
                authors_text = f"{source.authors[0]} and {source.authors[1]}"
            else:
                authors_text = f"{source.authors[0]} et al."
                
        title = f'"{source.title}"'
        publication = source.publication if source.publication else "Online"
        date = source.publication_date.strftime("%b. %Y") if source.publication_date else "n.d."
        
        return f"[{index}] {authors_text}, {title}, {publication}, {date}. Available: {source.url}"
    
    def _format_harvard_reference(self, source: Source) -> str:
        """Format a reference in Harvard style."""
        authors_text = "n.a."
        if source.authors:
            if len(source.authors) == 1:
                authors_text = f"{source.authors[0]}"
            elif len(source.authors) == 2:
                authors_text = f"{source.authors[0]} and {source.authors[1]}"
            else:
                authors_text = f"{source.authors[0]} et al."
                
        year = source.publication_date.year if source.publication_date else "n.d."
        title = source.title
        publication = source.publication if source.publication else "Online"
        access_date = source.access_date.strftime("%d %B %Y")
        
        return f"{authors_text} {year}, {title}, {publication}, viewed {access_date}, <{source.url}>"
    
    def _format_generic_reference(self, source: Source) -> str:
        """Format a reference in a simple generic style."""
        authors = ", ".join(source.authors) if source.authors else "n.a."
        date = source.publication_date.strftime("%Y-%m-%d") if source.publication_date else "n.d."
        title = source.title
        
        return f"{authors} ({date}). {title}. URL: {source.url}"


# ----- Output Formatter -----

class OutputFormatter:
    """
    Converts report data to different output formats.
    
    The OutputFormatter:
    - Transforms report models into formatted output
    - Supports multiple output formats (Markdown, PDF, Word, HTML)
    - Handles template-based formatting
    - Controls layout and styling
    """
    
    def __init__(self):
        """Initialize the output formatter."""
        pass
        
    def format_report(self, report: Report, output_format: ReportFormat) -> str:
        """
        Format a report in the specified output format.
        
        Args:
            report: The report to format
            output_format: The desired output format
            
        Returns:
            The formatted report as a string
        """
        if output_format == ReportFormat.MARKDOWN:
            return self.format_markdown(report)
        elif output_format == ReportFormat.HTML:
            return self.format_html(report)
        elif output_format == ReportFormat.PDF:
            raise NotImplementedError("PDF formatting requires additional dependencies")
        elif output_format == ReportFormat.WORD:
            raise NotImplementedError("Word formatting requires additional dependencies")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def format_markdown(self, report: Report) -> str:
        """
        Format a report as Markdown.
        
        Args:
            report: The report to format
            
        Returns:
            The report formatted as Markdown
        """
        lines = []
        
        # Title
        lines.append(f"# {report.title}")
        lines.append("")
        
        # Executive Summary (if included)
        if report.configuration.include_executive_summary:
            summary_section = self._find_section_by_title(report, "Executive Summary")
            if summary_section:
                lines.append("## Executive Summary")
                lines.append("")
                lines.append(summary_section.content)
                lines.append("")
                
        # Table of Contents (if included)
        if report.configuration.include_table_of_contents:
            lines.append("## Table of Contents")
            lines.append("")
            
            # Generate TOC from sections
            toc_lines = []
            root_sections = [s for s in report.sections if s.parent_id is None and s.title != "Executive Summary" and s.title != "References"]
            root_sections.sort(key=lambda s: s.order)
            
            for section in root_sections:
                toc_lines.append(f"* [{section.title}](#{self._slugify(section.title)})")
                child_sections = [s for s in report.sections if s.parent_id == section.id]
                child_sections.sort(key=lambda s: s.order)
                
                for child in child_sections:
                    toc_lines.append(f"  * [{child.title}](#{self._slugify(child.title)})")
                    
            lines.extend(toc_lines)
            lines.append("")
            
        # Main Content
        main_sections = [s for s in report.sections if s.parent_id is None and s.title != "Executive Summary" and s.title != "References"]
        main_sections.sort(key=lambda s: s.order)
        
        for section in main_sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            
            # Add subsections
            child_sections = [s for s in report.sections if s.parent_id == section.id]
            child_sections.sort(key=lambda s: s.order)
            
            for child in child_sections:
                lines.append(f"### {child.title}")
                lines.append("")
                lines.append(child.content)
                lines.append("")
                
        # References (if included)
        if report.configuration.include_references and report.sources:
            lines.append("## References")
            lines.append("")
            
            # Find references section or generate one
            references_section = self._find_section_by_title(report, "References")
            if references_section:
                lines.append(references_section.content)
            else:
                # Create a citation manager to generate references
                citation_manager = CitationManager(report.configuration.citation_style)
                for source_id, source in report.sources.items():
                    citation_manager.add_source(source)
                
                lines.append(citation_manager.generate_references_list())
                
            lines.append("")
            
        # Join all lines
        return "\n".join(lines)
    
    def format_html(self, report: Report) -> str:
        """
        Format a report as HTML.
        
        Args:
            report: The report to format
            
        Returns:
            The report formatted as HTML
        """
        # First convert to Markdown
        markdown = self.format_markdown(report)
        
        # Basic HTML wrapper with some styling
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        h1 {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }}
        blockquote {{
            border-left: 3px solid #ccc;
            padding-left: 10px;
            color: #555;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }}
        .references {{
            margin-top: 40px;
            border-top: 1px solid #ccc;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="report-content">
        <!-- Simple conversion of markdown to HTML -->
        {self._convert_markdown_to_html(markdown)}
    </div>
</body>
</html>"""
        
        return html
    
    def save_to_file(self, content: str, output_path: str) -> bool:
        """
        Save formatted content to a file.
        
        Args:
            content: The formatted content to save
            output_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Report saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving report to {output_path}: {str(e)}")
            return False
    
    def _find_section_by_title(self, report: Report, title: str) -> Optional[ReportSection]:
        """Find a section by its title."""
        for section in report.sections:
            if section.title.lower() == title.lower():
                return section
        return None
    
    def _slugify(self, text: str) -> str:
        """Convert text to a URL-friendly slug."""
        text = text.lower()
        # Replace spaces with hyphens
        text = re.sub(r'\s+', '-', text)
        # Remove special characters
        text = re.sub(r'[^\w\-]', '', text)
        return text
    
    def _convert_markdown_to_html(self, markdown: str) -> str:
        """
        Simple converter from Markdown to HTML.
        
        Note: This is a basic implementation for the PoC.
        In production, we would use a proper Markdown parser like mistune or markdown.
        """
        html = markdown
        
        # Convert headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        
        # Convert emphasis
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Convert links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        
        # Convert lists (simple version)
        html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.+</li>\n)+', r'<ul>\n\g<0></ul>', html, flags=re.MULTILINE)
        
        # Convert paragraphs (simple version)
        html = re.sub(r'(?<!\n)\n(?!\n)(?!<[uo]l|<li|<h[1-6]|<p)', ' ', html)  # Join lines that don't end with double newline
        html = re.sub(r'\n\n(?!<[uo]l|<li|<h[1-6]|<p)', r'\n\n<p>', html)  # Start paragraph
        html = re.sub(r'<p>(.+?)(\n\n|$)', r'<p>\1</p>\2', html, flags=re.DOTALL)  # Close paragraph
        
        return html


# ----- Report Generator -----

class ReportGenerator:
    """
    Main class for generating research reports.
    
    The ReportGenerator:
    - Takes research results and organizes them into coherent reports
    - Uses LLM for content generation and organization
    - Manages the overall report structure and flow
    - Coordinates with CitationManager and OutputFormatter
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.citation_manager = CitationManager()
        self.output_formatter = OutputFormatter()
    
    async def generate_report(
        self, 
        research_results: Dict[str, Any], 
        config: ReportConfiguration
    ) -> Report:
        """
        Generate a report from research results.
        
        Args:
            research_results: Research results from the knowledge base
            config: Report configuration parameters
            
        Returns:
            A generated Report object
        """
        logger.info(f"Generating {config.report_type.value} report: {config.title}")
        
        # Create the report
        report_id = str(uuid.uuid4())[:8]
        report = Report(
            id=report_id,
            title=config.title,
            configuration=config,
            sections=[],
            sources={},
            citations=[],
            metadata={
                "generation_timestamp": datetime.now().isoformat(),
                "topic_id": research_results.get("topic_id", "unknown")
            }
        )
        
        # Set up citation manager with the right style
        self.citation_manager = CitationManager(config.citation_style)
        
        # Process sources first
        sources = self._process_sources(research_results)
        for source_id, source in sources.items():
            report.sources[source_id] = source
            self.citation_manager.add_source(source)
        
        # Generate report outline based on the report type
        outline = self._generate_outline(config.report_type, research_results)
        
        # Generate each section
        for section_id, section_info in outline.items():
            section = await self._generate_section(
                section_id,
                section_info,
                research_results,
                config,
                report.sources
            )
            report.sections.append(section)
        
        # Add executive summary if needed
        if config.include_executive_summary:
            summary = await self._generate_executive_summary(report)
            report.sections.insert(0, summary)
        
        # Add references if needed
        if config.include_references:
            references = self._generate_references_section(report)
            report.sections.append(references)
        
        # Collect all citations
        report.citations = self.citation_manager.citations
        
        logger.info(f"Generated report with {len(report.sections)} sections and {len(report.sources)} sources")
        return report
    
    async def generate_and_save_report(
        self,
        research_results: Dict[str, Any],
        config: ReportConfiguration,
        output_dir: str = "output/reports"
    ) -> Dict[str, str]:
        """
        Generate a report and save it in the requested formats.
        
        Args:
            research_results: Research results from the knowledge base
            config: Report configuration parameters
            output_dir: Directory to save the report files
            
        Returns:
            Dictionary mapping format to output file path
        """
        # Generate the report
        report = await self.generate_report(research_results, config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in each requested format
        output_files = {}
        base_filename = f"{report.id}_{self._slugify(report.title)}"
        
        for output_format in config.output_formats:
            try:
                formatted_content = self.output_formatter.format_report(report, output_format)
                
                # Determine file extension and path
                if output_format == ReportFormat.MARKDOWN:
                    file_extension = ".md"
                elif output_format == ReportFormat.HTML:
                    file_extension = ".html"
                elif output_format == ReportFormat.PDF:
                    file_extension = ".pdf"
                elif output_format == ReportFormat.WORD:
                    file_extension = ".docx"
                else:
                    file_extension = ".txt"
                    
                output_path = os.path.join(output_dir, f"{base_filename}{file_extension}")
                
                # Save the file
                if self.output_formatter.save_to_file(formatted_content, output_path):
                    output_files[output_format.value] = output_path
                
            except NotImplementedError as e:
                logger.warning(f"Format {output_format.value} not implemented: {str(e)}")
            except Exception as e:
                logger.error(f"Error saving report in {output_format.value} format: {str(e)}")
        
        return output_files
    
    def _process_sources(self, research_results: Dict[str, Any]) -> Dict[str, Source]:
        """
        Process and extract sources from research results.
        
        Args:
            research_results: Research results from the knowledge base
            
        Returns:
            Dictionary of source ID to Source objects
        """
        sources = {}
        
        # Extract sources from research results
        if "sources" in research_results:
            for source_data in research_results["sources"]:
                url = source_data.get("url", "")
                title = source_data.get("title", "Unknown Source")
                
                # Extract additional metadata if available
                authors = None
                publication_date = None
                publication = None
                credibility = None
                
                if "evaluation" in source_data:
                    evaluation = source_data["evaluation"]
                    credibility = evaluation.get("credibility")
                    
                    # Try to extract author information
                    if "author_credentials" in evaluation:
                        authors = [evaluation["author_credentials"]]
                        
                    # Try to extract publication date
                    if "publication_date" in evaluation:
                        try:
                            pub_date = evaluation["publication_date"]
                            if isinstance(pub_date, str):
                                publication_date = datetime.fromisoformat(pub_date)
                            elif isinstance(pub_date, datetime):
                                publication_date = pub_date
                        except (ValueError, TypeError):
                            pass
                            
                # Create source object
                source = Source(
                    url=url,
                    title=title,
                    authors=authors,
                    publication_date=publication_date,
                    publication=publication,
                    credibility=credibility,
                    metadata={
                        "source_data": source_data,
                        "id": f"source_{len(sources) + 1}"
                    }
                )
                
                # Add to sources dictionary
                source_id = source.metadata["id"]
                sources[source_id] = source
        
        logger.info(f"Processed {len(sources)} sources from research results")
        return sources
    
    def _generate_outline(
        self, 
        report_type: ReportType,
        research_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a report outline based on the report type and research results.
        
        Args:
            report_type: The type of report to generate
            research_results: Research results from the knowledge base
            
        Returns:
            Dictionary mapping section IDs to section information
        """
        # Extract information from research results
        topic_id = research_results.get("topic_id", "unknown")
        total_items = research_results.get("total_information_items", 0)
        
        # Build basic outline based on report type
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            # Very brief report with just key findings
            outline = {
                f"section_summary": {
                    "title": "Key Findings",
                    "level": 1,
                    "order": 1,
                    "parent_id": None,
                }
            }
            
        elif report_type == ReportType.PRESENTATION:
            # Streamlined report suitable for presentation slides
            outline = {
                f"section_overview": {
                    "title": "Overview",
                    "level": 1,
                    "order": 1,
                    "parent_id": None,
                },
                f"section_key_findings": {
                    "title": "Key Findings",
                    "level": 1,
                    "order": 2,
                    "parent_id": None,
                },
                f"section_conclusion": {
                    "title": "Conclusion",
                    "level": 1,
                    "order": 3,
                    "parent_id": None,
                }
            }
            
        elif report_type == ReportType.ANALYTICAL_REPORT:
            # Detailed analytical report with in-depth sections
            outline = {
                f"section_introduction": {
                    "title": "Introduction",
                    "level": 1,
                    "order": 1,
                    "parent_id": None,
                },
                f"section_methodology": {
                    "title": "Methodology",
                    "level": 1,
                    "order": 2,
                    "parent_id": None,
                },
                f"section_analysis": {
                    "title": "Analysis and Findings",
                    "level": 1,
                    "order": 3,
                    "parent_id": None,
                },
                f"section_discussion": {
                    "title": "Discussion",
                    "level": 1,
                    "order": 4,
                    "parent_id": None,
                },
                f"section_conclusion": {
                    "title": "Conclusion",
                    "level": 1,
                    "order": 5,
                    "parent_id": None,
                }
            }
            
            # Add subsections to analysis
            outline[f"section_analysis_key"] = {
                "title": "Key Findings",
                "level": 2,
                "order": 1,
                "parent_id": "section_analysis",
            }
            
            outline[f"section_analysis_implications"] = {
                "title": "Implications",
                "level": 2,
                "order": 2,
                "parent_id": "section_analysis",
            }
            
        else:  # Default: StandardReport
            # Standard balanced report
            outline = {
                f"section_introduction": {
                    "title": "Introduction",
                    "level": 1,
                    "order": 1,
                    "parent_id": None,
                },
                f"section_findings": {
                    "title": "Key Findings",
                    "level": 1,
                    "order": 2,
                    "parent_id": None,
                },
                f"section_discussion": {
                    "title": "Discussion",
                    "level": 1,
                    "order": 3,
                    "parent_id": None,
                },
                f"section_conclusion": {
                    "title": "Conclusion",
                    "level": 1,
                    "order": 4,
                    "parent_id": None,
                }
            }
        
        # Attempt to create topical subsections based on the research results
        if "sources" in research_results and len(research_results["sources"]) > 0:
            # Extract common themes from information items
            themes = self._extract_themes(research_results)
            
            # For standard and analytical reports, add thematic subsections
            if report_type in [ReportType.STANDARD_REPORT, ReportType.ANALYTICAL_REPORT]:
                parent_section = "section_findings" if report_type == ReportType.STANDARD_REPORT else "section_analysis"
                
                for i, theme in enumerate(themes[:3]):  # Limit to top 3 themes
                    section_id = f"{parent_section}_theme_{i+1}"
                    outline[section_id] = {
                        "title": theme,
                        "level": 2,
                        "order": i+1,
                        "parent_id": parent_section,
                        "theme": theme
                    }
        
        logger.info(f"Generated outline with {len(outline)} sections for {report_type.value} report")
        return outline
    
    def _extract_themes(self, research_results: Dict[str, Any]) -> List[str]:
        """
        Extract common themes from research results.
        
        In a production implementation, this would use sophisticated topic modeling
        or LLM-assisted clustering to identify themes.
        
        Args:
            research_results: Research results from the knowledge base
            
        Returns:
            List of theme titles
        """
        # This is a simplified implementation for the PoC
        # In production, we would use NLP techniques or LLM analysis
        
        # Default themes in case we can't extract better ones
        default_themes = [
            "Primary Findings", 
            "Secondary Aspects", 
            "Additional Considerations"
        ]
        
        # Try to extract meaningful themes from research results
        themes = []
        
        # If we have source information with specific themes
        if "sources" in research_results:
            source_titles = [s.get("title", "") for s in research_results["sources"]]
            
            # Very simple keyword extraction - in production this would be more sophisticated
            keywords = set()
            for title in source_titles:
                # Split and filter words
                words = [w for w in title.split() if len(w) > 4]  # Simple heuristic
                keywords.update(words)
            
            # Use top keywords as themes (very simplified approach)
            themes = list(keywords)[:5]
            
            # If we extracted too few themes, add some defaults
            if len(themes) < 3:
                themes.extend(default_themes[:3 - len(themes)])
        else:
            themes = default_themes
            
        return themes
            
    async def _generate_section(
        self,
        section_id: str,
        section_info: Dict[str, Any],
        research_results: Dict[str, Any],
        config: ReportConfiguration,
        sources: Dict[str, Source]
    ) -> ReportSection:
        """
        Generate content for a report section.
        
        In a production implementation, this would use an LLM to generate
        content based on the research results.
        
        Args:
            section_id: ID of the section
            section_info: Information about the section
            research_results: Research results from the knowledge base
            config: Report configuration
            sources: Dictionary of sources
            
        Returns:
            A ReportSection object with generated content
        """
        title = section_info["title"]
        level = section_info["level"]
        order = section_info["order"]
        parent_id = section_info["parent_id"]
        
        logger.info(f"Generating content for section: {title}")
        
        # In a production implementation, this would call an LLM to generate content
        # based on the research results, section type, etc.
        
        # For this PoC, we'll generate placeholder content
        content = f"This section on {title} would contain content generated by an LLM based on the research results."
        
        # Add some example content based on section type
        if "Introduction" in title:
            content = f"""This report examines the key aspects and findings related to the research topic. 
            
Based on analysis of {len(sources)} sources, we have identified several important patterns and insights that will be discussed in detail throughout this report.

The following sections present an organized overview of the research findings, their implications, and potential conclusions."""
            
        elif "Methodology" in title:
            content = f"""The research methodology involved gathering information from {len(sources)} distinct sources, 
evaluating their credibility, and synthesizing the extracted information into cohesive findings.

Sources were selected based on relevance to the topic and assessed for reliability using a multi-factor evaluation system."""
            
        elif "Findings" in title or "Analysis" in title:
            # For findings/analysis section, add some placeholder content with citations
            content = f"""Our research reveals several significant findings related to this topic. 
            
The analysis of available information indicates patterns and trends that provide valuable insights."""
            
            # Add citations from a couple of sources
            if sources:
                source_ids = list(sources.keys())
                if len(source_ids) >= 2:
                    # Add a citation to the content
                    first_cite = self.citation_manager.format_in_text_citation(source_ids[0])
                    second_cite = self.citation_manager.format_in_text_citation(source_ids[1])
                    
                    # Add citations to the section content
                    content += f"""

According to one source {first_cite}, the primary aspects of this topic demonstrate important relationships that merit further investigation.

Additional research {second_cite} suggests complementary findings that help to build a more complete understanding of the subject matter."""
                    
                    # Record these citations
                    self.citation_manager.add_citation(
                        source_ids[0], 
                        first_cite, 
                        section_id, 
                        "the primary aspects of this topic demonstrate important relationships"
                    )
                    
                    self.citation_manager.add_citation(
                        source_ids[1], 
                        second_cite, 
                        section_id, 
                        "suggests complementary findings that help to build a more complete understanding"
                    )
            
        elif "Discussion" in title:
            content = f"""The findings presented in this report have several important implications.
            
When considering these results in a broader context, several patterns emerge that warrant further consideration.

The limitations of the current research should be acknowledged, including potential gaps in available information and the constraints of the methodology employed."""
            
        elif "Conclusion" in title:
            content = f"""In conclusion, this research has identified several key insights about the topic.
            
The findings suggest important considerations for understanding the subject matter comprehensively.

Further research would be beneficial to explore additional aspects and develop a more nuanced understanding of the topic."""
            
        elif "theme" in section_info:
            # This is a thematic subsection
            theme = section_info["theme"]
            content = f"""This section examines {theme} as a key aspect of the research topic.
            
Analysis of the information related to this theme reveals important patterns and insights."""
            
            # Add a citation if we have sources
            if sources:
                source_id = list(sources.keys())[0]
                citation = self.citation_manager.format_in_text_citation(source_id)
                content += f"""

Evidence suggests {citation} that this theme represents a significant component of the overall topic and merits detailed consideration."""
                
                # Record the citation
                self.citation_manager.add_citation(
                    source_id, 
                    citation, 
                    section_id, 
                    "this theme represents a significant component of the overall topic"
                )
        
        # Create the section
        section = ReportSection(
            id=section_id,
            title=title,
            content=content,
            level=level,
            order=order,
            parent_id=parent_id,
            citations=[],  # Citations added by CitationManager
            metadata={
                "section_type": title.lower().replace(" ", "_"),
                "word_count": len(content.split())
            }
        )
        
        return section
    
    async def _generate_executive_summary(self, report: Report) -> ReportSection:
        """
        Generate an executive summary for the report.
        
        Args:
            report: The report object
            
        Returns:
            A ReportSection with the executive summary
        """
        logger.info("Generating executive summary")
        
        # In a production implementation, this would use an LLM to generate
        # a concise summary based on the report contents
        
        # For this PoC, we'll create a simple summary
        major_sections = [s for s in report.sections if s.level == 1 and s.title != "References"]
        section_titles = [s.title for s in major_sections]
        
        content = f"""This report presents findings on {report.title}.
        
Based on research from {len(report.sources)} sources, the report examines {", ".join(section_titles[:-1])}, and {section_titles[-1] if section_titles else ""}.

Key insights include the significance of the main findings and their implications for the broader understanding of the topic."""
        
        # Create the summary section
        summary = ReportSection(
            id="section_executive_summary",
            title="Executive Summary",
            content=content,
            level=1,
            order=0,
            parent_id=None,
            citations=[],
            metadata={
                "section_type": "executive_summary",
                "word_count": len(content.split())
            }
        )
        
        return summary
    
    def _generate_references_section(self, report: Report) -> ReportSection:
        """
        Generate a references section for the report.
        
        Args:
            report: The report object
            
        Returns:
            A ReportSection with the references
        """
        logger.info("Generating references section")
        
        # Use the citation manager to generate formatted references
        references_content = self.citation_manager.generate_references_list()
        
        # Create the references section
        references = ReportSection(
            id="section_references",
            title="References",
            content=references_content,
            level=1,
            order=999,  # Always last
            parent_id=None,
            citations=[],
            metadata={
                "section_type": "references",
                "source_count": len(report.sources)
            }
        )
        
        return references
    
    def _slugify(self, text: str) -> str:
        """Convert text to a URL-friendly slug."""
        text = text.lower()
        # Replace spaces with hyphens
        text = re.sub(r'\s+', '-', text)
        # Remove special characters
        text = re.sub(r'[^\w\-]', '', text)
        return text


# ----- Example Usage -----

async def main():
    """Example usage of the report generation module."""
    print("Report Generation Demo")
    print("-" * 50)
    
    # Create a sample research results object
    research_results = {
        "topic_id": "topic_001",
        "total_information_items": 15,
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
                        "text": "Climate change is affecting agricultural productivity worldwide.",
                        "relevance_score": 0.95
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
                    }
                ]
            }
        ]
    }
    
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
    
    print(f"Generating {config.report_type.value} report: {config.title}")
    print(f"Citation style: {config.citation_style.value}")
    print(f"Output formats: {[f.value for f in config.output_formats]}")
    print("-" * 50)
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate the report
    print("\nGenerating report...")
    report = await generator.generate_report(research_results, config)
    
    print(f"\nReport generated with {len(report.sections)} sections:")
    for section in report.sections:
        print(f"- {section.title} (Level {section.level}, {len(section.content.split())} words)")
    
    print(f"\nSources cited: {len(report.sources)}")
    print(f"Citations: {len(report.citations)}")
    
    # Generate output files
    output_dir = "output/reports"
    print(f"\nSaving report to {output_dir}...")
    output_files = await generator.generate_and_save_report(research_results, config, output_dir)
    
    print("\nOutput files:")
    for format_name, file_path in output_files.items():
        print(f"- {format_name}: {file_path}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
