"""
Content Extraction Module

This module implements the ContentExtractor class that uses Playwright
to navigate to web pages and extract their content for the Research Assistant.

It provides a real implementation to replace the mock ContentRetriever
in the research_pipeline.py.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Error as PlaywrightError
from research_pipeline import WebContent, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("content_extraction")


class ContentExtractor:
    """
    Extracts content from web pages using Playwright.
    
    This class provides a production implementation to replace the mock
    ContentRetriever in the research pipeline.
    """
    
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        user_agent: Optional[str] = None,
        max_retries: int = 2,
    ):
        """
        Initialize the ContentExtractor.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Navigation timeout in milliseconds
            user_agent: Custom user agent string
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self._browser = None
        self._context = None
        
    async def __aenter__(self):
        """Initialize browser when used as context manager."""
        await self._ensure_browser()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close browser when exiting context manager."""
        await self.close()
        
    async def _ensure_browser(self):
        """Ensure browser is launched."""
        if self._browser is None:
            try:
                playwright = await async_playwright().start()
                self._playwright = playwright
                self._browser = await playwright.chromium.launch(headless=self.headless)
                self._context = await self._browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1280, "height": 800},
                )
                
                # Set default timeout
                self._context.set_default_timeout(self.timeout)
                
                logger.info(f"Browser launched successfully (headless={self.headless})")
            except Exception as e:
                logger.error(f"Failed to launch browser: {str(e)}")
                if self._browser:
                    await self._browser.close()
                    self._browser = None
                if hasattr(self, '_playwright'):
                    await self._playwright.stop()
                raise
                
    async def close(self):
        """Close browser and playwright."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._context = None
        if hasattr(self, '_playwright'):
            await self._playwright.stop()
            
    async def extract_content(self, result: SearchResult) -> Optional[WebContent]:
        """
        Navigate to a URL and extract its content.
        
        Args:
            result: SearchResult containing the URL to extract content from
            
        Returns:
            WebContent object containing extracted content or None if extraction failed
        """
        url = result.url
        logger.info(f"Extracting content from URL: {url}")
        
        await self._ensure_browser()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Create new page
                page = await self._context.new_page()
                
                try:
                    # Navigate to URL with timeout
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                    
                    # Wait for content to load
                    await self._wait_for_content(page)
                    
                    # Extract title, content and HTML
                    title = await self._extract_title(page)
                    content = await self._extract_text_content(page)
                    html = await page.content()
                    
                    # Get metadata
                    metadata = await self._extract_metadata(page, result)
                    
                    # Create WebContent object
                    web_content = WebContent(
                        url=url,
                        title=title or result.title,
                        content=content,
                        html=html,
                        timestamp=datetime.now(),
                        metadata=metadata
                    )
                    
                    logger.info(f"Successfully extracted content from URL: {url} (Title: {title})")
                    return web_content
                    
                finally:
                    # Close the page regardless of success/failure
                    await page.close()
                    
            except PlaywrightError as e:
                if "ERR_NAME_NOT_RESOLVED" in str(e):
                    logger.error(f"DNS resolution failed for URL: {url}")
                    return None
                elif "ERR_CONNECTION_REFUSED" in str(e):
                    logger.error(f"Connection refused for URL: {url}")
                    return None
                elif "ERR_CONNECTION_TIMED_OUT" in str(e):
                    if attempt < self.max_retries:
                        logger.warning(f"Connection timed out for URL: {url}, retrying ({attempt+1}/{self.max_retries})")
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    else:
                        logger.error(f"Connection timed out for URL: {url} after {self.max_retries} retries")
                        return None
                else:
                    logger.error(f"Error navigating to URL: {url} - {str(e)}")
                    if attempt < self.max_retries:
                        logger.warning(f"Retrying... ({attempt+1}/{self.max_retries})")
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    return None
            except Exception as e:
                logger.error(f"Unexpected error extracting content from URL: {url} - {str(e)}")
                if attempt < self.max_retries:
                    logger.warning(f"Retrying... ({attempt+1}/{self.max_retries})")
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                return None
                
        return None
        
    async def _wait_for_content(self, page: Page):
        """
        Wait for content to be fully loaded.
        
        Args:
            page: Playwright Page object
        """
        # Wait for page to be fully loaded
        try:
            # Wait for load state
            await page.wait_for_load_state("load", timeout=self.timeout)
            
            # Wait for network to be idle
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightError:
            # Continue even if timeout - we'll work with what we have
            pass
            
        # Additional waiting for JavaScript-heavy sites
        try:
            # Wait for main content elements that commonly appear in articles/blogs
            selectors = [
                "article", 
                "main", 
                ".content", 
                "#content", 
                ".article", 
                "#article",
                ".post-content",
                ".entry-content"
            ]
            
            for selector in selectors:
                try:
                    # Use a short timeout per selector
                    await page.wait_for_selector(selector, timeout=1000)
                    # If found, no need to check others
                    break
                except PlaywrightError:
                    # Selector not found, try next one
                    continue
        except PlaywrightError:
            # Continue even if we can't find content containers
            pass
            
    async def _extract_title(self, page: Page) -> str:
        """
        Extract the title of the page.
        
        Args:
            page: Playwright Page object
            
        Returns:
            Title string
        """
        try:
            # Try to get the article title first
            for selector in [
                "article h1", 
                ".article-title", 
                ".post-title", 
                ".entry-title",
                "article header h1",
                "main h1"
            ]:
                try:
                    title_element = await page.query_selector(selector)
                    if title_element:
                        title = await title_element.text_content()
                        title = title.strip()
                        if title:
                            return title
                except Exception:
                    continue
                    
            # Fall back to page title
            return await page.title()
        except Exception as e:
            logger.warning(f"Error extracting title: {str(e)}")
            return "Unknown Title"
            
    async def _extract_text_content(self, page: Page) -> str:
        """
        Extract the main text content from the page.
        
        Args:
            page: Playwright Page object
            
        Returns:
            Extracted text content
        """
        content = ""
        
        try:
            # Try to extract from main content containers first
            content_selectors = [
                "article", 
                "main", 
                ".article-content", 
                ".post-content", 
                ".entry-content",
                "#content",
                ".content"
            ]
            
            for selector in content_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        element_content = await element.text_content()
                        if element_content and len(element_content.strip()) > 200:  # Ensure it has substantial content
                            content = element_content
                            break
                except Exception:
                    continue
                    
            # If we didn't find content with selectors, use body text
            if not content:
                # Get body text excluding scripts, styles and navigation
                content = await page.evaluate("""() => {
                    // Helper to check if element should be skipped
                    function shouldSkip(element) {
                        if (!element) return true;
                        
                        const tag = element.tagName.toLowerCase();
                        const role = element.getAttribute('role');
                        const classes = element.className;
                        const id = element.id;
                        
                        // Skip navigation, scripts, styles, footers
                        if (tag === 'script' || tag === 'style' || tag === 'nav' || 
                            tag === 'footer' || tag === 'header' || 
                            (role === 'navigation') ||
                            (classes && (
                                classes.includes('nav') || 
                                classes.includes('menu') || 
                                classes.includes('footer') || 
                                classes.includes('header') ||
                                classes.includes('comment')
                            )) ||
                            (id && (
                                id.includes('nav') || 
                                id.includes('menu') || 
                                id.includes('footer') || 
                                id.includes('header') ||
                                id.includes('comment')
                            ))) {
                            return true;
                        }
                        return false;
                    }
                    
                    // Get all text nodes from the body
                    let textContent = '';
                    const walk = document.createTreeWalker(
                        document.body, 
                        NodeFilter.SHOW_TEXT, 
                        {
                            acceptNode: function(node) {
                                if (shouldSkip(node.parentElement)) {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );
                    
                    let node;
                    while(node = walk.nextNode()) {
                        const trimmed = node.textContent.trim();
                        if (trimmed) {
                            textContent += trimmed + '\\n';
                        }
                    }
                    
                    return textContent;
                }""")
            
            # Clean up the content
            content = self._clean_text(content)
            
            # If the content is still too short, fall back to all text from body
            if len(content.strip()) < 200:
                body_text = await page.evaluate('document.body.innerText')
                content = self._clean_text(body_text)
                
            return content
            
        except Exception as e:
            logger.warning(f"Error extracting text content: {str(e)}")
            # Last resort: get all body text
            try:
                return await page.evaluate('document.body.innerText')
            except Exception:
                return ""
                
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
            
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with two newlines (paragraph break)
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Rejoin lines, filtering out empty ones
        text = '\n'.join(line for line in lines if line)
        
        return text
        
    async def _extract_metadata(self, page: Page, result: SearchResult) -> Dict[str, Any]:
        """
        Extract metadata from the page.
        
        Args:
            page: Playwright Page object
            result: Original SearchResult
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "domain": self._extract_domain(result.url),
            "query": result.metadata.get("query", ""),
            "position": result.position,
            "extraction_timestamp": datetime.now().isoformat(),
        }
        
        try:
            # Add word count
            body_text = await page.evaluate('document.body.innerText')
            metadata["word_count"] = len(body_text.split())
            
            # Extract meta tags
            meta_tags = await page.evaluate("""() => {
                const metaTags = {};
                
                // Get meta description
                const descriptionTag = document.querySelector('meta[name="description"]') || 
                                      document.querySelector('meta[property="og:description"]');
                if (descriptionTag) {
                    metaTags.description = descriptionTag.content;
                }
                
                // Get publishing date
                const dateTag = document.querySelector('meta[name="date"]') || 
                               document.querySelector('meta[property="article:published_time"]') ||
                               document.querySelector('meta[property="og:published_time"]');
                if (dateTag) {
                    metaTags.published_date = dateTag.content;
                }
                
                // Get author
                const authorTag = document.querySelector('meta[name="author"]') || 
                                 document.querySelector('meta[property="article:author"]') ||
                                 document.querySelector('meta[property="og:author"]');
                if (authorTag) {
                    metaTags.author = authorTag.content;
                }
                
                // Get keywords
                const keywordsTag = document.querySelector('meta[name="keywords"]');
                if (keywordsTag) {
                    metaTags.keywords = keywordsTag.content;
                }
                
                return metaTags;
            }""")
            
            # Add meta tags to metadata
            metadata.update(meta_tags)
            
            # Try to extract reading time
            try:
                text_content = await self._extract_text_content(page)
                word_count = len(text_content.split())
                # Average reading speed: 200-250 words per minute
                reading_time_minutes = max(1, round(word_count / 225))
                metadata["estimated_reading_time_minutes"] = reading_time_minutes
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            
        return metadata
        
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain name from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain name
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except Exception:
            return ""


async def test_content_extractor():
    """Test the ContentExtractor with a sample URL."""
    print("ContentExtractor Test")
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
    
    # Create the ContentExtractor
    extractor = ContentExtractor(headless=True)
    
    try:
        print(f"Extracting content from: {test_url}")
        content = await extractor.extract_content(result)
        
        if content:
            print("\nExtraction successful:")
            print(f"Title: {content.title}")
            print(f"Content length: {len(content.content)} characters")
            print(f"Word count: {content.metadata.get('word_count', 'unknown')}")
            print(f"\nFirst 300 characters of content:")
            print(content.content[:300] + "...")
            
            print("\nMetadata:")
            for key, value in content.metadata.items():
                print(f"  {key}: {value}")
        else:
            print("Extraction failed")
    finally:
        await extractor.close()
        print("\nTest completed")


if __name__ == "__main__":
    # Run test function
    asyncio.run(test_content_extractor())
