"""
Run the Research Assistant Web Application

This script starts the Flask web application for the Research Assistant,
handling the integration between Flask and asyncio.
"""

import os
import asyncio
import logging
from waitress import serve
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("research_web_app")

# Set up asyncio policy for Windows if needed
try:
    from asyncio import WindowsSelectorEventLoopPolicy, set_event_loop_policy
    if os.name == 'nt':  # Windows
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
except ImportError:
    pass  # Not on Windows or asyncio.WindowsSelectorEventLoopPolicy not available

def main():
    """Start the web application."""
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Determine whether to run in development or production mode
    debug_mode = os.environ.get("FLASK_ENV", "development").lower() == "development"
    
    if debug_mode:
        logger.info(f"Starting development server on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    else:
        logger.info(f"Starting production server on {host}:{port}")
        serve(app, host=host, port=port)

if __name__ == "__main__":
    main()
