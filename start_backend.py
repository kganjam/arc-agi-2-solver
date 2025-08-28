#!/usr/bin/env python3
"""
Start the backend with proper settings to prevent crashes and restarts
"""

import uvicorn
import sys
import signal
import logging
from arc_integrated_app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def signal_handler(sig, frame):
    """Handle shutdown gracefully"""
    print('\nShutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    print("Starting ARC AGI Backend (stable mode)")
    print("Open http://localhost:8050 to view the application")
    print("Press Ctrl+C to stop")
    
    try:
        # Run with explicit settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8050,
            reload=False,  # Disable auto-reload
            workers=1,     # Single worker to prevent conflicts
            log_level="info",
            access_log=True,
            loop="asyncio",  # Explicit event loop
            timeout_keep_alive=30,  # Timeout for keep-alive connections
            limit_max_requests=1000  # Restart worker after 1000 requests to prevent memory leaks
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)