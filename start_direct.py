#!/usr/bin/env python3
"""
Direct backend startup without uvicorn wrapper
"""

import sys
import os

# Ensure we're using the right Python path
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Import and run the app directly
print("Starting ARC AGI Backend...")

try:
    import uvicorn
    from arc_integrated_app import app
    
    print("✓ Modules loaded successfully")
    
    # Run with minimal configuration
    print("Starting server on http://0.0.0.0:8050")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8050,
        log_level="info",
        reload=False
    )
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)