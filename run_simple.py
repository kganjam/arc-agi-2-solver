#!/usr/bin/env python3
"""
Simple launcher for the ARC AGI Challenge Solver
Starts the web application on http://localhost:8050
"""

import subprocess
import sys
import time
import webbrowser
import signal
import os

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down...")
    sys.exit(0)

def main():
    print("="*60)
    print("  ARC AGI Challenge Solver")
    print("  Starting application...")
    print("="*60)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\n📥 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed")
    
    # Start server
    print("\n🚀 Starting server...")
    print("   URL: http://localhost:8050")
    print("   Press Ctrl+C to stop\n")
    
    # Delay browser launch
    def launch_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8050")
    
    # Start browser in background after delay
    import threading
    browser_thread = threading.Thread(target=launch_browser, daemon=True)
    browser_thread.start()
    
    # Run the server
    try:
        subprocess.run([sys.executable, "test_simple.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()