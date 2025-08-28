#!/usr/bin/env python3
"""
Run the enhanced ARC puzzle editor standalone
"""

import uvicorn
from arc_puzzle_editor_enhanced import app

if __name__ == "__main__":
    print("Starting Enhanced ARC Puzzle Editor")
    print("Open http://localhost:8000/enhanced-editor to view")
    uvicorn.run(app, host="0.0.0.0", port=8000)