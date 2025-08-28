#!/usr/bin/env python3
"""
Test fibonacci function directly to diagnose the issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the fibonacci tool
from arc_fibonacci_tool import fibonacci

print("Testing fibonacci function directly...")
print(f"fibonacci(20) = {fibonacci(20)}")
print("Test completed successfully!")