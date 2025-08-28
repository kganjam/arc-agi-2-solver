#!/usr/bin/env python3
"""
Test the specific fibonacci issue without going through the web API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arc_puzzle_ai_assistant_v2 import SmartPuzzleAIAssistant

def test_fibonacci_natural_language():
    """Test processing natural language fibonacci request"""
    
    print("Testing natural language fibonacci request...")
    print("-" * 50)
    
    # Initialize AI assistant
    assistant = SmartPuzzleAIAssistant()
    
    # Test different variations
    test_inputs = [
        "What is fibonacci(20)?",
        "print fibonacci(20)",
        "Calculate the 20th fibonacci number"
    ]
    
    for input_text in test_inputs:
        print(f"\nTesting: {input_text}")
        
        # First, try basic processing (without Bedrock)
        print("Using basic processing:")
        result = assistant._basic_process(input_text)
        print(f"Result: {result}")
        
        # Try to extract and execute fibonacci if mentioned
        if "fibonacci" in input_text.lower():
            # Extract number
            import re
            match = re.search(r'fibonacci\((\d+)\)', input_text.lower())
            if not match:
                match = re.search(r'(\d+)(?:th|st|nd|rd)?\s+fibonacci', input_text.lower())
            
            if match:
                n = int(match.group(1))
                print(f"\nDirect fibonacci execution for n={n}:")
                
                # Execute directly
                result = assistant.execute_function(
                    "invoke_generated_tool",
                    {"tool_name": "fibonacci", "parameters": {"n": n}}
                )
                print(f"Result: {result}")

if __name__ == "__main__":
    test_fibonacci_natural_language()