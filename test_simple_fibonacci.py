#!/usr/bin/env python3
"""
Simple test for fibonacci function call issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arc_puzzle_ai_assistant_v2 import SmartPuzzleAIAssistant

def test_fibonacci_issue():
    """Test the specific issue with fibonacci(20)"""
    
    print("Testing fibonacci(20) issue...")
    
    # Initialize AI assistant
    assistant = SmartPuzzleAIAssistant()
    
    # Test the exact function call that's causing issues
    print("\nTesting invoke_generated_tool with fibonacci...")
    
    try:
        result = assistant.execute_function(
            "invoke_generated_tool",
            {
                "tool_name": "fibonacci",
                "parameters": {"n": 20}
            }
        )
        print(f"Result: {result}")
        
        if 'error' in result:
            print(f"Error detected: {result['error']}")
        elif 'result' in result:
            print(f"Success! fibonacci(20) = {result['result']}")
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest complete")

if __name__ == "__main__":
    test_fibonacci_issue()