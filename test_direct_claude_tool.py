#!/usr/bin/env python3
"""
Direct test of Claude Code tool generation through the AI assistant
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arc_puzzle_ai_assistant_v2 import SmartPuzzleAIAssistant

def test_direct_claude_integration():
    """Test Claude Code integration directly"""
    
    print("Testing Direct Claude Code Integration")
    print("=" * 50)
    
    # Initialize AI assistant
    assistant = SmartPuzzleAIAssistant()
    
    # Test 1: List generated tools
    print("\n1. Testing list of generated tools...")
    result = assistant.execute_function("list_generated_tools", {})
    print(f"Result: {result}")
    
    # Test 2: Generate a new tool with Claude Code
    print("\n2. Testing tool generation with Claude Code...")
    
    result = assistant.execute_function(
        "generate_tool_with_claude",
        {
            "description": """A tool that counts the number of unique colors in a grid.
            It should:
            - Take a 2D list (grid) as input
            - Return the count of unique colors
            - Handle empty grids
            - Include a function to get color frequency distribution""",
            "name": "color_counter"
        }
    )
    
    print(f"Generation result: {result}")
    
    if result.get('success'):
        print("✅ Tool generated successfully!")
        
        # Test 3: List tools again to see if new one appears
        print("\n3. Checking if new tool appears in list...")
        result = assistant.execute_function("list_generated_tools", {})
        print(f"Updated tool list: {result}")
        
        # Test 4: Try to invoke the generated tool
        print("\n4. Testing invocation of generated tool...")
        test_grid = [
            [1, 1, 2, 3],
            [1, 2, 2, 3],
            [4, 4, 4, 3],
            [0, 0, 0, 0]
        ]
        
        result = assistant.execute_function(
            "invoke_generated_tool",
            {
                "tool_name": "color_counter",
                "parameters": {"grid": test_grid}
            }
        )
        print(f"Tool invocation result: {result}")
    else:
        print(f"⚠️ Tool generation failed: {result.get('message', 'Unknown error')}")
    
    # Test 5: Test Fibonacci tool invocation
    print("\n5. Testing Fibonacci tool...")
    result = assistant.execute_function(
        "invoke_generated_tool",
        {
            "tool_name": "fibonacci",
            "parameters": {"n": 15}
        }
    )
    print(f"Fibonacci(15) = {result}")
    
    print("\n" + "=" * 50)
    print("Direct test complete!")
    return True

if __name__ == "__main__":
    try:
        success = test_direct_claude_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)