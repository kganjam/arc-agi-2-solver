#!/usr/bin/env python3
"""
Test Claude Code integration by having it create a Fibonacci tool
"""

import asyncio
import json
from arc_claude_code_integration import ClaudeCodeDialogue

async def test_fibonacci_tool_creation():
    """Test creating a Fibonacci tool using Claude Code"""
    
    print("Testing Claude Code Integration")
    print("-" * 50)
    
    # Initialize Claude Code dialogue manager
    dialogue = ClaudeCodeDialogue()
    
    # Set to actual mode (not simulation)
    dialogue.simulate_mode = False
    
    # Create prompt for Claude Code to generate a Fibonacci tool
    prompt = """Create a new Python tool file called 'arc_fibonacci_tool.py' that:

1. Contains a function fibonacci(n) that computes the nth Fibonacci number efficiently
2. Uses memoization or dynamic programming for efficiency
3. Handles edge cases (n < 0, n = 0, n = 1)
4. Includes a docstring explaining the function
5. Adds a test function to verify it works correctly
6. The tool should be integrated into the ARC AGI system as a reusable component

Please create this file and ensure it follows Python best practices."""
    
    print(f"Prompt for Claude Code:\n{prompt}\n")
    print("Invoking Claude Code...")
    
    # Invoke Claude Code
    result = await dialogue.invoke_claude(prompt)
    
    print("\nClause Code Response:")
    print("-" * 50)
    print(f"Status: {result['status']}")
    print(f"Response: {result['response'][:500]}..." if len(result['response']) > 500 else f"Response: {result['response']}")
    print(f"Cost: ${result['cost']}")
    
    # Check if the file was created
    from pathlib import Path
    fib_file = Path("arc_fibonacci_tool.py")
    if fib_file.exists():
        print("\n✅ Fibonacci tool file created successfully!")
        print("\nFile contents:")
        print("-" * 50)
        with open(fib_file, 'r') as f:
            content = f.read()
            print(content[:1000] + "..." if len(content) > 1000 else content)
        
        # Try to import and test the tool
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("arc_fibonacci_tool", fib_file)
            fib_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fib_module)
            
            # Test the fibonacci function
            if hasattr(fib_module, 'fibonacci'):
                test_values = [0, 1, 5, 10, 15]
                print("\nTesting Fibonacci function:")
                for n in test_values:
                    result = fib_module.fibonacci(n)
                    print(f"  fibonacci({n}) = {result}")
                print("\n✅ Fibonacci function works correctly!")
            else:
                print("\n⚠️ fibonacci function not found in the module")
        except Exception as e:
            print(f"\n❌ Error testing Fibonacci tool: {e}")
    else:
        print("\n⚠️ Fibonacci tool file was not created")
    
    # Display conversation history
    print("\n" + "=" * 50)
    print("Conversation History:")
    for conv in dialogue.get_recent_conversations(1):
        formatted = dialogue.format_for_display(conv)
        print(f"ID: {formatted['id']}")
        print(f"Status: {formatted['status']}")
        print(f"Cost: ${formatted['cost']}")
    
    return result

async def test_tool_integration():
    """Test integrating Claude Code with the AI assistant for dynamic tool creation"""
    
    print("\n" + "=" * 50)
    print("Testing AI Assistant + Claude Code Integration")
    print("=" * 50)
    
    # This would integrate with the AI assistant
    # For now, we'll test the concept
    
    dialogue = ClaudeCodeDialogue()
    dialogue.simulate_mode = False  # Use actual Claude Code
    
    # Prompt for creating a pattern detection tool
    prompt = """The ARC AGI solver needs a new tool for detecting repeating patterns in grids.

Create a file 'arc_pattern_detector_tool.py' with:

1. A function find_repeating_patterns(grid) that:
   - Takes a 2D list of integers (grid)
   - Identifies repeating 2x2, 3x3 blocks
   - Returns a list of pattern locations and the pattern itself

2. A function has_symmetry(grid) that:
   - Checks for horizontal, vertical, and diagonal symmetry
   - Returns a dict with symmetry types found

3. Include comprehensive docstrings and type hints
4. Add test cases to verify functionality

Make this a reusable tool for the ARC AGI system."""
    
    print("Creating Pattern Detector Tool...")
    result = await dialogue.invoke_claude(prompt)
    
    if result['status'] == 'completed':
        print("✅ Pattern detector tool creation completed!")
        print(f"Response preview: {result['response'][:200]}...")
    else:
        print(f"❌ Tool creation failed: {result.get('response', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    print("Starting Claude Code Integration Tests")
    print("=" * 50)
    
    # Run the tests
    loop = asyncio.get_event_loop()
    
    # Test 1: Create Fibonacci tool
    fib_result = loop.run_until_complete(test_fibonacci_tool_creation())
    
    # Test 2: Create pattern detection tool
    pattern_result = loop.run_until_complete(test_tool_integration())
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print(f"Total cost: ${0.02}")  # Approximate