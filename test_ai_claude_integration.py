#!/usr/bin/env python3
"""
Test AI Assistant with Claude Code integration for dynamic tool generation
"""

import requests
import json
import time

def test_ai_claude_integration():
    """Test the AI assistant's ability to generate tools using Claude Code"""
    
    print("Testing AI Assistant + Claude Code Integration")
    print("=" * 50)
    
    base_url = "http://localhost:8050"
    
    # First, load a puzzle for context
    print("Loading puzzle...")
    puzzle_response = requests.get(f"{base_url}/api/puzzle/current")
    if puzzle_response.status_code != 200:
        print("❌ Failed to load puzzle")
        return False
    
    puzzle_data = puzzle_response.json()
    puzzle_id = puzzle_data.get('puzzle', {}).get('id', 'unknown')
    print(f"✓ Loaded puzzle: {puzzle_id}")
    
    # Test 1: List existing generated tools
    print("\n1. Testing list of generated tools...")
    response = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": "List all generated tools using the list_generated_tools function",
            "puzzle_id": puzzle_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result.get('message', '')[:200]}...")
        if 'function_results' in result:
            print(f"✓ Function executed, found {len(result['function_results'])} result(s)")
    
    # Test 2: Generate a new tool using Claude Code
    print("\n2. Testing tool generation with Claude Code...")
    
    tool_request = """Use the generate_tool_with_claude function to create a new tool called 'symmetry_detector' with this description:
    
    'A tool that detects symmetrical patterns in grids. It should:
    - Check for horizontal symmetry (top-bottom mirror)
    - Check for vertical symmetry (left-right mirror)
    - Check for rotational symmetry (90, 180, 270 degrees)
    - Check for diagonal symmetry
    - Return a dict with all symmetry types found
    - Handle non-square grids appropriately'
    
    Please execute the function and generate this tool."""
    
    response = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": tool_request,
            "puzzle_id": puzzle_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result.get('message', '')[:300]}...")
        
        if 'function_results' in result and result['function_results']:
            for func_result in result['function_results']:
                if func_result.get('function') == 'generate_tool_with_claude':
                    tool_result = func_result.get('result', {})
                    if isinstance(tool_result, dict) and tool_result.get('success'):
                        print(f"✅ Tool generated successfully: {tool_result.get('file_path')}")
                    else:
                        print(f"⚠️ Tool generation result: {tool_result}")
        else:
            print("⚠️ No function results returned")
    
    # Test 3: Test the Fibonacci tool invocation
    print("\n3. Testing Fibonacci tool invocation...")
    
    fib_request = """Use the invoke_generated_tool function to call the 'fibonacci' tool with parameters {'n': 10}"""
    
    response = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": fib_request,
            "puzzle_id": puzzle_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if 'function_results' in result and result['function_results']:
            for func_result in result['function_results']:
                if func_result.get('function') == 'invoke_generated_tool':
                    tool_result = func_result.get('result', {})
                    print(f"Fibonacci tool result: {tool_result}")
                    if 'success' in tool_result or 'result' in tool_result:
                        print("✅ Fibonacci tool executed successfully!")
    
    # Test 4: Complex pattern analysis with tool generation
    print("\n4. Testing complex pattern analysis...")
    
    complex_request = """Analyze the current puzzle and determine if we need any new tools. 
    If you identify a missing capability, use generate_tool_with_claude to create it.
    Execute all necessary functions to provide a complete analysis."""
    
    response = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": complex_request,
            "puzzle_id": puzzle_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis response: {result.get('message', '')[:400]}...")
        
        if 'function_results' in result:
            print(f"Functions executed: {len(result['function_results'])}")
            for func_result in result['function_results']:
                print(f"  - {func_result.get('function', 'unknown')}")
    
    print("\n" + "=" * 50)
    print("Integration test complete!")
    return True

if __name__ == "__main__":
    import sys
    
    # Make sure the app is running
    print("Note: Make sure arc_integrated_app.py is running on port 8050")
    time.sleep(2)
    
    try:
        success = test_ai_claude_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)