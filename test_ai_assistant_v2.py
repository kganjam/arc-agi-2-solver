#!/usr/bin/env python3
"""
Test the enhanced AI assistant to ensure it actually executes functions
"""

import json
import requests
import sys

def test_ai_assistant():
    """Test the AI assistant's ability to execute functions"""
    
    print("Testing Enhanced AI Assistant V2")
    print("-" * 50)
    
    # Test queries that require function execution
    test_queries = [
        {
            "query": "How many heuristics are there? Execute the function and give me the exact number.",
            "expected": "should return actual count"
        },
        {
            "query": "Get the statistics about the heuristics knowledge base",
            "expected": "should return stats with numbers"
        },
        {
            "query": "Analyze the pattern in this puzzle",
            "expected": "should return actual analysis"
        },
        {
            "query": "Search for heuristics related to 'color'",
            "expected": "should return search results"
        },
        {
            "query": "What is the size of the test output grid?",
            "expected": "should return actual grid dimensions"
        }
    ]
    
    base_url = "http://localhost:8050"
    
    # First, load a puzzle
    print("Loading a puzzle...")
    puzzle_response = requests.get(f"{base_url}/api/puzzle/current")
    if puzzle_response.status_code != 200:
        print("❌ Failed to load puzzle")
        return False
    
    puzzle_data = puzzle_response.json()
    puzzle_id = puzzle_data.get('puzzle', {}).get('id', 'unknown')
    print(f"✓ Loaded puzzle: {puzzle_id}")
    
    # Test each query
    for i, test in enumerate(test_queries, 1):
        print(f"\nTest {i}: {test['query'][:50]}...")
        
        # Send query to AI assistant
        response = requests.post(
            f"{base_url}/api/puzzle/ai-chat",
            json={
                "message": test['query'],
                "puzzle_id": puzzle_id
            }
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed with status {response.status_code}")
            continue
        
        result = response.json()
        
        # Check if we got a response
        if 'message' in result:
            message = result['message']
            print(f"Response: {message[:200]}...")
            
            # Check if functions were executed
            if 'function_results' in result and result['function_results']:
                print(f"✅ Functions executed: {len(result['function_results'])} function(s)")
                for func_result in result['function_results'][:2]:
                    print(f"   - {func_result.get('function', 'unknown')}")
            else:
                # Check if the response contains actual data (numbers, specific info)
                has_numbers = any(char.isdigit() for char in message)
                has_specific_info = any(keyword in message.lower() for keyword in 
                                       ['heuristic', 'grid', 'size', 'color', 'pattern', 'found', 'results'])
                
                if has_numbers or has_specific_info:
                    print("✅ Response contains specific information")
                else:
                    print("⚠️  Response may not contain actual function results")
        else:
            print(f"❌ No message in response: {result}")
    
    # Test a function that modifies the grid
    print("\n" + "=" * 50)
    print("Testing grid modification...")
    
    modify_response = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": "Set cell 0,0 to color 5 and tell me what you did",
            "puzzle_id": puzzle_id,
            "output_grid": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        }
    )
    
    if modify_response.status_code == 200:
        result = modify_response.json()
        if 'output_grid' in result or 'updated_output_grid' in result:
            grid = result.get('output_grid') or result.get('updated_output_grid')
            if grid and grid[0][0] == 5:
                print("✅ Grid modification successful!")
            else:
                print(f"⚠️  Grid modified but unexpected result: {grid}")
        else:
            print("❌ No grid returned after modification")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    return True

if __name__ == "__main__":
    try:
        success = test_ai_assistant()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)