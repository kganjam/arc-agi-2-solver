#!/usr/bin/env python3
"""Test AI improvements for grid manipulation and solution verification"""

import json
import requests

# Test the AI chat endpoint
def test_ai_functions():
    base_url = "http://localhost:8050"
    
    # Get current puzzle
    resp = requests.get(f"{base_url}/api/puzzle/current")
    puzzle_data = resp.json()
    print(f"✓ Current puzzle loaded: {puzzle_data.get('position', [0, 0])}")
    
    # Test AI chat with new functions
    test_commands = [
        {
            "message": "get_input_grid",
            "expected": "grid"
        },
        {
            "message": "copy the input grid to a 6x6 output using copy_grid function",
            "expected": "success"
        },
        {
            "message": "check if the solution is correct",
            "expected": "verification"
        },
        {
            "message": "analyze the pattern and suggest next step",
            "expected": "suggestion"
        }
    ]
    
    for test in test_commands:
        print(f"\n📝 Testing: {test['message']}")
        
        resp = requests.post(
            f"{base_url}/api/puzzle/ai-chat",
            json={
                "message": test["message"],
                "puzzle_id": puzzle_data.get("puzzle", {}).get("id"),
                "output_grid": [[0, 0], [0, 0]]
            }
        )
        
        if resp.status_code == 200:
            result = resp.json()
            
            # Check if function was executed
            if result.get("function_call"):
                print(f"  ✓ Function called: {result['function_call']['name']}")
            
            if result.get("function_result"):
                print(f"  ✓ Function result: {json.dumps(result['function_result'], indent=2)[:200]}...")
            
            if result.get("message"):
                print(f"  ✓ AI response: {result['message'][:100]}...")
                
            if result.get("output_grid"):
                grid = result["output_grid"]
                print(f"  ✓ Output grid updated: {len(grid)}x{len(grid[0]) if grid else 0}")
        else:
            print(f"  ✗ Error: {resp.status_code}")

if __name__ == "__main__":
    print("🧪 Testing AI Improvements")
    print("=" * 50)
    
    try:
        test_ai_functions()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")