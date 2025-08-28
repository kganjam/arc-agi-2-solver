#!/usr/bin/env python3
"""
Test fibonacci through the web API to diagnose the issue
"""

import requests
import json

def test_fibonacci_web():
    """Test fibonacci through the web API"""
    
    base_url = "http://localhost:8050"
    
    # Get current puzzle for context
    puzzle_response = requests.get(f"{base_url}/api/puzzle/current")
    if puzzle_response.status_code != 200:
        print("Failed to load puzzle")
        return
    
    puzzle_data = puzzle_response.json()
    puzzle_id = puzzle_data.get('puzzle', {}).get('id', 'unknown')
    
    # Test different ways to request fibonacci
    test_requests = [
        "print fibonacci(20)",
        "Calculate fibonacci(20)",
        "What is fibonacci(20)?",
        "Use the invoke_generated_tool function to call the 'fibonacci' tool with parameters {'n': 20}"
    ]
    
    for request_text in test_requests:
        print(f"\nTesting: {request_text}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/api/puzzle/ai-chat",
                json={
                    "message": request_text,
                    "puzzle_id": puzzle_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Status: 200 OK")
                print(f"Message: {result.get('message', '')[:200]}")
                
                if 'function_results' in result:
                    print(f"Functions executed: {len(result.get('function_results', []))}")
                    for func_result in result.get('function_results', []):
                        print(f"  - {func_result.get('function')}: {func_result.get('result', {})}")
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
            else:
                print(f"Status: {response.status_code}")
                print(f"Error: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print("Request timed out!")
        except Exception as e:
            print(f"Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete")

if __name__ == "__main__":
    test_fibonacci_web()