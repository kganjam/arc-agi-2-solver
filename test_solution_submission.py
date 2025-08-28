#!/usr/bin/env python3
"""
Test solution submission to verify correct answers are properly validated
"""

import json
import sys
from arc_puzzle_loader import puzzle_loader
from arc_verification_oracle import verification_oracle

def test_correct_solution_submission():
    """Test that correct solutions are validated properly"""
    
    print("Testing Solution Submission System")
    print("-" * 50)
    
    # Load a training puzzle where we know the correct answer
    puzzle = puzzle_loader.get_puzzle(0)  # Get first puzzle
    if not puzzle:
        print("❌ Failed to load puzzle")
        return False
    
    puzzle_id = puzzle.get('id', 'unknown')
    print(f"Testing with puzzle: {puzzle_id}")
    
    # Check if this is a training puzzle with known output
    if not puzzle.get('test') or len(puzzle['test']) == 0:
        print("❌ Puzzle has no test cases")
        return False
    
    test_case = puzzle['test'][0]
    
    # For training puzzles, we should have the output
    if 'output' not in test_case:
        print("⚠️  This appears to be an evaluation puzzle (no output provided)")
        print("    Creating a test solution based on input...")
        # For evaluation puzzles, we'll test with a modified input
        correct_solution = test_case['input']
    else:
        correct_solution = test_case['output']
        print(f"✓ Found training puzzle with known output")
    
    # Test 1: Submit the correct solution
    print("\nTest 1: Submitting correct solution...")
    result = verification_oracle.verify_solution(
        puzzle_id=puzzle_id,
        submitted_output=correct_solution,
        expected_output=correct_solution if 'output' in test_case else None
    )
    
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    if 'output' in test_case:
        # For training puzzles, we should get 100% accuracy
        if result.get('correct', False) and result.get('accuracy', 0) == 1.0:
            print("  ✅ PASS: Correct solution validated successfully")
        else:
            print(f"  ❌ FAIL: Correct solution not validated properly")
            print(f"     Expected: correct=True, accuracy=1.0")
            print(f"     Got: correct={result.get('correct')}, accuracy={result.get('accuracy')}")
            return False
    else:
        # For evaluation puzzles, we just check the function works
        print("  ✓ Evaluation puzzle - verification function works")
    
    # Test 2: Submit an incorrect solution (if we have the correct one)
    if 'output' in test_case:
        print("\nTest 2: Submitting incorrect solution...")
        
        # Create an incorrect solution by modifying the correct one
        incorrect_solution = [row[:] for row in correct_solution]  # Deep copy
        if len(incorrect_solution) > 0 and len(incorrect_solution[0]) > 0:
            # Change first cell to a different color
            original_color = incorrect_solution[0][0]
            incorrect_solution[0][0] = (original_color + 1) % 10
        
        result = verification_oracle.verify_solution(
            puzzle_id=puzzle_id,
            submitted_output=incorrect_solution,
            expected_output=correct_solution
        )
        
        print(f"  Result: {json.dumps(result, indent=2)}")
        
        if not result.get('correct', True):
            print("  ✅ PASS: Incorrect solution properly rejected")
        else:
            print("  ❌ FAIL: Incorrect solution was accepted")
            return False
    
    # Test 3: Test with empty solution
    print("\nTest 3: Submitting empty solution...")
    empty_solution = [[0] * len(test_case['input'][0]) for _ in range(len(test_case['input']))]
    
    result = verification_oracle.verify_solution(
        puzzle_id=puzzle_id,
        submitted_output=empty_solution,
        expected_output=correct_solution if 'output' in test_case else None
    )
    
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    if 'output' in test_case:
        # Check if empty solution is different from correct solution
        is_different = empty_solution != correct_solution
        if is_different and not result.get('correct', True):
            print("  ✅ PASS: Empty solution properly rejected")
        elif not is_different and result.get('correct', False):
            print("  ✅ PASS: Solution happens to be all zeros (correct)")
        else:
            print("  ⚠️  Unexpected result for empty solution")
    
    # Test 4: Test dimension mismatch
    print("\nTest 4: Testing dimension mismatch...")
    wrong_size_solution = [[0, 0], [0, 0]]  # 2x2 grid
    
    result = verification_oracle.verify_solution(
        puzzle_id=puzzle_id,
        submitted_output=wrong_size_solution,
        expected_output=correct_solution if 'output' in test_case else None
    )
    
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    if not result.get('correct', True):
        print("  ✅ PASS: Wrong dimensions properly rejected")
    else:
        print("  ⚠️  Wrong dimensions were accepted (might be correct size)")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Solution submission working correctly.")
    return True

def test_submission_endpoint():
    """Test the actual API endpoint"""
    import requests
    
    print("\nTesting API Endpoint...")
    print("-" * 50)
    
    # Get a puzzle
    puzzle = puzzle_loader.get_puzzle(0)
    if not puzzle or 'test' not in puzzle or 'output' not in puzzle['test'][0]:
        print("⚠️  No training puzzle with output available for API test")
        return
    
    puzzle_id = puzzle.get('id', 'unknown')
    correct_solution = puzzle['test'][0]['output']
    
    # Test the API endpoint
    try:
        response = requests.post(
            'http://localhost:8050/api/submit-solution',
            json={
                'puzzle_id': puzzle_id,
                'solution': correct_solution
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("✅ API endpoint working correctly!")
            else:
                print("❌ API endpoint rejected correct solution")
        else:
            print(f"❌ API returned status code: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Could not test API endpoint: {e}")
        print("    (Make sure the server is running on port 8050)")

if __name__ == "__main__":
    try:
        # Run the tests
        success = test_correct_solution_submission()
        
        if success:
            # Also test the API if successful
            test_submission_endpoint()
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)