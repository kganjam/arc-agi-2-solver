#!/usr/bin/env python3
"""
Test for AI heuristic creation capability
Verifies that the AI can create new heuristics and add them to the knowledge base
"""

import sys
import json
from arc_puzzle_ai_assistant import PuzzleAIAssistant
from arc_unified_heuristics import unified_heuristics as heuristics_manager

def test_heuristic_creation():
    """Test that AI can create a new heuristic and it gets added to the knowledge base"""
    
    print("Testing AI Heuristic Creation...")
    print("-" * 50)
    
    # Initialize AI assistant
    ai = PuzzleAIAssistant()
    
    # Get initial heuristic count
    initial_heuristics = heuristics_manager.get_all_heuristics()
    initial_count = len(initial_heuristics)
    print(f"Initial heuristics count: {initial_count}")
    
    # Create a test heuristic
    test_heuristic = {
        "name": "Test Pattern Detector",
        "description": "A test heuristic for detecting diagonal patterns",
        "pattern_type": "pattern_completion",
        "conditions": ["diagonal_pattern", "test_condition"],
        "transformations": ["diagonal_fill", "test_transform"],
        "complexity": 3,
        "tags": ["test", "diagonal", "pattern"]
    }
    
    print("\nCreating new heuristic:")
    print(f"  Name: {test_heuristic['name']}")
    print(f"  Description: {test_heuristic['description']}")
    print(f"  Pattern Type: {test_heuristic['pattern_type']}")
    
    # Execute the create_heuristic function
    result = ai.execute_function("create_heuristic", test_heuristic)
    
    # Check if creation was successful
    if result.get("success"):
        print(f"\n✓ Heuristic created successfully!")
        print(f"  Heuristic ID: {result.get('heuristic_id')}")
        
        # Get updated heuristic count
        updated_heuristics = heuristics_manager.get_all_heuristics()
        updated_count = len(updated_heuristics)
        print(f"\nUpdated heuristics count: {updated_count}")
        
        # Verify the heuristic was added
        if updated_count == initial_count + 1:
            print("✓ Heuristic count increased by 1")
            
            # Find and verify the new heuristic
            new_heuristic = None
            for h in updated_heuristics:
                if h['name'] == test_heuristic['name']:
                    new_heuristic = h
                    break
            
            if new_heuristic:
                print("✓ New heuristic found in knowledge base")
                print("\nVerifying heuristic properties:")
                
                # Verify all properties
                checks = [
                    ("Name", new_heuristic['name'] == test_heuristic['name']),
                    ("Description", new_heuristic['description'] == test_heuristic['description']),
                    ("Pattern Type", new_heuristic['pattern_type'] == test_heuristic['pattern_type']),
                    ("Conditions", new_heuristic['conditions'] == test_heuristic['conditions']),
                    ("Transformations", new_heuristic['transformations'] == test_heuristic['transformations']),
                    ("Complexity", new_heuristic['complexity'] == test_heuristic['complexity']),
                    ("Tags", set(new_heuristic['tags']) == set(test_heuristic['tags']))
                ]
                
                all_passed = True
                for prop_name, check in checks:
                    if check:
                        print(f"  ✓ {prop_name} matches")
                    else:
                        print(f"  ✗ {prop_name} does not match")
                        all_passed = False
                
                # Test that we can retrieve the heuristic
                print("\nTesting heuristic retrieval:")
                search_result = heuristics_manager.search_heuristics("Test Pattern Detector")
                if search_result and len(search_result) > 0:
                    print("✓ Heuristic can be retrieved via search")
                else:
                    print("✗ Failed to retrieve heuristic via search")
                    all_passed = False
                
                # Test that the heuristic can be used
                print("\nTesting heuristic application:")
                test_puzzle_data = {
                    "puzzle_id": "test_puzzle",
                    "train": [],
                    "test": [],
                    "colors": [0, 1, 2],
                    "size_change": False
                }
                
                apply_result = heuristics_manager.apply_heuristic(
                    result.get('heuristic_id'),
                    test_puzzle_data
                )
                
                if apply_result.get("applied"):
                    print("✓ Heuristic can be applied")
                else:
                    print("✗ Failed to apply heuristic")
                    all_passed = False
                
                # Final result
                print("\n" + "=" * 50)
                if all_passed:
                    print("✅ TEST PASSED: AI successfully created and added a new heuristic")
                    return True
                else:
                    print("❌ TEST FAILED: Some verification checks failed")
                    return False
            else:
                print("✗ New heuristic not found in knowledge base")
                return False
        else:
            print(f"✗ Heuristic count mismatch. Expected {initial_count + 1}, got {updated_count}")
            return False
    else:
        print(f"\n✗ Failed to create heuristic: {result.get('message', 'Unknown error')}")
        return False

def cleanup_test_heuristic():
    """Remove the test heuristic from the knowledge base"""
    print("\nCleaning up test heuristic...")
    
    # Find and remove the test heuristic
    all_heuristics = heuristics_manager.get_all_heuristics()
    for h in all_heuristics:
        if h['name'] == "Test Pattern Detector":
            # Remove using the delete method
            if heuristics_manager.delete_heuristic(h['id']):
                print("✓ Test heuristic removed from knowledge base")
                return
    
    print("  Test heuristic not found (may have been already removed)")

if __name__ == "__main__":
    try:
        # Run the test
        success = test_heuristic_creation()
        
        # Cleanup
        cleanup_test_heuristic()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)