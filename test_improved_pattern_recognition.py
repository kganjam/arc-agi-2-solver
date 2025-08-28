#!/usr/bin/env python3
"""
Test script to verify the improved pattern recognition capabilities
Specifically tests the 2x2 -> 6x6 cell expansion pattern that previously failed
"""

import sys
import os
sys.path.append('.')

from arc_puzzle_ai_assistant_v2 import SmartPuzzleAIAssistant

def create_test_puzzle():
    """Create a test puzzle with 2x2 -> 6x6 cell expansion pattern"""
    # Training example: 2x2 input -> 6x6 output (each cell becomes 3x3 block)
    train_input = [
        [3, 1],
        [1, 2]
    ]
    
    # Each cell expanded to 3x3 block
    train_output = [
        [3, 3, 3, 1, 1, 1],
        [3, 3, 3, 1, 1, 1], 
        [3, 3, 3, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2]
    ]
    
    # Test input (different 2x2 pattern)
    test_input = [
        [2, 4],
        [0, 3]
    ]
    
    puzzle = {
        'id': 'test_cell_expansion',
        'train': [
            {
                'input': train_input,
                'output': train_output
            }
        ],
        'test': [
            {
                'input': test_input
            }
        ]
    }
    
    return puzzle

def test_pattern_recognition():
    """Test the improved pattern recognition"""
    print("🧪 Testing Improved Pattern Recognition")
    print("="*50)
    
    # Create AI assistant
    ai = SmartPuzzleAIAssistant()
    
    # Set up test puzzle
    puzzle = create_test_puzzle()
    ai.set_puzzle(puzzle)
    
    print("📊 Test Puzzle:")
    print(f"  Input size: 2x2 -> Output size: 6x6")
    print(f"  Expected pattern: Each cell becomes 3x3 block")
    print()
    
    # Test 1: Detailed pattern analysis
    print("🔍 Test 1: Pattern Analysis")
    analysis = ai._analyze_pattern_detailed()
    
    print(f"  Cell expansion detected: {analysis.get('cell_expansion_detected', False)}")
    print(f"  Scaling factor: {analysis.get('scaling_factor', 'None')}")
    print(f"  Size changes: {len(analysis.get('size_changes', []))}")
    
    if analysis.get('scaling_patterns'):
        for pattern in analysis['scaling_patterns']:
            print(f"  Pattern: {pattern['pattern']}")
    print()
    
    # Test 2: Transformation detection
    print("🎯 Test 2: Transformation Detection")
    transformation = ai._find_transformation()
    
    print(f"  Primary pattern: {transformation.get('primary_pattern', 'unknown')}")
    print(f"  Suggestions count: {len(transformation.get('suggestions', []))}")
    
    for suggestion in transformation.get('suggestions', [])[:3]:  # Show first 3
        print(f"  - {suggestion}")
    print()
    
    # Test 3: Pattern type detection
    print("🔎 Test 3: Pattern Type Detection")
    pattern_type = ai._detect_pattern_type()
    
    print(f"  Primary pattern: {pattern_type.get('primary_pattern', 'unknown')}")
    print(f"  Confidence: {pattern_type.get('confidence', 0):.2f}")
    print(f"  Pattern scores: {pattern_type.get('pattern_scores', {})}")
    print()
    
    # Test 4: Next step suggestion
    print("💡 Test 4: Next Step Suggestion")
    suggestion = ai._suggest_next_step()
    
    print(f"  Suggestion: {suggestion.get('suggestion', 'None')}")
    print(f"  Confidence: {suggestion.get('confidence', 0):.2f}")
    print(f"  Reasoning: {suggestion.get('reasoning', 'None')}")
    
    if suggestion.get('function_call'):
        func_call = suggestion['function_call']
        print(f"  Recommended function: {func_call['name']}")
        print(f"  Parameters: {func_call['parameters']}")
    print()
    
    # Test 5: Apply cell expansion
    print("🔧 Test 5: Applying Cell Expansion")
    
    if analysis.get('scaling_factor'):
        factor = analysis['scaling_factor']
        result = ai._apply_cell_expansion(factor)
        
        if result.get('success'):
            print(f"  ✅ Success: {result['message']}")
            print(f"  Input size: {result['input_size']}")
            print(f"  Output size: {result['output_size']}")
            print(f"  Transformation: {result['transformation']}")
            
            # Verify the output grid
            if ai.output_grid:
                expected_size = (6, 6)  # 2x2 * 3 = 6x6
                actual_size = (len(ai.output_grid), len(ai.output_grid[0]))
                print(f"  Grid verification: Expected {expected_size}, Got {actual_size}")
                
                if actual_size == expected_size:
                    print("  ✅ Grid size is correct!")
                    
                    # Check first few cells to verify expansion
                    print("  Sample output (first row):", ai.output_grid[0])
                else:
                    print("  ❌ Grid size mismatch!")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    else:
        print("  ⚠️  No scaling factor detected")
    print()
    
    # Test 6: Function execution via execute_function
    print("🎮 Test 6: Function Execution Interface")
    
    # Test the execute_function interface
    result = ai.execute_function("detect_pattern_type", {})
    print(f"  detect_pattern_type result: {result.get('primary_pattern', 'error')}")
    
    result = ai.execute_function("suggest_next_step", {})
    print(f"  suggest_next_step result: {result.get('suggestion', 'error')[:50]}...")
    
    if analysis.get('scaling_factor'):
        result = ai.execute_function("apply_cell_expansion", {"scaling_factor": analysis['scaling_factor']})
        print(f"  apply_cell_expansion result: {result.get('success', False)}")
    
    print("\n" + "="*50)
    print("🎉 Testing Complete!")
    
    # Summary
    success_indicators = [
        analysis.get('cell_expansion_detected', False),
        analysis.get('scaling_factor') == 3,
        transformation.get('primary_pattern') == 'cell_expansion',
        pattern_type.get('confidence', 0) > 0.8
    ]
    
    success_count = sum(success_indicators)
    print(f"📈 Success Rate: {success_count}/4 tests passed")
    
    if success_count >= 3:
        print("✅ PASS: The improved AI should now correctly recognize cell expansion patterns!")
        return True
    else:
        print("❌ FAIL: Some improvements may need additional work")
        return False

if __name__ == "__main__":
    test_pattern_recognition()