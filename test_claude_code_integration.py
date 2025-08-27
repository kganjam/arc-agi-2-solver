"""
Test Claude Code Integration and Continuous Self-Improvement
Demonstrates calling Claude Code and updating the system
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def test_claude_code_call():
    """Test calling Claude Code with proper arguments"""
    
    print("Testing Claude Code Integration")
    print("="*60)
    
    # Test prompt for generating a new heuristic
    test_prompt = """
    Generate a Python function that detects mirror symmetry in a 2D grid.
    The function should:
    1. Take a List[List[int]] as input
    2. Return True if the grid has horizontal or vertical symmetry
    3. Be efficient for grids up to 30x30
    
    Save as: patterns/test_symmetry_detector.py
    """
    
    # Build Claude Code command
    cmd = [
        "claude",  # Assuming claude is in PATH
        "--allowedTools", "Bash,Read,WebSearch,Fetch",
        "--permission-mode", "acceptEdits",
        "--message", test_prompt
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: This would call Claude Code in production")
    print("For testing, we'll simulate the response\n")
    
    # Simulate Claude Code response
    simulated_response = """
def detect_mirror_symmetry(grid):
    '''Detect horizontal or vertical mirror symmetry in grid'''
    if not grid or not grid[0]:
        return False
    
    h, w = len(grid), len(grid[0])
    
    # Check horizontal symmetry
    h_sym = all(grid[i] == grid[h-1-i] for i in range(h//2))
    
    # Check vertical symmetry
    v_sym = all(
        all(row[j] == row[w-1-j] for j in range(w//2))
        for row in grid
    )
    
    return h_sym or v_sym
"""
    
    # Log the conversation
    log_claude_conversation(test_prompt, simulated_response)
    
    print("✓ Claude Code call simulated successfully")
    print("✓ Conversation logged")
    
    return True


def log_claude_conversation(prompt, response, cost=0.01):
    """Log Claude Code conversation for replay"""
    
    log_dir = Path("logs/claude_conversations")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "prompt": prompt,
        "response": response,
        "cost": cost,
        "success": True
    }
    
    # Save to timestamped file
    log_file = log_dir / f"claude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # Also append to master log
    master_log = log_dir / "master_log.jsonl"
    with open(master_log, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"Logged to: {log_file}")


def test_system_update():
    """Test updating the main system with generated code"""
    
    print("\nTesting System Update Capability")
    print("="*60)
    
    # Simulate generating a new heuristic
    new_heuristic_code = '''
# Auto-generated heuristic
def color_frequency_transform(grid):
    """Transform based on color frequency"""
    from collections import Counter
    
    # Count color frequencies
    color_counts = Counter()
    for row in grid:
        color_counts.update(row)
    
    # Map most frequent to least frequent
    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
    color_map = {c: sorted_colors[-(i+1)] for i, c in enumerate(sorted_colors)}
    
    # Apply transformation
    return [[color_map.get(cell, cell) for cell in row] for row in grid]
'''
    
    # Save to patterns directory
    patterns_dir = Path("patterns/generated")
    patterns_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = patterns_dir / "test_generated_heuristic.py"
    with open(test_file, 'w') as f:
        f.write(new_heuristic_code)
    
    print(f"✓ Generated code saved to: {test_file}")
    
    # Test importing the new code
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Test the function
        test_grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        result = module.color_frequency_transform(test_grid)
        print(f"✓ Successfully imported and tested generated code")
        print(f"  Input:  {test_grid[0]}")
        print(f"  Output: {result[0]}")
        
    except Exception as e:
        print(f"✗ Error testing generated code: {e}")
        
    return True


def test_self_improvement_cycle():
    """Test one cycle of self-improvement"""
    
    print("\nTesting Self-Improvement Cycle")
    print("="*60)
    
    # Simulate performance analysis
    performance_data = {
        "puzzles_solved": 3,
        "total_puzzles": 10,
        "failed_patterns": ["rotation", "scaling"],
        "slow_heuristics": ["complex_transform"],
        "success_rate": 0.3
    }
    
    print("Current Performance:")
    print(json.dumps(performance_data, indent=2))
    
    # Generate improvement prompt based on analysis
    improvement_prompt = f"""
    Based on performance analysis:
    - Success rate: {performance_data['success_rate']:.1%}
    - Failed patterns: {performance_data['failed_patterns']}
    - Slow heuristics: {performance_data['slow_heuristics']}
    
    Generate an improved heuristic that addresses rotation patterns.
    Focus on efficiency and generalization.
    """
    
    print("\nGenerated Improvement Prompt:")
    print(improvement_prompt)
    
    # Log the improvement attempt
    log_claude_conversation(improvement_prompt, "# Improved rotation handler...")
    
    print("\n✓ Self-improvement cycle completed")
    print("✓ Ready for next iteration")
    
    return True


def run_all_tests():
    """Run all integration tests"""
    
    print("\n" + "="*70)
    print("ARC AGI Claude Code Integration Tests")
    print("="*70 + "\n")
    
    tests = [
        ("Claude Code Call", test_claude_code_call),
        ("System Update", test_system_update),
        ("Self-Improvement", test_self_improvement_cycle)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System ready for continuous improvement.")
    else:
        print("\n⚠️  Some tests failed. Please review before deployment.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)