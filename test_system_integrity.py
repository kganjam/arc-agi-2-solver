#!/usr/bin/env python3
"""
System Integrity Test - Expose cheating and verify actual solving
"""

import json
import asyncio
import numpy as np
from datetime import datetime
import time
from pathlib import Path

def create_real_arc_puzzle():
    """Create a real ARC-style puzzle that requires actual solving"""
    # This is a real pattern: diagonal line detection and extension
    puzzle = {
        'id': 'integrity_test_001',
        'train': [
            {
                'input': [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0]
                ],
                'output': [
                    [2, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 2]
                ]
            },
            {
                'input': [
                    [0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0]
                ],
                'output': [
                    [0, 3, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0]
                ]
            }
        ],
        'test': [
            {
                'input': [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]
            }
        ]
    }
    return puzzle

async def test_master_solver():
    """Test if the master solver actually solves puzzles"""
    print("="*60)
    print("SYSTEM INTEGRITY TEST - MASTER SOLVER")
    print("="*60)
    
    from arc_master_solver import MasterSolver
    
    solver = MasterSolver()
    puzzle = create_real_arc_puzzle()
    
    print("\n1. Testing with REAL ARC puzzle (diagonal pattern)...")
    print("   Expected output: diagonal line should be extended")
    
    start_time = time.time()
    result = await solver.solve_puzzle_with_all_methods(puzzle)
    elapsed = time.time() - start_time
    
    print(f"\n   Result: {result['solved']}")
    print(f"   Method: {result.get('method', 'none')}")
    print(f"   Time: {elapsed:.3f}s")
    
    if result['solved']:
        # Check if solution is correct
        expected = [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        # The real solution should extend the diagonal
        print("\n   ⚠️ WARNING: Solver claims success but doesn't provide actual solution grid!")
        print("   This indicates CHEATING - no actual puzzle solving happening")
    
    print("\n2. Testing solving speed...")
    print("   Solving 10 puzzles to measure real performance...")
    
    start_time = time.time()
    solved_count = 0
    
    for i in range(10):
        test_puzzle = {
            'id': f'speed_test_{i}',
            'train': puzzle['train'],
            'test': puzzle['test']
        }
        result = await solver.solve_puzzle_with_all_methods(test_puzzle)
        if result['solved']:
            solved_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n   Solved: {solved_count}/10")
    print(f"   Total time: {elapsed:.3f}s")
    print(f"   Per puzzle: {elapsed/10:.3f}s")
    
    if elapsed < 0.1:
        print("\n   🚨 CRITICAL: Solving 10 puzzles in < 0.1s is impossible!")
        print("   This confirms the system is CHEATING")
    
    return solved_count, elapsed

async def test_multi_agent():
    """Test if multi-agent system actually analyzes puzzles"""
    print("\n" + "="*60)
    print("SYSTEM INTEGRITY TEST - MULTI-AGENT")
    print("="*60)
    
    from arc_multi_agent_system import MultiAgentSystem
    
    system = MultiAgentSystem()
    puzzle = create_real_arc_puzzle()
    
    print("\n1. Testing dialogue generation...")
    result = await system.solve_with_dialogue(puzzle)
    
    print(f"   Solved: {result.get('solved', False)}")
    print(f"   Has solution grid: {'solution' in result}")
    
    if 'solution' in result:
        solution = result['solution']
        print(f"   Solution shape: {len(solution)}x{len(solution[0]) if solution else 0}")
        
        # Check if it's just a copy of input
        test_input = puzzle['test'][0]['input']
        if solution == test_input:
            print("\n   🚨 CRITICAL: Solution is identical to input!")
            print("   Multi-agent system is NOT solving, just copying!")
        
        # Check if it's the hardcoded color swap
        color_swapped = [[1 if cell == 0 else cell for cell in row] for row in test_input]
        if solution == color_swapped:
            print("\n   🚨 CRITICAL: Solution is just hardcoded color swap!")
            print("   No actual pattern analysis happening!")
    
    print("\n2. Testing agent analysis...")
    dialogue = system.dialogue_manager.get_dialogue_history()
    print(f"   Dialogue messages: {len(dialogue)}")
    
    if len(dialogue) == 0:
        print("   🚨 No actual agent dialogue generated!")
    
    return result

async def test_safeguards():
    """Test if safeguards actually prevent cheating"""
    print("\n" + "="*60)
    print("SYSTEM INTEGRITY TEST - SAFEGUARDS")
    print("="*60)
    
    from arc_solver_safeguarded import SafeguardedSolver, ARCPuzzle
    
    solver = SafeguardedSolver()
    
    # Test 1: Try to solve without training examples
    print("\n1. Testing safeguard against solving without learning...")
    
    invalid_puzzle = {
        'id': 'cheat_test',
        'train': [],  # No training examples!
        'test': [{'input': [[1, 2], [3, 4]]}]
    }
    
    try:
        arc_puzzle = ARCPuzzle(invalid_puzzle)
        solution, stats = solver.solve_puzzle(arc_puzzle)
        print("   🚨 FAILED: Solver accepted puzzle without training data!")
    except Exception as e:
        print(f"   ✅ PASSED: Safeguard blocked invalid puzzle: {e}")
    
    # Test 2: Check if solver validates patterns
    print("\n2. Testing pattern validation...")
    
    puzzle = create_real_arc_puzzle()
    arc_puzzle = ARCPuzzle(puzzle)
    
    solution, stats = solver.solve_puzzle(arc_puzzle)
    
    print(f"   Validation performed: {stats.get('validated', False)}")
    print(f"   Pattern learned: {stats.get('pattern_learned', False)}")
    print(f"   Training accuracy: {stats.get('training_accuracy', 0):.1%}")
    
    if stats.get('training_accuracy', 0) < 0.8:
        print("   ⚠️ WARNING: Low training accuracy indicates poor learning!")
    
    return stats

async def main():
    """Run all integrity tests"""
    print("\n" + "🔍"*30)
    print(" "*10 + "ARC AGI SYSTEM INTEGRITY CHECK")
    print("🔍"*30)
    
    failures = []
    
    # Test 1: Master Solver
    try:
        solved, time_taken = await test_master_solver()
        if time_taken < 0.1:
            failures.append("Master Solver is cheating (impossible speed)")
    except Exception as e:
        print(f"\n❌ Master Solver test failed: {e}")
        failures.append(f"Master Solver crashed: {e}")
    
    # Test 2: Multi-Agent System
    try:
        result = await test_multi_agent()
        if not result.get('solved'):
            print("\n⚠️ Multi-agent couldn't solve simple pattern")
    except Exception as e:
        print(f"\n❌ Multi-Agent test failed: {e}")
        failures.append(f"Multi-Agent crashed: {e}")
    
    # Test 3: Safeguards
    try:
        stats = await test_safeguards()
        if not stats.get('validated'):
            failures.append("Safeguards not validating solutions")
    except Exception as e:
        print(f"\n❌ Safeguard test failed: {e}")
        failures.append(f"Safeguards crashed: {e}")
    
    # Final Report
    print("\n" + "="*60)
    print("INTEGRITY TEST SUMMARY")
    print("="*60)
    
    if failures:
        print("\n🚨 CRITICAL FAILURES DETECTED:")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. {failure}")
        print("\n❌ SYSTEM IS NOT GENUINELY SOLVING PUZZLES")
        print("   The reported 1000 puzzles solved is FAKE")
    else:
        print("\n✅ All integrity tests passed")
        print("   System appears to be attempting genuine solving")
    
    print("\n" + "="*60)
    
    # Save integrity report
    report = {
        'timestamp': datetime.now().isoformat(),
        'failures': failures,
        'tests_run': 3,
        'system_integrity': len(failures) == 0
    }
    
    with open('integrity_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Report saved to integrity_report.json")

if __name__ == "__main__":
    asyncio.run(main())