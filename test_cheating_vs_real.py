#!/usr/bin/env python3
"""
Comparison Test: Cheating System vs Real System
Demonstrates the difference between fake and real solving
"""

import asyncio
import time
import json
from datetime import datetime

async def test_cheating_system():
    """Test the original cheating system"""
    print("\n" + "="*60)
    print("TESTING ORIGINAL (CHEATING) SYSTEM")
    print("="*60)
    
    from arc_master_solver import MasterSolver
    
    solver = MasterSolver()
    solver.target_puzzles = 10
    
    # Generate puzzles
    puzzles = solver.generate_synthetic_puzzles(10)
    
    start_time = time.time()
    solved_count = 0
    
    for puzzle in puzzles:
        result = await solver.solve_puzzle_with_all_methods(puzzle)
        if result['solved']:
            solved_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 Cheating System Results:")
    print(f"  Puzzles solved: {solved_count}/10")
    print(f"  Time taken: {elapsed:.4f}s")
    print(f"  Speed: {solved_count/elapsed:.1f} puzzles/second")
    print(f"  Solution provided: {result.get('solution') is not None}")
    print(f"  Training accuracy tracked: No")
    print(f"  Pattern learning: No")
    
    return {
        'solved': solved_count,
        'time': elapsed,
        'has_solutions': result.get('solution') is not None
    }

async def test_real_system():
    """Test the fixed real system"""
    print("\n" + "="*60)
    print("TESTING FIXED (REAL) SYSTEM")
    print("="*60)
    
    from arc_master_solver_fixed import FixedMasterSolver
    
    solver = FixedMasterSolver()
    
    # Create test puzzles
    puzzles = solver.create_test_puzzles(10)
    
    start_time = time.time()
    solved_count = 0
    total_confidence = 0
    total_training_acc = 0
    solutions = []
    
    for puzzle in puzzles:
        result = await solver.solve_puzzle_with_all_methods(puzzle)
        if result['solved']:
            solved_count += 1
            total_confidence += result.get('confidence', 0)
            total_training_acc += result.get('training_accuracy', 0)
            if result.get('solution'):
                solutions.append(result['solution'])
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 Real System Results:")
    print(f"  Puzzles solved: {solved_count}/10")
    print(f"  Time taken: {elapsed:.4f}s")
    print(f"  Speed: {solved_count/elapsed:.1f} puzzles/second")
    print(f"  Solutions provided: {len(solutions)}")
    print(f"  Average confidence: {total_confidence/max(1,solved_count):.1%}")
    print(f"  Average training accuracy: {total_training_acc/max(1,solved_count):.1%}")
    print(f"  Pattern learning: Yes")
    
    # Show a sample solution
    if solutions:
        print(f"\n  Sample solution grid:")
        for row in solutions[0][:3]:  # Show first 3 rows
            print(f"    {row}")
    
    return {
        'solved': solved_count,
        'time': elapsed,
        'has_solutions': len(solutions) > 0,
        'avg_confidence': total_confidence/max(1,solved_count),
        'avg_training_acc': total_training_acc/max(1,solved_count)
    }

async def main():
    """Run comparison test"""
    print("\n" + "🔬"*30)
    print(" "*10 + "CHEATING vs REAL SYSTEM COMPARISON")
    print("🔬"*30)
    
    # Test both systems
    cheating_results = await test_cheating_system()
    real_results = await test_real_system()
    
    # Comparison
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    print("\n🚨 CHEATING INDICATORS:")
    
    # Speed check
    cheating_speed = cheating_results['solved'] / max(0.001, cheating_results['time'])
    real_speed = real_results['solved'] / max(0.001, real_results['time'])
    
    print(f"\n1. SPEED:")
    print(f"   Cheating: {cheating_speed:.1f} puzzles/sec")
    print(f"   Real: {real_speed:.1f} puzzles/sec")
    
    if cheating_speed > 100:
        print("   ⚠️ CHEATING DETECTED: Impossible solving speed!")
    
    # Solution check
    print(f"\n2. SOLUTIONS:")
    print(f"   Cheating provides solutions: {cheating_results['has_solutions']}")
    print(f"   Real provides solutions: {real_results['has_solutions']}")
    
    if not cheating_results['has_solutions']:
        print("   ⚠️ CHEATING DETECTED: No actual solutions generated!")
    
    # Learning check
    print(f"\n3. LEARNING:")
    print(f"   Real system training accuracy: {real_results.get('avg_training_acc', 0):.1%}")
    print(f"   Real system confidence: {real_results.get('avg_confidence', 0):.1%}")
    print("   Cheating system: No learning metrics")
    
    # Final verdict
    print("\n" + "="*60)
    print("🏁 FINAL VERDICT")
    print("="*60)
    
    if cheating_speed > 100 or not cheating_results['has_solutions']:
        print("\n❌ ORIGINAL SYSTEM IS CHEATING")
        print("   - Solves puzzles instantly (impossible)")
        print("   - Doesn't generate actual solution grids")
        print("   - No pattern learning from training examples")
        print("   - Claims 1000 puzzles solved are FAKE")
    
    if real_results['has_solutions'] and real_results.get('avg_training_acc', 0) > 0:
        print("\n✅ FIXED SYSTEM IS LEGITIMATE")
        print("   - Takes realistic time to solve")
        print("   - Generates actual solution grids")
        print("   - Learns patterns from training examples")
        print("   - Validates solutions against training data")
        print("   - Has realistic success rates")
    
    # Save comparison report
    report = {
        'timestamp': datetime.now().isoformat(),
        'cheating_system': cheating_results,
        'real_system': real_results,
        'verdict': {
            'original_is_cheating': cheating_speed > 100 or not cheating_results['has_solutions'],
            'fixed_is_legitimate': real_results['has_solutions'] and real_results.get('avg_training_acc', 0) > 0
        }
    }
    
    with open('cheating_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nReport saved to cheating_comparison_report.json")

if __name__ == "__main__":
    asyncio.run(main())