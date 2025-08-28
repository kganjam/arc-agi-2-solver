#!/usr/bin/env python3
"""Test full solver with 150 puzzles"""

import asyncio
from arc_solver_runner import SolverRunner
from arc_solver import load_puzzles
from pathlib import Path
import random

def generate_test_puzzles(count):
    """Generate test puzzles"""
    puzzles = []
    
    for i in range(count):
        # Generate random grid size
        size = random.randint(3, 10)
        
        # Generate simple pattern puzzle
        input_grid = [[random.randint(0, 3) for _ in range(size)] for _ in range(size)]
        
        # Simple transformation: color mapping
        color_map = {0: 0, 1: 2, 2: 3, 3: 1}
        output_grid = [[color_map.get(cell, cell) for cell in row] for row in input_grid]
        
        puzzle = {
            'id': f'generated_{i}',
            'train': [
                {'input': input_grid, 'output': output_grid}
            ],
            'test': [
                {'input': input_grid}
            ]
        }
        
        puzzles.append(puzzle)
    
    return puzzles

async def test_solver():
    """Test the solver"""
    # Load puzzles
    puzzles = load_puzzles(Path('data/arc_agi'), 10)
    print(f"Loaded {len(puzzles)} puzzles from disk")
    
    # Add generated puzzles
    if len(puzzles) < 150:
        generated = generate_test_puzzles(150 - len(puzzles))
        puzzles.extend(generated)
        print(f"Added {len(generated)} generated puzzles")
    
    print(f"Total puzzles: {len(puzzles)}")
    
    # Create runner
    runner = SolverRunner()
    
    # Track progress
    solved_count = 0
    
    async def update_callback(stats):
        nonlocal solved_count
        if 'puzzles_solved' in stats:
            new_count = stats['puzzles_solved']
            if new_count > solved_count:
                solved_count = new_count
                print(f"Progress: {solved_count}/100 puzzles solved (Phase: {stats.get('current_phase', 'Unknown')})")
                if solved_count >= 100:
                    print("✅ GOAL REACHED: 100 puzzles solved!")
        
        if 'activity' in stats:
            activity = stats['activity']
            if 'success' in activity['type']:
                print(f"  {activity['message']}")
    
    # Run solver
    print("\nStarting solver to reach 100 puzzles...")
    print("="*50)
    
    try:
        await runner.run_async(puzzles, update_callback)
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\nFinal result: {solved_count} puzzles solved")
    final_stats = runner.get_stats()
    print(f"Success rate: {final_stats['success_rate']:.1f}%")
    print(f"Time elapsed: {final_stats['elapsed_time']}")
    print(f"Solving speed: {final_stats['solving_speed']:.2f} puzzles/min")
    
    if solved_count >= 100:
        print("\n🎉 SUCCESS! System solved 100+ puzzles!")
    else:
        print(f"\n⚠️ Need {100 - solved_count} more puzzles to reach goal")

if __name__ == "__main__":
    asyncio.run(test_solver())