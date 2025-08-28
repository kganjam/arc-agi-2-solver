#!/usr/bin/env python3
"""Test solver runner"""

import asyncio
from arc_solver_runner import SolverRunner
from arc_solver import load_puzzles
from pathlib import Path

async def test_solver():
    """Test the solver"""
    # Load puzzles
    puzzles = load_puzzles(Path('data/arc_agi'), 10)
    print(f"Loaded {len(puzzles)} puzzles")
    
    if not puzzles:
        print("No puzzles found!")
        return
    
    # Create runner
    runner = SolverRunner()
    
    # Track updates
    updates = []
    
    async def update_callback(stats):
        print(f"Update: {stats}")
        updates.append(stats)
    
    # Run solver
    print("Starting solver...")
    await runner.run_async(puzzles, update_callback)
    
    print(f"Solver completed. Total updates: {len(updates)}")
    print(f"Final stats: {runner.get_stats()}")

if __name__ == "__main__":
    asyncio.run(test_solver())