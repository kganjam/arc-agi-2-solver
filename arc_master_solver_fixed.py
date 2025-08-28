#!/usr/bin/env python3
"""
Fixed Master Solver - Uses REAL solving, no cheating
Integrates all subsystems with actual puzzle solving
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

# Import real solver
from arc_real_solver import RealARCSolver, PatternAnalyzer

class FixedMultiAgentSolver:
    """Fixed multi-agent system that uses real solving"""
    
    def __init__(self):
        self.real_solver = RealARCSolver()
        self.agents = {
            'pattern': 'Pattern Analyst',
            'transform': 'Transform Specialist',
            'validator': 'Validator',
            'strategist': 'Strategist'
        }
        
    async def solve_with_dialogue(self, puzzle: Dict) -> Dict:
        """Solve puzzle with agent dialogue using real solver"""
        result = {
            'puzzle_id': puzzle.get('id', 'unknown'),
            'solved': False,
            'solution': None,
            'dialogue': [],
            'method': 'multi_agent_real'
        }
        
        # Generate dialogue
        result['dialogue'].append({
            'agent': 'Pattern Analyst',
            'message': 'Analyzing training examples for patterns...'
        })
        
        # Use real solver
        real_result = self.real_solver.solve_puzzle(puzzle)
        
        # Add agent dialogue based on real results
        for step in real_result.get('explanation', []):
            agent = random.choice(list(self.agents.values()))
            result['dialogue'].append({
                'agent': agent,
                'message': step
            })
        
        # Copy real results
        result['solved'] = real_result['solved']
        result['solution'] = real_result.get('solution')
        result['confidence'] = real_result.get('confidence', 0.0)
        result['training_accuracy'] = real_result.get('training_accuracy', 0.0)
        
        if result['solved']:
            result['dialogue'].append({
                'agent': 'Validator',
                'message': f"Solution validated with {result['training_accuracy']:.1%} accuracy"
            })
        
        return result

class FixedMasterSolver:
    """Fixed master solver with real puzzle solving"""
    
    def __init__(self):
        self.real_solver = RealARCSolver()
        self.multi_agent = FixedMultiAgentSolver()
        
        # Tracking
        self.total_puzzles_solved = 0
        self.total_attempts = 0
        self.solving_strategies = {
            'real_solver': 0,
            'multi_agent': 0,
            'pattern_learning': 0
        }
        self.solved_puzzles = []
        self.failed_puzzles = []
        
    async def solve_puzzle_with_all_methods(self, puzzle: Dict) -> Dict:
        """Try to solve puzzle using real methods"""
        puzzle_id = puzzle.get('id', 'unknown')
        
        result = {
            'puzzle_id': puzzle_id,
            'solved': False,
            'solution': None,
            'method': None,
            'attempts': 0,
            'time_taken': 0,
            'confidence': 0.0,
            'training_accuracy': 0.0
        }
        
        start_time = time.time()
        
        # Method 1: Direct real solver
        try:
            real_result = self.real_solver.solve_puzzle(puzzle)
            if real_result['solved']:
                result['solved'] = True
                result['solution'] = real_result['solution']
                result['method'] = 'real_solver'
                result['confidence'] = real_result['confidence']
                result['training_accuracy'] = real_result['training_accuracy']
                self.solving_strategies['real_solver'] += 1
        except Exception as e:
            print(f"Real solver error: {e}")
        
        # Method 2: Multi-agent with real solving
        if not result['solved']:
            try:
                ma_result = await self.multi_agent.solve_with_dialogue(puzzle)
                if ma_result['solved']:
                    result['solved'] = True
                    result['solution'] = ma_result['solution']
                    result['method'] = 'multi_agent'
                    result['confidence'] = ma_result.get('confidence', 0.5)
                    result['training_accuracy'] = ma_result.get('training_accuracy', 0.0)
                    self.solving_strategies['multi_agent'] += 1
            except Exception as e:
                print(f"Multi-agent error: {e}")
        
        result['time_taken'] = time.time() - start_time
        result['attempts'] = self.total_attempts
        
        return result
    
    def load_real_arc_puzzles(self, limit: int = 10) -> List[Dict]:
        """Load real ARC puzzles from files"""
        puzzles = []
        
        # Try to load from data directory
        data_paths = [
            Path("data/arc_agi_full/data/training"),
            Path("data/arc_agi/training"),
            Path("data/training")
        ]
        
        for data_path in data_paths:
            if data_path.exists():
                json_files = list(data_path.glob("*.json"))[:limit]
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            puzzle = json.load(f)
                            puzzle['id'] = json_file.stem
                            puzzles.append(puzzle)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                        
                if puzzles:
                    break
        
        # If no real puzzles found, create test puzzles
        if not puzzles:
            print("No real ARC puzzles found. Creating test puzzles...")
            puzzles = self.create_test_puzzles(limit)
        
        return puzzles
    
    def create_test_puzzles(self, count: int) -> List[Dict]:
        """Create test puzzles with real patterns"""
        puzzles = []
        
        patterns = [
            # Rotation pattern
            {
                'id': f'test_rotation_{i}',
                'train': [
                    {'input': [[1, 0, 0], [0, 0, 0], [0, 0, 0]], 
                     'output': [[0, 0, 1], [0, 0, 0], [0, 0, 0]]},
                    {'input': [[2, 0, 0], [0, 0, 0], [0, 0, 0]], 
                     'output': [[0, 0, 2], [0, 0, 0], [0, 0, 0]]}
                ],
                'test': [{'input': [[3, 0, 0], [0, 0, 0], [0, 0, 0]]}]
            }
            for i in range(min(count, 3))
        ]
        
        # Color mapping pattern
        patterns.extend([
            {
                'id': f'test_color_{i}',
                'train': [
                    {'input': [[1, 2], [2, 1]], 'output': [[2, 1], [1, 2]]},
                    {'input': [[1, 1], [2, 2]], 'output': [[2, 2], [1, 1]]}
                ],
                'test': [{'input': [[2, 1], [1, 2]]}]
            }
            for i in range(min(count - 3, 3))
        ])
        
        # Flip pattern
        patterns.extend([
            {
                'id': f'test_flip_{i}',
                'train': [
                    {'input': [[1, 0], [0, 0]], 'output': [[0, 1], [0, 0]]},
                    {'input': [[0, 2], [0, 0]], 'output': [[2, 0], [0, 0]]}
                ],
                'test': [{'input': [[3, 0], [0, 0]]}]
            }
            for i in range(min(count - 6, 4))
        ])
        
        return patterns[:count]
    
    async def run_test(self, num_puzzles: int = 10):
        """Run test with specified number of puzzles"""
        print(f"\nStarting Fixed Master Solver Test - {num_puzzles} puzzles")
        print("="*60)
        
        # Load puzzles
        puzzles = self.load_real_arc_puzzles(num_puzzles)
        print(f"Loaded {len(puzzles)} puzzles")
        
        start_time = time.time()
        
        # Solve puzzles
        for i, puzzle in enumerate(puzzles):
            print(f"\nPuzzle {i+1}/{len(puzzles)}: {puzzle.get('id', 'unknown')}")
            
            self.total_attempts += 1
            result = await self.solve_puzzle_with_all_methods(puzzle)
            
            if result['solved']:
                self.total_puzzles_solved += 1
                self.solved_puzzles.append(result)
                print(f"  ✓ Solved using {result['method']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  Training accuracy: {result['training_accuracy']:.1%}")
                print(f"  Time: {result['time_taken']:.3f}s")
                
                # Show solution grid
                if result['solution'] and len(result['solution']) <= 5:
                    print("  Solution:")
                    for row in result['solution']:
                        print(f"    {row}")
            else:
                self.failed_puzzles.append(result)
                print(f"  ✗ Failed to solve")
        
        total_time = time.time() - start_time
        
        # Print statistics
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Total puzzles: {len(puzzles)}")
        print(f"Solved: {self.total_puzzles_solved}")
        print(f"Failed: {len(self.failed_puzzles)}")
        print(f"Success rate: {self.total_puzzles_solved/len(puzzles)*100:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per puzzle: {total_time/len(puzzles):.3f}s")
        
        print("\nSolving strategies used:")
        for strategy, count in self.solving_strategies.items():
            if count > 0:
                print(f"  {strategy}: {count}")
        
        # Calculate average confidence
        if self.solved_puzzles:
            avg_confidence = np.mean([p['confidence'] for p in self.solved_puzzles])
            avg_training_acc = np.mean([p['training_accuracy'] for p in self.solved_puzzles])
            print(f"\nAverage confidence: {avg_confidence:.1%}")
            print(f"Average training accuracy: {avg_training_acc:.1%}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_puzzles': len(puzzles),
            'total_solved': self.total_puzzles_solved,
            'success_rate': self.total_puzzles_solved/len(puzzles) if puzzles else 0,
            'total_time_seconds': total_time,
            'solving_strategies': self.solving_strategies,
            'average_time_per_puzzle': total_time/len(puzzles) if puzzles else 0
        }
        
        with open('fixed_solver_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to fixed_solver_results.json")
        
        return results

async def main():
    """Run the fixed master solver test"""
    solver = FixedMasterSolver()
    await solver.run_test(10)

if __name__ == "__main__":
    asyncio.run(main())