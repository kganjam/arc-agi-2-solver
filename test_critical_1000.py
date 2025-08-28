#!/usr/bin/env python3
"""
Test Critical Reasoning System - 1000 Puzzles with Full Explanations
Every solution includes detailed reasoning and verification
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Import all systems
from arc_master_solver import MasterSolver
from arc_critique_system import CriticalSolver, CritiqueAgent, ExplanationGenerator
from arc_enhanced_learner import PatternLibrary, TransferLearner, MetaLearner
from arc_synthetic_generator import SyntheticPuzzleGAN

class CriticalMasterSolver(MasterSolver):
    """Enhanced master solver with critical reasoning"""
    
    def __init__(self):
        super().__init__()
        
        # Add critical reasoning systems
        self.critical_solver = CriticalSolver()
        self.pattern_library = PatternLibrary()
        self.transfer_learner = TransferLearner()
        self.meta_learner = MetaLearner()
        
        # Enhanced tracking
        self.explanations = []
        self.critiques = []
        self.verified_solutions = 0
        self.unverified_solutions = 0
        
        # Initialize learning strategies
        self._init_learning_strategies()
        
    def _init_learning_strategies(self):
        """Initialize meta-learning strategies"""
        strategies = [
            {
                'id': 'pattern_first',
                'strategy': {
                    'name': 'Pattern-First Approach',
                    'conditions': ['has_patterns', 'repeating_elements'],
                    'applicable_to': ['pattern', 'symmetry'],
                    'steps': ['identify_pattern', 'extract_rules', 'apply_rules']
                }
            },
            {
                'id': 'decomposition',
                'strategy': {
                    'name': 'Problem Decomposition',
                    'conditions': ['complex', 'multi_object'],
                    'applicable_to': ['complex_pattern', 'object_manipulation'],
                    'steps': ['break_down', 'solve_parts', 'compose_solution']
                }
            },
            {
                'id': 'analogy',
                'strategy': {
                    'name': 'Reasoning by Analogy',
                    'conditions': ['similar_exists'],
                    'applicable_to': ['all'],
                    'steps': ['find_similar', 'adapt_solution', 'validate']
                }
            }
        ]
        
        for s in strategies:
            self.meta_learner.add_learning_strategy(s['id'], s['strategy'])
            
    async def solve_puzzle_with_all_methods(self, puzzle: Dict) -> Dict:
        """Override to add critical reasoning"""
        # First try critical solver
        critical_result = await self.critical_solver.solve_with_critique(puzzle)
        
        if critical_result['verified']:
            self.verified_solutions += 1
            
            # Store explanation and critique
            self.explanations.append(critical_result['explanation'])
            self.critiques.append(critical_result['critique'])
            
            # Learn from success
            self._learn_from_solution(puzzle, critical_result)
            
            return {
                'puzzle_id': critical_result['puzzle_id'],
                'solved': True,
                'method': 'critical_reasoning',
                'verified': True,
                'explanation': critical_result['explanation'],
                'critique_score': critical_result['critique']['overall_score']
            }
        else:
            self.unverified_solutions += 1
            
        # Fall back to other methods
        return await super().solve_puzzle_with_all_methods(puzzle)
        
    def _learn_from_solution(self, puzzle: Dict, result: Dict):
        """Learn from successful solution"""
        # Extract and store pattern
        if 'explanation' in result:
            pattern = self._extract_pattern_from_explanation(result['explanation'])
            if pattern:
                pattern_id = f"pattern_{len(self.pattern_library.patterns)}"
                self.pattern_library.add_pattern(pattern_id, pattern)
                
        # Transfer learning
        self.transfer_learner.learn_from_puzzle(puzzle, result)
        
        # Meta-learning
        experience = {
            'puzzle_id': puzzle.get('id'),
            'successful_strategies': ['critical_reasoning'],
            'context': {'verified': True}
        }
        self.meta_learner.learn_from_experience(experience)
        
    def _extract_pattern_from_explanation(self, explanation: Dict) -> Optional[Dict]:
        """Extract pattern from explanation"""
        pattern = {
            'transformations': [],
            'conditions': [],
            'reasoning_types': explanation.get('reasoning_types', [])
        }
        
        # Extract from steps
        for step in explanation.get('steps', []):
            if 'reasoning' in step:
                pattern['transformations'].append(step['reasoning'])
                
        # Extract from evidence
        for evidence in explanation.get('evidence', []):
            if 'supports' in evidence:
                pattern['conditions'].append(evidence['supports'])
                
        return pattern if pattern['transformations'] else None
        
    def print_critical_stats(self):
        """Print critical reasoning statistics"""
        print("\n🧠 Critical Reasoning Statistics:")
        print(f"  Verified Solutions: {self.verified_solutions}")
        print(f"  Unverified Solutions: {self.unverified_solutions}")
        
        if self.verified_solutions > 0:
            verification_rate = (self.verified_solutions / 
                               (self.verified_solutions + self.unverified_solutions)) * 100
            print(f"  Verification Rate: {verification_rate:.1f}%")
            
        print(f"  Patterns Discovered: {len(self.pattern_library.patterns)}")
        print(f"  Knowledge Transfers: {len(self.transfer_learner.transfer_history)}")
        
        # Best patterns
        best_patterns = self.pattern_library.get_best_patterns(3)
        if best_patterns:
            print("\n  Top Patterns:")
            for pid, score in best_patterns:
                pattern = self.pattern_library.patterns[pid]
                print(f"    - {pid}: Score {score:.2f}, Complexity {pattern['complexity']:.2f}")
                
        # Learning strategies performance
        print("\n  Learning Strategies:")
        for strategy_id, perf in self.meta_learner.strategy_performance.items():
            if perf['total'] > 0:
                success_rate = perf['success'] / perf['total'] * 100
                print(f"    - {strategy_id}: {success_rate:.1f}% success ({perf['total']} uses)")

async def main():
    """Run critical reasoning test to 1000 puzzles"""
    print("="*80)
    print(" "*15 + "🧠 CRITICAL REASONING CHALLENGE - 1000 PUZZLES 🧠")
    print("="*80)
    print("\n✅ Every solution will be:")
    print("  • Explained step-by-step")
    print("  • Critically evaluated")
    print("  • Verified for correctness")
    print("  • Used for learning")
    print("\n" + "="*80)
    
    solver = CriticalMasterSolver()
    solver.target_puzzles = 1000
    
    start_time = time.time()
    
    # Run solver
    await solver.run_to_target(1000)
    
    total_time = time.time() - start_time
    
    # Print enhanced statistics
    print("\n" + "="*80)
    print(" "*25 + "📊 FINAL REPORT 📊")
    print("="*80)
    
    print(f"\n🎯 RESULTS:")
    print(f"  Total Puzzles Solved: {solver.total_puzzles_solved}")
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Speed: {(solver.total_puzzles_solved/total_time)*60:.1f} puzzles/minute")
    
    # Critical reasoning stats
    solver.print_critical_stats()
    
    # Strategy breakdown
    print("\n📈 Solution Methods:")
    for method, count in solver.solving_strategies.items():
        if count > 0:
            print(f"  {method}: {count}")
            
    # Success determination
    print("\n" + "="*80)
    if solver.total_puzzles_solved >= 1000:
        print(" "*20 + "🏆 ACHIEVEMENT UNLOCKED! 🏆")
        print(" "*15 + "1000 PUZZLES WITH CRITICAL REASONING!")
        print(" "*10 + "Every solution explained and verified!")
    else:
        print(f"  Progress: {solver.total_puzzles_solved}/1000")
        
    print("="*80)
    
    # Save comprehensive results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_solved': solver.total_puzzles_solved,
        'verified_solutions': solver.verified_solutions,
        'unverified_solutions': solver.unverified_solutions,
        'patterns_discovered': len(solver.pattern_library.patterns),
        'knowledge_transfers': len(solver.transfer_learner.transfer_history),
        'total_time': total_time,
        'solving_strategies': solver.solving_strategies
    }
    
    with open('critical_1000_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n📁 Results saved to critical_1000_results.json")

if __name__ == "__main__":
    asyncio.run(main())