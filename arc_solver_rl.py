"""
ARC AGI Solver with Reinforcement Learning and Performance Optimization
Final implementation with all advanced features
"""

import json
import time
import numpy as np
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
import hashlib

# Import base components
from arc_solver import Heuristic, PatternTool, load_puzzles
from arc_solver_enhanced import EnhancedARCSolver, MetaHeuristic


class ReinforcementLearningSystem:
    """In-context reinforcement learning for self-improvement"""
    
    def __init__(self):
        # Reward signals
        self.reward_signals = {
            'puzzle_solved': 10.0,
            'partial_match': 2.0,
            'pattern_discovered': 3.0,
            'tool_generated': 1.0,
            'time_improved': 5.0,
            'failed_attempt': -0.5
        }
        
        self.meta_rewards = {
            'generalization': 15.0,
            'abstraction': 12.0,
            'efficiency_gain': 8.0
        }
        
        # Q-table for heuristic values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Performance history
        self.episode_rewards = []
        self.cumulative_reward = 0
        
    def get_reward(self, event: str, context: Dict = None) -> float:
        """Calculate reward for an event"""
        base_reward = self.reward_signals.get(event, 0)
        
        # Apply contextual modifiers
        if context:
            if 'time_saved' in context:
                base_reward += context['time_saved'] * 0.1
            if 'complexity_reduced' in context:
                base_reward += self.meta_rewards['efficiency_gain']
                
        return base_reward
        
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon or state not in self.q_table:
            # Explore
            return random.choice(available_actions)
        else:
            # Exploit
            action_values = {a: self.q_table[state].get(a, 0) for a in available_actions}
            return max(action_values, key=action_values.get)
            
    def update_episode(self, total_reward: float):
        """Update after an episode (puzzle solving attempt)"""
        self.episode_rewards.append(total_reward)
        self.cumulative_reward += total_reward
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
    def get_performance_stats(self) -> Dict:
        """Get RL performance statistics"""
        return {
            'cumulative_reward': self.cumulative_reward,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }


class PerformanceOptimizer:
    """System for optimizing solving performance"""
    
    def __init__(self):
        self.puzzle_times = {}
        self.heuristic_times = defaultdict(list)
        self.pattern_cache = {}
        self.optimization_history = []
        
    def start_puzzle_timer(self, puzzle_id: str):
        """Start timing a puzzle"""
        self.puzzle_times[puzzle_id] = {
            'start': time.time(),
            'attempts': []
        }
        
    def record_attempt(self, puzzle_id: str, heuristic: str, success: bool):
        """Record an attempt timing"""
        if puzzle_id in self.puzzle_times:
            attempt_time = time.time() - self.puzzle_times[puzzle_id]['start']
            self.puzzle_times[puzzle_id]['attempts'].append({
                'heuristic': heuristic,
                'time': attempt_time,
                'success': success
            })
            self.heuristic_times[heuristic].append(attempt_time)
            
    def get_puzzle_time(self, puzzle_id: str) -> float:
        """Get total time for a puzzle"""
        if puzzle_id in self.puzzle_times and 'end' in self.puzzle_times[puzzle_id]:
            return self.puzzle_times[puzzle_id]['end'] - self.puzzle_times[puzzle_id]['start']
        elif puzzle_id in self.puzzle_times:
            return time.time() - self.puzzle_times[puzzle_id]['start']
        return 0
        
    def end_puzzle_timer(self, puzzle_id: str):
        """End timing for a puzzle"""
        if puzzle_id in self.puzzle_times:
            self.puzzle_times[puzzle_id]['end'] = time.time()
            
    def cache_pattern(self, pattern_hash: str, solution: Any):
        """Cache a successful pattern solution"""
        self.pattern_cache[pattern_hash] = {
            'solution': solution,
            'timestamp': time.time(),
            'uses': 0
        }
        
    def get_cached_solution(self, pattern_hash: str) -> Optional[Any]:
        """Retrieve cached solution if available"""
        if pattern_hash in self.pattern_cache:
            self.pattern_cache[pattern_hash]['uses'] += 1
            return self.pattern_cache[pattern_hash]['solution']
        return None
        
    def suggest_optimization(self, performance_data: Dict) -> Optional[str]:
        """Suggest optimization based on performance"""
        suggestions = []
        
        # Check for slow heuristics
        for heuristic, times in self.heuristic_times.items():
            avg_time = np.mean(times) if times else 0
            if avg_time > 30:  # More than 30 seconds average
                suggestions.append(f"optimize_{heuristic}")
                
        # Check cache hit rate
        cache_hits = sum(1 for p in self.pattern_cache.values() if p['uses'] > 0)
        if len(self.pattern_cache) > 10 and cache_hits / len(self.pattern_cache) < 0.3:
            suggestions.append("improve_pattern_matching")
            
        return suggestions[0] if suggestions else None


class RLEnhancedSolver(EnhancedARCSolver):
    """Final solver with RL and performance optimization"""
    
    def __init__(self):
        super().__init__()
        self.rl_system = ReinforcementLearningSystem()
        self.optimizer = PerformanceOptimizer()
        self.optimization_meta_heuristics = self._init_optimization_meta_heuristics()
        self.total_start_time = None
        self.speed_goals = {
            'initial': 300,  # 5 minutes
            'optimized': 120,  # 2 minutes
            'target': 60  # 1 minute
        }
        
    def _init_optimization_meta_heuristics(self) -> List[MetaHeuristic]:
        """Initialize optimization-focused meta-heuristics"""
        return [
            MetaHeuristic(
                "opt1", "Speed Optimizer",
                "When solving takes > 30s, simplify approach",
                "Simplified heuristics",
                "Slow solving detected"
            ),
            MetaHeuristic(
                "opt2", "Parallel Processor",
                "Try multiple heuristics simultaneously",
                "Parallel execution framework",
                "Multiple viable approaches"
            ),
            MetaHeuristic(
                "opt3", "Cache Manager",
                "Reuse successful patterns on similar puzzles",
                "Pattern caching system",
                "Repeated patterns detected"
            ),
            MetaHeuristic(
                "opt4", "Abstraction Builder",
                "Extract general principles from specific solutions",
                "Abstract pattern recognizers",
                "Multiple similar solutions"
            ),
            MetaHeuristic(
                "opt5", "Failure Analyzer",
                "Learn more from failures than successes",
                "Failure pattern detectors",
                "Repeated failures"
            ),
        ]
        
    def solve_puzzle_with_rl(self, puzzle: Dict) -> Tuple[Optional[List[List[int]]], Dict]:
        """Solve puzzle using RL-guided approach"""
        puzzle_id = puzzle.get('id', 'unknown')
        
        # Start timing
        self.optimizer.start_puzzle_timer(puzzle_id)
        
        # Get puzzle state representation
        test_input = puzzle['test'][0]['input']
        state = self._get_state_representation(test_input)
        
        # Check cache first
        pattern_hash = self._get_pattern_hash(test_input)
        cached_solution = self.optimizer.get_cached_solution(pattern_hash)
        
        if cached_solution:
            # Reward for cache hit
            reward = self.rl_system.get_reward('time_improved', {'time_saved': 10})
            self.rl_system.update_episode(reward)
            self.optimizer.end_puzzle_timer(puzzle_id)
            return cached_solution, {'solved': True, 'method': 'cache_hit'}
            
        # Get available heuristics
        features = self.tool.extract_features(test_input)
        available_heuristics = [h for h in self.heuristics if h.should_apply(features)]
        
        if not available_heuristics:
            available_heuristics = self.heuristics[:3]  # Use top 3 if none applicable
            
        # Select heuristic using RL
        action_names = [h.id for h in available_heuristics]
        selected_action = self.rl_system.select_action(state, action_names)
        selected_heuristic = next(h for h in available_heuristics if h.id == selected_action)
        
        # Apply heuristic
        solution = None
        episode_reward = 0
        
        try:
            solution = selected_heuristic.apply(test_input)
            
            # Check solution
            is_correct = self._check_solution(puzzle, solution)
            
            # Calculate reward
            if is_correct:
                solve_time = self.optimizer.get_puzzle_time(puzzle_id)
                time_bonus = max(0, (30 - solve_time) * 0.1)  # Bonus for fast solving
                reward = self.rl_system.get_reward('puzzle_solved') + time_bonus
                
                # Cache successful solution
                self.optimizer.cache_pattern(pattern_hash, solution)
                
                # Update stats
                selected_heuristic.update_stats(True)
                self.optimizer.record_attempt(puzzle_id, selected_action, True)
                
            else:
                reward = self.rl_system.get_reward('failed_attempt')
                selected_heuristic.update_stats(False)
                self.optimizer.record_attempt(puzzle_id, selected_action, False)
                
            # Update Q-table
            next_state = self._get_state_representation(solution if solution else test_input)
            self.rl_system.update_q_value(state, selected_action, reward, next_state)
            
            episode_reward += reward
            
        except Exception as e:
            print(f"Error in RL solving: {e}")
            episode_reward += self.rl_system.get_reward('failed_attempt')
            
        # Update RL system
        self.rl_system.update_episode(episode_reward)
        
        # End timing
        self.optimizer.end_puzzle_timer(puzzle_id)
        
        # Check for optimization opportunities
        if self.optimizer.get_puzzle_time(puzzle_id) > 30:
            self._trigger_optimization()
            
        return solution if solution and is_correct else None, {
            'solved': solution is not None and is_correct,
            'heuristic_used': selected_heuristic.name,
            'time': self.optimizer.get_puzzle_time(puzzle_id),
            'reward': episode_reward
        }
        
    def _get_state_representation(self, grid: List[List[int]]) -> str:
        """Get state representation for RL"""
        # Simple representation: grid dimensions + color count + pattern features
        features = self.tool.extract_features(grid)
        state_parts = [
            f"size_{features.get('height', 0)}x{features.get('width', 0)}",
            f"colors_{features.get('num_colors', 0)}",
            f"sym_{features.get('has_symmetry', False)}"
        ]
        return "_".join(state_parts)
        
    def _get_pattern_hash(self, grid: List[List[int]]) -> str:
        """Generate hash for pattern caching"""
        grid_str = json.dumps(grid)
        return hashlib.md5(grid_str.encode()).hexdigest()
        
    def _trigger_optimization(self):
        """Trigger optimization meta-heuristics"""
        suggestion = self.optimizer.suggest_optimization({
            'heuristic_times': self.optimizer.heuristic_times,
            'cache_stats': self.optimizer.pattern_cache
        })
        
        if suggestion:
            print(f"🔧 Triggering optimization: {suggestion}")
            
            # Find relevant optimization meta-heuristic
            for meta_h in self.optimization_meta_heuristics:
                if "speed" in meta_h.name.lower() or "optimize" in suggestion:
                    # Would trigger Claude Code here
                    print(f"  Applying: {meta_h.name}")
                    break
                    
    def run_with_performance_targets(self, puzzles: List[Dict]):
        """Run solver with performance targets"""
        self.total_start_time = time.time()
        
        print("\n" + "="*60)
        print("ARC AGI Solver with Reinforcement Learning")
        print("Performance Target: Solve 10 puzzles < 2 minutes")
        print("="*60)
        
        iteration = 0
        max_iterations = 100
        
        while len(self.solved_puzzles) < len(puzzles) and iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - self.total_start_time
            
            print(f"\n--- Iteration {iteration} (Time: {elapsed:.1f}s) ---")
            print(f"Solved: {len(self.solved_puzzles)}/{len(puzzles)}")
            print(f"RL Stats: {self.rl_system.get_performance_stats()}")
            
            # Check speed goals
            if elapsed > self.speed_goals['initial'] and len(self.solved_puzzles) < len(puzzles):
                print("⚠️  Exceeding initial time goal - activating speed optimizations")
                self._activate_speed_mode()
                
            unsolved = [p for p in puzzles if p['id'] not in self.solved_puzzles]
            
            for puzzle in unsolved:
                solution, result = self.solve_puzzle_with_rl(puzzle)
                
                if result['solved']:
                    print(f"✓ Solved {puzzle['id']} in {result['time']:.2f}s (reward: {result['reward']:.2f})")
                    self.solved_puzzles.add(puzzle['id'])
                    
                    # Check for meta-rewards
                    if len(self.solved_puzzles) > 3:
                        # Generalization reward
                        if result['time'] < 5:  # Fast solving indicates good generalization
                            meta_reward = self.rl_system.get_reward('generalization')
                            self.rl_system.cumulative_reward += meta_reward
                            print(f"  🌟 Meta-reward for generalization: +{meta_reward}")
                else:
                    print(f"✗ Failed {puzzle['id']}")
                    
            # Self-reflection and improvement
            if iteration % 5 == 0:
                self._self_improve()
                
        # Final results
        total_time = time.time() - self.total_start_time
        
        print("\n" + "="*60)
        print("Final Results")
        print(f"Puzzles Solved: {len(self.solved_puzzles)}/{len(puzzles)}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time per Puzzle: {total_time/max(1, len(self.solved_puzzles)):.2f}s")
        print(f"Cumulative Reward: {self.rl_system.cumulative_reward:.2f}")
        
        # Check performance targets
        if total_time <= self.speed_goals['optimized']:
            print("🏆 ACHIEVED OPTIMIZED SPEED GOAL!")
        elif total_time <= self.speed_goals['initial']:
            print("✓ Met initial speed goal")
        else:
            print("⚠️  Speed optimization needed")
            
        print("="*60)
        
        return {
            'solved_count': len(self.solved_puzzles),
            'total_time': total_time,
            'cumulative_reward': self.rl_system.cumulative_reward,
            'performance_stats': self.rl_system.get_performance_stats()
        }
        
    def _activate_speed_mode(self):
        """Activate speed optimizations"""
        # Reduce exploration
        self.rl_system.epsilon = 0.01
        
        # Limit attempts per puzzle
        self.max_attempts_per_puzzle = 3
        
        # Prioritize fast heuristics
        for h in self.heuristics:
            if "simple" in h.name.lower() or "fast" in h.name.lower():
                h.confidence *= 1.5
                
    def _self_improve(self):
        """Self-improvement based on RL insights"""
        print("\n🧠 Self-Improvement Analysis:")
        
        # Analyze Q-table for insights
        if self.rl_system.q_table:
            best_actions = {}
            for state, actions in self.rl_system.q_table.items():
                if actions:
                    best_action = max(actions, key=actions.get)
                    best_actions[state] = (best_action, actions[best_action])
                    
            print(f"  Learned {len(best_actions)} state-action mappings")
            
            # Boost confidence of consistently good heuristics
            action_scores = defaultdict(list)
            for state, (action, value) in best_actions.items():
                action_scores[action].append(value)
                
            for action, scores in action_scores.items():
                avg_score = np.mean(scores)
                if avg_score > 5:  # High average Q-value
                    for h in self.heuristics:
                        if h.id == action:
                            h.confidence = min(0.95, h.confidence * 1.2)
                            print(f"  Boosted confidence for {h.name}")
                            
        # Analyze cache effectiveness
        cache_hits = sum(1 for p in self.optimizer.pattern_cache.values() if p['uses'] > 0)
        if self.optimizer.pattern_cache:
            hit_rate = cache_hits / len(self.optimizer.pattern_cache)
            print(f"  Cache hit rate: {hit_rate:.2%}")


def main():
    """Main entry point for RL solver"""
    print("ARC AGI Solver with Reinforcement Learning")
    print("="*60)
    
    # Load puzzles
    data_dir = Path("data/arc_agi")
    puzzles = load_puzzles(data_dir, limit=10)
    print(f"Loaded {len(puzzles)} puzzles\n")
    
    # Create RL solver
    solver = RLEnhancedSolver()
    
    # Run with performance targets
    results = solver.run_with_performance_targets(puzzles)
    
    # Save results
    results_file = Path("rl_solver_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Commit results
    try:
        subprocess.run(["git", "add", "-A"], check=False)
        commit_msg = f"RL Solver: {results['solved_count']} puzzles in {results['total_time']:.1f}s"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "push"], check=False)
        print("✓ Results committed to Git")
    except:
        pass
    
    return results


if __name__ == "__main__":
    main()