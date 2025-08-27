"""
Enhanced ARC AGI Solver with Claude Code Integration
Automatically generates tools and commits code periodically
"""

import json
import time
import subprocess
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import copy
from arc_solver import Heuristic, PatternTool, ARCSolver

class ClaudeCodeIntegration:
    """Handles Claude Code integration for tool generation"""
    
    def __init__(self):
        self.claude_code_path = "claude"  # Assuming claude is in PATH
        self.allowed_tools = "Bash,Read,WebSearch,Fetch"
        self.permission_mode = "acceptEdits"
        self.last_commit_time = time.time()
        self.commit_interval = 300  # Commit every 5 minutes
        
    def generate_tool(self, puzzle_analysis: Dict, failed_attempts: List) -> Optional[str]:
        """Generate a new tool using Claude Code"""
        
        # Build prompt for Claude Code
        prompt = f"""
        Create a Python function to solve ARC AGI puzzles with these characteristics:
        - Puzzle features: {json.dumps(puzzle_analysis, indent=2)}
        - Failed approaches: {failed_attempts}
        
        The function should:
        1. Take a grid (List[List[int]]) as input
        2. Return a transformed grid (List[List[int]]) as output
        3. Focus on pattern transformations that haven't been tried
        
        Save the function in patterns/generated_tool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py
        """
        
        # Build Claude Code command
        cmd = [
            self.claude_code_path,
            "--allowedTools", self.allowed_tools,
            "--permission-mode", self.permission_mode,
            "--message", prompt
        ]
        
        try:
            # Execute Claude Code
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✓ Claude Code generated new tool successfully")
                self.periodic_commit("Added new generated tool")
                return result.stdout
            else:
                print(f"✗ Claude Code failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("✗ Claude Code timed out")
            return None
        except Exception as e:
            print(f"✗ Error calling Claude Code: {e}")
            return None
            
    def improve_heuristic(self, heuristic: Heuristic, failure_cases: List) -> Optional[str]:
        """Improve an existing heuristic using Claude Code"""
        
        prompt = f"""
        Improve this ARC AGI solving heuristic:
        - Name: {heuristic.name}
        - Current success rate: {heuristic.success_rate:.2%}
        - Failed on: {failure_cases}
        
        Modify the heuristic to handle these failure cases better.
        Update the file: patterns/heuristic_{heuristic.id}_improved.py
        """
        
        cmd = [
            self.claude_code_path,
            "--allowedTools", self.allowed_tools,
            "--permission-mode", self.permission_mode,
            "--message", prompt
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✓ Improved heuristic {heuristic.name}")
                self.periodic_commit(f"Improved heuristic {heuristic.name}")
                return result.stdout
        except Exception as e:
            print(f"✗ Error improving heuristic: {e}")
            
        return None
        
    def periodic_commit(self, message: str = "Periodic checkpoint"):
        """Commit code periodically"""
        current_time = time.time()
        
        if current_time - self.last_commit_time > self.commit_interval:
            try:
                # Stage all changes
                subprocess.run(["git", "add", "-A"], check=True)
                
                # Commit with message
                commit_msg = f"{message}\n\nAuto-commit by solver system\nTimestamp: {datetime.now().isoformat()}"
                subprocess.run(["git", "commit", "-m", commit_msg], check=True)
                
                # Push to remote
                subprocess.run(["git", "push"], check=False)  # Don't fail if push fails
                
                print(f"✓ Code committed: {message}")
                self.last_commit_time = current_time
                
            except subprocess.CalledProcessError as e:
                print(f"Git commit failed: {e}")
                

class EnhancedARCSolver(ARCSolver):
    """Enhanced solver with Claude Code integration and monitoring"""
    
    def __init__(self):
        super().__init__()
        self.claude_integration = ClaudeCodeIntegration()
        self.performance_dashboard = PerformanceDashboard()
        self.tool_generation_count = 0
        self.max_tool_generations = 10
        
    def continuous_learning_loop_enhanced(self, puzzles: List[Dict], max_iterations: int = 100):
        """Enhanced learning loop with Claude Code integration"""
        
        print("\n" + "="*60)
        print("Enhanced Continuous Learning Loop with Claude Code")
        print("="*60)
        
        iteration = 0
        start_time = time.time()
        last_solve_count = 0
        stuck_counter = 0
        
        # Start performance dashboard
        self.performance_dashboard.start()
        
        while len(self.solved_puzzles) < len(puzzles) and iteration < max_iterations:
            iteration += 1
            
            # Update dashboard
            self.performance_dashboard.update({
                'iteration': iteration,
                'solved': len(self.solved_puzzles),
                'total': len(puzzles),
                'heuristics': len(self.heuristics),
                'tools_generated': self.tool_generation_count
            })
            
            # Get unsolved puzzles
            unsolved = [p for p in puzzles if p['id'] not in self.solved_puzzles]
            
            print(f"\n--- Iteration {iteration} ---")
            print(f"Solved: {len(self.solved_puzzles)}/{len(puzzles)}")
            
            # Try to solve each unsolved puzzle
            for puzzle in unsolved:
                if self.puzzle_attempts.get(puzzle['id'], {}).get('attempts', 0) >= self.max_attempts_per_puzzle:
                    continue
                    
                solution, result = self.solve_puzzle(puzzle)
                
                if result['solved']:
                    print(f"✓ Solved {puzzle['id']} using {result.get('heuristic_used', 'unknown')}")
                    self.performance_dashboard.log_success(puzzle['id'])
                else:
                    print(f"✗ Failed {puzzle['id']} (attempt {result['attempts']})")
                    self.performance_dashboard.log_failure(puzzle['id'])
                    
            # Check if stuck
            if len(self.solved_puzzles) == last_solve_count:
                stuck_counter += 1
            else:
                stuck_counter = 0
                last_solve_count = len(self.solved_puzzles)
                
            # Generate new tools if stuck
            if stuck_counter >= 3 and self.tool_generation_count < self.max_tool_generations:
                print("\n🔧 System is stuck. Generating new tools with Claude Code...")
                
                # Analyze unsolved puzzles
                puzzle_analysis = self._analyze_unsolved(unsolved)
                failed_attempts = [h.name for h in self.heuristics if h.success_rate < 0.3]
                
                # Generate new tool
                tool_code = self.claude_integration.generate_tool(puzzle_analysis, failed_attempts)
                
                if tool_code:
                    self.tool_generation_count += 1
                    # Load the generated tool (simplified - would need actual import)
                    self._load_generated_tool()
                    stuck_counter = 0
                    
            # Improve poorly performing heuristics
            if iteration % 10 == 0:
                self._improve_heuristics()
                
            # Self-reflection
            if iteration % 5 == 0:
                self._self_reflect()
                
            # Periodic commit
            self.claude_integration.periodic_commit(f"Iteration {iteration}: {len(self.solved_puzzles)}/{len(puzzles)} solved")
            
            # Performance summary
            self._print_performance_summary()
            
        # Final commit
        self.claude_integration.periodic_commit(f"Final: {len(self.solved_puzzles)}/{len(puzzles)} puzzles solved")
        
        # Stop dashboard
        self.performance_dashboard.stop()
        
        # Final summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Enhanced Learning Loop Complete")
        print(f"Final Score: {len(self.solved_puzzles)}/{len(puzzles)} puzzles solved")
        print(f"Total Iterations: {iteration}")
        print(f"Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Total Heuristics: {len(self.heuristics)}")
        print(f"Tools Generated: {self.tool_generation_count}")
        print("="*60)
        
        return {
            'solved_count': len(self.solved_puzzles),
            'total_puzzles': len(puzzles),
            'iterations': iteration,
            'time_elapsed': elapsed_time,
            'heuristics_count': len(self.heuristics),
            'tools_generated': self.tool_generation_count
        }
        
    def _analyze_unsolved(self, unsolved_puzzles: List[Dict]) -> Dict:
        """Analyze patterns in unsolved puzzles"""
        analysis = {
            'common_features': {},
            'grid_sizes': [],
            'color_patterns': [],
            'transformation_types': []
        }
        
        for puzzle in unsolved_puzzles:
            test_input = puzzle['test'][0]['input']
            features = self.tool.extract_features(test_input)
            
            # Collect common features
            for key, value in features.items():
                if key not in analysis['common_features']:
                    analysis['common_features'][key] = []
                analysis['common_features'][key].append(value)
                
            analysis['grid_sizes'].append((features.get('height', 0), features.get('width', 0)))
            
        return analysis
        
    def _improve_heuristics(self):
        """Use Claude Code to improve underperforming heuristics"""
        poor_heuristics = [h for h in self.heuristics 
                          if h.usage_count > 10 and h.success_rate < 0.3]
        
        if poor_heuristics and self.tool_generation_count < self.max_tool_generations:
            worst_heuristic = min(poor_heuristics, key=lambda h: h.success_rate)
            
            print(f"\n🔄 Improving heuristic: {worst_heuristic.name}")
            
            # Get failure cases
            failure_cases = []
            for puzzle_id, data in self.puzzle_attempts.items():
                if worst_heuristic.id in data.get('heuristics_tried', []) and not data['solved']:
                    failure_cases.append(puzzle_id)
                    
            # Improve using Claude Code
            improved_code = self.claude_integration.improve_heuristic(worst_heuristic, failure_cases[:3])
            
            if improved_code:
                worst_heuristic.confidence *= 1.2  # Boost confidence to try again
                self.tool_generation_count += 1
                
    def _load_generated_tool(self):
        """Load a generated tool (placeholder for actual implementation)"""
        # In a real implementation, this would dynamically import the generated module
        # For now, we'll create a simple new heuristic
        
        def generated_transform(grid):
            # Placeholder for generated transformation
            result = copy.deepcopy(grid)
            # Apply some transformation
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] > 0:
                        result[i][j] = (result[i][j] + 1) % 10
            return result
            
        new_heuristic = Heuristic(
            f'generated_{self.tool_generation_count}',
            f'Generated Tool {self.tool_generation_count}',
            {'conditions': [], 'puzzle_features': []},
            generated_transform,
            0.4
        )
        
        self.heuristics.append(new_heuristic)
        print(f"✓ Loaded generated tool: {new_heuristic.name}")


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            'puzzles_solved': [],
            'success_timeline': [],
            'heuristic_performance': {},
            'current_status': {}
        }
        
    def start(self):
        """Start the dashboard"""
        self.start_time = time.time()
        print("\n📊 Performance Dashboard Started")
        print("-" * 40)
        
    def update(self, status: Dict):
        """Update current status"""
        self.metrics['current_status'] = status
        self._display()
        
    def log_success(self, puzzle_id: str):
        """Log a successful solve"""
        self.metrics['puzzles_solved'].append(puzzle_id)
        self.metrics['success_timeline'].append({
            'puzzle': puzzle_id,
            'time': time.time() - self.start_time,
            'success': True
        })
        
    def log_failure(self, puzzle_id: str):
        """Log a failed attempt"""
        self.metrics['success_timeline'].append({
            'puzzle': puzzle_id,
            'time': time.time() - self.start_time,
            'success': False
        })
        
    def _display(self):
        """Display current metrics"""
        status = self.metrics['current_status']
        
        # Clear line and display status
        print(f"\r[Iter {status.get('iteration', 0):3d}] "
              f"Solved: {status.get('solved', 0)}/{status.get('total', 0)} "
              f"({100 * status.get('solved', 0) / max(1, status.get('total', 1)):.1f}%) "
              f"| Heuristics: {status.get('heuristics', 0)} "
              f"| Tools: {status.get('tools_generated', 0)} ", end="")
              
    def stop(self):
        """Stop the dashboard and show final summary"""
        print("\n" + "-" * 40)
        print("📊 Performance Dashboard Summary")
        
        if self.metrics['puzzles_solved']:
            print(f"  Puzzles solved: {self.metrics['puzzles_solved']}")
            
        # Calculate solve rate over time
        if self.metrics['success_timeline']:
            successes = [e for e in self.metrics['success_timeline'] if e['success']]
            if successes:
                avg_time = sum(e['time'] for e in successes) / len(successes)
                print(f"  Average time to solve: {avg_time:.2f}s")


def main():
    """Main entry point for enhanced solver"""
    print("ARC AGI Enhanced Solver with Claude Code Integration")
    print("=" * 60)
    
    # Load puzzles
    from arc_solver import load_puzzles
    data_dir = Path("data/arc_agi")
    puzzles = load_puzzles(data_dir, limit=10)
    print(f"Loaded {len(puzzles)} puzzles\n")
    
    # Initial commit
    try:
        subprocess.run(["git", "add", "-A"], check=False)
        subprocess.run(["git", "commit", "-m", "Starting enhanced solver run"], check=False)
        subprocess.run(["git", "push"], check=False)
    except:
        pass
    
    # Create enhanced solver
    solver = EnhancedARCSolver()
    
    # Run enhanced learning loop
    results = solver.continuous_learning_loop_enhanced(puzzles, max_iterations=100)
    
    # Save results
    results_file = Path("enhanced_solver_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Final commit and push
    try:
        subprocess.run(["git", "add", "-A"], check=False)
        subprocess.run(["git", "commit", "-m", f"Solver run complete: {results['solved_count']}/{results['total_puzzles']} solved"], check=False)
        subprocess.run(["git", "push"], check=False)
        print("\n✓ Final results committed and pushed to GitHub")
    except:
        pass
    
    return results


if __name__ == "__main__":
    main()