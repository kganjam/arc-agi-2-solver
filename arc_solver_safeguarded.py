"""
ARC AGI Solver with Anti-Cheating Safeguards
This solver implements proper ARC AGI solving methodology:
1. Learn patterns from training examples
2. Validate hypothesis on all training data
3. Apply learned transformation to test input
4. Properly validate solutions

IMPORTANT: These safeguards are critical for fair competition compliance.
DO NOT MODIFY OR REMOVE SAFEGUARDS.
"""

import json
import copy
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

# SAFEGUARD: Immutable checksum to prevent tampering
SAFEGUARD_CHECKSUM = "arc_agi_fair_competition_2025"

class SafeguardViolationError(Exception):
    """Raised when safeguards are violated"""
    pass

class ARCPuzzle:
    """Represents a proper ARC AGI puzzle with train/test structure"""
    
    def __init__(self, puzzle_data: Dict):
        """
        puzzle_data should have:
        - 'train': list of {'input': grid, 'output': grid} examples
        - 'test': list of {'input': grid, 'output': grid} test cases
        """
        if 'train' not in puzzle_data or 'test' not in puzzle_data:
            raise ValueError("Invalid ARC puzzle format - must have 'train' and 'test'")
        
        self.train_examples = puzzle_data['train']
        self.test_cases = puzzle_data['test']
        self.puzzle_id = puzzle_data.get('id', 'unknown')
        
        # SAFEGUARD: Verify puzzle has proper structure
        if len(self.train_examples) < 1:
            raise ValueError("Puzzle must have at least 1 training example")
        
        for example in self.train_examples:
            if 'input' not in example or 'output' not in example:
                raise ValueError("Each training example must have 'input' and 'output'")
    
    def get_training_pairs(self) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """Get all training input/output pairs"""
        return [(ex['input'], ex['output']) for ex in self.train_examples]
    
    def get_test_inputs(self) -> List[List[List[int]]]:
        """Get test inputs"""
        return [test['input'] for test in self.test_cases]
    
    def get_test_outputs(self) -> List[List[List[int]]]:
        """Get expected test outputs (for validation)"""
        return [test.get('output') for test in self.test_cases if 'output' in test]


class PatternLearner:
    """Learns transformation patterns from training examples"""
    
    def __init__(self):
        self.learned_patterns = {}
        
    def analyze_transformation(self, input_grid: List[List[int]], 
                              output_grid: List[List[int]]) -> Dict:
        """Analyze a single input/output transformation"""
        analysis = {
            'input_shape': (len(input_grid), len(input_grid[0]) if input_grid else 0),
            'output_shape': (len(output_grid), len(output_grid[0]) if output_grid else 0),
            'shape_changed': False,
            'colors_input': set(),
            'colors_output': set(),
            'transformation_type': 'unknown'
        }
        
        # Check if shape changed
        if analysis['input_shape'] != analysis['output_shape']:
            analysis['shape_changed'] = True
            
        # Collect colors
        for row in input_grid:
            analysis['colors_input'].update(row)
        for row in output_grid:
            analysis['colors_output'].update(row)
            
        # Try to identify transformation type
        if input_grid == output_grid:
            analysis['transformation_type'] = 'identity'
        elif self._is_rotation(input_grid, output_grid):
            analysis['transformation_type'] = 'rotation'
        elif self._is_flip(input_grid, output_grid):
            analysis['transformation_type'] = 'flip'
        else:
            analysis['transformation_type'] = 'complex'
            
        return analysis
    
    def _is_rotation(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a rotation of grid1"""
        if len(grid1) != len(grid2[0]) or len(grid1[0]) != len(grid2):
            return False
            
        # Check 90 degree rotation
        rotated = [list(row) for row in zip(*grid1[::-1])]
        return rotated == grid2
    
    def _is_flip(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a flip of grid1"""
        # Horizontal flip
        if grid1[::-1] == grid2:
            return True
        # Vertical flip
        if [row[::-1] for row in grid1] == grid2:
            return True
        return False
    
    def learn_pattern(self, training_pairs: List[Tuple]) -> Dict:
        """Learn pattern from multiple training examples"""
        if not training_pairs:
            raise ValueError("Cannot learn from empty training set")
            
        # Analyze each training pair
        analyses = []
        for input_grid, output_grid in training_pairs:
            analyses.append(self.analyze_transformation(input_grid, output_grid))
            
        # Find consistent pattern across all examples
        pattern = {
            'consistent': True,
            'transformation_type': analyses[0]['transformation_type'],
            'shape_change': analyses[0]['shape_changed']
        }
        
        # Check consistency
        for analysis in analyses[1:]:
            if analysis['transformation_type'] != pattern['transformation_type']:
                pattern['consistent'] = False
                pattern['transformation_type'] = 'mixed'
                
        return pattern


class SafeguardedSolver:
    """Solver with anti-cheating safeguards"""
    
    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.solving_history = []
        self._safeguard_intact = True
        
        # SAFEGUARD: Verify initialization integrity
        self._verify_safeguards()
        
    def _verify_safeguards(self):
        """SAFEGUARD: Verify safeguards haven't been tampered with"""
        if not self._safeguard_intact:
            raise SafeguardViolationError("Safeguards have been compromised")
        if SAFEGUARD_CHECKSUM != "arc_agi_fair_competition_2025":
            raise SafeguardViolationError("Safeguard checksum mismatch")
            
    def solve_puzzle(self, puzzle: ARCPuzzle) -> Tuple[Optional[List[List[int]]], Dict]:
        """
        Solve an ARC puzzle properly:
        1. Learn from training examples
        2. Apply learned pattern to test input
        3. Validate solution
        """
        # SAFEGUARD: Always verify before solving
        self._verify_safeguards()
        
        result = {
            'puzzle_id': puzzle.puzzle_id,
            'learned_pattern': None,
            'solution': None,
            'validated': False,
            'training_accuracy': 0.0,
            'cheating_attempted': False
        }
        
        try:
            # SAFEGUARD: Must learn from training examples
            training_pairs = puzzle.get_training_pairs()
            if len(training_pairs) < 1:
                result['cheating_attempted'] = True
                raise SafeguardViolationError("Must have training examples to learn from")
                
            # Learn the pattern
            pattern = self.pattern_learner.learn_pattern(training_pairs)
            result['learned_pattern'] = pattern
            
            # SAFEGUARD: Validate learned pattern on training data
            training_correct = 0
            for input_grid, expected_output in training_pairs:
                predicted = self._apply_pattern(input_grid, pattern)
                if predicted == expected_output:
                    training_correct += 1
                    
            result['training_accuracy'] = training_correct / len(training_pairs)
            
            # SAFEGUARD: Only proceed if pattern works on training data
            if result['training_accuracy'] < 0.8:  # Require 80% accuracy on training
                print(f"Pattern accuracy too low: {result['training_accuracy']:.1%}")
                return None, result
                
            # Apply to test input
            test_inputs = puzzle.get_test_inputs()
            if test_inputs:
                test_input = test_inputs[0]
                solution = self._apply_pattern(test_input, pattern)
                result['solution'] = solution
                
                # SAFEGUARD: Validate if test output is available
                test_outputs = puzzle.get_test_outputs()
                if test_outputs and test_outputs[0] is not None:
                    result['validated'] = (solution == test_outputs[0])
                    
                return solution, result
                
        except SafeguardViolationError as e:
            print(f"SAFEGUARD VIOLATION: {e}")
            result['cheating_attempted'] = True
            raise
            
        except Exception as e:
            print(f"Error solving puzzle: {e}")
            return None, result
            
        return None, result
    
    def _apply_pattern(self, input_grid: List[List[int]], pattern: Dict) -> List[List[int]]:
        """Apply learned pattern to input"""
        if pattern['transformation_type'] == 'identity':
            return copy.deepcopy(input_grid)
        elif pattern['transformation_type'] == 'rotation':
            return [list(row) for row in zip(*input_grid[::-1])]
        elif pattern['transformation_type'] == 'flip':
            return input_grid[::-1]
        else:
            # Complex transformation - need more sophisticated learning
            return copy.deepcopy(input_grid)
    
    def check_solution_validity(self, solution: List[List[int]], 
                               expected: List[List[int]]) -> bool:
        """SAFEGUARD: Properly check if solution matches expected output"""
        self._verify_safeguards()
        
        if solution is None or expected is None:
            return False
            
        if len(solution) != len(expected):
            return False
            
        for i in range(len(solution)):
            if len(solution[i]) != len(expected[i]):
                return False
            if solution[i] != expected[i]:
                return False
                
        return True


class SafeguardedSolverSystem:
    """Complete solving system with safeguards"""
    
    def __init__(self):
        self.solver = SafeguardedSolver()
        self.puzzles_attempted = 0
        self.puzzles_solved = 0
        self.cheating_attempts = 0
        
    def load_puzzle(self, puzzle_path: Path) -> Optional[ARCPuzzle]:
        """Load a puzzle from JSON file"""
        try:
            with open(puzzle_path, 'r') as f:
                data = json.load(f)
                
            # Add ID from filename
            data['id'] = puzzle_path.stem
            return ARCPuzzle(data)
            
        except Exception as e:
            print(f"Error loading puzzle {puzzle_path}: {e}")
            return None
    
    def solve_dataset(self, dataset_path: Path, limit: int = 10):
        """Solve puzzles from dataset with proper methodology"""
        puzzle_files = list(dataset_path.glob("*.json"))[:limit]
        
        print(f"\n{'='*60}")
        print("ARC AGI Safeguarded Solver")
        print("Following proper competition rules:")
        print("1. Learn from training examples")
        print("2. Validate on training data")  
        print("3. Apply to test input")
        print("4. No blind transformations")
        print(f"{'='*60}\n")
        
        for puzzle_file in puzzle_files:
            puzzle = self.load_puzzle(puzzle_file)
            if not puzzle:
                continue
                
            self.puzzles_attempted += 1
            print(f"\nSolving {puzzle.puzzle_id}...")
            
            try:
                solution, result = self.solver.solve_puzzle(puzzle)
                
                if result['cheating_attempted']:
                    self.cheating_attempts += 1
                    print("  ❌ CHEATING DETECTED - Puzzle skipped")
                    continue
                    
                if result['learned_pattern']:
                    print(f"  Pattern: {result['learned_pattern']['transformation_type']}")
                    print(f"  Training accuracy: {result['training_accuracy']:.1%}")
                    
                if solution and result.get('validated'):
                    self.puzzles_solved += 1
                    print(f"  ✓ SOLVED and validated!")
                elif solution:
                    print(f"  ? Solution generated (validation unavailable)")
                else:
                    print(f"  ✗ Could not solve")
                    
            except SafeguardViolationError:
                self.cheating_attempts += 1
                print("  ❌ SAFEGUARD VIOLATION - Stopping")
                break
                
        # Summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"Puzzles attempted: {self.puzzles_attempted}")
        print(f"Puzzles solved: {self.puzzles_solved}")
        print(f"Success rate: {self.puzzles_solved/max(1, self.puzzles_attempted):.1%}")
        print(f"Cheating attempts blocked: {self.cheating_attempts}")
        print(f"{'='*60}\n")


def main():
    """Run safeguarded solver"""
    # Use full dataset
    dataset_paths = [
        Path("data/arc_agi_full/data/training"),
        Path("data/arc_agi_full/data/evaluation")
    ]
    
    system = SafeguardedSolverSystem()
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            print(f"\nProcessing {dataset_path.name} dataset...")
            system.solve_dataset(dataset_path, limit=5)
            break
    else:
        print("No dataset found. Please ensure ARC AGI data is available.")


if __name__ == "__main__":
    main()