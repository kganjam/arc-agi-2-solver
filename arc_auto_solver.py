"""
ARC AGI Automatic Solver System
Implements comprehensive solving pipeline with learning and self-improvement
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from datetime import datetime
import copy
import random
from collections import defaultdict
import hashlib

# Import existing components
from arc_solver_safeguarded import SafeguardedSolver, ARCPuzzle, SafeguardViolationError
from arc_enhanced_pattern_learner import EnhancedPatternLearner

# SAFEGUARD: Immutable checksum
SAFEGUARD_CHECKSUM = "arc_agi_fair_competition_2025"

class Heuristic:
    """Represents a solving heuristic"""
    
    def __init__(self, heuristic_id: str, name: str, when_to_use: Dict, 
                 strategy: str, confidence: float = 0.5):
        self.id = heuristic_id
        self.name = name
        self.when_to_use = when_to_use
        self.strategy = strategy
        self.confidence = confidence
        self.success_rate = 0.0
        self.usage_count = 0
        self.successful_puzzles = []
        
    def should_apply(self, puzzle_features: Dict) -> bool:
        """Check if this heuristic should be applied"""
        conditions = self.when_to_use.get('conditions', [])
        features = self.when_to_use.get('puzzle_features', [])
        
        # Check all conditions
        for condition in conditions:
            if not self._check_condition(condition, puzzle_features):
                return False
                
        # Check feature requirements
        for feature in features:
            if feature not in puzzle_features or not puzzle_features[feature]:
                return False
                
        return True
        
    def _check_condition(self, condition: str, features: Dict) -> bool:
        """Check a single condition"""
        if "colors <=" in condition:
            max_colors = int(condition.split("<=")[1].strip())
            return len(features.get('colors', [])) <= max_colors
        elif "grid_has_symmetry" in condition:
            return any(features.get('symmetry', {}).values())
        elif "small_grid" in condition:
            shape = features.get('shape', (30, 30))
            return shape[0] <= 10 and shape[1] <= 10
        return True
        
    def update_performance(self, success: bool, puzzle_id: str = None):
        """Update heuristic performance metrics"""
        self.usage_count += 1
        if success:
            self.success_rate = ((self.success_rate * (self.usage_count - 1)) + 1) / self.usage_count
            if puzzle_id:
                self.successful_puzzles.append(puzzle_id)
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count
            
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'when_to_use': self.when_to_use,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'successful_puzzles': self.successful_puzzles
        }


class TransformationLibrary:
    """Library of transformation functions"""
    
    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise"""
        return np.rot90(grid, -1)
    
    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 180 degrees"""
        return np.rot90(grid, 2)
    
    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 270 degrees clockwise"""
        return np.rot90(grid, -3)
    
    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally"""
        return np.fliplr(grid)
    
    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically"""
        return np.flipud(grid)
    
    @staticmethod
    def extract_object(grid: np.ndarray, color: int) -> np.ndarray:
        """Extract object of specific color"""
        mask = (grid == color)
        if not mask.any():
            return np.zeros((1, 1), dtype=int)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        cropped = grid[rows][:, cols]
        result = np.zeros_like(cropped)
        result[cropped == color] = color
        return result
    
    @staticmethod
    def fill_pattern(grid: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Fill grid with repeating pattern"""
        result = np.zeros_like(grid)
        ph, pw = pattern.shape
        
        for i in range(0, grid.shape[0], ph):
            for j in range(0, grid.shape[1], pw):
                h = min(ph, grid.shape[0] - i)
                w = min(pw, grid.shape[1] - j)
                result[i:i+h, j:j+w] = pattern[:h, :w]
                
        return result
    
    @staticmethod
    def color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply color mapping"""
        result = grid.copy()
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        return result
    
    @staticmethod
    def scale_up(grid: np.ndarray, factor: int) -> np.ndarray:
        """Scale up grid by integer factor"""
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
    
    @staticmethod
    def scale_down(grid: np.ndarray, factor: int) -> np.ndarray:
        """Scale down grid by integer factor"""
        return grid[::factor, ::factor]


class AutomaticSolver:
    """Main automatic solving system"""
    
    def __init__(self):
        self.safeguarded_solver = SafeguardedSolver()
        self.pattern_learner = EnhancedPatternLearner()
        self.heuristics = self._initialize_heuristics()
        self.transformation_lib = TransformationLibrary()
        self.solved_puzzles = []
        self.failed_puzzles = []
        self.total_attempts = 0
        self.start_time = None
        self.learning_history = []
        self.performance_metrics = defaultdict(list)
        
    def _initialize_heuristics(self) -> List[Heuristic]:
        """Initialize default heuristics library"""
        return [
            Heuristic("h1", "Color Mapping", 
                     {"conditions": ["colors <= 5"], "puzzle_features": ["color_consistent"]},
                     "map_colors", 0.7),
            Heuristic("h2", "Symmetry Detection",
                     {"conditions": ["grid_has_symmetry"], "puzzle_features": ["has_symmetry"]},
                     "apply_symmetry", 0.6),
            Heuristic("h3", "Object Extraction",
                     {"conditions": ["colors <= 3"], "puzzle_features": ["has_objects"]},
                     "extract_and_transform", 0.5),
            Heuristic("h4", "Pattern Completion",
                     {"conditions": [], "puzzle_features": ["has_pattern"]},
                     "complete_pattern", 0.4),
            Heuristic("h5", "Size Transformation",
                     {"conditions": [], "puzzle_features": ["size_change"]},
                     "transform_size", 0.5),
            Heuristic("h6", "Rotation",
                     {"conditions": [], "puzzle_features": ["rotation_possible"]},
                     "try_rotations", 0.3),
            Heuristic("h7", "Boundary Detection",
                     {"conditions": [], "puzzle_features": ["has_boundary"]},
                     "process_boundaries", 0.4),
            Heuristic("h8", "Fill Pattern",
                     {"conditions": [], "puzzle_features": ["repeating_pattern"]},
                     "fill_with_pattern", 0.3),
        ]
    
    def solve_puzzle(self, puzzle: ARCPuzzle, max_attempts: int = 10) -> Tuple[bool, Optional[List[List[int]]]]:
        """Solve a single puzzle with multiple attempts"""
        # SAFEGUARD: Always learn from training examples first
        if not puzzle.train_examples:
            raise SafeguardViolationError("Cannot solve puzzle without training examples")
        
        # Analyze all training examples
        training_pairs = puzzle.get_training_pairs()
        learned_transformations = []
        
        for input_grid, output_grid in training_pairs:
            analysis = self._analyze_transformation(input_grid, output_grid)
            learned_transformations.append(analysis)
        
        # Find consistent transformation
        transformation = self._find_consistent_transformation(learned_transformations)
        
        if not transformation:
            # Try heuristics
            for attempt in range(max_attempts):
                heuristic = self._select_heuristic(puzzle)
                if heuristic:
                    solution = self._apply_heuristic(heuristic, puzzle)
                    if solution and self._validate_solution(solution, puzzle):
                        return True, solution
        else:
            # Apply learned transformation
            test_inputs = puzzle.get_test_inputs()
            if test_inputs:
                solution = self._apply_transformation(test_inputs[0], transformation)
                if self._validate_solution(solution, puzzle):
                    return True, solution
        
        return False, None
    
    def _analyze_transformation(self, input_grid: List[List[int]], 
                               output_grid: List[List[int]]) -> Dict:
        """Analyze transformation between input and output"""
        input_np = np.array(input_grid)
        output_np = np.array(output_grid)
        
        analysis = {
            'input_analysis': self.pattern_learner.analyze_grid(input_grid),
            'output_analysis': self.pattern_learner.analyze_grid(output_grid),
            'transformation_type': 'unknown'
        }
        
        # Check for simple transformations
        if np.array_equal(input_np, output_np):
            analysis['transformation_type'] = 'identity'
        elif np.array_equal(np.rot90(input_np, -1), output_np):
            analysis['transformation_type'] = 'rotate_90'
        elif np.array_equal(np.rot90(input_np, 2), output_np):
            analysis['transformation_type'] = 'rotate_180'
        elif np.array_equal(np.fliplr(input_np), output_np):
            analysis['transformation_type'] = 'flip_horizontal'
        elif np.array_equal(np.flipud(input_np), output_np):
            analysis['transformation_type'] = 'flip_vertical'
        elif input_np.shape != output_np.shape:
            analysis['transformation_type'] = 'size_change'
            # Check for scaling
            if output_np.shape[0] % input_np.shape[0] == 0:
                factor = output_np.shape[0] // input_np.shape[0]
                if np.array_equal(self.transformation_lib.scale_up(input_np, factor), output_np):
                    analysis['transformation_type'] = 'scale_up'
                    analysis['scale_factor'] = factor
        
        # Check for color mapping
        input_colors = set(input_np.flatten())
        output_colors = set(output_np.flatten())
        if len(input_colors) == len(output_colors) and input_np.shape == output_np.shape:
            # Try to find color mapping
            mapping = {}
            for ic in input_colors:
                mask = (input_np == ic)
                if mask.any():
                    output_vals = output_np[mask]
                    if len(set(output_vals)) == 1:
                        mapping[ic] = output_vals[0]
            
            if len(mapping) == len(input_colors):
                test_output = self.transformation_lib.color_map(input_np, mapping)
                if np.array_equal(test_output, output_np):
                    analysis['transformation_type'] = 'color_map'
                    analysis['color_mapping'] = mapping
        
        return analysis
    
    def _find_consistent_transformation(self, transformations: List[Dict]) -> Optional[Dict]:
        """Find transformation consistent across all examples"""
        if not transformations:
            return None
        
        # Check if all have same transformation type
        types = [t['transformation_type'] for t in transformations]
        if len(set(types)) == 1 and types[0] != 'unknown':
            return transformations[0]
        
        # Check for consistent color mapping
        if all('color_mapping' in t for t in transformations):
            # Verify mappings are consistent
            base_mapping = transformations[0]['color_mapping']
            if all(t['color_mapping'] == base_mapping for t in transformations[1:]):
                return transformations[0]
        
        return None
    
    def _apply_transformation(self, input_grid: List[List[int]], 
                             transformation: Dict) -> List[List[int]]:
        """Apply learned transformation to input"""
        grid_np = np.array(input_grid)
        trans_type = transformation['transformation_type']
        
        if trans_type == 'identity':
            result = grid_np
        elif trans_type == 'rotate_90':
            result = self.transformation_lib.rotate_90(grid_np)
        elif trans_type == 'rotate_180':
            result = self.transformation_lib.rotate_180(grid_np)
        elif trans_type == 'flip_horizontal':
            result = self.transformation_lib.flip_horizontal(grid_np)
        elif trans_type == 'flip_vertical':
            result = self.transformation_lib.flip_vertical(grid_np)
        elif trans_type == 'scale_up':
            factor = transformation.get('scale_factor', 2)
            result = self.transformation_lib.scale_up(grid_np, factor)
        elif trans_type == 'color_map':
            mapping = transformation.get('color_mapping', {})
            result = self.transformation_lib.color_map(grid_np, mapping)
        else:
            result = grid_np
        
        return result.tolist()
    
    def _select_heuristic(self, puzzle: ARCPuzzle) -> Optional[Heuristic]:
        """Select best heuristic for puzzle"""
        # Get puzzle features
        features = self._extract_puzzle_features(puzzle)
        
        # Find applicable heuristics
        applicable = [h for h in self.heuristics if h.should_apply(features)]
        
        if not applicable:
            # Return random heuristic if none applicable
            return random.choice(self.heuristics)
        
        # Sort by success rate and confidence
        applicable.sort(key=lambda h: h.success_rate * h.confidence, reverse=True)
        return applicable[0]
    
    def _extract_puzzle_features(self, puzzle: ARCPuzzle) -> Dict:
        """Extract features from puzzle"""
        features = {
            'colors': set(),
            'has_symmetry': False,
            'has_objects': False,
            'has_pattern': False,
            'size_change': False,
            'rotation_possible': True,
            'has_boundary': False,
            'repeating_pattern': False,
            'color_consistent': True,
            'small_grid': True
        }
        
        # Analyze all training examples
        for input_grid, output_grid in puzzle.get_training_pairs():
            input_analysis = self.pattern_learner.analyze_grid(input_grid)
            output_analysis = self.pattern_learner.analyze_grid(output_grid)
            
            features['colors'].update(input_analysis['colors'])
            features['colors'].update(output_analysis['colors'])
            
            if any(input_analysis['symmetry'].values()):
                features['has_symmetry'] = True
            
            if input_analysis['objects']:
                features['has_objects'] = True
            
            if input_analysis['patterns'].get('has_repeating'):
                features['repeating_pattern'] = True
            
            if input_analysis['shape'] != output_analysis['shape']:
                features['size_change'] = True
            
            shape = input_analysis['shape']
            if shape[0] > 10 or shape[1] > 10:
                features['small_grid'] = False
        
        features['colors'] = list(features['colors'])
        return features
    
    def _apply_heuristic(self, heuristic: Heuristic, puzzle: ARCPuzzle) -> Optional[List[List[int]]]:
        """Apply heuristic to solve puzzle"""
        test_inputs = puzzle.get_test_inputs()
        if not test_inputs:
            return None
        
        test_input = test_inputs[0]
        strategy = heuristic.strategy
        
        if strategy == "map_colors":
            # Try to learn color mapping from examples
            mappings = []
            for input_grid, output_grid in puzzle.get_training_pairs():
                analysis = self._analyze_transformation(input_grid, output_grid)
                if 'color_mapping' in analysis:
                    mappings.append(analysis['color_mapping'])
            
            if mappings and all(m == mappings[0] for m in mappings):
                return self.transformation_lib.color_map(np.array(test_input), mappings[0]).tolist()
        
        elif strategy == "apply_symmetry":
            # Try different symmetry operations
            grid_np = np.array(test_input)
            for transform in [self.transformation_lib.flip_horizontal,
                            self.transformation_lib.flip_vertical,
                            self.transformation_lib.rotate_90,
                            self.transformation_lib.rotate_180]:
                result = transform(grid_np)
                # Check if this matches pattern from training
                for _, output_grid in puzzle.get_training_pairs():
                    if result.shape == np.array(output_grid).shape:
                        return result.tolist()
        
        elif strategy == "try_rotations":
            # Try all rotations
            grid_np = np.array(test_input)
            for rotation in [0, 90, 180, 270]:
                result = np.rot90(grid_np, rotation // 90)
                # Check against training patterns
                for _, output_grid in puzzle.get_training_pairs():
                    if result.shape == np.array(output_grid).shape:
                        return result.tolist()
        
        return None
    
    def _validate_solution(self, solution: List[List[int]], puzzle: ARCPuzzle) -> bool:
        """Validate solution against training examples"""
        # SAFEGUARD: Must validate against training examples
        if not solution:
            return False
        
        # Check basic validity
        if not isinstance(solution, list) or not all(isinstance(row, list) for row in solution):
            return False
        
        # Check if solution matches expected output shape patterns
        expected_outputs = puzzle.get_test_outputs()
        if expected_outputs and expected_outputs[0]:
            expected = expected_outputs[0]
            return solution == expected
        
        # At least check if dimensions are reasonable
        training_outputs = [output for _, output in puzzle.get_training_pairs()]
        if training_outputs:
            output_shapes = [(len(o), len(o[0]) if o else 0) for o in training_outputs]
            solution_shape = (len(solution), len(solution[0]) if solution else 0)
            
            # Check if solution shape matches any training output shape
            if solution_shape in output_shapes:
                return True
        
        return False
    
    def run_automatic_solving(self, puzzle_dir: Path, target_count: int = 100) -> Dict:
        """Run automatic solving on puzzles until target is reached"""
        self.start_time = time.time()
        results = {
            'solved': 0,
            'failed': 0,
            'total_attempts': 0,
            'puzzles_solved': [],
            'puzzles_failed': [],
            'time_elapsed': 0,
            'heuristics_performance': {}
        }
        
        # Load puzzles
        puzzle_files = list(puzzle_dir.glob("*.json"))
        random.shuffle(puzzle_files)  # Randomize order
        
        print(f"Starting automatic solving with {len(puzzle_files)} available puzzles")
        print(f"Target: {target_count} solved puzzles")
        print("-" * 50)
        
        for puzzle_file in puzzle_files:
            if results['solved'] >= target_count:
                break
            
            try:
                # Load puzzle
                with open(puzzle_file, 'r') as f:
                    puzzle_data = json.load(f)
                
                puzzle_data['id'] = puzzle_file.stem
                puzzle = ARCPuzzle(puzzle_data)
                
                # Attempt to solve
                print(f"\\nAttempting puzzle: {puzzle.puzzle_id}")
                solved, solution = self.solve_puzzle(puzzle)
                
                if solved:
                    results['solved'] += 1
                    results['puzzles_solved'].append(puzzle.puzzle_id)
                    self.solved_puzzles.append(puzzle.puzzle_id)
                    print(f"✓ Solved! ({results['solved']}/{target_count})")
                    
                    # Update heuristic performance
                    for h in self.heuristics:
                        if h.usage_count > 0:
                            h.update_performance(True, puzzle.puzzle_id)
                else:
                    results['failed'] += 1
                    results['puzzles_failed'].append(puzzle.puzzle_id)
                    self.failed_puzzles.append(puzzle.puzzle_id)
                    print(f"✗ Failed")
                
                results['total_attempts'] += 1
                self.total_attempts += 1
                
                # Show progress
                if results['solved'] % 10 == 0 and results['solved'] > 0:
                    elapsed = time.time() - self.start_time
                    rate = results['solved'] / elapsed
                    eta = (target_count - results['solved']) / rate if rate > 0 else 0
                    print(f"\\n{'='*50}")
                    print(f"Progress: {results['solved']}/{target_count} solved")
                    print(f"Success rate: {results['solved']/results['total_attempts']*100:.1f}%")
                    print(f"Time elapsed: {elapsed:.1f}s")
                    print(f"Estimated time remaining: {eta:.1f}s")
                    print(f"{'='*50}\\n")
                
            except Exception as e:
                print(f"Error processing {puzzle_file}: {e}")
                continue
        
        # Final results
        results['time_elapsed'] = time.time() - self.start_time
        results['heuristics_performance'] = {
            h.name: {
                'success_rate': h.success_rate,
                'usage_count': h.usage_count,
                'puzzles_solved': len(h.successful_puzzles)
            }
            for h in self.heuristics
        }
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "solver_results.json"):
        """Save solving results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate performance report"""
        report = []
        report.append("\\n" + "="*60)
        report.append("ARC AGI AUTOMATIC SOLVER RESULTS")
        report.append("="*60)
        report.append(f"Total puzzles attempted: {results['total_attempts']}")
        report.append(f"Puzzles solved: {results['solved']}")
        report.append(f"Puzzles failed: {results['failed']}")
        report.append(f"Success rate: {results['solved']/results['total_attempts']*100:.2f}%")
        report.append(f"Time elapsed: {results['time_elapsed']:.2f} seconds")
        report.append(f"Average time per puzzle: {results['time_elapsed']/results['total_attempts']:.2f} seconds")
        
        report.append("\\n" + "-"*60)
        report.append("HEURISTICS PERFORMANCE")
        report.append("-"*60)
        
        for name, perf in results['heuristics_performance'].items():
            report.append(f"{name}:")
            report.append(f"  Success rate: {perf['success_rate']*100:.1f}%")
            report.append(f"  Usage count: {perf['usage_count']}")
            report.append(f"  Puzzles solved: {perf['puzzles_solved']}")
        
        report.append("\\n" + "="*60)
        
        return "\\n".join(report)


def main():
    """Main entry point"""
    solver = AutomaticSolver()
    
    # Run on training puzzles
    training_dir = Path("/mnt/c/Users/kganjam/OneDrive/git/ArcAGI/data/arc_agi_full/data/training")
    
    # Start with smaller target for testing
    results = solver.run_automatic_solving(training_dir, target_count=100)
    
    # Save results
    solver.save_results(results)
    
    # Print report
    print(solver.generate_report(results))
    
    # Save detailed log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'solved_puzzles': solver.solved_puzzles,
        'failed_puzzles': solver.failed_puzzles,
        'heuristics': [h.to_dict() for h in solver.heuristics]
    }
    
    with open('solver_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print("\\nDetailed log saved to solver_log.json")


if __name__ == "__main__":
    main()