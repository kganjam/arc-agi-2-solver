"""
Synthetic Puzzle Generator with GAN-inspired Architecture
Generates novel ARC-style puzzles for training
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random
from enum import Enum

class TransformationType(Enum):
    """Types of transformations for puzzle generation"""
    COLOR_MAPPING = "color_mapping"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    SCALING = "scaling"
    PATTERN_COMPLETION = "pattern_completion"
    OBJECT_MANIPULATION = "object_manipulation"
    SYMMETRY = "symmetry"
    BOUNDARY = "boundary"
    TILING = "tiling"
    EXTRACTION = "extraction"

class PuzzleGenerator:
    """Generates synthetic ARC-style puzzles"""
    
    def __init__(self, complexity_level: int = 1):
        self.complexity_level = complexity_level
        self.generated_count = 0
        self.transformation_history = []
        
    def generate_puzzle(self, transformation_type: Optional[TransformationType] = None) -> Dict:
        """Generate a synthetic puzzle"""
        if transformation_type is None:
            transformation_type = random.choice(list(TransformationType))
            
        # Generate base grid
        size = random.randint(3 + self.complexity_level, 10 + self.complexity_level * 2)
        num_colors = min(2 + self.complexity_level, 9)
        
        # Create training examples
        train_examples = []
        for _ in range(random.randint(2, 4)):
            input_grid = self._generate_base_grid(size, num_colors)
            output_grid = self._apply_transformation(input_grid, transformation_type)
            train_examples.append({
                'input': input_grid.tolist(),
                'output': output_grid.tolist()
            })
            
        # Create test example
        test_input = self._generate_base_grid(size, num_colors)
        test_output = self._apply_transformation(test_input, transformation_type)
        
        puzzle = {
            'id': f'synthetic_{self.generated_count}',
            'train': train_examples,
            'test': [{
                'input': test_input.tolist(),
                'output': test_output.tolist()  # Include for validation
            }],
            'metadata': {
                'transformation': transformation_type.value,
                'complexity': self.complexity_level,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        self.generated_count += 1
        self.transformation_history.append(transformation_type)
        
        return puzzle
        
    def _generate_base_grid(self, size: int, num_colors: int) -> np.ndarray:
        """Generate a base grid with patterns"""
        grid = np.zeros((size, size), dtype=int)
        
        # Choose pattern type
        pattern_type = random.choice(['random', 'structured', 'objects', 'symmetric'])
        
        if pattern_type == 'random':
            # Random colors
            grid = np.random.randint(0, num_colors, (size, size))
            
        elif pattern_type == 'structured':
            # Create structured pattern
            for i in range(size):
                for j in range(size):
                    grid[i, j] = (i + j) % num_colors
                    
        elif pattern_type == 'objects':
            # Place random objects
            num_objects = random.randint(1, min(5, size // 2))
            for _ in range(num_objects):
                obj_size = random.randint(1, min(3, size // 3))
                x = random.randint(0, size - obj_size)
                y = random.randint(0, size - obj_size)
                color = random.randint(1, num_colors - 1)
                grid[x:x+obj_size, y:y+obj_size] = color
                
        elif pattern_type == 'symmetric':
            # Create symmetric pattern
            half_size = size // 2
            for i in range(half_size):
                for j in range(size):
                    color = random.randint(0, num_colors - 1)
                    grid[i, j] = color
                    grid[size - 1 - i, j] = color  # Mirror vertically
                    
        return grid
        
    def _apply_transformation(self, grid: np.ndarray, 
                            transformation_type: TransformationType) -> np.ndarray:
        """Apply transformation to grid"""
        
        if transformation_type == TransformationType.COLOR_MAPPING:
            return self._transform_color_mapping(grid)
        elif transformation_type == TransformationType.ROTATION:
            return self._transform_rotation(grid)
        elif transformation_type == TransformationType.REFLECTION:
            return self._transform_reflection(grid)
        elif transformation_type == TransformationType.SCALING:
            return self._transform_scaling(grid)
        elif transformation_type == TransformationType.PATTERN_COMPLETION:
            return self._transform_pattern_completion(grid)
        elif transformation_type == TransformationType.OBJECT_MANIPULATION:
            return self._transform_object_manipulation(grid)
        elif transformation_type == TransformationType.SYMMETRY:
            return self._transform_symmetry(grid)
        elif transformation_type == TransformationType.BOUNDARY:
            return self._transform_boundary(grid)
        elif transformation_type == TransformationType.TILING:
            return self._transform_tiling(grid)
        elif transformation_type == TransformationType.EXTRACTION:
            return self._transform_extraction(grid)
        else:
            return grid
            
    def _transform_color_mapping(self, grid: np.ndarray) -> np.ndarray:
        """Apply color mapping transformation"""
        unique_colors = np.unique(grid)
        color_map = {c: (c + 1) % len(unique_colors) for c in unique_colors}
        
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
            
        return result
        
    def _transform_rotation(self, grid: np.ndarray) -> np.ndarray:
        """Apply rotation transformation"""
        rotations = random.choice([1, 2, 3])
        return np.rot90(grid, rotations)
        
    def _transform_reflection(self, grid: np.ndarray) -> np.ndarray:
        """Apply reflection transformation"""
        if random.random() < 0.5:
            return np.fliplr(grid)  # Horizontal flip
        else:
            return np.flipud(grid)  # Vertical flip
            
    def _transform_scaling(self, grid: np.ndarray) -> np.ndarray:
        """Apply scaling transformation"""
        scale = random.choice([2, 3])
        new_size = min(grid.shape[0] * scale, 30)
        
        result = np.zeros((new_size, new_size), dtype=int)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                result[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = grid[i, j]
                
        return result[:new_size, :new_size]
        
    def _transform_pattern_completion(self, grid: np.ndarray) -> np.ndarray:
        """Complete partial patterns"""
        result = grid.copy()
        
        # Find the most common non-zero value
        non_zero = grid[grid > 0]
        if len(non_zero) > 0:
            common_value = np.bincount(non_zero).argmax()
            
            # Fill some zeros with the common value
            zero_positions = np.where(grid == 0)
            if len(zero_positions[0]) > 0:
                num_to_fill = len(zero_positions[0]) // 2
                indices = np.random.choice(len(zero_positions[0]), num_to_fill, replace=False)
                for idx in indices:
                    result[zero_positions[0][idx], zero_positions[1][idx]] = common_value
                    
        return result
        
    def _transform_object_manipulation(self, grid: np.ndarray) -> np.ndarray:
        """Manipulate objects in grid"""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(grid > 0)
        
        if num_features > 0:
            # Move or resize largest object
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_idx = np.argmax(sizes) + 1
            
            # Extract largest object
            obj_mask = (labeled == largest_idx)
            obj_value = grid[obj_mask][0]
            
            # Create result with object moved
            result = grid.copy()
            result[obj_mask] = 0  # Remove object
            
            # Place object in new position
            obj_coords = np.where(obj_mask)
            min_r, max_r = obj_coords[0].min(), obj_coords[0].max()
            min_c, max_c = obj_coords[1].min(), obj_coords[1].max()
            
            # Shift object
            shift_r = random.randint(-2, 2)
            shift_c = random.randint(-2, 2)
            
            for i in range(len(obj_coords[0])):
                new_r = obj_coords[0][i] + shift_r
                new_c = obj_coords[1][i] + shift_c
                if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                    result[new_r, new_c] = obj_value
                    
            return result
        else:
            return grid
            
    def _transform_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """Make grid symmetric"""
        result = grid.copy()
        
        if random.random() < 0.5:
            # Horizontal symmetry
            for i in range(grid.shape[0] // 2):
                result[grid.shape[0] - 1 - i, :] = result[i, :]
        else:
            # Vertical symmetry
            for j in range(grid.shape[1] // 2):
                result[:, grid.shape[1] - 1 - j] = result[:, j]
                
        return result
        
    def _transform_boundary(self, grid: np.ndarray) -> np.ndarray:
        """Add or modify boundary"""
        result = grid.copy()
        boundary_color = random.randint(1, np.max(grid) + 1)
        
        # Add boundary
        result[0, :] = boundary_color
        result[-1, :] = boundary_color
        result[:, 0] = boundary_color
        result[:, -1] = boundary_color
        
        return result
        
    def _transform_tiling(self, grid: np.ndarray) -> np.ndarray:
        """Create tiled pattern"""
        tile_size = 2
        new_size = grid.shape[0] * tile_size
        
        if new_size <= 30:
            result = np.tile(grid, (tile_size, tile_size))
            return result
        else:
            return grid
            
    def _transform_extraction(self, grid: np.ndarray) -> np.ndarray:
        """Extract a region"""
        size = grid.shape[0]
        extract_size = max(2, size // 2)
        
        start_r = random.randint(0, size - extract_size)
        start_c = random.randint(0, size - extract_size)
        
        return grid[start_r:start_r+extract_size, start_c:start_c+extract_size]

class PuzzleDiscriminator:
    """Discriminator to validate puzzle quality"""
    
    def __init__(self):
        self.validation_history = []
        
    def validate_puzzle(self, puzzle: Dict) -> Dict:
        """Validate if puzzle is well-formed and solvable"""
        validation_result = {
            'valid': True,
            'score': 0.0,
            'issues': []
        }
        
        # Check structure
        if 'train' not in puzzle or 'test' not in puzzle:
            validation_result['valid'] = False
            validation_result['issues'].append('Missing train or test data')
            return validation_result
            
        # Check training examples
        if len(puzzle['train']) < 1:
            validation_result['valid'] = False
            validation_result['issues'].append('No training examples')
            return validation_result
            
        # Check consistency
        input_shapes = []
        output_shapes = []
        
        for example in puzzle['train']:
            if 'input' not in example or 'output' not in example:
                validation_result['valid'] = False
                validation_result['issues'].append('Incomplete training example')
                return validation_result
                
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            input_shapes.append(input_grid.shape)
            output_shapes.append(output_grid.shape)
            
        # Check if transformation is consistent
        if len(set(input_shapes)) > 2:
            validation_result['issues'].append('Inconsistent input shapes')
            validation_result['score'] -= 0.2
            
        if len(set(output_shapes)) > 2:
            validation_result['issues'].append('Inconsistent output shapes')
            validation_result['score'] -= 0.2
            
        # Check complexity
        complexity_score = self._calculate_complexity(puzzle)
        validation_result['score'] += complexity_score
        
        # Check learnability
        learnability_score = self._calculate_learnability(puzzle)
        validation_result['score'] += learnability_score
        
        # Normalize score
        validation_result['score'] = max(0, min(1, validation_result['score']))
        
        self.validation_history.append(validation_result)
        
        return validation_result
        
    def _calculate_complexity(self, puzzle: Dict) -> float:
        """Calculate puzzle complexity"""
        complexity = 0.0
        
        if len(puzzle['train']) > 0:
            example = puzzle['train'][0]
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Size complexity
            size_factor = min(input_grid.size / 100, 1.0)
            complexity += size_factor * 0.2
            
            # Color complexity
            num_colors = len(np.unique(input_grid))
            color_factor = min(num_colors / 10, 1.0)
            complexity += color_factor * 0.2
            
            # Transformation complexity
            if input_grid.shape != output_grid.shape:
                complexity += 0.2
                
            # Pattern complexity
            if not np.array_equal(input_grid, output_grid):
                complexity += 0.2
                
        return complexity
        
    def _calculate_learnability(self, puzzle: Dict) -> float:
        """Calculate how learnable the pattern is"""
        if len(puzzle['train']) < 2:
            return 0.1
            
        # Check if transformation is consistent across examples
        transformations_consistent = True
        
        for i in range(len(puzzle['train']) - 1):
            ex1 = puzzle['train'][i]
            ex2 = puzzle['train'][i + 1]
            
            # Simple check: output should have similar properties
            out1 = np.array(ex1['output'])
            out2 = np.array(ex2['output'])
            
            if out1.shape != out2.shape:
                transformations_consistent = False
                break
                
        return 0.4 if transformations_consistent else 0.2

class SyntheticPuzzleGAN:
    """GAN-inspired system for generating high-quality puzzles"""
    
    def __init__(self):
        self.generator = PuzzleGenerator()
        self.discriminator = PuzzleDiscriminator()
        self.generated_puzzles = []
        self.quality_threshold = 0.6
        
    def generate_batch(self, batch_size: int, 
                       min_quality: float = 0.6) -> List[Dict]:
        """Generate batch of high-quality puzzles"""
        batch = []
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(batch) < batch_size and attempts < max_attempts:
            # Generate puzzle
            puzzle = self.generator.generate_puzzle()
            
            # Validate puzzle
            validation = self.discriminator.validate_puzzle(puzzle)
            
            # Accept if quality is sufficient
            if validation['valid'] and validation['score'] >= min_quality:
                batch.append(puzzle)
                self.generated_puzzles.append(puzzle)
                
            attempts += 1
            
            # Increase complexity if generating too easily
            if len(batch) > batch_size // 2 and attempts < batch_size * 2:
                self.generator.complexity_level = min(5, self.generator.complexity_level + 1)
                
        return batch
        
    def improve_generator(self):
        """Improve generator based on discriminator feedback"""
        if len(self.discriminator.validation_history) >= 10:
            # Analyze recent validations
            recent_validations = self.discriminator.validation_history[-10:]
            avg_score = np.mean([v['score'] for v in recent_validations])
            
            # Adjust complexity based on performance
            if avg_score < 0.5:
                self.generator.complexity_level = max(1, self.generator.complexity_level - 1)
            elif avg_score > 0.8:
                self.generator.complexity_level = min(5, self.generator.complexity_level + 1)
                
            # Analyze common issues
            all_issues = []
            for v in recent_validations:
                all_issues.extend(v['issues'])
                
            # Adjust generation strategy based on issues
            if 'Inconsistent input shapes' in all_issues:
                # Focus on consistent shapes
                pass  # Would implement shape consistency improvements
                
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        if not self.discriminator.validation_history:
            return {'generated': 0, 'average_quality': 0}
            
        scores = [v['score'] for v in self.discriminator.validation_history]
        valid_count = sum(1 for v in self.discriminator.validation_history if v['valid'])
        
        return {
            'generated': len(self.generated_puzzles),
            'validated': len(self.discriminator.validation_history),
            'valid_count': valid_count,
            'average_quality': np.mean(scores),
            'max_quality': np.max(scores),
            'min_quality': np.min(scores),
            'complexity_level': self.generator.complexity_level
        }