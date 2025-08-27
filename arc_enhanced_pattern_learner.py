"""
Enhanced Pattern Learner for ARC AGI Puzzles
Implements sophisticated pattern recognition while maintaining safeguards
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import copy
from collections import Counter

class EnhancedPatternLearner:
    """Advanced pattern learning for ARC puzzles"""
    
    def __init__(self):
        self.patterns = []
        self.transformations = []
        
    def analyze_grid(self, grid: List[List[int]]) -> Dict:
        """Comprehensive grid analysis"""
        grid_np = np.array(grid)
        h, w = grid_np.shape
        
        analysis = {
            'shape': (h, w),
            'colors': np.unique(grid_np).tolist(),
            'color_counts': dict(zip(*np.unique(grid_np, return_counts=True))),
            'objects': self._find_objects(grid_np),
            'symmetry': self._check_symmetry(grid_np),
            'patterns': self._detect_patterns(grid_np),
            'boundaries': self._find_boundaries(grid_np)
        }
        
        return analysis
    
    def _find_objects(self, grid: np.ndarray) -> List[Dict]:
        """Find connected components (objects) in grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj = self._flood_fill(grid, visited, i, j, grid[i, j])
                    if obj:
                        objects.append(obj)
                        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, 
                    start_i: int, start_j: int, color: int) -> Dict:
        """Extract connected component using flood fill"""
        stack = [(start_i, start_j)]
        cells = []
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or
                visited[i, j] or grid[i, j] != color):
                continue
                
            visited[i, j] = True
            cells.append((i, j))
            
            # Add neighbors
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
            
        if cells:
            min_i = min(c[0] for c in cells)
            max_i = max(c[0] for c in cells)
            min_j = min(c[1] for c in cells)
            max_j = max(c[1] for c in cells)
            
            return {
                'color': int(color),
                'cells': cells,
                'bounding_box': (min_i, min_j, max_i, max_j),
                'size': len(cells),
                'width': max_j - min_j + 1,
                'height': max_i - min_i + 1
            }
        return None
    
    def _check_symmetry(self, grid: np.ndarray) -> Dict:
        """Check for various symmetries"""
        return {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'diagonal': np.array_equal(grid, grid.T),
            'rotational_90': np.array_equal(grid, np.rot90(grid)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2))
        }
    
    def _detect_patterns(self, grid: np.ndarray) -> Dict:
        """Detect common patterns in grid"""
        patterns = {
            'has_border': self._has_border(grid),
            'has_filled_rect': self._has_filled_rectangle(grid),
            'has_repeating': self._has_repeating_pattern(grid),
            'has_diagonal': self._has_diagonal_pattern(grid)
        }
        return patterns
    
    def _has_border(self, grid: np.ndarray) -> bool:
        """Check if grid has a colored border"""
        if grid.size == 0:
            return False
        # Check if edges have same non-zero color
        edges = np.concatenate([
            grid[0, :], grid[-1, :],
            grid[1:-1, 0], grid[1:-1, -1]
        ])
        non_zero = edges[edges != 0]
        return len(non_zero) > 0 and len(np.unique(non_zero)) == 1
    
    def _has_filled_rectangle(self, grid: np.ndarray) -> bool:
        """Check for filled rectangles"""
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = (grid == color)
            if np.any(mask):
                rows, cols = np.where(mask)
                if len(rows) > 1 and len(cols) > 1:
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()
                    expected = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if np.sum(mask) == expected:
                        return True
        return False
    
    def _has_repeating_pattern(self, grid: np.ndarray) -> bool:
        """Check for repeating patterns"""
        h, w = grid.shape
        # Check for 2x2 repeating patterns
        if h >= 4 and w >= 4:
            for i in range(h-3):
                for j in range(w-3):
                    pattern = grid[i:i+2, j:j+2]
                    if np.array_equal(pattern, grid[i+2:i+4, j:j+2]):
                        return True
        return False
    
    def _has_diagonal_pattern(self, grid: np.ndarray) -> bool:
        """Check for diagonal patterns"""
        h, w = grid.shape
        if h != w:
            return False
        
        # Check main diagonal
        diagonal = np.diag(grid)
        if len(np.unique(diagonal[diagonal != 0])) == 1:
            return True
            
        # Check anti-diagonal
        anti_diagonal = np.diag(np.fliplr(grid))
        if len(np.unique(anti_diagonal[anti_diagonal != 0])) == 1:
            return True
            
        return False
    
    def _find_boundaries(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find boundary cells between different colors"""
        boundaries = []
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                # Check if this cell is on a boundary
                current = grid[i, j]
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni, nj] != current:
                            boundaries.append((i, j))
                            break
                            
        return boundaries
    
    def learn_transformation(self, input_grids: List[List[List[int]]], 
                            output_grids: List[List[List[int]]]) -> Dict:
        """Learn transformation from multiple examples"""
        if len(input_grids) != len(output_grids):
            raise ValueError("Input and output counts must match")
            
        transformations = []
        
        for inp, out in zip(input_grids, output_grids):
            inp_analysis = self.analyze_grid(inp)
            out_analysis = self.analyze_grid(out)
            
            transform = {
                'shape_change': inp_analysis['shape'] != out_analysis['shape'],
                'shape_ratio': (out_analysis['shape'][0] / inp_analysis['shape'][0],
                               out_analysis['shape'][1] / inp_analysis['shape'][1])
                               if inp_analysis['shape'][0] > 0 and inp_analysis['shape'][1] > 0 else (1, 1),
                'color_mapping': self._infer_color_mapping(inp, out),
                'object_changes': self._analyze_object_changes(inp_analysis['objects'], 
                                                              out_analysis['objects']),
                'pattern_type': self._infer_pattern_type(inp, out, inp_analysis, out_analysis)
            }
            transformations.append(transform)
            
        # Find consistent transformation
        consistent_transform = self._find_consistent_transform(transformations)
        
        return consistent_transform
    
    def _infer_color_mapping(self, inp: List[List[int]], out: List[List[int]]) -> Dict:
        """Infer color transformation mapping"""
        inp_flat = [c for row in inp for c in row]
        out_flat = [c for row in out for c in row]
        
        if len(inp_flat) == len(out_flat):
            # Direct mapping possible
            mapping = {}
            for i_color, o_color in zip(inp_flat, out_flat):
                if i_color not in mapping:
                    mapping[i_color] = o_color
                elif mapping[i_color] != o_color:
                    # Inconsistent mapping
                    return {}
            return mapping
            
        return {}
    
    def _analyze_object_changes(self, inp_objects: List[Dict], 
                                out_objects: List[Dict]) -> Dict:
        """Analyze how objects change between input and output"""
        changes = {
            'count_change': len(out_objects) - len(inp_objects),
            'preserved': 0,
            'modified': 0,
            'removed': 0,
            'added': 0
        }
        
        # Simple heuristic: match objects by color and size
        matched = []
        for inp_obj in inp_objects:
            for out_obj in out_objects:
                if (out_obj not in matched and 
                    inp_obj['color'] == out_obj['color'] and
                    abs(inp_obj['size'] - out_obj['size']) < 3):
                    changes['preserved'] += 1
                    matched.append(out_obj)
                    break
                    
        changes['added'] = len(out_objects) - len(matched)
        changes['removed'] = len(inp_objects) - changes['preserved']
        
        return changes
    
    def _infer_pattern_type(self, inp: List[List[int]], out: List[List[int]], 
                           inp_analysis: Dict, out_analysis: Dict) -> str:
        """Infer the type of transformation pattern"""
        inp_np = np.array(inp)
        out_np = np.array(out)
        
        # Check for simple transformations
        if np.array_equal(inp_np, out_np):
            return 'identity'
            
        if inp_np.shape == out_np.shape:
            # Same size transformations
            if np.array_equal(np.rot90(inp_np), out_np):
                return 'rotate_90'
            if np.array_equal(np.rot90(inp_np, 2), out_np):
                return 'rotate_180'
            if np.array_equal(np.rot90(inp_np, 3), out_np):
                return 'rotate_270'
            if np.array_equal(np.flipud(inp_np), out_np):
                return 'flip_horizontal'
            if np.array_equal(np.fliplr(inp_np), out_np):
                return 'flip_vertical'
            if np.array_equal(inp_np.T, out_np):
                return 'transpose'
                
        # Check for scaling
        if out_np.shape[0] == inp_np.shape[0] * 2 and out_np.shape[1] == inp_np.shape[1] * 2:
            # Check if it's a 2x scaling
            scaled = np.repeat(np.repeat(inp_np, 2, axis=0), 2, axis=1)
            if np.array_equal(scaled, out_np):
                return 'scale_2x'
                
        # Check for cropping
        if out_np.shape[0] < inp_np.shape[0] or out_np.shape[1] < inp_np.shape[1]:
            return 'crop'
            
        # Check for padding
        if out_np.shape[0] > inp_np.shape[0] or out_np.shape[1] > inp_np.shape[1]:
            return 'pad'
            
        # Complex transformation
        return 'complex'
    
    def _find_consistent_transform(self, transformations: List[Dict]) -> Dict:
        """Find transformation that's consistent across all examples"""
        if not transformations:
            return {}
            
        # Check if all have same pattern type
        pattern_types = [t['pattern_type'] for t in transformations]
        most_common_pattern = Counter(pattern_types).most_common(1)[0][0]
        
        # Check shape changes
        shape_changes = [t['shape_change'] for t in transformations]
        consistent_shape_change = all(shape_changes) or not any(shape_changes)
        
        return {
            'pattern_type': most_common_pattern,
            'consistent_shape_change': consistent_shape_change,
            'shape_ratios': [t['shape_ratio'] for t in transformations],
            'transformations': transformations
        }
    
    def apply_learned_transformation(self, grid: List[List[int]], 
                                    transformation: Dict) -> List[List[int]]:
        """Apply learned transformation to new grid"""
        grid_np = np.array(grid)
        pattern_type = transformation.get('pattern_type', 'identity')
        
        # Apply simple transformations
        if pattern_type == 'identity':
            return grid
        elif pattern_type == 'rotate_90':
            return np.rot90(grid_np).tolist()
        elif pattern_type == 'rotate_180':
            return np.rot90(grid_np, 2).tolist()
        elif pattern_type == 'rotate_270':
            return np.rot90(grid_np, 3).tolist()
        elif pattern_type == 'flip_horizontal':
            return np.flipud(grid_np).tolist()
        elif pattern_type == 'flip_vertical':
            return np.fliplr(grid_np).tolist()
        elif pattern_type == 'transpose':
            return grid_np.T.tolist()
        elif pattern_type == 'scale_2x':
            return np.repeat(np.repeat(grid_np, 2, axis=0), 2, axis=1).tolist()
        else:
            # For complex transformations, try to apply shape ratio
            if transformation.get('consistent_shape_change'):
                ratios = transformation.get('shape_ratios', [])
                if ratios and len(ratios) > 0:
                    avg_ratio = (
                        np.mean([r[0] for r in ratios]),
                        np.mean([r[1] for r in ratios])
                    )
                    
                    # Simple scaling based on ratio
                    if avg_ratio[0] == avg_ratio[1] and avg_ratio[0] > 1:
                        scale = int(avg_ratio[0])
                        return np.repeat(np.repeat(grid_np, scale, axis=0), scale, axis=1).tolist()
                        
            # Default: return original
            return grid