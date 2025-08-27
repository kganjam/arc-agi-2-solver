"""
Pattern Detector Tool for ARC AGI
Detects various patterns in grids
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

class PatternDetector:
    """Main pattern detection tool"""
    
    def __init__(self):
        self.min_object_size = 1
        
    def analyze(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze grid for patterns"""
        grid_np = np.array(grid)
        
        return {
            'symmetry': self.detect_symmetry(grid_np),
            'objects': self.find_objects(grid_np),
            'patterns': self.detect_patterns(grid_np),
            'boundaries': self.find_boundaries(grid_np),
            'colors': self.analyze_colors(grid_np),
            'shape': {
                'height': grid_np.shape[0],
                'width': grid_np.shape[1],
                'is_square': grid_np.shape[0] == grid_np.shape[1]
            }
        }
    
    def detect_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """Detect various types of symmetry"""
        return {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'diagonal': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False,
            'anti_diagonal': np.array_equal(grid, np.fliplr(grid).T) if grid.shape[0] == grid.shape[1] else False,
            'rotational_90': np.array_equal(grid, np.rot90(grid)) if grid.shape[0] == grid.shape[1] else False,
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2))
        }
    
    def find_objects(self, grid: np.ndarray) -> List[Dict]:
        """Find connected components (objects) in the grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj = self._flood_fill(grid, visited, i, j, grid[i, j])
                    if obj and len(obj['cells']) >= self.min_object_size:
                        objects.append(obj)
        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, 
                    start_i: int, start_j: int, color: int) -> Dict:
        """Extract connected component using flood fill"""
        stack = [(start_i, start_j)]
        cells = []
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= grid.shape[0] or 
                j < 0 or j >= grid.shape[1] or
                visited[i, j] or grid[i, j] != color):
                continue
            
            visited[i, j] = True
            cells.append((i, j))
            
            # Add 4-connected neighbors
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
                'height': max_i - min_i + 1,
                'center': ((min_i + max_i) / 2, (min_j + max_j) / 2)
            }
        
        return None
    
    def detect_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect repeating patterns"""
        patterns = {
            'has_repeating_2x2': self._check_repeating_block(grid, 2, 2),
            'has_repeating_3x3': self._check_repeating_block(grid, 3, 3),
            'has_checkerboard': self._check_checkerboard(grid),
            'has_stripes': self._check_stripes(grid),
            'has_border': self._check_border(grid)
        }
        
        return patterns
    
    def _check_repeating_block(self, grid: np.ndarray, block_h: int, block_w: int) -> bool:
        """Check for repeating block patterns"""
        h, w = grid.shape
        
        if h < block_h * 2 or w < block_w * 2:
            return False
        
        for i in range(h - block_h * 2 + 1):
            for j in range(w - block_w * 2 + 1):
                block1 = grid[i:i+block_h, j:j+block_w]
                block2 = grid[i+block_h:i+block_h*2, j:j+block_w]
                
                if np.array_equal(block1, block2):
                    return True
        
        return False
    
    def _check_checkerboard(self, grid: np.ndarray) -> bool:
        """Check for checkerboard pattern"""
        if len(np.unique(grid)) != 2:
            return False
        
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                if grid[i, j] == grid[i+1, j+1] and grid[i, j] != grid[i, j+1]:
                    continue
                else:
                    return False
        
        return True
    
    def _check_stripes(self, grid: np.ndarray) -> Dict[str, bool]:
        """Check for striped patterns"""
        h_stripes = True
        v_stripes = True
        
        # Check horizontal stripes
        for i in range(grid.shape[0]):
            if len(np.unique(grid[i, :])) > 1:
                h_stripes = False
                break
        
        # Check vertical stripes
        for j in range(grid.shape[1]):
            if len(np.unique(grid[:, j])) > 1:
                v_stripes = False
                break
        
        return {
            'horizontal': h_stripes,
            'vertical': v_stripes,
            'any': h_stripes or v_stripes
        }
    
    def _check_border(self, grid: np.ndarray) -> bool:
        """Check if grid has a uniform border"""
        if grid.size == 0:
            return False
        
        # Get border elements
        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]
        
        border_elements = np.concatenate([top, bottom, left[1:-1], right[1:-1]])
        
        # Check if all border elements are the same non-zero color
        unique_border = np.unique(border_elements)
        return len(unique_border) == 1 and unique_border[0] != 0
    
    def find_boundaries(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find cells on boundaries between different colors"""
        boundaries = []
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                is_boundary = False
                current = grid[i, j]
                
                # Check 4-connected neighbors
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni, nj] != current:
                            is_boundary = True
                            break
                
                if is_boundary:
                    boundaries.append((i, j))
        
        return boundaries
    
    def analyze_colors(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution"""
        flat = grid.flatten()
        color_counts = Counter(flat)
        
        return {
            'unique_colors': len(color_counts),
            'color_counts': dict(color_counts),
            'dominant_color': max(color_counts, key=color_counts.get),
            'background_color': 0 if 0 in color_counts else min(color_counts, key=color_counts.get),
            'color_ratios': {
                color: count / len(flat) 
                for color, count in color_counts.items()
            }
        }


def run_tool(grid: List[List[int]]) -> Dict[str, Any]:
    """Entry point for the tool"""
    detector = PatternDetector()
    return detector.analyze(grid)