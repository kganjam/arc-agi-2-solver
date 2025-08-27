"""
Pattern Recognition and Feature Detection for ARC Puzzles
Automatically detects common patterns and features in puzzle grids
"""

from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

# Minimal numpy replacement for basic operations
class SimpleArray:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
            self.shape = (len(data), len(data[0]) if data else 0)
        else:
            self.data = data
            self.shape = getattr(data, 'shape', (0, 0))
    
    def __getitem__(self, key):
        return self.data[key]
    
    def flatten(self):
        result = []
        for row in self.data:
            result.extend(row)
        return result
    
    def unique(self):
        flat = self.flatten()
        return list(set(flat))
    
    @property
    def size(self):
        return self.shape[0] * self.shape[1]

def array(data):
    return SimpleArray(data)

class FeatureDetector:
    """Detects various features and patterns in ARC puzzle grids"""
    
    def __init__(self):
        self.detected_features = {}
    
    def analyze_grid(self, grid: np.ndarray) -> Dict[str, Any]:
        """Comprehensive analysis of a grid"""
        features = {}
        
        # Basic properties
        features['dimensions'] = grid.shape
        features['colors'] = self._analyze_colors(grid)
        features['symmetries'] = self._detect_symmetries(grid)
        features['objects'] = self._detect_objects(grid)
        features['patterns'] = self._detect_patterns(grid)
        features['transformations'] = self._suggest_transformations(grid)
        
        return features
    
    def _analyze_colors(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution and properties"""
        unique_colors, counts = np.unique(grid, return_counts=True)
        total_cells = grid.size
        
        return {
            'unique_colors': unique_colors.tolist(),
            'color_counts': dict(zip(unique_colors.tolist(), counts.tolist())),
            'dominant_color': unique_colors[np.argmax(counts)],
            'color_diversity': len(unique_colors),
            'background_color': self._detect_background_color(grid),
            'color_distribution': (counts / total_cells).tolist()
        }
    
    def _detect_background_color(self, grid: np.ndarray) -> int:
        """Detect the most likely background color"""
        # Usually the most frequent color or color at corners
        corner_colors = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
        corner_mode = Counter(corner_colors).most_common(1)[0][0]
        
        # Check if corner mode is also the most frequent overall
        unique_colors, counts = np.unique(grid, return_counts=True)
        most_frequent = unique_colors[np.argmax(counts)]
        
        return corner_mode if corner_colors.count(corner_mode) >= 3 else most_frequent
    
    def _detect_symmetries(self, grid: np.ndarray) -> Dict[str, bool]:
        """Detect various types of symmetries"""
        return {
            'horizontal': np.array_equal(grid, np.fliplr(grid)),
            'vertical': np.array_equal(grid, np.flipud(grid)),
            'diagonal_main': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False,
            'diagonal_anti': np.array_equal(grid, np.rot90(grid, 2).T) if grid.shape[0] == grid.shape[1] else False,
            'rotational_90': np.array_equal(grid, np.rot90(grid, 1)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2))
        }
    
    def _detect_objects(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect connected objects in the grid"""
        background = self._detect_background_color(grid)
        
        # Create binary mask for non-background pixels
        object_mask = grid != background
        
        # Find connected components
        labeled_array, num_objects = ndimage.label(object_mask)
        
        objects = []
        for i in range(1, num_objects + 1):
            obj_mask = labeled_array == i
            obj_coords = np.where(obj_mask)
            
            # Object properties
            obj_info = {
                'id': i,
                'size': np.sum(obj_mask),
                'bounding_box': (
                    int(np.min(obj_coords[0])), int(np.min(obj_coords[1])),
                    int(np.max(obj_coords[0])), int(np.max(obj_coords[1]))
                ),
                'colors': np.unique(grid[obj_mask]).tolist(),
                'shape': self._analyze_object_shape(obj_mask),
                'position': (int(np.mean(obj_coords[0])), int(np.mean(obj_coords[1])))
            }
            objects.append(obj_info)
        
        return {
            'count': num_objects,
            'objects': objects,
            'total_object_area': np.sum(object_mask),
            'background_area': np.sum(~object_mask)
        }
    
    def _analyze_object_shape(self, obj_mask: np.ndarray) -> Dict[str, Any]:
        """Analyze the shape properties of an object"""
        coords = np.where(obj_mask)
        
        if len(coords[0]) == 0:
            return {'type': 'empty'}
        
        # Basic shape metrics
        height = np.max(coords[0]) - np.min(coords[0]) + 1
        width = np.max(coords[1]) - np.min(coords[1]) + 1
        area = len(coords[0])
        
        # Shape classification
        shape_type = 'irregular'
        if area == 1:
            shape_type = 'point'
        elif height == 1 or width == 1:
            shape_type = 'line'
        elif area == height * width:
            shape_type = 'rectangle'
        elif self._is_square_shape(obj_mask):
            shape_type = 'square'
        
        return {
            'type': shape_type,
            'area': area,
            'height': height,
            'width': width,
            'aspect_ratio': width / height if height > 0 else 0,
            'compactness': area / (height * width) if height * width > 0 else 0
        }
    
    def _is_square_shape(self, obj_mask: np.ndarray) -> bool:
        """Check if object forms a square shape"""
        coords = np.where(obj_mask)
        if len(coords[0]) == 0:
            return False
        
        min_row, max_row = np.min(coords[0]), np.max(coords[0])
        min_col, max_col = np.min(coords[1]), np.max(coords[1])
        
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        if height != width:
            return False
        
        # Check if all cells in the bounding square are filled
        expected_area = height * width
        actual_area = len(coords[0])
        
        return expected_area == actual_area
    
    def _detect_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect repeating patterns in the grid"""
        patterns = {}
        
        # Row patterns
        patterns['row_repetition'] = self._detect_row_patterns(grid)
        
        # Column patterns
        patterns['column_repetition'] = self._detect_column_patterns(grid)
        
        # Checkerboard pattern
        patterns['checkerboard'] = self._detect_checkerboard(grid)
        
        # Periodic patterns
        patterns['periodic'] = self._detect_periodic_patterns(grid)
        
        return patterns
    
    def _detect_row_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect patterns in rows"""
        rows = [tuple(row) for row in grid]
        row_counts = Counter(rows)
        
        return {
            'unique_rows': len(row_counts),
            'most_common_row': row_counts.most_common(1)[0] if row_counts else None,
            'all_rows_identical': len(row_counts) == 1,
            'alternating_rows': self._check_alternating_rows(rows)
        }
    
    def _detect_column_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect patterns in columns"""
        cols = [tuple(grid[:, i]) for i in range(grid.shape[1])]
        col_counts = Counter(cols)
        
        return {
            'unique_columns': len(col_counts),
            'most_common_column': col_counts.most_common(1)[0] if col_counts else None,
            'all_columns_identical': len(col_counts) == 1,
            'alternating_columns': self._check_alternating_columns(cols)
        }
    
    def _check_alternating_rows(self, rows: List[Tuple]) -> bool:
        """Check if rows alternate between two patterns"""
        if len(rows) < 2:
            return False
        
        pattern1, pattern2 = rows[0], rows[1]
        if pattern1 == pattern2:
            return False
        
        for i, row in enumerate(rows):
            expected = pattern1 if i % 2 == 0 else pattern2
            if row != expected:
                return False
        
        return True
    
    def _check_alternating_columns(self, cols: List[Tuple]) -> bool:
        """Check if columns alternate between two patterns"""
        if len(cols) < 2:
            return False
        
        pattern1, pattern2 = cols[0], cols[1]
        if pattern1 == pattern2:
            return False
        
        for i, col in enumerate(cols):
            expected = pattern1 if i % 2 == 0 else pattern2
            if col != expected:
                return False
        
        return True
    
    def _detect_checkerboard(self, grid: np.ndarray) -> bool:
        """Detect checkerboard pattern"""
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return False
        
        # Check if adjacent cells have different values
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                expected_value = grid[0, 0] if (i + j) % 2 == 0 else grid[0, 1]
                if grid[i, j] != expected_value:
                    return False
        
        return grid[0, 0] != grid[0, 1]  # Ensure there are actually two different values
    
    def _detect_periodic_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns in the grid"""
        patterns = {}
        
        # Check for horizontal periodicity
        for period in range(1, grid.shape[1] // 2 + 1):
            if self._check_horizontal_period(grid, period):
                patterns['horizontal_period'] = period
                break
        
        # Check for vertical periodicity
        for period in range(1, grid.shape[0] // 2 + 1):
            if self._check_vertical_period(grid, period):
                patterns['vertical_period'] = period
                break
        
        return patterns
    
    def _check_horizontal_period(self, grid: np.ndarray, period: int) -> bool:
        """Check if grid has horizontal periodicity with given period"""
        for i in range(grid.shape[0]):
            for j in range(period, grid.shape[1]):
                if grid[i, j] != grid[i, j % period]:
                    return False
        return True
    
    def _check_vertical_period(self, grid: np.ndarray, period: int) -> bool:
        """Check if grid has vertical periodicity with given period"""
        for i in range(period, grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != grid[i % period, j]:
                    return False
        return True
    
    def _suggest_transformations(self, grid: np.ndarray) -> List[str]:
        """Suggest possible transformations based on detected features"""
        suggestions = []
        
        # Analyze features to suggest transformations
        colors = self._analyze_colors(grid)
        symmetries = self._detect_symmetries(grid)
        
        if colors['color_diversity'] == 2:
            suggestions.append("binary_transformation")
        
        if symmetries['horizontal']:
            suggestions.append("horizontal_mirror")
        
        if symmetries['vertical']:
            suggestions.append("vertical_mirror")
        
        if symmetries['rotational_90']:
            suggestions.append("rotation_90")
        
        if symmetries['rotational_180']:
            suggestions.append("rotation_180")
        
        return suggestions
    
    def compare_grids(self, grid1: np.ndarray, grid2: np.ndarray) -> Dict[str, Any]:
        """Compare two grids and identify differences/transformations"""
        if grid1.shape != grid2.shape:
            return {'error': 'Grid shapes do not match'}
        
        differences = grid1 != grid2
        diff_count = np.sum(differences)
        
        analysis = {
            'identical': diff_count == 0,
            'differences_count': int(diff_count),
            'differences_percentage': float(diff_count / grid1.size * 100),
            'changed_positions': np.where(differences),
            'transformations': self._identify_transformation(grid1, grid2)
        }
        
        return analysis
    
    def _identify_transformation(self, grid1: np.ndarray, grid2: np.ndarray) -> List[str]:
        """Identify the type of transformation between two grids"""
        transformations = []
        
        # Check for simple transformations
        if np.array_equal(grid2, np.fliplr(grid1)):
            transformations.append("horizontal_flip")
        
        if np.array_equal(grid2, np.flipud(grid1)):
            transformations.append("vertical_flip")
        
        if np.array_equal(grid2, np.rot90(grid1, 1)):
            transformations.append("rotate_90_cw")
        
        if np.array_equal(grid2, np.rot90(grid1, -1)):
            transformations.append("rotate_90_ccw")
        
        if np.array_equal(grid2, np.rot90(grid1, 2)):
            transformations.append("rotate_180")
        
        # Check for color transformations
        if grid1.shape == grid2.shape and not np.array_equal(grid1, grid2):
            unique1 = set(grid1.flatten())
            unique2 = set(grid2.flatten())
            
            if len(unique1) == len(unique2):
                transformations.append("color_mapping")
        
        return transformations

class PatternLibrary:
    """Library of common ARC patterns and their solutions"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize library with common ARC patterns"""
        return {
            "identity": {
                "description": "Output is identical to input",
                "code": "return input_grid.copy()",
                "conditions": ["grid1 == grid2"]
            },
            "horizontal_flip": {
                "description": "Flip grid horizontally",
                "code": "return np.fliplr(input_grid)",
                "conditions": ["horizontal_symmetry"]
            },
            "vertical_flip": {
                "description": "Flip grid vertically", 
                "code": "return np.flipud(input_grid)",
                "conditions": ["vertical_symmetry"]
            },
            "color_replace": {
                "description": "Replace one color with another",
                "code": "output = input_grid.copy(); output[output == old_color] = new_color; return output",
                "conditions": ["color_mapping"]
            },
            "object_completion": {
                "description": "Complete partial objects",
                "code": "# Detect incomplete objects and complete them",
                "conditions": ["incomplete_objects"]
            }
        }
    
    def match_pattern(self, features: Dict[str, Any]) -> List[str]:
        """Match detected features to known patterns"""
        matches = []
        
        for pattern_name, pattern_info in self.patterns.items():
            if self._pattern_matches(features, pattern_info["conditions"]):
                matches.append(pattern_name)
        
        return matches
    
    def _pattern_matches(self, features: Dict[str, Any], conditions: List[str]) -> bool:
        """Check if features match pattern conditions"""
        # Simplified pattern matching logic
        for condition in conditions:
            if condition == "horizontal_symmetry" and features.get("symmetries", {}).get("horizontal"):
                return True
            elif condition == "vertical_symmetry" and features.get("symmetries", {}).get("vertical"):
                return True
            elif condition == "color_mapping" and len(features.get("colors", {}).get("unique_colors", [])) <= 5:
                return True
        
        return False
