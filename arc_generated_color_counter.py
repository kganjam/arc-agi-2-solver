"""
ARC AGI Color Counter Tool
A reusable component for analyzing color distributions in ARC puzzle grids.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import Counter
import numpy as np


class ColorCounter:
    """
    A tool for analyzing color distributions in ARC AGI puzzle grids.
    
    ARC AGI uses colors 0-9 to represent different grid cell values.
    This tool provides utilities for counting and analyzing these colors.
    """
    
    def __init__(self):
        """Initialize the ColorCounter with ARC AGI color palette."""
        self.valid_colors = set(range(10))  # ARC uses colors 0-9
        
    def count_unique_colors(self, grid: Union[List[List[int]], np.ndarray]) -> int:
        """
        Count the number of unique colors in a grid.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            
        Returns:
            Number of unique colors in the grid
            
        Raises:
            ValueError: If grid is None or contains invalid color values
            TypeError: If grid is not a list or numpy array
        """
        if grid is None:
            raise ValueError("Grid cannot be None")
            
        # Convert to numpy array for easier processing
        try:
            if isinstance(grid, list):
                if len(grid) == 0:
                    return 0
                grid_array = np.array(grid)
            elif isinstance(grid, np.ndarray):
                if grid.size == 0:
                    return 0
                grid_array = grid
            else:
                raise TypeError(f"Grid must be a list or numpy array, got {type(grid)}")
        except Exception as e:
            raise TypeError(f"Error processing grid: {e}")
            
        # Flatten the grid and get unique colors
        unique_colors = set(grid_array.flatten())
        
        # Validate colors are in valid range
        invalid_colors = unique_colors - self.valid_colors
        if invalid_colors:
            raise ValueError(f"Invalid colors found: {invalid_colors}. ARC colors must be 0-9")
            
        return len(unique_colors)
    
    def get_color_frequency(self, grid: Union[List[List[int]], np.ndarray]) -> Dict[int, int]:
        """
        Get the frequency distribution of colors in a grid.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            
        Returns:
            Dictionary mapping color values to their frequency counts
            
        Raises:
            ValueError: If grid is None or contains invalid color values
            TypeError: If grid is not a list or numpy array
        """
        if grid is None:
            raise ValueError("Grid cannot be None")
            
        # Convert to numpy array
        try:
            if isinstance(grid, list):
                if len(grid) == 0:
                    return {}
                grid_array = np.array(grid)
            elif isinstance(grid, np.ndarray):
                if grid.size == 0:
                    return {}
                grid_array = grid
            else:
                raise TypeError(f"Grid must be a list or numpy array, got {type(grid)}")
        except Exception as e:
            raise TypeError(f"Error processing grid: {e}")
            
        # Flatten and count frequencies
        flat_grid = grid_array.flatten()
        color_counts = Counter(flat_grid)
        
        # Validate colors
        invalid_colors = set(color_counts.keys()) - self.valid_colors
        if invalid_colors:
            raise ValueError(f"Invalid colors found: {invalid_colors}. ARC colors must be 0-9")
            
        return dict(color_counts)
    
    def get_dominant_color(self, grid: Union[List[List[int]], np.ndarray], 
                          exclude_background: bool = False) -> Optional[int]:
        """
        Get the most frequent color in the grid.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            exclude_background: If True, excludes color 0 (typically background)
            
        Returns:
            The most frequent color, or None if grid is empty
        """
        freq = self.get_color_frequency(grid)
        
        if not freq:
            return None
            
        if exclude_background and 0 in freq:
            del freq[0]
            
        if not freq:
            return None
            
        return max(freq, key=freq.get)
    
    def get_color_positions(self, grid: Union[List[List[int]], np.ndarray], 
                           color: int) -> List[Tuple[int, int]]:
        """
        Get all positions (row, col) where a specific color appears.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            color: The color value to search for (0-9)
            
        Returns:
            List of (row, col) tuples where the color appears
            
        Raises:
            ValueError: If color is not in valid range (0-9)
        """
        if color not in self.valid_colors:
            raise ValueError(f"Invalid color: {color}. Must be 0-9")
            
        if grid is None:
            return []
            
        # Convert to numpy array
        if isinstance(grid, list):
            if len(grid) == 0:
                return []
            grid_array = np.array(grid)
        elif isinstance(grid, np.ndarray):
            if grid.size == 0:
                return []
            grid_array = grid
        else:
            raise TypeError(f"Grid must be a list or numpy array")
            
        # Find positions
        positions = np.argwhere(grid_array == color)
        return [(int(row), int(col)) for row, col in positions]
    
    def get_color_regions(self, grid: Union[List[List[int]], np.ndarray]) -> Dict[int, int]:
        """
        Count the number of connected regions for each color.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            
        Returns:
            Dictionary mapping colors to number of connected regions
        """
        if grid is None or (isinstance(grid, list) and len(grid) == 0):
            return {}
            
        # Convert to numpy array
        if isinstance(grid, list):
            grid_array = np.array(grid)
        else:
            grid_array = grid
            
        if grid_array.size == 0:
            return {}
            
        region_counts = {}
        unique_colors = set(grid_array.flatten())
        
        for color in unique_colors:
            if color not in self.valid_colors:
                continue
            # Simple connected component counting (4-connectivity)
            mask = (grid_array == color).astype(int)
            regions = self._count_connected_components(mask)
            region_counts[color] = regions
            
        return region_counts
    
    def _count_connected_components(self, mask: np.ndarray) -> int:
        """Count connected components in a binary mask using flood fill."""
        if mask.size == 0:
            return 0
            
        visited = np.zeros_like(mask, dtype=bool)
        count = 0
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1 and not visited[i, j]:
                    self._flood_fill(mask, visited, i, j)
                    count += 1
                    
        return count
    
    def _flood_fill(self, mask: np.ndarray, visited: np.ndarray, i: int, j: int):
        """Flood fill algorithm for marking connected components."""
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return
        if visited[i, j] or mask[i, j] == 0:
            return
            
        visited[i, j] = True
        
        # Check 4-connected neighbors
        self._flood_fill(mask, visited, i+1, j)
        self._flood_fill(mask, visited, i-1, j)
        self._flood_fill(mask, visited, i, j+1)
        self._flood_fill(mask, visited, i, j-1)
    
    def analyze_grid(self, grid: Union[List[List[int]], np.ndarray]) -> Dict[str, any]:
        """
        Perform comprehensive color analysis on a grid.
        
        Args:
            grid: 2D list or numpy array representing an ARC puzzle grid
            
        Returns:
            Dictionary containing:
                - unique_colors: Number of unique colors
                - color_frequency: Frequency distribution
                - dominant_color: Most frequent color
                - dominant_non_background: Most frequent non-zero color
                - color_regions: Connected regions per color
                - grid_size: (rows, cols) tuple
        """
        if grid is None:
            return {
                'unique_colors': 0,
                'color_frequency': {},
                'dominant_color': None,
                'dominant_non_background': None,
                'color_regions': {},
                'grid_size': (0, 0)
            }
            
        # Convert to numpy array for consistent processing
        if isinstance(grid, list):
            if len(grid) == 0:
                grid_array = np.array([])
            else:
                grid_array = np.array(grid)
        else:
            grid_array = grid
            
        if grid_array.size == 0:
            return {
                'unique_colors': 0,
                'color_frequency': {},
                'dominant_color': None,
                'dominant_non_background': None,
                'color_regions': {},
                'grid_size': (0, 0)
            }
            
        return {
            'unique_colors': self.count_unique_colors(grid_array),
            'color_frequency': self.get_color_frequency(grid_array),
            'dominant_color': self.get_dominant_color(grid_array, exclude_background=False),
            'dominant_non_background': self.get_dominant_color(grid_array, exclude_background=True),
            'color_regions': self.get_color_regions(grid_array),
            'grid_size': grid_array.shape
        }


# Convenience functions for direct usage
def count_unique_colors(grid: Union[List[List[int]], np.ndarray]) -> int:
    """
    Count unique colors in a grid (convenience function).
    
    Args:
        grid: 2D list or numpy array representing an ARC puzzle grid
        
    Returns:
        Number of unique colors
    """
    counter = ColorCounter()
    return counter.count_unique_colors(grid)


def get_color_frequency(grid: Union[List[List[int]], np.ndarray]) -> Dict[int, int]:
    """
    Get color frequency distribution (convenience function).
    
    Args:
        grid: 2D list or numpy array representing an ARC puzzle grid
        
    Returns:
        Dictionary mapping colors to frequencies
    """
    counter = ColorCounter()
    return counter.get_color_frequency(grid)


def analyze_grid(grid: Union[List[List[int]], np.ndarray]) -> Dict[str, any]:
    """
    Perform comprehensive grid analysis (convenience function).
    
    Args:
        grid: 2D list or numpy array representing an ARC puzzle grid
        
    Returns:
        Dictionary with analysis results
    """
    counter = ColorCounter()
    return counter.analyze_grid(grid)


# Test functions
def test_color_counter():
    """Test suite for ColorCounter functionality."""
    counter = ColorCounter()
    
    print("Running ColorCounter tests...")
    
    # Test 1: Empty grid
    empty_grid = []
    assert counter.count_unique_colors(empty_grid) == 0
    assert counter.get_color_frequency(empty_grid) == {}
    print("✓ Test 1: Empty grid handling")
    
    # Test 2: Single color grid
    single_color = [[1, 1], [1, 1]]
    assert counter.count_unique_colors(single_color) == 1
    assert counter.get_color_frequency(single_color) == {1: 4}
    print("✓ Test 2: Single color grid")
    
    # Test 3: Multiple colors
    multi_color = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert counter.count_unique_colors(multi_color) == 9
    freq = counter.get_color_frequency(multi_color)
    assert len(freq) == 9
    assert all(count == 1 for count in freq.values())
    print("✓ Test 3: Multiple colors")
    
    # Test 4: Dominant color
    dominant_grid = [[0, 0, 1], [0, 2, 0], [0, 0, 3]]
    assert counter.get_dominant_color(dominant_grid) == 0
    assert counter.get_dominant_color(dominant_grid, exclude_background=True) in [1, 2, 3]
    print("✓ Test 4: Dominant color detection")
    
    # Test 5: Color positions
    position_grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    positions = counter.get_color_positions(position_grid, 1)
    assert len(positions) == 4
    assert (0, 1) in positions
    assert (1, 0) in positions
    assert (1, 2) in positions
    assert (2, 1) in positions
    print("✓ Test 5: Color position finding")
    
    # Test 6: Connected regions
    region_grid = [[1, 0, 1], [1, 0, 1], [0, 0, 0]]
    regions = counter.get_color_regions(region_grid)
    assert regions[1] == 2  # Two separate regions of 1s
    assert regions[0] == 1  # One connected region of 0s
    print("✓ Test 6: Connected region counting")
    
    # Test 7: Comprehensive analysis
    test_grid = [[0, 1, 2], [0, 1, 3], [0, 0, 3]]
    analysis = counter.analyze_grid(test_grid)
    assert analysis['unique_colors'] == 4
    assert analysis['dominant_color'] == 0
    assert analysis['grid_size'] == (3, 3)
    print("✓ Test 7: Comprehensive analysis")
    
    # Test 8: NumPy array input
    np_grid = np.array([[1, 2], [3, 4]])
    assert counter.count_unique_colors(np_grid) == 4
    print("✓ Test 8: NumPy array input")
    
    # Test 9: Error handling for invalid colors
    try:
        invalid_grid = [[0, 10, 2]]  # 10 is invalid
        counter.count_unique_colors(invalid_grid)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid colors" in str(e)
        print("✓ Test 9: Invalid color detection")
    
    # Test 10: Convenience functions
    test_grid = [[1, 2, 3], [4, 5, 6]]
    assert count_unique_colors(test_grid) == 6
    assert len(get_color_frequency(test_grid)) == 6
    analysis = analyze_grid(test_grid)
    assert analysis['unique_colors'] == 6
    print("✓ Test 10: Convenience functions")
    
    print("\nAll tests passed! ✓")
    return True


# Run tests if executed directly
if __name__ == "__main__":
    test_color_counter()
    
    # Example usage demonstration
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50)
    
    # Create a sample ARC-style grid
    sample_grid = [
        [0, 0, 1, 1, 0],
        [0, 2, 2, 2, 0],
        [3, 3, 3, 3, 3],
        [0, 4, 0, 4, 0],
        [5, 5, 5, 5, 5]
    ]
    
    counter = ColorCounter()
    
    print("\nSample Grid:")
    for row in sample_grid:
        print(row)
    
    print(f"\nUnique colors: {counter.count_unique_colors(sample_grid)}")
    print(f"Color frequency: {counter.get_color_frequency(sample_grid)}")
    print(f"Dominant color: {counter.get_dominant_color(sample_grid)}")
    print(f"Dominant non-background: {counter.get_dominant_color(sample_grid, exclude_background=True)}")
    
    print("\nPositions of color 3:")
    positions = counter.get_color_positions(sample_grid, 3)
    for pos in positions:
        print(f"  {pos}")
    
    print("\nFull analysis:")
    analysis = counter.analyze_grid(sample_grid)
    for key, value in analysis.items():
        print(f"  {key}: {value}")