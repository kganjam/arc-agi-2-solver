"""
ARC AGI Color Counter Tool
A utility for analyzing color distribution in ARC puzzle grids.
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import Counter


class ColorCounter:
    """Analyzes color distribution in ARC AGI puzzle grids."""
    
    def __init__(self):
        """Initialize the ColorCounter."""
        self.valid_colors = set(range(10))  # ARC uses colors 0-9
    
    def count_unique_colors(self, grid: List[List[int]]) -> int:
        """
        Count the number of unique colors in a grid.
        
        Args:
            grid: 2D list representing the puzzle grid
            
        Returns:
            Number of unique colors in the grid
            
        Raises:
            ValueError: If grid contains invalid color values
        """
        if not grid:
            return 0
        
        unique_colors = set()
        for row in grid:
            if not row:
                continue
            for cell in row:
                if not isinstance(cell, (int, np.integer)):
                    raise ValueError(f"Invalid cell value: {cell}. Expected integer.")
                if cell not in self.valid_colors:
                    raise ValueError(f"Invalid color value: {cell}. Must be 0-9.")
                unique_colors.add(cell)
        
        return len(unique_colors)
    
    def get_color_frequency(self, grid: List[List[int]]) -> Dict[int, int]:
        """
        Get the frequency distribution of colors in a grid.
        
        Args:
            grid: 2D list representing the puzzle grid
            
        Returns:
            Dictionary mapping color values to their frequencies
            
        Raises:
            ValueError: If grid contains invalid color values
        """
        if not grid:
            return {}
        
        color_counter = Counter()
        for row in grid:
            if not row:
                continue
            for cell in row:
                if not isinstance(cell, (int, np.integer)):
                    raise ValueError(f"Invalid cell value: {cell}. Expected integer.")
                if cell not in self.valid_colors:
                    raise ValueError(f"Invalid color value: {cell}. Must be 0-9.")
                color_counter[cell] += 1
        
        return dict(color_counter)
    
    def get_dominant_color(self, grid: List[List[int]], 
                          exclude_background: bool = True) -> Optional[int]:
        """
        Get the most frequent color in the grid.
        
        Args:
            grid: 2D list representing the puzzle grid
            exclude_background: If True, excludes color 0 (typically background)
            
        Returns:
            The most frequent color value, or None if grid is empty
        """
        if not grid:
            return None
        
        freq = self.get_color_frequency(grid)
        if not freq:
            return None
        
        if exclude_background and 0 in freq and len(freq) > 1:
            del freq[0]
        
        if not freq:
            return 0  # Only background color present
        
        return max(freq, key=freq.get)
    
    def get_color_statistics(self, grid: List[List[int]]) -> Dict[str, any]:
        """
        Get comprehensive color statistics for a grid.
        
        Args:
            grid: 2D list representing the puzzle grid
            
        Returns:
            Dictionary containing:
                - unique_count: Number of unique colors
                - frequency: Color frequency distribution
                - dominant_color: Most frequent color
                - color_list: Sorted list of unique colors
                - grid_size: Total number of cells
                - color_coverage: Percentage of non-zero cells
        """
        if not grid:
            return {
                'unique_count': 0,
                'frequency': {},
                'dominant_color': None,
                'color_list': [],
                'grid_size': 0,
                'color_coverage': 0.0
            }
        
        unique_count = self.count_unique_colors(grid)
        frequency = self.get_color_frequency(grid)
        dominant = self.get_dominant_color(grid, exclude_background=True)
        color_list = sorted(frequency.keys())
        
        grid_size = sum(len(row) for row in grid)
        non_zero_cells = sum(1 for row in grid for cell in row if cell != 0)
        coverage = (non_zero_cells / grid_size * 100) if grid_size > 0 else 0.0
        
        return {
            'unique_count': unique_count,
            'frequency': frequency,
            'dominant_color': dominant,
            'color_list': color_list,
            'grid_size': grid_size,
            'color_coverage': round(coverage, 2)
        }
    
    def compare_color_distributions(self, grid1: List[List[int]], 
                                   grid2: List[List[int]]) -> Dict[str, any]:
        """
        Compare color distributions between two grids.
        
        Args:
            grid1: First grid to compare
            grid2: Second grid to compare
            
        Returns:
            Dictionary containing comparison results
        """
        stats1 = self.get_color_statistics(grid1)
        stats2 = self.get_color_statistics(grid2)
        
        colors1 = set(stats1['color_list'])
        colors2 = set(stats2['color_list'])
        
        return {
            'grid1_stats': stats1,
            'grid2_stats': stats2,
            'colors_added': list(colors2 - colors1),
            'colors_removed': list(colors1 - colors2),
            'colors_common': list(colors1 & colors2),
            'unique_count_diff': stats2['unique_count'] - stats1['unique_count'],
            'coverage_diff': stats2['color_coverage'] - stats1['color_coverage']
        }
    
    def find_color_mapping(self, input_grid: List[List[int]], 
                          output_grid: List[List[int]]) -> Optional[Dict[int, int]]:
        """
        Attempt to find a direct color mapping between input and output grids.
        
        Args:
            input_grid: Input puzzle grid
            output_grid: Output puzzle grid
            
        Returns:
            Dictionary mapping input colors to output colors if consistent mapping exists,
            None otherwise
        """
        if not input_grid or not output_grid:
            return None
        
        # Ensure grids have same dimensions
        if len(input_grid) != len(output_grid):
            return None
        if any(len(row1) != len(row2) for row1, row2 in zip(input_grid, output_grid)):
            return None
        
        mapping = {}
        for i, (in_row, out_row) in enumerate(zip(input_grid, output_grid)):
            for j, (in_cell, out_cell) in enumerate(zip(in_row, out_row)):
                if in_cell in mapping:
                    if mapping[in_cell] != out_cell:
                        # Inconsistent mapping
                        return None
                else:
                    mapping[in_cell] = out_cell
        
        return mapping


# Standalone utility functions for easy integration
def count_unique_colors(grid: List[List[int]]) -> int:
    """
    Count unique colors in a grid (standalone function).
    
    Args:
        grid: 2D list representing the puzzle grid
        
    Returns:
        Number of unique colors
    """
    counter = ColorCounter()
    return counter.count_unique_colors(grid)


def get_color_frequency(grid: List[List[int]]) -> Dict[int, int]:
    """
    Get color frequency distribution (standalone function).
    
    Args:
        grid: 2D list representing the puzzle grid
        
    Returns:
        Dictionary of color frequencies
    """
    counter = ColorCounter()
    return counter.get_color_frequency(grid)


def get_color_stats(grid: List[List[int]]) -> Dict[str, any]:
    """
    Get comprehensive color statistics (standalone function).
    
    Args:
        grid: 2D list representing the puzzle grid
        
    Returns:
        Dictionary of color statistics
    """
    counter = ColorCounter()
    return counter.get_color_statistics(grid)


# Test functions
def test_color_counter():
    """Test the ColorCounter functionality."""
    print("Testing ColorCounter...")
    
    # Test 1: Empty grid
    counter = ColorCounter()
    assert counter.count_unique_colors([]) == 0
    assert counter.get_color_frequency([]) == {}
    print("✓ Empty grid test passed")
    
    # Test 2: Simple grid
    grid1 = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0]
    ]
    assert counter.count_unique_colors(grid1) == 4
    freq1 = counter.get_color_frequency(grid1)
    assert freq1[0] == 2
    assert freq1[1] == 2
    assert freq1[2] == 3
    assert freq1[3] == 2
    print("✓ Simple grid test passed")
    
    # Test 3: Dominant color
    grid2 = [
        [5, 5, 5],
        [5, 1, 5],
        [5, 5, 5]
    ]
    assert counter.get_dominant_color(grid2, exclude_background=True) == 5
    print("✓ Dominant color test passed")
    
    # Test 4: Color statistics
    stats = counter.get_color_statistics(grid2)
    assert stats['unique_count'] == 2
    assert stats['dominant_color'] == 5
    assert stats['grid_size'] == 9
    print("✓ Color statistics test passed")
    
    # Test 5: Color mapping
    input_grid = [
        [1, 2],
        [2, 1]
    ]
    output_grid = [
        [3, 4],
        [4, 3]
    ]
    mapping = counter.find_color_mapping(input_grid, output_grid)
    assert mapping == {1: 3, 2: 4}
    print("✓ Color mapping test passed")
    
    # Test 6: Grid comparison
    comparison = counter.compare_color_distributions(grid1, grid2)
    assert 5 in comparison['colors_added']
    assert 0 in comparison['colors_removed']
    print("✓ Grid comparison test passed")
    
    # Test 7: Invalid color handling
    try:
        counter.count_unique_colors([[10]])  # Invalid color
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid color value" in str(e)
        print("✓ Invalid color handling test passed")
    
    # Test 8: Standalone functions
    assert count_unique_colors(grid1) == 4
    assert get_color_frequency(grid1) == freq1
    stats_standalone = get_color_stats(grid1)
    assert stats_standalone['unique_count'] == 4
    print("✓ Standalone functions test passed")
    
    print("\nAll tests passed! ✓")
    return True


if __name__ == "__main__":
    # Run tests when executed directly
    test_color_counter()
    
    # Example usage
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50)
    
    example_grid = [
        [0, 0, 1, 1, 0],
        [0, 2, 2, 2, 0],
        [3, 3, 3, 3, 3],
        [0, 4, 4, 4, 0],
        [0, 0, 5, 0, 0]
    ]
    
    counter = ColorCounter()
    
    print("\nExample Grid:")
    for row in example_grid:
        print(row)
    
    print(f"\nUnique colors: {counter.count_unique_colors(example_grid)}")
    print(f"Color frequency: {counter.get_color_frequency(example_grid)}")
    print(f"Dominant color (excluding background): {counter.get_dominant_color(example_grid)}")
    
    stats = counter.get_color_statistics(example_grid)
    print(f"\nFull statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")