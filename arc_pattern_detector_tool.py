"""
ARC Pattern Detector Tool
A reusable tool for detecting repeating patterns and symmetry in ARC AGI grids.
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict


def find_repeating_patterns(grid: List[List[int]]) -> Dict[str, List[Dict]]:
    """
    Find repeating patterns in a grid for various block sizes.
    
    Args:
        grid: 2D list of integers representing an ARC puzzle grid
        
    Returns:
        Dictionary with pattern sizes as keys and list of pattern info as values.
        Each pattern info contains:
        - 'pattern': The actual pattern as a list of lists
        - 'locations': List of (row, col) tuples where pattern starts
        - 'count': Number of times pattern appears
    
    Example:
        >>> grid = [[1, 2, 1, 2],
        ...         [3, 4, 3, 4],
        ...         [1, 2, 1, 2],
        ...         [3, 4, 3, 4]]
        >>> patterns = find_repeating_patterns(grid)
        >>> len(patterns['2x2']) > 0
        True
    """
    if not grid or not grid[0]:
        return {}
    
    rows, cols = len(grid), len(grid[0])
    results = defaultdict(list)
    
    # Check for 2x2 and 3x3 patterns
    for pattern_size in [2, 3]:
        pattern_key = f"{pattern_size}x{pattern_size}"
        pattern_map = defaultdict(list)
        
        # Extract all possible blocks of the given size
        for i in range(rows - pattern_size + 1):
            for j in range(cols - pattern_size + 1):
                # Extract the block
                block = []
                for di in range(pattern_size):
                    row = []
                    for dj in range(pattern_size):
                        row.append(grid[i + di][j + dj])
                    block.append(row)
                
                # Convert to tuple for hashing
                block_tuple = tuple(tuple(row) for row in block)
                pattern_map[block_tuple].append((i, j))
        
        # Store patterns that appear more than once
        for pattern_tuple, locations in pattern_map.items():
            if len(locations) > 1:
                pattern_list = [list(row) for row in pattern_tuple]
                results[pattern_key].append({
                    'pattern': pattern_list,
                    'locations': locations,
                    'count': len(locations)
                })
    
    return dict(results)


def has_symmetry(grid: List[List[int]]) -> Dict[str, bool]:
    """
    Check for various types of symmetry in a grid.
    
    Args:
        grid: 2D list of integers representing an ARC puzzle grid
        
    Returns:
        Dictionary with symmetry types as keys and boolean values indicating
        whether that symmetry exists in the grid.
        Keys: 'horizontal', 'vertical', 'diagonal_main', 'diagonal_anti'
    
    Example:
        >>> grid = [[1, 2, 1],
        ...         [3, 4, 3],
        ...         [1, 2, 1]]
        >>> symmetry = has_symmetry(grid)
        >>> symmetry['vertical']
        True
    """
    if not grid or not grid[0]:
        return {
            'horizontal': False,
            'vertical': False,
            'diagonal_main': False,
            'diagonal_anti': False
        }
    
    rows, cols = len(grid), len(grid[0])
    
    # Check horizontal symmetry (reflection across horizontal axis)
    horizontal = True
    for i in range(rows // 2):
        for j in range(cols):
            if grid[i][j] != grid[rows - 1 - i][j]:
                horizontal = False
                break
        if not horizontal:
            break
    
    # Check vertical symmetry (reflection across vertical axis)
    vertical = True
    for i in range(rows):
        for j in range(cols // 2):
            if grid[i][j] != grid[i][cols - 1 - j]:
                vertical = False
                break
        if not vertical:
            break
    
    # Check diagonal symmetry (main diagonal: top-left to bottom-right)
    diagonal_main = rows == cols  # Only square grids can have diagonal symmetry
    if diagonal_main:
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != grid[j][i]:
                    diagonal_main = False
                    break
            if not diagonal_main:
                break
    
    # Check anti-diagonal symmetry (top-right to bottom-left)
    diagonal_anti = rows == cols
    if diagonal_anti:
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != grid[cols - 1 - j][rows - 1 - i]:
                    diagonal_anti = False
                    break
            if not diagonal_anti:
                break
    
    return {
        'horizontal': horizontal,
        'vertical': vertical,
        'diagonal_main': diagonal_main,
        'diagonal_anti': diagonal_anti
    }


def find_periodic_patterns(grid: List[List[int]]) -> Dict[str, List[Dict]]:
    """
    Find periodic/repeating patterns along rows and columns.
    
    Args:
        grid: 2D list of integers representing an ARC puzzle grid
        
    Returns:
        Dictionary with 'row_periods' and 'col_periods' containing information
        about periodic patterns found.
    
    Example:
        >>> grid = [[1, 2, 1, 2],
        ...         [3, 4, 3, 4],
        ...         [5, 6, 5, 6],
        ...         [7, 8, 7, 8]]
        >>> patterns = find_periodic_patterns(grid)
        >>> len(patterns['row_periods']) > 0
        True
    """
    if not grid or not grid[0]:
        return {'row_periods': [], 'col_periods': []}
    
    rows, cols = len(grid), len(grid[0])
    results = {'row_periods': [], 'col_periods': []}
    
    # Check for row periodicity
    for i in range(rows):
        row = grid[i]
        for period in range(1, cols // 2 + 1):
            is_periodic = True
            for j in range(period, cols):
                if row[j] != row[j % period]:
                    is_periodic = False
                    break
            if is_periodic:
                results['row_periods'].append({
                    'row_index': i,
                    'period': period,
                    'pattern': row[:period]
                })
                break  # Found the smallest period
    
    # Check for column periodicity
    for j in range(cols):
        col = [grid[i][j] for i in range(rows)]
        for period in range(1, rows // 2 + 1):
            is_periodic = True
            for i in range(period, rows):
                if col[i] != col[i % period]:
                    is_periodic = False
                    break
            if is_periodic:
                results['col_periods'].append({
                    'col_index': j,
                    'period': period,
                    'pattern': col[:period]
                })
                break  # Found the smallest period
    
    return results


def detect_connected_components(grid: List[List[int]], background_color: int = 0) -> List[Dict]:
    """
    Detect connected components (objects) in the grid.
    
    Args:
        grid: 2D list of integers representing an ARC puzzle grid
        background_color: The color value considered as background (default: 0)
        
    Returns:
        List of dictionaries, each containing:
        - 'color': The color of the component
        - 'cells': List of (row, col) tuples belonging to the component
        - 'bounding_box': (min_row, min_col, max_row, max_col)
    
    Example:
        >>> grid = [[0, 1, 1, 0],
        ...         [0, 1, 1, 0],
        ...         [0, 0, 0, 0],
        ...         [2, 2, 0, 0]]
        >>> components = detect_connected_components(grid)
        >>> len(components)
        2
    """
    if not grid or not grid[0]:
        return []
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    
    def dfs(i: int, j: int, color: int) -> List[Tuple[int, int]]:
        """Depth-first search to find all connected cells of the same color."""
        if (i < 0 or i >= rows or j < 0 or j >= cols or 
            visited[i][j] or grid[i][j] != color):
            return []
        
        visited[i][j] = True
        cells = [(i, j)]
        
        # Check 4-connected neighbors
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            cells.extend(dfs(i + di, j + dj, color))
        
        return cells
    
    # Find all connected components
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] != background_color:
                color = grid[i][j]
                cells = dfs(i, j, color)
                
                if cells:
                    # Calculate bounding box
                    min_row = min(r for r, c in cells)
                    max_row = max(r for r, c in cells)
                    min_col = min(c for r, c in cells)
                    max_col = max(c for r, c in cells)
                    
                    components.append({
                        'color': color,
                        'cells': cells,
                        'bounding_box': (min_row, min_col, max_row, max_col)
                    })
    
    return components


# Test cases
def run_tests():
    """Run test cases to verify the functionality of pattern detection tools."""
    
    print("Running Pattern Detector Tool Tests...")
    
    # Test 1: Repeating 2x2 patterns
    print("\nTest 1: Repeating 2x2 patterns")
    grid1 = [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 5, 6],
        [3, 4, 7, 8]
    ]
    patterns1 = find_repeating_patterns(grid1)
    assert '2x2' in patterns1
    assert len(patterns1['2x2']) > 0
    print(f"✓ Found {len(patterns1['2x2'])} repeating 2x2 patterns")
    
    # Test 2: Vertical symmetry
    print("\nTest 2: Vertical symmetry")
    grid2 = [
        [1, 2, 3, 2, 1],
        [4, 5, 6, 5, 4],
        [7, 8, 9, 8, 7]
    ]
    symmetry2 = has_symmetry(grid2)
    assert symmetry2['vertical'] == True
    assert symmetry2['horizontal'] == False
    print("✓ Correctly detected vertical symmetry")
    
    # Test 3: Horizontal symmetry
    print("\nTest 3: Horizontal symmetry")
    grid3 = [
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6],
        [1, 2, 3]
    ]
    symmetry3 = has_symmetry(grid3)
    assert symmetry3['horizontal'] == True
    print("✓ Correctly detected horizontal symmetry")
    
    # Test 4: Diagonal symmetry
    print("\nTest 4: Diagonal symmetry")
    grid4 = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ]
    symmetry4 = has_symmetry(grid4)
    assert symmetry4['diagonal_main'] == True
    print("✓ Correctly detected diagonal symmetry")
    
    # Test 5: Periodic patterns
    print("\nTest 5: Periodic patterns")
    grid5 = [
        [1, 2, 1, 2, 1, 2],
        [3, 4, 5, 3, 4, 5],
        [6, 6, 6, 6, 6, 6]
    ]
    periodic5 = find_periodic_patterns(grid5)
    assert len(periodic5['row_periods']) > 0
    print(f"✓ Found {len(periodic5['row_periods'])} rows with periodic patterns")
    
    # Test 6: Connected components
    print("\nTest 6: Connected components")
    grid6 = [
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
        [3, 3, 0, 0, 2],
        [3, 3, 0, 0, 0]
    ]
    components6 = detect_connected_components(grid6)
    assert len(components6) == 3
    print(f"✓ Found {len(components6)} connected components")
    
    # Test 7: 3x3 repeating patterns
    print("\nTest 7: Repeating 3x3 patterns")
    grid7 = [
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9],
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9]
    ]
    patterns7 = find_repeating_patterns(grid7)
    assert '3x3' in patterns7
    assert len(patterns7['3x3']) > 0
    print(f"✓ Found {len(patterns7['3x3'])} repeating 3x3 patterns")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_tests()