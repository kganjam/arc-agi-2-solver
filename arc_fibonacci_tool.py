"""
ARC AGI Fibonacci Tool
A reusable component for computing Fibonacci numbers efficiently.
This tool can be used for pattern recognition and sequence analysis in ARC puzzles.
"""

from functools import lru_cache
from typing import Union


class FibonacciTool:
    """Tool for computing Fibonacci numbers with various methods."""
    
    @staticmethod
    @lru_cache(maxsize=None)
    def fibonacci(n: int) -> Union[int, None]:
        """
        Compute the nth Fibonacci number using memoization.
        
        The Fibonacci sequence is defined as:
        F(0) = 0
        F(1) = 1
        F(n) = F(n-1) + F(n-2) for n >= 2
        
        Args:
            n: The position in the Fibonacci sequence (0-indexed)
        
        Returns:
            The nth Fibonacci number, or None if n is negative
            
        Examples:
            >>> FibonacciTool.fibonacci(0)
            0
            >>> FibonacciTool.fibonacci(1)
            1
            >>> FibonacciTool.fibonacci(10)
            55
        """
        if n < 0:
            return None
        if n == 0:
            return 0
        if n == 1:
            return 1
        return FibonacciTool.fibonacci(n - 1) + FibonacciTool.fibonacci(n - 2)
    
    @staticmethod
    def fibonacci_dp(n: int) -> Union[int, None]:
        """
        Compute the nth Fibonacci number using dynamic programming (iterative).
        More memory efficient for large n compared to recursive approach.
        
        Args:
            n: The position in the Fibonacci sequence (0-indexed)
        
        Returns:
            The nth Fibonacci number, or None if n is negative
        """
        if n < 0:
            return None
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        prev2, prev1 = 0, 1
        for _ in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    @staticmethod
    def fibonacci_sequence(count: int) -> list:
        """
        Generate a list of the first 'count' Fibonacci numbers.
        
        Args:
            count: Number of Fibonacci numbers to generate
        
        Returns:
            List of the first 'count' Fibonacci numbers
        """
        if count <= 0:
            return []
        
        sequence = []
        for i in range(count):
            sequence.append(FibonacciTool.fibonacci(i))
        
        return sequence
    
    @staticmethod
    def is_fibonacci(num: int) -> bool:
        """
        Check if a number is in the Fibonacci sequence.
        
        Args:
            num: The number to check
        
        Returns:
            True if the number is a Fibonacci number, False otherwise
        """
        if num < 0:
            return False
        
        a, b = 0, 1
        while a < num:
            a, b = b, a + b
        
        return a == num


def test_fibonacci_tool():
    """Test function to verify the Fibonacci tool works correctly."""
    tool = FibonacciTool()
    
    # Test edge cases
    assert tool.fibonacci(-1) is None, "Negative input should return None"
    assert tool.fibonacci(0) == 0, "F(0) should be 0"
    assert tool.fibonacci(1) == 1, "F(1) should be 1"
    
    # Test known values
    expected_values = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for i, expected in enumerate(expected_values):
        assert tool.fibonacci(i) == expected, f"F({i}) should be {expected}"
    
    # Test dynamic programming version
    for i in range(20):
        assert tool.fibonacci(i) == tool.fibonacci_dp(i), \
            f"Memoized and DP versions should match for n={i}"
    
    # Test sequence generation
    sequence = tool.fibonacci_sequence(10)
    assert sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], \
        "Sequence generation failed"
    
    # Test is_fibonacci
    fibonacci_numbers = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    non_fibonacci = [4, 6, 7, 9, 10, 11, 12, 14, 15]
    
    for num in fibonacci_numbers:
        assert tool.is_fibonacci(num), f"{num} should be identified as Fibonacci"
    
    for num in non_fibonacci:
        assert not tool.is_fibonacci(num), f"{num} should not be Fibonacci"
    
    print("✓ All tests passed!")
    print(f"✓ F(20) = {tool.fibonacci(20)}")
    print(f"✓ F(30) = {tool.fibonacci(30)}")
    print(f"✓ First 15 Fibonacci numbers: {tool.fibonacci_sequence(15)}")
    
    return True


# Integration with ARC AGI system
class ARCFibonacciAnalyzer:
    """
    Analyzer for detecting Fibonacci patterns in ARC puzzles.
    Can be used to identify sequences, spiral patterns, or growth patterns.
    """
    
    def __init__(self):
        self.tool = FibonacciTool()
    
    def analyze_grid_for_fibonacci(self, grid: list) -> dict:
        """
        Analyze a grid for Fibonacci-related patterns.
        
        Args:
            grid: 2D list representing an ARC puzzle grid
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'contains_fibonacci_dimensions': False,
            'fibonacci_counts': [],
            'fibonacci_positions': []
        }
        
        if not grid:
            return results
        
        # Check if dimensions are Fibonacci numbers
        height = len(grid)
        width = len(grid[0]) if grid[0] else 0
        
        results['contains_fibonacci_dimensions'] = (
            self.tool.is_fibonacci(height) or 
            self.tool.is_fibonacci(width)
        )
        
        # Count occurrences of each value
        value_counts = {}
        for row in grid:
            for val in row:
                value_counts[val] = value_counts.get(val, 0) + 1
        
        # Check if any counts are Fibonacci numbers
        for val, count in value_counts.items():
            if self.tool.is_fibonacci(count):
                results['fibonacci_counts'].append((val, count))
        
        return results


if __name__ == "__main__":
    # Run tests when executed directly
    test_fibonacci_tool()
    
    # Demo integration with ARC system
    analyzer = ARCFibonacciAnalyzer()
    sample_grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 2, 2]
    ]
    
    analysis = analyzer.analyze_grid_for_fibonacci(sample_grid)
    print(f"\nGrid analysis: {analysis}")