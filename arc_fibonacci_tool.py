"""
ARC AGI Fibonacci Tool
A reusable component for computing Fibonacci numbers efficiently.
Part of the ARC AGI system's tool library for mathematical operations.
"""

from functools import lru_cache
from typing import Union


@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """
    Compute the nth Fibonacci number using memoization for efficiency.
    
    The Fibonacci sequence is defined as:
    - F(0) = 0
    - F(1) = 1
    - F(n) = F(n-1) + F(n-2) for n > 1
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    
    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    
    Time Complexity: O(n) for first call, O(1) for cached calls
    Space Complexity: O(n) for cache storage
    """
    if not isinstance(n, int):
        raise TypeError(f"Input must be an integer, got {type(n).__name__}")
    
    if n < 0:
        raise ValueError(f"Fibonacci is not defined for negative numbers: n={n}")
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_iterative(n: int) -> int:
    """
    Compute the nth Fibonacci number using dynamic programming (iterative approach).
    
    This is an alternative implementation that uses O(1) space complexity.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    if not isinstance(n, int):
        raise TypeError(f"Input must be an integer, got {type(n).__name__}")
    
    if n < 0:
        raise ValueError(f"Fibonacci is not defined for negative numbers: n={n}")
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def fibonacci_sequence(count: int) -> list:
    """
    Generate a list of the first 'count' Fibonacci numbers.
    
    Args:
        count: Number of Fibonacci numbers to generate
    
    Returns:
        List containing the first 'count' Fibonacci numbers
    
    Raises:
        ValueError: If count is negative or zero
    """
    if count <= 0:
        raise ValueError(f"Count must be positive, got {count}")
    
    return [fibonacci(i) for i in range(count)]


def test_fibonacci():
    """
    Test function to verify the Fibonacci implementation works correctly.
    
    Returns:
        bool: True if all tests pass, raises AssertionError otherwise
    """
    print("Running Fibonacci tool tests...")
    
    # Test base cases
    assert fibonacci(0) == 0, "F(0) should be 0"
    assert fibonacci(1) == 1, "F(1) should be 1"
    
    # Test known values
    known_values = [
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (7, 13),
        (8, 21),
        (9, 34),
        (10, 55),
        (15, 610),
        (20, 6765),
    ]
    
    for n, expected in known_values:
        result = fibonacci(n)
        assert result == expected, f"F({n}) should be {expected}, got {result}"
        # Also test iterative version
        result_iter = fibonacci_iterative(n)
        assert result_iter == expected, f"Iterative F({n}) should be {expected}, got {result_iter}"
    
    # Test error handling
    try:
        fibonacci(-1)
        assert False, "Should raise ValueError for negative input"
    except ValueError as e:
        assert "negative" in str(e).lower()
    
    try:
        fibonacci(3.14)
        assert False, "Should raise TypeError for non-integer input"
    except TypeError as e:
        assert "integer" in str(e).lower()
    
    # Test sequence generation
    seq = fibonacci_sequence(10)
    expected_seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert seq == expected_seq, f"Sequence mismatch: expected {expected_seq}, got {seq}"
    
    # Test performance with larger numbers
    import time
    start = time.time()
    result = fibonacci(100)
    end = time.time()
    assert result == 354224848179261915075, "F(100) calculation incorrect"
    print(f"  F(100) = {result} (computed in {end-start:.6f} seconds)")
    
    # Test cache effectiveness
    start = time.time()
    result2 = fibonacci(100)  # Should be instant due to cache
    end = time.time()
    assert result2 == result, "Cached result mismatch"
    print(f"  F(100) cached = {result2} (retrieved in {end-start:.6f} seconds)")
    
    print("✓ All Fibonacci tool tests passed!")
    return True


def integrate_with_arc_system():
    """
    Integration point for the ARC AGI system.
    This function can be called to register the Fibonacci tool with the system.
    
    Returns:
        dict: Tool metadata for integration
    """
    return {
        "tool_name": "Fibonacci Calculator",
        "tool_id": "fibonacci_tool",
        "version": "1.0.0",
        "category": "mathematical",
        "functions": {
            "fibonacci": {
                "description": "Compute nth Fibonacci number (memoized)",
                "input": "integer n >= 0",
                "output": "integer Fibonacci value",
                "complexity": "O(n) first call, O(1) cached"
            },
            "fibonacci_iterative": {
                "description": "Compute nth Fibonacci number (iterative)",
                "input": "integer n >= 0",
                "output": "integer Fibonacci value",
                "complexity": "O(n) time, O(1) space"
            },
            "fibonacci_sequence": {
                "description": "Generate first n Fibonacci numbers",
                "input": "integer count > 0",
                "output": "list of Fibonacci numbers",
                "complexity": "O(n) time and space"
            }
        },
        "test_function": test_fibonacci,
        "use_cases": [
            "Pattern generation for ARC puzzles",
            "Sequence analysis in puzzle solving",
            "Mathematical transformations",
            "Numerical pattern recognition"
        ]
    }


if __name__ == "__main__":
    # Run tests when executed directly
    test_fibonacci()
    
    # Demonstrate usage
    print("\nDemonstration:")
    print(f"First 15 Fibonacci numbers: {fibonacci_sequence(15)}")
    print(f"F(30) = {fibonacci(30)}")
    print(f"F(50) = {fibonacci(50)}")
    
    # Show integration metadata
    print("\nTool Integration Metadata:")
    metadata = integrate_with_arc_system()
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")