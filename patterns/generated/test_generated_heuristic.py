
# Auto-generated heuristic
def color_frequency_transform(grid):
    """Transform based on color frequency"""
    from collections import Counter
    
    # Count color frequencies
    color_counts = Counter()
    for row in grid:
        color_counts.update(row)
    
    # Map most frequent to least frequent
    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
    color_map = {c: sorted_colors[-(i+1)] for i, c in enumerate(sorted_colors)}
    
    # Apply transformation
    return [[color_map.get(cell, cell) for cell in row] for row in grid]
