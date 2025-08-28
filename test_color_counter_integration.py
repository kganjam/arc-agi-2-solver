"""
Integration example for the ColorCounter tool with ARC AGI puzzles.
"""

import json
from arc_generated_color_counter import ColorCounter, count_unique_colors, get_color_stats


def analyze_arc_puzzle(puzzle_data):
    """
    Analyze an ARC puzzle using the ColorCounter tool.
    
    Args:
        puzzle_data: Dictionary containing 'train' and 'test' examples
    """
    counter = ColorCounter()
    
    print("ARC Puzzle Color Analysis")
    print("="*60)
    
    # Analyze training examples
    if 'train' in puzzle_data:
        print("\nTraining Examples:")
        for i, example in enumerate(puzzle_data['train']):
            print(f"\n  Example {i+1}:")
            
            # Analyze input
            input_grid = example['input']
            input_stats = counter.get_color_statistics(input_grid)
            print(f"    Input: {input_stats['unique_count']} unique colors")
            print(f"           Colors: {input_stats['color_list']}")
            print(f"           Coverage: {input_stats['color_coverage']}%")
            
            # Analyze output
            output_grid = example['output']
            output_stats = counter.get_color_statistics(output_grid)
            print(f"    Output: {output_stats['unique_count']} unique colors")
            print(f"            Colors: {output_stats['color_list']}")
            print(f"            Coverage: {output_stats['color_coverage']}%")
            
            # Check for color mapping
            mapping = counter.find_color_mapping(input_grid, output_grid)
            if mapping:
                print(f"    Color mapping detected: {mapping}")
            
            # Compare distributions
            comparison = counter.compare_color_distributions(input_grid, output_grid)
            if comparison['colors_added']:
                print(f"    Colors added: {comparison['colors_added']}")
            if comparison['colors_removed']:
                print(f"    Colors removed: {comparison['colors_removed']}")
    
    # Analyze test example
    if 'test' in puzzle_data:
        print("\nTest Examples:")
        for i, example in enumerate(puzzle_data['test']):
            print(f"\n  Test {i+1}:")
            input_grid = example['input']
            input_stats = counter.get_color_statistics(input_grid)
            print(f"    Input: {input_stats['unique_count']} unique colors")
            print(f"           Colors: {input_stats['color_list']}")
            print(f"           Dominant: {input_stats['dominant_color']}")
            print(f"           Grid size: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")


def identify_color_pattern(train_examples):
    """
    Identify common color patterns across training examples.
    
    Args:
        train_examples: List of training examples with 'input' and 'output'
    
    Returns:
        Dictionary describing identified patterns
    """
    counter = ColorCounter()
    patterns = {
        'consistent_mapping': True,
        'color_reduction': False,
        'color_expansion': False,
        'background_preserved': True,
        'mappings': []
    }
    
    all_mappings = []
    for example in train_examples:
        input_stats = counter.get_color_statistics(example['input'])
        output_stats = counter.get_color_statistics(example['output'])
        
        # Check for color count changes
        if output_stats['unique_count'] < input_stats['unique_count']:
            patterns['color_reduction'] = True
        elif output_stats['unique_count'] > input_stats['unique_count']:
            patterns['color_expansion'] = True
        
        # Check for consistent color mapping
        mapping = counter.find_color_mapping(example['input'], example['output'])
        if mapping:
            all_mappings.append(mapping)
        else:
            patterns['consistent_mapping'] = False
        
        # Check if background (0) is preserved
        if 0 in input_stats['frequency'] and 0 in output_stats['frequency']:
            if input_stats['frequency'][0] != output_stats['frequency'][0]:
                patterns['background_preserved'] = False
    
    # Check if all mappings are consistent
    if all_mappings and patterns['consistent_mapping']:
        first_mapping = all_mappings[0]
        for mapping in all_mappings[1:]:
            if mapping != first_mapping:
                patterns['consistent_mapping'] = False
                break
        if patterns['consistent_mapping']:
            patterns['mappings'] = first_mapping
    
    return patterns


# Example usage with a sample ARC puzzle structure
if __name__ == "__main__":
    # Sample ARC puzzle data (simplified)
    sample_puzzle = {
        'train': [
            {
                'input': [
                    [0, 1, 0],
                    [2, 2, 2],
                    [0, 3, 0]
                ],
                'output': [
                    [0, 4, 0],
                    [5, 5, 5],
                    [0, 6, 0]
                ]
            },
            {
                'input': [
                    [1, 1, 0],
                    [0, 2, 2],
                    [3, 0, 3]
                ],
                'output': [
                    [4, 4, 0],
                    [0, 5, 5],
                    [6, 0, 6]
                ]
            }
        ],
        'test': [
            {
                'input': [
                    [0, 0, 1],
                    [2, 0, 0],
                    [3, 3, 3]
                ]
            }
        ]
    }
    
    # Analyze the puzzle
    analyze_arc_puzzle(sample_puzzle)
    
    # Identify patterns
    print("\n" + "="*60)
    print("Pattern Analysis:")
    patterns = identify_color_pattern(sample_puzzle['train'])
    for key, value in patterns.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Quick analysis using standalone functions:")
    test_grid = sample_puzzle['test'][0]['input']
    print(f"  Unique colors in test: {count_unique_colors(test_grid)}")
    print(f"  Full stats: {get_color_stats(test_grid)}")