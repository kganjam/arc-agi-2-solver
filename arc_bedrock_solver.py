#!/usr/bin/env python3
"""
ARC-AGI Solver with AWS Bedrock AI Integration
Uses AI assistance for pattern recognition and verification oracle
"""

import json
import boto3
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import asyncio
import os

class BedrockAISolver:
    """Integrates AWS Bedrock AI for puzzle analysis and solving"""
    
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """Initialize Bedrock client"""
        self.model_id = model_id
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            self.bedrock_available = True
        except Exception as e:
            print(f"Warning: Bedrock not available: {e}")
            self.bedrock_available = False
            self.bedrock_client = None
    
    def analyze_puzzle_with_ai(self, puzzle: Dict) -> Dict:
        """Use Bedrock AI to analyze puzzle and suggest solving approach"""
        
        if not self.bedrock_available:
            return self._fallback_analysis(puzzle)
        
        # Prepare puzzle description for AI
        prompt = self._create_analysis_prompt(puzzle)
        
        try:
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            ai_analysis = response_body.get('content', [{}])[0].get('text', '')
            
            return self._parse_ai_response(ai_analysis)
            
        except Exception as e:
            print(f"Bedrock API error: {e}")
            return self._fallback_analysis(puzzle)
    
    def _create_analysis_prompt(self, puzzle: Dict) -> str:
        """Create detailed prompt for AI analysis"""
        prompt = """Analyze this ARC-AGI puzzle and provide solving strategy.

PUZZLE DATA:
"""
        
        # Add training examples
        if 'train' in puzzle:
            for i, example in enumerate(puzzle['train']):
                prompt += f"\nTraining Example {i+1}:\n"
                prompt += f"Input Grid ({len(example['input'])}x{len(example['input'][0])}):\n"
                prompt += self._grid_to_string(example['input'])
                prompt += f"Output Grid ({len(example['output'])}x{len(example['output'][0])}):\n"
                prompt += self._grid_to_string(example['output'])
        
        # Add test input
        if 'test' in puzzle and puzzle['test']:
            prompt += f"\nTest Input ({len(puzzle['test'][0]['input'])}x{len(puzzle['test'][0]['input'][0])}):\n"
            prompt += self._grid_to_string(puzzle['test'][0]['input'])
        
        prompt += """
TASK: Analyze the transformation pattern from inputs to outputs. Consider:
1. Grid size changes (cropping, padding, scaling)
2. Color transformations (mapping, replacement, conditional changes)
3. Spatial transformations (rotation, reflection, translation)
4. Pattern-based rules (symmetry, repetition, completion)
5. Object manipulation (extraction, counting, movement)
6. Logical operations (AND, OR, XOR on patterns)

Provide your analysis in this JSON format:
{
    "pattern_type": "describe the main transformation",
    "grid_size_change": "none/crop/expand/scale",
    "color_rules": ["list of color transformation rules"],
    "spatial_transform": "none/rotate/flip/translate",
    "key_observation": "critical insight about the pattern",
    "solution_approach": "step-by-step approach to solve",
    "confidence": 0.0-1.0
}
"""
        return prompt
    
    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert grid to readable string format"""
        lines = []
        for row in grid:
            lines.append(' '.join(str(cell) for cell in row))
        return '\n'.join(lines) + '\n'
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        return {
            'pattern_type': 'unknown',
            'analysis': response,
            'confidence': 0.5
        }
    
    def _fallback_analysis(self, puzzle: Dict) -> Dict:
        """Fallback analysis when Bedrock is not available"""
        return {
            'pattern_type': 'fallback',
            'analysis': 'Bedrock not available, using basic pattern matching',
            'confidence': 0.3
        }

class VerificationOracle:
    """Verification oracle to check solution correctness"""
    
    def __init__(self):
        self.verification_history = []
    
    def verify_solution(self, puzzle: Dict, proposed_solution: List[List[int]]) -> Dict:
        """Verify if proposed solution matches expected output"""
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'puzzle_id': puzzle.get('id', 'unknown'),
            'verified': False,
            'accuracy': 0.0,
            'grid_size_correct': False,
            'cell_accuracy': 0.0,
            'feedback': []
        }
        
        # Check if we have expected output (for training puzzles)
        if 'test' in puzzle and puzzle['test'] and 'output' in puzzle['test'][0]:
            expected = puzzle['test'][0]['output']
            result['has_expected'] = True
            
            # Verify grid dimensions
            expected_shape = (len(expected), len(expected[0]) if expected else 0)
            solution_shape = (len(proposed_solution), len(proposed_solution[0]) if proposed_solution else 0)
            
            result['expected_shape'] = expected_shape
            result['solution_shape'] = solution_shape
            result['grid_size_correct'] = expected_shape == solution_shape
            
            if not result['grid_size_correct']:
                result['feedback'].append(
                    f"Grid size mismatch: expected {expected_shape}, got {solution_shape}"
                )
                result['accuracy'] = 0.0
            else:
                # Check cell-by-cell accuracy
                correct_cells = 0
                total_cells = expected_shape[0] * expected_shape[1]
                
                for i in range(expected_shape[0]):
                    for j in range(expected_shape[1]):
                        if expected[i][j] == proposed_solution[i][j]:
                            correct_cells += 1
                
                result['cell_accuracy'] = correct_cells / total_cells if total_cells > 0 else 0
                result['accuracy'] = result['cell_accuracy']
                result['verified'] = result['accuracy'] == 1.0
                
                if result['verified']:
                    result['feedback'].append("Perfect match! Solution is correct.")
                else:
                    result['feedback'].append(
                        f"Partial match: {result['accuracy']:.1%} cells correct"
                    )
        else:
            # No expected output available (test puzzle)
            result['has_expected'] = False
            result['feedback'].append("No expected output available for verification")
            
            # Basic validation checks
            if proposed_solution:
                result['solution_shape'] = (len(proposed_solution), len(proposed_solution[0]) if proposed_solution else 0)
                
                # Check if solution is reasonable
                if result['solution_shape'][0] > 0 and result['solution_shape'][1] > 0:
                    # Check if solution is not just a copy of input
                    if 'test' in puzzle and puzzle['test']:
                        test_input = puzzle['test'][0].get('input', [])
                        if not self._grids_equal(test_input, proposed_solution):
                            result['feedback'].append("Solution differs from input (good)")
                            result['confidence'] = 0.5
                        else:
                            result['feedback'].append("Warning: Solution identical to input")
                            result['confidence'] = 0.1
        
        # Store in history
        self.verification_history.append(result)
        
        return result
    
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are equal"""
        if len(grid1) != len(grid2):
            return False
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def verify_training_accuracy(self, puzzle: Dict, transformation_func) -> float:
        """Verify transformation accuracy on training examples"""
        
        if 'train' not in puzzle:
            return 0.0
        
        correct = 0
        total = 0
        
        for example in puzzle['train']:
            if 'input' in example and 'output' in example:
                predicted = transformation_func(example['input'])
                if predicted:
                    expected = example['output']
                    
                    # Check if shapes match
                    if (len(predicted) == len(expected) and 
                        len(predicted[0]) == len(expected[0])):
                        
                        # Check if grids match
                        if self._grids_equal(predicted, expected):
                            correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0

class EnhancedARCSolver:
    """Enhanced ARC solver with AI assistance and verification"""
    
    def __init__(self):
        self.bedrock_solver = BedrockAISolver()
        self.oracle = VerificationOracle()
        self.solving_history = []
    
    async def solve_puzzle(self, puzzle: Dict) -> Dict:
        """Solve puzzle with AI assistance and verification"""
        
        result = {
            'puzzle_id': puzzle.get('id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'solved': False,
            'solution': None,
            'confidence': 0.0,
            'ai_analysis': None,
            'verification': None,
            'method': 'ai_assisted',
            'explanation': []
        }
        
        # Step 1: Get AI analysis
        print(f"\n🤖 Analyzing puzzle {result['puzzle_id']} with AI...")
        ai_analysis = self.bedrock_solver.analyze_puzzle_with_ai(puzzle)
        result['ai_analysis'] = ai_analysis
        result['explanation'].append(f"AI identified pattern: {ai_analysis.get('pattern_type', 'unknown')}")
        
        # Step 2: Apply transformation based on AI suggestions
        solution = self._apply_ai_suggested_transform(puzzle, ai_analysis)
        
        if solution:
            result['solution'] = solution
            result['explanation'].append("Applied AI-suggested transformation")
            
            # Step 3: Verify solution
            verification = self.oracle.verify_solution(puzzle, solution)
            result['verification'] = verification
            
            if verification['verified']:
                result['solved'] = True
                result['confidence'] = 0.9
                result['explanation'].append("✅ Solution verified correct!")
            elif verification.get('accuracy', 0) > 0.8:
                result['solved'] = True
                result['confidence'] = verification['accuracy']
                result['explanation'].append(f"⚠️ Partial solution: {verification['accuracy']:.1%} accurate")
            else:
                result['explanation'].append(f"❌ Solution incorrect: {verification.get('feedback', ['Unknown error'])}")
        
        # Store in history
        self.solving_history.append(result)
        
        return result
    
    def _apply_ai_suggested_transform(self, puzzle: Dict, ai_analysis: Dict) -> Optional[List[List[int]]]:
        """Apply transformation based on AI analysis"""
        
        if 'test' not in puzzle or not puzzle['test']:
            return None
        
        test_input = puzzle['test'][0].get('input', [])
        if not test_input:
            return None
        
        pattern_type = ai_analysis.get('pattern_type', '').lower()
        
        # Grid size changes
        grid_size_change = ai_analysis.get('grid_size_change', 'none').lower()
        
        # Apply transformations based on AI suggestions
        result = np.array(test_input)
        
        # Handle grid size changes first
        if grid_size_change == 'crop':
            # Implement cropping logic
            result = self._crop_grid(result, puzzle)
        elif grid_size_change == 'expand':
            # Implement expansion logic
            result = self._expand_grid(result, puzzle)
        
        # Apply spatial transformations
        spatial = ai_analysis.get('spatial_transform', 'none').lower()
        if 'rotate' in spatial:
            result = np.rot90(result)
        elif 'flip' in spatial:
            result = np.fliplr(result)
        
        # Apply color rules
        color_rules = ai_analysis.get('color_rules', [])
        for rule in color_rules:
            result = self._apply_color_rule(result, rule)
        
        return result.tolist()
    
    def _crop_grid(self, grid: np.ndarray, puzzle: Dict) -> np.ndarray:
        """Crop grid based on training examples"""
        
        # Analyze training examples for crop pattern
        if 'train' in puzzle and puzzle['train']:
            # Get typical output size from training
            output_sizes = []
            for example in puzzle['train']:
                if 'output' in example:
                    output_sizes.append(np.array(example['output']).shape)
            
            if output_sizes:
                # Use most common output size
                target_shape = max(set(output_sizes), key=output_sizes.count)
                
                # Crop to target shape
                if grid.shape[0] > target_shape[0]:
                    grid = grid[:target_shape[0], :]
                if grid.shape[1] > target_shape[1]:
                    grid = grid[:, :target_shape[1]]
        
        return grid
    
    def _expand_grid(self, grid: np.ndarray, puzzle: Dict) -> np.ndarray:
        """Expand grid based on training examples"""
        
        # Analyze training examples for expansion pattern
        if 'train' in puzzle and puzzle['train']:
            # Get typical output size from training
            output_sizes = []
            for example in puzzle['train']:
                if 'output' in example:
                    output_sizes.append(np.array(example['output']).shape)
            
            if output_sizes:
                # Use most common output size
                target_shape = max(set(output_sizes), key=output_sizes.count)
                
                # Pad to target shape
                if grid.shape[0] < target_shape[0]:
                    pad_rows = target_shape[0] - grid.shape[0]
                    grid = np.pad(grid, ((0, pad_rows), (0, 0)), mode='constant')
                if grid.shape[1] < target_shape[1]:
                    pad_cols = target_shape[1] - grid.shape[1]
                    grid = np.pad(grid, ((0, 0), (0, pad_cols)), mode='constant')
        
        return grid
    
    def _apply_color_rule(self, grid: np.ndarray, rule: str) -> np.ndarray:
        """Apply color transformation rule"""
        
        # Parse simple color rules
        if 'swap' in rule.lower():
            # Swap colors
            if '0' in rule and '1' in rule:
                grid_copy = grid.copy()
                grid[grid_copy == 0] = 1
                grid[grid_copy == 1] = 0
        elif 'invert' in rule.lower():
            # Invert non-zero colors
            grid[grid > 0] = 10 - grid[grid > 0]
        
        return grid
    
    def get_statistics(self) -> Dict:
        """Get solver statistics"""
        
        stats = {
            'total_puzzles': len(self.solving_history),
            'solved': sum(1 for h in self.solving_history if h['solved']),
            'verified': sum(1 for h in self.solving_history if h.get('verification', {}).get('verified')),
            'average_confidence': 0.0,
            'with_ai_analysis': sum(1 for h in self.solving_history if h.get('ai_analysis'))
        }
        
        if self.solving_history:
            confidences = [h['confidence'] for h in self.solving_history if h['confidence'] > 0]
            if confidences:
                stats['average_confidence'] = np.mean(confidences)
        
        return stats

async def test_enhanced_solver():
    """Test the enhanced solver with AI and verification"""
    
    print("="*60)
    print("Testing Enhanced ARC Solver with AI and Verification")
    print("="*60)
    
    solver = EnhancedARCSolver()
    
    # Create test puzzle with known solution
    puzzle = {
        'id': 'test_verification',
        'train': [
            {
                'input': [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                'output': [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
            },
            {
                'input': [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
                'output': [[0, 0, 2], [0, 0, 0], [0, 0, 0]]
            }
        ],
        'test': [
            {
                'input': [[3, 0, 0], [0, 0, 0], [0, 0, 0]],
                'output': [[0, 0, 3], [0, 0, 0], [0, 0, 0]]  # Known solution for verification
            }
        ]
    }
    
    # Solve puzzle
    result = await solver.solve_puzzle(puzzle)
    
    print(f"\n📊 Results:")
    print(f"  Puzzle ID: {result['puzzle_id']}")
    print(f"  Solved: {result['solved']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    
    if result['ai_analysis']:
        print(f"\n🤖 AI Analysis:")
        print(f"  Pattern: {result['ai_analysis'].get('pattern_type', 'unknown')}")
        print(f"  Confidence: {result['ai_analysis'].get('confidence', 0):.1%}")
    
    if result['verification']:
        print(f"\n✅ Verification:")
        print(f"  Verified: {result['verification']['verified']}")
        print(f"  Accuracy: {result['verification']['accuracy']:.1%}")
        print(f"  Grid size correct: {result['verification']['grid_size_correct']}")
        for feedback in result['verification'].get('feedback', []):
            print(f"  - {feedback}")
    
    if result['solution']:
        print(f"\n📋 Solution:")
        for row in result['solution']:
            print(f"  {row}")
    
    print(f"\n📝 Explanation:")
    for step in result['explanation']:
        print(f"  - {step}")
    
    # Get statistics
    stats = solver.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total puzzles: {stats['total_puzzles']}")
    print(f"  Solved: {stats['solved']}")
    print(f"  Verified: {stats['verified']}")
    print(f"  Average confidence: {stats['average_confidence']:.1%}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_solver())