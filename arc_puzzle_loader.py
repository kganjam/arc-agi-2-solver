#!/usr/bin/env python3
"""
ARC Puzzle Loader - Loads and manages real ARC puzzles
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import glob

class ARCPuzzleLoader:
    """Loads and manages ARC puzzles from files"""
    
    def __init__(self):
        self.puzzles = []
        self.puzzle_index = {}
        self.current_index = 0
        self.current_set = "evaluation"  # Default to evaluation/hard set
        self.load_puzzles()
    
    def load_puzzles(self, puzzle_set="evaluation"):
        """Load all available ARC puzzles from specified set"""
        self.puzzles = []
        self.puzzle_index = {}
        self.current_index = 0
        self.current_set = puzzle_set
        
        # Try different possible paths for ARC puzzles based on set type
        if puzzle_set == "training":
            search_paths = [
                Path("data/arc_agi_full/data/training"),
                Path("data/arc_agi/training"),
                Path("data/training"),
            ]
        else:  # evaluation
            search_paths = [
                Path("data/arc_agi_full/data/evaluation"),
                Path("data/arc_agi/evaluation"),
                Path("data/evaluation"),
            ]
        
        # Add fallback paths
        search_paths.extend([
            Path("data/arc_agi_full/data"),
            Path("data/arc_agi"),
            Path("data")
        ])
        
        loaded_files = set()
        
        for base_path in search_paths:
            if base_path.exists():
                # Load JSON files
                json_files = list(base_path.glob("*.json"))
                
                for json_file in json_files:
                    if json_file.name not in loaded_files:
                        try:
                            with open(json_file, 'r') as f:
                                puzzle_data = json.load(f)
                                
                            # Add puzzle ID from filename
                            puzzle_id = json_file.stem
                            puzzle_data['id'] = puzzle_id
                            
                            self.puzzles.append(puzzle_data)
                            self.puzzle_index[puzzle_id] = len(self.puzzles) - 1
                            loaded_files.add(json_file.name)
                            
                        except Exception as e:
                            print(f"Error loading {json_file}: {e}")
        
        # If no real puzzles found, create sample puzzles
        if not self.puzzles:
            self.puzzles = self.create_sample_puzzles()
            for i, puzzle in enumerate(self.puzzles):
                self.puzzle_index[puzzle['id']] = i
        
        print(f"Loaded {len(self.puzzles)} puzzles")
    
    def create_sample_puzzles(self) -> List[Dict]:
        """Create sample puzzles for testing"""
        samples = []
        
        # Sample 1: Color mapping
        samples.append({
            'id': 'sample_001',
            'train': [
                {
                    'input': [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                    'output': [[3, 0, 0], [0, 2, 0], [0, 0, 1]]
                },
                {
                    'input': [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
                    'output': [[6, 0, 0], [0, 5, 0], [0, 0, 4]]
                }
            ],
            'test': [
                {
                    'input': [[7, 0, 0], [0, 8, 0], [0, 0, 9]]
                }
            ]
        })
        
        # Sample 2: Pattern fill
        samples.append({
            'id': 'sample_002',
            'train': [
                {
                    'input': [
                        [1, 1, 1, 1],
                        [1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 1, 1]
                    ],
                    'output': [
                        [1, 1, 1, 1],
                        [1, 2, 2, 1],
                        [1, 2, 2, 1],
                        [1, 1, 1, 1]
                    ]
                }
            ],
            'test': [
                {
                    'input': [
                        [3, 3, 3],
                        [3, 0, 3],
                        [3, 3, 3]
                    ]
                }
            ]
        })
        
        # Sample 3: Rotation
        samples.append({
            'id': 'sample_003',
            'train': [
                {
                    'input': [[1, 0], [0, 0]],
                    'output': [[0, 1], [0, 0]]
                },
                {
                    'input': [[2, 0], [0, 0]],
                    'output': [[0, 2], [0, 0]]
                }
            ],
            'test': [
                {
                    'input': [[3, 0], [0, 0]]
                }
            ]
        })
        
        # Sample 4: Symmetry
        samples.append({
            'id': 'sample_004',
            'train': [
                {
                    'input': [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                    'output': [[1, 0, 1], [2, 0, 2], [3, 0, 3]]
                }
            ],
            'test': [
                {
                    'input': [[4, 5, 0], [6, 7, 0], [8, 9, 0]]
                }
            ]
        })
        
        # Sample 5: Grid cropping
        samples.append({
            'id': 'sample_005',
            'train': [
                {
                    'input': [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 0, 1, 2],
                        [3, 4, 5, 6]
                    ],
                    'output': [[1, 2], [5, 6]]
                },
                {
                    'input': [
                        [2, 3, 4, 5],
                        [6, 7, 8, 9],
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                    ],
                    'output': [[2, 3], [6, 7]]
                }
            ],
            'test': [
                {
                    'input': [
                        [3, 4, 5, 6],
                        [7, 8, 9, 0],
                        [1, 2, 3, 4],
                        [5, 6, 7, 8]
                    ]
                }
            ]
        })
        
        return samples
    
    def get_puzzle(self, index: int) -> Optional[Dict]:
        """Get puzzle by index"""
        if 0 <= index < len(self.puzzles):
            self.current_index = index
            return self.puzzles[index]
        return None
    
    def get_puzzle_by_id(self, puzzle_id: str) -> Optional[Dict]:
        """Get puzzle by ID"""
        if puzzle_id in self.puzzle_index:
            index = self.puzzle_index[puzzle_id]
            return self.get_puzzle(index)
        return None
    
    def next_puzzle(self) -> Optional[Dict]:
        """Get next puzzle"""
        if self.current_index < len(self.puzzles) - 1:
            self.current_index += 1
            return self.puzzles[self.current_index]
        return None
    
    def previous_puzzle(self) -> Optional[Dict]:
        """Get previous puzzle"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.puzzles[self.current_index]
        return None
    
    def get_current_puzzle(self) -> Optional[Dict]:
        """Get current puzzle"""
        return self.get_puzzle(self.current_index)
    
    def get_puzzle_count(self) -> int:
        """Get total number of puzzles"""
        return len(self.puzzles)
    
    def get_current_position(self) -> tuple:
        """Get current position as (current, total)"""
        return (self.current_index + 1, len(self.puzzles))
    
    def switch_puzzle_set(self, puzzle_set: str) -> bool:
        """Switch to a different puzzle set"""
        if puzzle_set in ["training", "evaluation"]:
            self.load_puzzles(puzzle_set)
            return True
        return False
    
    def get_current_set(self) -> str:
        """Get the currently loaded puzzle set"""
        return self.current_set

# Global puzzle loader instance
puzzle_loader = ARCPuzzleLoader()