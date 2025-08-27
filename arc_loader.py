"""
ARC Challenge Data Loader
Handles loading and parsing of ARC challenge datasets
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

class ARCChallenge:
    """Represents a single ARC challenge with training and test examples"""
    
    def __init__(self, challenge_id: str, data: Dict):
        self.id = challenge_id
        self.train = data.get('train', [])
        self.test = data.get('test', [])
    
    def get_training_examples(self) -> List[Dict]:
        """Get all training input-output pairs"""
        return self.train
    
    def get_test_examples(self) -> List[Dict]:
        """Get all test examples (input only)"""
        return self.test
    
    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Get maximum grid dimensions across all examples"""
        max_height, max_width = 0, 0
        
        for example in self.train + self.test:
            for grid_type in ['input', 'output']:
                if grid_type in example:
                    grid = np.array(example[grid_type])
                    max_height = max(max_height, grid.shape[0])
                    max_width = max(max_width, grid.shape[1])
        
        return max_height, max_width

class ARCLoader:
    """Loads and manages ARC challenge datasets"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.challenges: Dict[str, ARCChallenge] = {}
        self.current_challenge: Optional[ARCChallenge] = None
    
    def load_challenges_from_directory(self, directory: str) -> int:
        """Load all JSON challenge files from a directory"""
        loaded_count = 0
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating sample data...")
            self._create_sample_data()
            return 0
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                challenge_id = filename.replace('.json', '')
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    self.challenges[challenge_id] = ARCChallenge(challenge_id, data)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return loaded_count
    
    def _create_sample_data(self):
        """Create sample ARC challenges for demonstration"""
        os.makedirs(self.data_path, exist_ok=True)
        
        # Sample challenge 1: Simple pattern copying
        sample1 = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[1, 0], [0, 1]]
                },
                {
                    "input": [[2, 3], [3, 2]],
                    "output": [[2, 3], [3, 2]]
                }
            ],
            "test": [
                {
                    "input": [[4, 5], [5, 4]]
                }
            ]
        }
        
        # Sample challenge 2: Color transformation
        sample2 = {
            "train": [
                {
                    "input": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                    "output": [[3, 3, 3], [3, 2, 3], [3, 3, 3]]
                }
            ],
            "test": [
                {
                    "input": [[2, 2, 2], [2, 1, 2], [2, 2, 2]]
                }
            ]
        }
        
        with open(os.path.join(self.data_path, "sample1.json"), 'w') as f:
            json.dump(sample1, f, indent=2)
        
        with open(os.path.join(self.data_path, "sample2.json"), 'w') as f:
            json.dump(sample2, f, indent=2)
        
        print("Created sample ARC challenges in data/ directory")
    
    def get_challenge_list(self) -> List[str]:
        """Get list of all loaded challenge IDs"""
        return list(self.challenges.keys())
    
    def select_challenge(self, challenge_id: str) -> Optional[ARCChallenge]:
        """Select a specific challenge by ID"""
        if challenge_id in self.challenges:
            self.current_challenge = self.challenges[challenge_id]
            return self.current_challenge
        return None
    
    def get_current_challenge(self) -> Optional[ARCChallenge]:
        """Get the currently selected challenge"""
        return self.current_challenge
    
    def load_from_url(self, url: str) -> bool:
        """Load challenges from a remote URL (for ARC dataset)"""
        try:
            import requests
            response = requests.get(url)
            data = response.json()
            
            for challenge_id, challenge_data in data.items():
                self.challenges[challenge_id] = ARCChallenge(challenge_id, challenge_data)
            
            return True
        except Exception as e:
            print(f"Error loading from URL: {e}")
            return False
