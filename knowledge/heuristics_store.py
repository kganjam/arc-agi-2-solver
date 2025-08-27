"""
Knowledge and Heuristics Storage System
Manages storage and retrieval of solving strategies and patterns
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle

class Heuristic:
    """Represents a single heuristic or solving strategy"""
    
    def __init__(self, name: str, description: str, pattern_type: str, 
                 code_snippet: str = "", success_rate: float = 0.0):
        self.name = name
        self.description = description
        self.pattern_type = pattern_type
        self.code_snippet = code_snippet
        self.success_rate = success_rate
        self.created_at = datetime.now().isoformat()
        self.usage_count = 0
        self.successful_applications = []
    
    def to_dict(self) -> Dict:
        """Convert heuristic to dictionary for storage"""
        return {
            'name': self.name,
            'description': self.description,
            'pattern_type': self.pattern_type,
            'code_snippet': self.code_snippet,
            'success_rate': self.success_rate,
            'created_at': self.created_at,
            'usage_count': self.usage_count,
            'successful_applications': self.successful_applications
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Heuristic':
        """Create heuristic from dictionary"""
        heuristic = cls(
            name=data['name'],
            description=data['description'],
            pattern_type=data['pattern_type'],
            code_snippet=data.get('code_snippet', ''),
            success_rate=data.get('success_rate', 0.0)
        )
        heuristic.created_at = data.get('created_at', datetime.now().isoformat())
        heuristic.usage_count = data.get('usage_count', 0)
        heuristic.successful_applications = data.get('successful_applications', [])
        return heuristic

class KnowledgeStore:
    """Central knowledge storage system for ARC solving strategies"""
    
    def __init__(self, storage_path: str = "knowledge/store/"):
        self.storage_path = storage_path
        self.heuristics: Dict[str, Heuristic] = {}
        self.patterns: Dict[str, Any] = {}
        self.solutions: Dict[str, Dict] = {}
        
        os.makedirs(storage_path, exist_ok=True)
        self.load_knowledge()
    
    def add_heuristic(self, heuristic: Heuristic) -> bool:
        """Add a new heuristic to the knowledge base"""
        try:
            self.heuristics[heuristic.name] = heuristic
            self.save_heuristics()
            return True
        except Exception as e:
            print(f"Error adding heuristic: {e}")
            return False
    
    def get_heuristic(self, name: str) -> Optional[Heuristic]:
        """Retrieve a specific heuristic by name"""
        return self.heuristics.get(name)
    
    def get_heuristics_by_pattern(self, pattern_type: str) -> List[Heuristic]:
        """Get all heuristics for a specific pattern type"""
        return [h for h in self.heuristics.values() if h.pattern_type == pattern_type]
    
    def get_all_heuristics(self) -> List[Heuristic]:
        """Get all stored heuristics"""
        return list(self.heuristics.values())
    
    def update_heuristic_success(self, name: str, challenge_id: str, success: bool):
        """Update success tracking for a heuristic"""
        if name in self.heuristics:
            heuristic = self.heuristics[name]
            heuristic.usage_count += 1
            
            if success:
                heuristic.successful_applications.append({
                    'challenge_id': challenge_id,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Recalculate success rate
            if heuristic.usage_count > 0:
                heuristic.success_rate = len(heuristic.successful_applications) / heuristic.usage_count
            
            self.save_heuristics()
    
    def store_solution(self, challenge_id: str, solution: Dict):
        """Store a successful solution"""
        self.solutions[challenge_id] = {
            'solution': solution,
            'timestamp': datetime.now().isoformat(),
            'heuristics_used': solution.get('heuristics_used', [])
        }
        self.save_solutions()
    
    def get_solution(self, challenge_id: str) -> Optional[Dict]:
        """Retrieve a stored solution"""
        return self.solutions.get(challenge_id)
    
    def store_pattern(self, pattern_name: str, pattern_data: Any):
        """Store a recognized pattern"""
        self.patterns[pattern_name] = {
            'data': pattern_data,
            'timestamp': datetime.now().isoformat()
        }
        self.save_patterns()
    
    def get_pattern(self, pattern_name: str) -> Optional[Any]:
        """Retrieve a stored pattern"""
        pattern_info = self.patterns.get(pattern_name)
        return pattern_info['data'] if pattern_info else None
    
    def save_heuristics(self):
        """Save heuristics to file"""
        filepath = os.path.join(self.storage_path, "heuristics.json")
        data = {name: h.to_dict() for name, h in self.heuristics.items()}
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_solutions(self):
        """Save solutions to file"""
        filepath = os.path.join(self.storage_path, "solutions.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.solutions, f, indent=2)
    
    def save_patterns(self):
        """Save patterns to file"""
        filepath = os.path.join(self.storage_path, "patterns.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.patterns, f)
    
    def load_knowledge(self):
        """Load all knowledge from storage"""
        self.load_heuristics()
        self.load_solutions()
        self.load_patterns()
        self._create_default_heuristics()
    
    def load_heuristics(self):
        """Load heuristics from file"""
        filepath = os.path.join(self.storage_path, "heuristics.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.heuristics = {name: Heuristic.from_dict(h_data) 
                                 for name, h_data in data.items()}
            except Exception as e:
                print(f"Error loading heuristics: {e}")
    
    def load_solutions(self):
        """Load solutions from file"""
        filepath = os.path.join(self.storage_path, "solutions.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.solutions = json.load(f)
            except Exception as e:
                print(f"Error loading solutions: {e}")
    
    def load_patterns(self):
        """Load patterns from file"""
        filepath = os.path.join(self.storage_path, "patterns.pkl")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.patterns = pickle.load(f)
            except Exception as e:
                print(f"Error loading patterns: {e}")
    
    def _create_default_heuristics(self):
        """Create some default heuristics if none exist"""
        if not self.heuristics:
            default_heuristics = [
                Heuristic(
                    name="Color Replacement",
                    description="Replace all instances of one color with another",
                    pattern_type="transformation",
                    code_snippet="grid[grid == old_color] = new_color"
                ),
                Heuristic(
                    name="Mirror Horizontal",
                    description="Mirror the grid horizontally",
                    pattern_type="symmetry",
                    code_snippet="np.fliplr(grid)"
                ),
                Heuristic(
                    name="Mirror Vertical",
                    description="Mirror the grid vertically",
                    pattern_type="symmetry",
                    code_snippet="np.flipud(grid)"
                ),
                Heuristic(
                    name="Pattern Completion",
                    description="Complete a repeating pattern in the grid",
                    pattern_type="pattern",
                    code_snippet="# Detect and extend pattern"
                ),
                Heuristic(
                    name="Object Counting",
                    description="Count distinct objects and apply transformation based on count",
                    pattern_type="counting",
                    code_snippet="# Count connected components"
                )
            ]
            
            for heuristic in default_heuristics:
                self.add_heuristic(heuristic)
    
    def search_heuristics(self, query: str) -> List[Heuristic]:
        """Search heuristics by name or description"""
        query = query.lower()
        results = []
        
        for heuristic in self.heuristics.values():
            if (query in heuristic.name.lower() or 
                query in heuristic.description.lower() or
                query in heuristic.pattern_type.lower()):
                results.append(heuristic)
        
        return sorted(results, key=lambda h: h.success_rate, reverse=True)
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            'total_heuristics': len(self.heuristics),
            'total_solutions': len(self.solutions),
            'total_patterns': len(self.patterns),
            'avg_success_rate': sum(h.success_rate for h in self.heuristics.values()) / len(self.heuristics) if self.heuristics else 0,
            'most_successful_heuristic': max(self.heuristics.values(), key=lambda h: h.success_rate).name if self.heuristics else None
        }
