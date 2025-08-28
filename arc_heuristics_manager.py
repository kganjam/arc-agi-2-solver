#!/usr/bin/env python3
"""
Heuristics Knowledge Base Manager for ARC AGI
Manages heuristics storage, retrieval, ranking, and application
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

@dataclass
class Heuristic:
    """Represents a puzzle-solving heuristic"""
    id: str
    name: str
    description: str
    pattern_type: str  # 'color_mapping', 'symmetry', 'size_transform', 'object_based', 'pattern_completion'
    conditions: List[str]  # When to apply this heuristic
    transformations: List[str]  # What transformations to apply
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: str = ""
    tags: List[str] = None
    complexity: int = 1  # 1-5 complexity rating
    examples: List[str] = None  # Puzzle IDs where this worked
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.examples is None:
            self.examples = []

class HeuristicsManager:
    """Manages the heuristics knowledge base with individual JSON files"""
    
    def __init__(self, knowledge_base_dir: str = "heuristics_kb"):
        self.kb_dir = knowledge_base_dir
        self.old_kb_path = "heuristics_kb.json"  # Legacy single file
        self.heuristics: Dict[str, Heuristic] = {}
        
        # Create heuristics directory if it doesn't exist
        os.makedirs(self.kb_dir, exist_ok=True)
        
        # Migrate from old format if necessary
        self._migrate_from_single_file()
        
        # Load heuristics
        self.load_knowledge_base()
        self._initialize_default_heuristics()
    
    def _migrate_from_single_file(self):
        """Migrate from old single-file format to individual files"""
        if os.path.exists(self.old_kb_path) and not os.listdir(self.kb_dir):
            print("Migrating heuristics from old format to individual files...")
            try:
                with open(self.old_kb_path, 'r') as f:
                    data = json.load(f)
                    for h_data in data.get('heuristics', []):
                        h = Heuristic(**h_data)
                        self._save_individual_heuristic(h)
                # Rename old file to backup
                os.rename(self.old_kb_path, f"{self.old_kb_path}.backup")
                print(f"Migration complete. Old file backed up as {self.old_kb_path}.backup")
            except Exception as e:
                print(f"Error during migration: {e}")
    
    def load_knowledge_base(self):
        """Load heuristics from individual JSON files"""
        try:
            # Load each JSON file in the heuristics directory
            for filename in os.listdir(self.kb_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.kb_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            h_data = json.load(f)
                            h = Heuristic(**h_data)
                            self.heuristics[h.id] = h
                    except Exception as e:
                        print(f"Error loading heuristic from {filename}: {e}")
            
            print(f"Loaded {len(self.heuristics)} heuristics from {self.kb_dir}")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    
    def _save_individual_heuristic(self, heuristic: Heuristic):
        """Save a single heuristic to its own JSON file"""
        try:
            # Create filename from heuristic ID
            filename = f"{heuristic.id}.json"
            filepath = os.path.join(self.kb_dir, filename)
            
            # Save heuristic data with metadata
            data = asdict(heuristic)
            data['_metadata'] = {
                'last_updated': datetime.now().isoformat(),
                'version': '2.0'  # Version 2.0 uses individual files
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving heuristic {heuristic.id}: {e}")
    
    def save_knowledge_base(self):
        """Save all heuristics to individual files"""
        try:
            # Save each heuristic to its own file
            for heuristic in self.heuristics.values():
                self._save_individual_heuristic(heuristic)
            
            # Create an index file for quick reference
            index_data = {
                'total_heuristics': len(self.heuristics),
                'heuristic_ids': list(self.heuristics.keys()),
                'last_updated': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            index_path = os.path.join(self.kb_dir, '_index.json')
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
                
            print(f"Saved {len(self.heuristics)} heuristics to {self.kb_dir}")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic and its file"""
        if heuristic_id in self.heuristics:
            try:
                # Delete the file
                filename = f"{heuristic_id}.json"
                filepath = os.path.join(self.kb_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Remove from memory
                del self.heuristics[heuristic_id]
                
                # Update index
                self.save_knowledge_base()
                
                print(f"Deleted heuristic {heuristic_id}")
                return True
            except Exception as e:
                print(f"Error deleting heuristic {heuristic_id}: {e}")
                return False
        return False
    
    def _initialize_default_heuristics(self):
        """Initialize with default heuristics if none exist"""
        if not self.heuristics:
            defaults = [
                Heuristic(
                    id="h_color_map",
                    name="Direct Color Mapping",
                    description="Map each color in input to a specific color in output",
                    pattern_type="color_mapping",
                    conditions=["consistent_color_transform", "same_grid_size"],
                    transformations=["color_replace"],
                    complexity=1,
                    tags=["basic", "color"]
                ),
                Heuristic(
                    id="h_symmetry",
                    name="Symmetry Detection",
                    description="Detect and complete symmetrical patterns",
                    pattern_type="symmetry",
                    conditions=["partial_symmetry", "mirror_pattern"],
                    transformations=["mirror_horizontal", "mirror_vertical", "rotate"],
                    complexity=2,
                    tags=["symmetry", "spatial"]
                ),
                Heuristic(
                    id="h_object_count",
                    name="Object Counting",
                    description="Count objects and adjust output based on count",
                    pattern_type="object_based",
                    conditions=["discrete_objects", "countable_elements"],
                    transformations=["count_objects", "resize_by_count"],
                    complexity=3,
                    tags=["counting", "objects"]
                ),
                Heuristic(
                    id="h_pattern_extend",
                    name="Pattern Extension",
                    description="Extend existing patterns to fill grid",
                    pattern_type="pattern_completion",
                    conditions=["repeating_pattern", "incomplete_pattern"],
                    transformations=["extend_pattern", "fill_pattern"],
                    complexity=2,
                    tags=["pattern", "completion"]
                ),
                Heuristic(
                    id="h_size_crop",
                    name="Size Cropping",
                    description="Crop to bounding box of non-background elements",
                    pattern_type="size_transform",
                    conditions=["sparse_content", "background_dominant"],
                    transformations=["find_bounding_box", "crop_to_content"],
                    complexity=1,
                    tags=["size", "crop"]
                ),
                Heuristic(
                    id="h_flood_fill",
                    name="Flood Fill Regions",
                    description="Fill connected regions with specific colors",
                    pattern_type="color_mapping",
                    conditions=["enclosed_regions", "boundary_detection"],
                    transformations=["flood_fill", "region_color"],
                    complexity=2,
                    tags=["fill", "regions"]
                )
            ]
            
            for h in defaults:
                self.heuristics[h.id] = h
            
            self.save_knowledge_base()
    
    def get_all_heuristics(self) -> List[Dict]:
        """Get all heuristics as dictionaries"""
        return [asdict(h) for h in self.heuristics.values()]
    
    def get_relevant_heuristics(self, puzzle_features: Dict) -> List[Dict]:
        """Get heuristics relevant to the given puzzle features"""
        relevant = []
        
        for h in self.heuristics.values():
            relevance_score = self._calculate_relevance(h, puzzle_features)
            if relevance_score > 0:
                h_dict = asdict(h)
                h_dict['relevance_score'] = relevance_score
                relevant.append(h_dict)
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant
    
    def _calculate_relevance(self, heuristic: Heuristic, features: Dict) -> float:
        """Calculate how relevant a heuristic is to puzzle features"""
        score = 0.0
        
        # Check pattern type match
        if features.get('pattern_type') == heuristic.pattern_type:
            score += 0.5
        
        # Check condition matches
        puzzle_conditions = features.get('conditions', [])
        for condition in heuristic.conditions:
            if condition in puzzle_conditions:
                score += 0.2
        
        # Consider success rate
        score += heuristic.success_rate * 0.3
        
        # Consider complexity vs puzzle difficulty
        puzzle_complexity = features.get('complexity', 3)
        if abs(puzzle_complexity - heuristic.complexity) <= 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def rank_heuristics(self, heuristics: List[str], puzzle_data: Dict) -> List[Dict]:
        """Rank given heuristics based on likely effectiveness"""
        ranked = []
        
        for h_id in heuristics:
            if h_id in self.heuristics:
                h = self.heuristics[h_id]
                score = self._calculate_effectiveness_score(h, puzzle_data)
                h_dict = asdict(h)
                h_dict['effectiveness_score'] = score
                ranked.append(h_dict)
        
        ranked.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        return ranked
    
    def _calculate_effectiveness_score(self, heuristic: Heuristic, puzzle_data: Dict) -> float:
        """Calculate effectiveness score for a heuristic on a puzzle"""
        score = 0.0
        
        # Base score from success rate
        score = heuristic.success_rate * 0.4
        
        # Check if input/output sizes match heuristic expectations
        if puzzle_data.get('size_change'):
            if heuristic.pattern_type == 'size_transform':
                score += 0.3
        else:
            if heuristic.pattern_type != 'size_transform':
                score += 0.2
        
        # Check color complexity
        num_colors = len(puzzle_data.get('colors', []))
        if num_colors <= 3 and heuristic.complexity <= 2:
            score += 0.2
        elif num_colors > 5 and heuristic.complexity >= 3:
            score += 0.2
        
        # Boost if recently successful
        if heuristic.usage_count > 0 and heuristic.success_rate > 0.7:
            score += 0.1
        
        return min(score, 1.0)
    
    def apply_heuristic(self, heuristic_id: str, puzzle_data: Dict) -> Dict:
        """Apply a heuristic to puzzle data and return result"""
        if heuristic_id not in self.heuristics:
            return {"error": f"Heuristic {heuristic_id} not found"}
        
        h = self.heuristics[heuristic_id]
        h.usage_count += 1
        
        # Simulate heuristic application
        result = {
            "heuristic_id": heuristic_id,
            "heuristic_name": h.name,
            "applied": True,
            "transformations_suggested": h.transformations,
            "confidence": 0.0,
            "suggested_output": None
        }
        
        # Apply based on pattern type
        if h.pattern_type == "color_mapping":
            result["confidence"] = 0.7
            result["analysis"] = "Detected color mapping pattern. Suggest mapping colors consistently."
        elif h.pattern_type == "symmetry":
            result["confidence"] = 0.6
            result["analysis"] = "Detected symmetry. Suggest completing the symmetrical pattern."
        elif h.pattern_type == "size_transform":
            result["confidence"] = 0.8
            result["analysis"] = "Size transformation detected. Suggest cropping or scaling."
        elif h.pattern_type == "object_based":
            result["confidence"] = 0.5
            result["analysis"] = "Object-based pattern. Count objects and transform accordingly."
        elif h.pattern_type == "pattern_completion":
            result["confidence"] = 0.6
            result["analysis"] = "Pattern needs completion. Extend the existing pattern."
        
        self.save_knowledge_base()
        return result
    
    def test_heuristic(self, heuristic_id: str, puzzle_data: Dict, expected_output: List[List[int]]) -> Dict:
        """Test a heuristic against known output"""
        if heuristic_id not in self.heuristics:
            return {"error": f"Heuristic {heuristic_id} not found"}
        
        h = self.heuristics[heuristic_id]
        
        # Apply heuristic
        result = self.apply_heuristic(heuristic_id, puzzle_data)
        
        # Simulate testing (in real implementation, would apply actual transformations)
        success = np.random.random() > 0.3  # Simulated success
        
        if success:
            h.success_rate = (h.success_rate * h.usage_count + 1) / (h.usage_count + 1)
            if puzzle_data.get('puzzle_id'):
                h.examples.append(puzzle_data['puzzle_id'])
        else:
            h.success_rate = (h.success_rate * h.usage_count) / (h.usage_count + 1)
        
        self.save_knowledge_base()
        
        return {
            "heuristic_id": heuristic_id,
            "success": success,
            "new_success_rate": h.success_rate,
            "analysis": result.get("analysis", "")
        }
    
    def add_heuristic(self, name: str, description: str, pattern_type: str, 
                     conditions: List[str], transformations: List[str], 
                     complexity: int = 1, tags: List[str] = None) -> Dict:
        """Add a new heuristic to the knowledge base"""
        # Generate ID
        h_id = f"h_{name.lower().replace(' ', '_')}_{len(self.heuristics)}"
        
        # Create new heuristic
        new_heuristic = Heuristic(
            id=h_id,
            name=name,
            description=description,
            pattern_type=pattern_type,
            conditions=conditions,
            transformations=transformations,
            complexity=complexity,
            tags=tags or []
        )
        
        # Add to knowledge base
        self.heuristics[h_id] = new_heuristic
        
        # Save individual heuristic immediately
        self._save_individual_heuristic(new_heuristic)
        
        # Update index
        self.save_knowledge_base()
        
        return {
            "success": True,
            "heuristic_id": h_id,
            "message": f"Added new heuristic: {name}",
            "heuristic": asdict(new_heuristic)
        }
    
    def update_heuristic_success(self, heuristic_id: str, success: bool, puzzle_id: str = None):
        """Update heuristic success rate after application"""
        if heuristic_id in self.heuristics:
            h = self.heuristics[heuristic_id]
            
            # Update success rate
            if success:
                h.success_rate = (h.success_rate * h.usage_count + 1) / (h.usage_count + 1)
                if puzzle_id and puzzle_id not in h.examples:
                    h.examples.append(puzzle_id)
            else:
                h.success_rate = (h.success_rate * h.usage_count) / (h.usage_count + 1)
            
            h.usage_count += 1
            
            # Save individual heuristic immediately
            self._save_individual_heuristic(h)
            
            # Update index
            self.save_knowledge_base()
    
    def search_heuristics(self, query: str) -> List[Dict]:
        """Search heuristics by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for h in self.heuristics.values():
            if (query_lower in h.name.lower() or 
                query_lower in h.description.lower() or
                any(query_lower in tag for tag in h.tags)):
                results.append(asdict(h))
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the heuristics knowledge base"""
        total = len(self.heuristics)
        avg_success = sum(h.success_rate for h in self.heuristics.values()) / total if total > 0 else 0
        most_used = max(self.heuristics.values(), key=lambda h: h.usage_count) if self.heuristics else None
        most_successful = max(self.heuristics.values(), key=lambda h: h.success_rate) if self.heuristics else None
        
        return {
            "total_heuristics": total,
            "average_success_rate": avg_success,
            "most_used": asdict(most_used) if most_used else None,
            "most_successful": asdict(most_successful) if most_successful else None,
            "pattern_types": list(set(h.pattern_type for h in self.heuristics.values()))
        }

# Global instance
heuristics_manager = HeuristicsManager()