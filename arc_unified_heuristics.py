#!/usr/bin/env python3
"""
Unified Heuristics Manager for ARC AGI
Combines the best features from all existing heuristics systems
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
from pathlib import Path
import shutil

@dataclass
class UnifiedHeuristic:
    """Unified heuristic structure combining all best features"""
    id: str
    name: str
    description: str
    pattern_type: str  # 'color_mapping', 'symmetry', 'size_transform', 'object_based', 'pattern_completion'
    
    # When to use (from base_heuristics.json)
    conditions: List[str]  # Conditions that should be met
    features: List[str] = field(default_factory=list)  # Required features
    confidence: float = 0.5  # Confidence threshold for application
    
    # Transformation details (from base_heuristics.json)
    transformation: Dict[str, Any] = field(default_factory=dict)  # Type and parameters
    transformations: List[str] = field(default_factory=list)  # List of applicable transformations
    
    # Code and implementation (from heuristics_store.py)
    code_snippet: str = ""  # Python code for the transformation
    
    # Performance tracking
    success_rate: float = 0.0
    usage_count: int = 0
    successful_applications: List[Dict] = field(default_factory=list)  # List of successful puzzle IDs with timestamps
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)
    complexity: int = 1  # 1-5 complexity rating
    examples: List[str] = field(default_factory=list)  # Example puzzle IDs
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

class UnifiedHeuristicsManager:
    """Unified manager for all heuristics with clean interface"""
    
    def __init__(self, storage_dir: str = "heuristics_unified"):
        """Initialize unified heuristics manager"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.heuristics_dir = self.storage_dir / "heuristics"
        self.patterns_dir = self.storage_dir / "patterns"
        self.solutions_dir = self.storage_dir / "solutions"
        
        # Create subdirectories
        self.heuristics_dir.mkdir(exist_ok=True)
        self.patterns_dir.mkdir(exist_ok=True)
        self.solutions_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self.heuristics: Dict[str, UnifiedHeuristic] = {}
        self.patterns: Dict[str, Any] = {}
        self.solutions: Dict[str, Dict] = {}
        
        # Migrate from old systems if needed
        self._migrate_existing_heuristics()
        
        # Load heuristics
        self.load_all()
        
        # Initialize defaults if empty
        if not self.heuristics:
            self._initialize_default_heuristics()
    
    def _migrate_existing_heuristics(self):
        """Migrate from all existing heuristics systems"""
        migrated_count = 0
        
        # 1. Migrate from heuristics_kb.json
        if os.path.exists("heuristics_kb.json"):
            print("Migrating from heuristics_kb.json...")
            try:
                with open("heuristics_kb.json", 'r') as f:
                    data = json.load(f)
                    for h_data in data.get('heuristics', []):
                        h = UnifiedHeuristic(
                            id=h_data.get('id', f"migrated_{migrated_count}"),
                            name=h_data.get('name', 'Unknown'),
                            description=h_data.get('description', ''),
                            pattern_type=h_data.get('pattern_type', 'unknown'),
                            conditions=h_data.get('conditions', []),
                            transformations=h_data.get('transformations', []),
                            success_rate=h_data.get('success_rate', 0.0),
                            usage_count=h_data.get('usage_count', 0),
                            created_at=h_data.get('created_at', ''),
                            tags=h_data.get('tags', []),
                            complexity=h_data.get('complexity', 1),
                            examples=h_data.get('examples', [])
                        )
                        self._save_heuristic(h)
                        migrated_count += 1
            except Exception as e:
                print(f"Error migrating from heuristics_kb.json: {e}")
        
        # 2. Migrate from heuristics/base_heuristics.json
        if os.path.exists("heuristics/base_heuristics.json"):
            print("Migrating from heuristics/base_heuristics.json...")
            try:
                with open("heuristics/base_heuristics.json", 'r') as f:
                    data = json.load(f)
                    for h_data in data.get('heuristics', []):
                        when_to_use = h_data.get('when_to_use', {})
                        transformation = h_data.get('transformation', {})
                        
                        h = UnifiedHeuristic(
                            id=h_data.get('id', f"base_{migrated_count}"),
                            name=h_data.get('name', 'Unknown'),
                            description=h_data.get('description', ''),
                            pattern_type=transformation.get('type', 'unknown'),
                            conditions=when_to_use.get('conditions', []),
                            features=when_to_use.get('features', []),
                            confidence=when_to_use.get('confidence', 0.5),
                            transformation=transformation,
                            success_rate=h_data.get('success_rate', 0.0),
                            usage_count=h_data.get('usage_count', 0)
                        )
                        self._save_heuristic(h)
                        migrated_count += 1
            except Exception as e:
                print(f"Error migrating from base_heuristics.json: {e}")
        
        # 3. Migrate from heuristics_kb directory (individual files)
        if os.path.exists("heuristics_kb"):
            print("Migrating from heuristics_kb directory...")
            for filename in os.listdir("heuristics_kb"):
                if filename.endswith('.json') and not filename.startswith('_'):
                    try:
                        with open(os.path.join("heuristics_kb", filename), 'r') as f:
                            h_data = json.load(f)
                            # Skip metadata fields
                            if '_metadata' in h_data:
                                del h_data['_metadata']
                            
                            h = UnifiedHeuristic(**h_data)
                            self._save_heuristic(h)
                            migrated_count += 1
                    except Exception as e:
                        print(f"Error migrating {filename}: {e}")
        
        # 4. Migrate from knowledge/store if it exists
        if os.path.exists("knowledge/store/heuristics.json"):
            print("Migrating from knowledge/store/heuristics.json...")
            try:
                with open("knowledge/store/heuristics.json", 'r') as f:
                    data = json.load(f)
                    for name, h_data in data.items():
                        h = UnifiedHeuristic(
                            id=f"knowledge_{name.lower().replace(' ', '_')}",
                            name=h_data.get('name', name),
                            description=h_data.get('description', ''),
                            pattern_type=h_data.get('pattern_type', 'unknown'),
                            code_snippet=h_data.get('code_snippet', ''),
                            success_rate=h_data.get('success_rate', 0.0),
                            usage_count=h_data.get('usage_count', 0),
                            created_at=h_data.get('created_at', ''),
                            successful_applications=h_data.get('successful_applications', []),
                            conditions=[],  # Not present in this format
                            transformations=[]
                        )
                        self._save_heuristic(h)
                        migrated_count += 1
            except Exception as e:
                print(f"Error migrating from knowledge store: {e}")
        
        if migrated_count > 0:
            print(f"Successfully migrated {migrated_count} heuristics")
    
    def _save_heuristic(self, heuristic: UnifiedHeuristic):
        """Save a single heuristic to its own file"""
        heuristic.updated_at = datetime.now().isoformat()
        
        filename = f"{heuristic.id}.json"
        filepath = self.heuristics_dir / filename
        
        data = asdict(heuristic)
        data['_metadata'] = {
            'version': '1.0',
            'last_updated': heuristic.updated_at
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_all(self):
        """Load all heuristics, patterns, and solutions"""
        self.load_heuristics()
        self.load_patterns()
        self.load_solutions()
    
    def load_heuristics(self):
        """Load all heuristics from individual files"""
        self.heuristics.clear()
        
        for filepath in self.heuristics_dir.glob("*.json"):
            if not filepath.name.startswith('_'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if '_metadata' in data:
                            del data['_metadata']
                        h = UnifiedHeuristic(**data)
                        self.heuristics[h.id] = h
                except Exception as e:
                    print(f"Error loading {filepath.name}: {e}")
        
        print(f"Loaded {len(self.heuristics)} heuristics")
    
    def load_patterns(self):
        """Load stored patterns"""
        patterns_file = self.patterns_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    self.patterns = json.load(f)
            except Exception as e:
                print(f"Error loading patterns: {e}")
    
    def load_solutions(self):
        """Load stored solutions"""
        solutions_file = self.solutions_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, 'r') as f:
                    self.solutions = json.load(f)
            except Exception as e:
                print(f"Error loading solutions: {e}")
    
    def save_all(self):
        """Save all data"""
        # Save index
        self._save_index()
        
        # Save patterns
        patterns_file = self.patterns_dir / "patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        
        # Save solutions
        solutions_file = self.solutions_dir / "solutions.json"
        with open(solutions_file, 'w') as f:
            json.dump(self.solutions, f, indent=2)
    
    def _save_index(self):
        """Save an index of all heuristics for quick reference"""
        index = {
            'total': len(self.heuristics),
            'heuristics': [
                {
                    'id': h.id,
                    'name': h.name,
                    'pattern_type': h.pattern_type,
                    'success_rate': h.success_rate,
                    'usage_count': h.usage_count
                }
                for h in self.heuristics.values()
            ],
            'last_updated': datetime.now().isoformat()
        }
        
        index_file = self.heuristics_dir / "_index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    # === HEURISTIC OPERATIONS ===
    
    def create_heuristic(self, name: str, description: str, pattern_type: str,
                        conditions: List[str] = None, transformations: List[str] = None,
                        code_snippet: str = "", confidence: float = 0.5,
                        complexity: int = 1, tags: List[str] = None) -> UnifiedHeuristic:
        """Create a new heuristic"""
        h_id = f"h_{name.lower().replace(' ', '_')}_{len(self.heuristics)}"
        
        heuristic = UnifiedHeuristic(
            id=h_id,
            name=name,
            description=description,
            pattern_type=pattern_type,
            conditions=conditions or [],
            transformations=transformations or [],
            code_snippet=code_snippet,
            confidence=confidence,
            complexity=complexity,
            tags=tags or []
        )
        
        self.heuristics[h_id] = heuristic
        self._save_heuristic(heuristic)
        self._save_index()
        
        return heuristic
    
    def get_heuristic(self, heuristic_id: str) -> Optional[UnifiedHeuristic]:
        """Get a specific heuristic by ID"""
        return self.heuristics.get(heuristic_id)
    
    def update_heuristic(self, heuristic_id: str, **kwargs) -> bool:
        """Update a heuristic's properties"""
        if heuristic_id not in self.heuristics:
            return False
        
        h = self.heuristics[heuristic_id]
        for key, value in kwargs.items():
            if hasattr(h, key):
                setattr(h, key, value)
        
        h.updated_at = datetime.now().isoformat()
        self._save_heuristic(h)
        return True
    
    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic"""
        if heuristic_id not in self.heuristics:
            return False
        
        # Delete file
        filepath = self.heuristics_dir / f"{heuristic_id}.json"
        if filepath.exists():
            filepath.unlink()
        
        # Remove from memory
        del self.heuristics[heuristic_id]
        self._save_index()
        
        return True
    
    def get_all_heuristics(self) -> List[Dict]:
        """Get all heuristics as dictionaries"""
        return [asdict(h) for h in self.heuristics.values()]
    
    def search_heuristics(self, query: str) -> List[UnifiedHeuristic]:
        """Search heuristics by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for h in self.heuristics.values():
            if (query_lower in h.name.lower() or
                query_lower in h.description.lower() or
                query_lower in h.pattern_type.lower() or
                any(query_lower in tag for tag in h.tags)):
                results.append(h)
        
        return sorted(results, key=lambda x: x.success_rate, reverse=True)
    
    # === RANKING AND RELEVANCE ===
    
    def get_relevant_heuristics(self, puzzle_features: Dict) -> List[Dict]:
        """Get heuristics relevant to puzzle features"""
        relevant = []
        
        for h in self.heuristics.values():
            score = self._calculate_relevance(h, puzzle_features)
            if score > 0:
                h_dict = asdict(h)
                h_dict['relevance_score'] = score
                relevant.append(h_dict)
        
        return sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)
    
    def _calculate_relevance(self, heuristic: UnifiedHeuristic, features: Dict) -> float:
        """Calculate relevance score for a heuristic"""
        score = 0.0
        
        # Pattern type match
        if features.get('pattern_type') == heuristic.pattern_type:
            score += 0.4
        
        # Condition matches
        puzzle_conditions = features.get('conditions', [])
        matching_conditions = sum(1 for c in heuristic.conditions if c in puzzle_conditions)
        if heuristic.conditions:
            score += 0.3 * (matching_conditions / len(heuristic.conditions))
        
        # Feature matches
        puzzle_features = features.get('features', [])
        matching_features = sum(1 for f in heuristic.features if f in puzzle_features)
        if heuristic.features:
            score += 0.2 * (matching_features / len(heuristic.features))
        
        # Success rate bonus
        score += 0.1 * heuristic.success_rate
        
        return min(score, 1.0)
    
    def rank_heuristics(self, heuristic_ids: List[str], puzzle_data: Dict) -> List[Dict]:
        """Rank heuristics by effectiveness for a puzzle"""
        ranked = []
        
        for h_id in heuristic_ids:
            if h_id in self.heuristics:
                h = self.heuristics[h_id]
                score = self._calculate_effectiveness(h, puzzle_data)
                h_dict = asdict(h)
                h_dict['effectiveness_score'] = score
                ranked.append(h_dict)
        
        return sorted(ranked, key=lambda x: x['effectiveness_score'], reverse=True)
    
    def _calculate_effectiveness(self, heuristic: UnifiedHeuristic, puzzle_data: Dict) -> float:
        """Calculate effectiveness score"""
        score = heuristic.success_rate * 0.4
        
        # Size change consideration
        if puzzle_data.get('size_change'):
            if heuristic.pattern_type == 'size_transform':
                score += 0.3
        else:
            if heuristic.pattern_type != 'size_transform':
                score += 0.2
        
        # Color complexity
        num_colors = len(puzzle_data.get('colors', []))
        if num_colors <= 3 and heuristic.complexity <= 2:
            score += 0.2
        elif num_colors > 5 and heuristic.complexity >= 3:
            score += 0.2
        
        # Recent success bonus
        if heuristic.usage_count > 0 and heuristic.success_rate > 0.7:
            score += 0.1
        
        return min(score, 1.0)
    
    # === APPLICATION AND TRACKING ===
    
    def apply_heuristic(self, heuristic_id: str, puzzle_data: Dict) -> Dict:
        """Apply a heuristic to puzzle data"""
        if heuristic_id not in self.heuristics:
            return {"error": f"Heuristic {heuristic_id} not found"}
        
        h = self.heuristics[heuristic_id]
        h.usage_count += 1
        
        result = {
            "heuristic_id": heuristic_id,
            "heuristic_name": h.name,
            "applied": True,
            "transformation": h.transformation,
            "transformations_suggested": h.transformations,
            "code_snippet": h.code_snippet,
            "confidence": h.confidence
        }
        
        # Add analysis based on pattern type
        if h.pattern_type == "color_mapping":
            result["analysis"] = "Apply color transformation rules"
        elif h.pattern_type == "symmetry":
            result["analysis"] = "Apply symmetry transformation"
        elif h.pattern_type == "size_transform":
            result["analysis"] = "Apply size transformation"
        elif h.pattern_type == "object_based":
            result["analysis"] = "Apply object-based transformation"
        elif h.pattern_type == "pattern_completion":
            result["analysis"] = "Complete the pattern"
        
        self._save_heuristic(h)
        return result
    
    def update_success(self, heuristic_id: str, success: bool, puzzle_id: str = None):
        """Update heuristic success tracking"""
        if heuristic_id not in self.heuristics:
            return
        
        h = self.heuristics[heuristic_id]
        
        if success:
            h.success_rate = (h.success_rate * max(h.usage_count - 1, 0) + 1) / h.usage_count
            if puzzle_id:
                h.successful_applications.append({
                    'puzzle_id': puzzle_id,
                    'timestamp': datetime.now().isoformat()
                })
                if puzzle_id not in h.examples:
                    h.examples.append(puzzle_id)
        else:
            h.success_rate = (h.success_rate * max(h.usage_count - 1, 0)) / h.usage_count
        
        self._save_heuristic(h)
    
    # === PATTERN AND SOLUTION STORAGE ===
    
    def store_pattern(self, pattern_name: str, pattern_data: Any):
        """Store a discovered pattern"""
        self.patterns[pattern_name] = {
            'data': pattern_data,
            'timestamp': datetime.now().isoformat()
        }
        self.save_all()
    
    def get_pattern(self, pattern_name: str) -> Optional[Any]:
        """Get a stored pattern"""
        pattern_info = self.patterns.get(pattern_name)
        return pattern_info['data'] if pattern_info else None
    
    def store_solution(self, puzzle_id: str, solution: Dict):
        """Store a puzzle solution"""
        self.solutions[puzzle_id] = {
            'solution': solution,
            'timestamp': datetime.now().isoformat(),
            'heuristics_used': solution.get('heuristics_used', [])
        }
        self.save_all()
    
    def get_solution(self, puzzle_id: str) -> Optional[Dict]:
        """Get a stored solution"""
        return self.solutions.get(puzzle_id)
    
    # === STATISTICS ===
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        total = len(self.heuristics)
        
        if total == 0:
            return {'total_heuristics': 0}
        
        # Calculate statistics
        avg_success = sum(h.success_rate for h in self.heuristics.values()) / total
        most_used = max(self.heuristics.values(), key=lambda h: h.usage_count)
        most_successful = max(self.heuristics.values(), key=lambda h: h.success_rate)
        
        # Pattern type distribution
        pattern_types = {}
        for h in self.heuristics.values():
            pattern_types[h.pattern_type] = pattern_types.get(h.pattern_type, 0) + 1
        
        return {
            'total_heuristics': total,
            'total_patterns': len(self.patterns),
            'total_solutions': len(self.solutions),
            'average_success_rate': avg_success,
            'most_used': asdict(most_used),
            'most_successful': asdict(most_successful),
            'pattern_types': pattern_types,
            'complexity_distribution': self._get_complexity_distribution()
        }
    
    def _get_complexity_distribution(self) -> Dict[int, int]:
        """Get distribution of heuristic complexities"""
        dist = {}
        for h in self.heuristics.values():
            dist[h.complexity] = dist.get(h.complexity, 0) + 1
        return dist
    
    def _initialize_default_heuristics(self):
        """Initialize with comprehensive default heuristics"""
        defaults = [
            # Color transformations
            self.create_heuristic(
                name="Direct Color Mapping",
                description="Map each color in input to a specific color in output",
                pattern_type="color_mapping",
                conditions=["consistent_color_transform", "same_grid_size"],
                transformations=["color_replace"],
                code_snippet="output[input == old_color] = new_color",
                confidence=0.8,
                complexity=1,
                tags=["basic", "color"]
            ),
            
            # Symmetry operations
            self.create_heuristic(
                name="Horizontal Mirror",
                description="Mirror the grid horizontally",
                pattern_type="symmetry",
                conditions=["horizontal_symmetry_possible"],
                transformations=["flip_horizontal"],
                code_snippet="np.fliplr(grid)",
                confidence=0.7,
                complexity=1,
                tags=["symmetry", "spatial"]
            ),
            
            self.create_heuristic(
                name="Vertical Mirror",
                description="Mirror the grid vertically",
                pattern_type="symmetry",
                conditions=["vertical_symmetry_possible"],
                transformations=["flip_vertical"],
                code_snippet="np.flipud(grid)",
                confidence=0.7,
                complexity=1,
                tags=["symmetry", "spatial"]
            ),
            
            self.create_heuristic(
                name="Rotate 90 Degrees",
                description="Rotate grid 90 degrees clockwise",
                pattern_type="symmetry",
                conditions=["square_grid", "rotational_pattern"],
                transformations=["rotate_90"],
                code_snippet="np.rot90(grid, -1)",
                confidence=0.6,
                complexity=2,
                tags=["rotation", "spatial"]
            ),
            
            # Size transformations
            self.create_heuristic(
                name="Crop to Content",
                description="Crop grid to bounding box of non-background elements",
                pattern_type="size_transform",
                conditions=["sparse_content", "background_dominant"],
                transformations=["find_bounding_box", "crop"],
                code_snippet="grid[min_row:max_row+1, min_col:max_col+1]",
                confidence=0.7,
                complexity=2,
                tags=["size", "crop"]
            ),
            
            self.create_heuristic(
                name="Scale 2x",
                description="Double the size of the grid",
                pattern_type="size_transform",
                conditions=["output_size_2x_input"],
                transformations=["scale"],
                code_snippet="np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)",
                confidence=0.8,
                complexity=2,
                tags=["scale", "size"]
            ),
            
            # Object-based operations
            self.create_heuristic(
                name="Extract Largest Object",
                description="Extract the largest connected component",
                pattern_type="object_based",
                conditions=["multiple_objects", "size_reduction"],
                transformations=["extract_object"],
                code_snippet="# Find and extract largest connected component",
                confidence=0.6,
                complexity=3,
                tags=["objects", "extraction"]
            ),
            
            self.create_heuristic(
                name="Count Objects",
                description="Count objects and transform based on count",
                pattern_type="object_based",
                conditions=["discrete_objects", "countable_elements"],
                transformations=["count_objects", "apply_count_rule"],
                code_snippet="# Count connected components and apply transformation",
                confidence=0.5,
                complexity=3,
                tags=["counting", "objects"]
            ),
            
            # Pattern completion
            self.create_heuristic(
                name="Complete Pattern",
                description="Complete a partial repeating pattern",
                pattern_type="pattern_completion",
                conditions=["repeating_pattern", "incomplete_pattern"],
                transformations=["detect_pattern", "extend_pattern"],
                code_snippet="# Detect and extend repeating pattern",
                confidence=0.6,
                complexity=3,
                tags=["pattern", "completion"]
            ),
            
            self.create_heuristic(
                name="Fill Enclosed Regions",
                description="Fill enclosed regions with specific colors",
                pattern_type="pattern_completion",
                conditions=["enclosed_regions", "boundary_detection"],
                transformations=["flood_fill"],
                code_snippet="# Flood fill enclosed regions",
                confidence=0.7,
                complexity=2,
                tags=["fill", "regions"]
            ),
            
            # Advanced transformations
            self.create_heuristic(
                name="Transpose",
                description="Transpose the grid (swap rows and columns)",
                pattern_type="symmetry",
                conditions=["diagonal_symmetry", "square_grid"],
                transformations=["transpose"],
                code_snippet="grid.T",
                confidence=0.6,
                complexity=1,
                tags=["transpose", "matrix"]
            ),
            
            self.create_heuristic(
                name="Line Extension",
                description="Extend lines to grid boundaries",
                pattern_type="pattern_completion",
                conditions=["partial_lines", "line_patterns"],
                transformations=["detect_lines", "extend_lines"],
                code_snippet="# Detect and extend lines to boundaries",
                confidence=0.5,
                complexity=3,
                tags=["lines", "extension"]
            )
        ]
        
        print(f"Initialized {len(defaults)} default heuristics")
        self._save_index()

# Global instance
unified_heuristics = UnifiedHeuristicsManager()