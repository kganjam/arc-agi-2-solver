"""
Enhanced Learning System with Pattern Recognition
Implements advanced learning algorithms for continuous improvement
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict
import hashlib

class PatternLibrary:
    """Library of discovered patterns with learning"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_success_rate = defaultdict(float)
        self.pattern_usage_count = defaultdict(int)
        self.pattern_relationships = defaultdict(set)
        
    def add_pattern(self, pattern_id: str, pattern: Dict):
        """Add a discovered pattern"""
        self.patterns[pattern_id] = {
            'pattern': pattern,
            'discovered_at': datetime.now().isoformat(),
            'complexity': self._calculate_complexity(pattern),
            'applications': []
        }
        
    def _calculate_complexity(self, pattern: Dict) -> float:
        """Calculate pattern complexity"""
        complexity = 0.0
        
        # Factors that increase complexity
        if 'transformations' in pattern:
            complexity += len(pattern['transformations']) * 0.1
        if 'conditions' in pattern:
            complexity += len(pattern['conditions']) * 0.15
        if 'nested' in pattern and pattern['nested']:
            complexity += 0.3
        if 'multi_step' in pattern and pattern['multi_step']:
            complexity += 0.25
            
        return min(1.0, complexity)
        
    def find_similar_patterns(self, pattern: Dict, threshold: float = 0.7) -> List[str]:
        """Find patterns similar to given pattern"""
        similar = []
        
        pattern_hash = self._pattern_hash(pattern)
        
        for pid, stored in self.patterns.items():
            similarity = self._calculate_similarity(pattern, stored['pattern'])
            if similarity >= threshold:
                similar.append(pid)
                # Track relationship
                self.pattern_relationships[pattern_hash].add(pid)
                
        return similar
        
    def _pattern_hash(self, pattern: Dict) -> str:
        """Generate hash for pattern"""
        pattern_str = json.dumps(pattern, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]
        
    def _calculate_similarity(self, p1: Dict, p2: Dict) -> float:
        """Calculate similarity between patterns"""
        similarity = 0.0
        
        # Check transformation similarity
        if 'transformations' in p1 and 'transformations' in p2:
            t1 = set(p1['transformations']) if isinstance(p1['transformations'], list) else {p1['transformations']}
            t2 = set(p2['transformations']) if isinstance(p2['transformations'], list) else {p2['transformations']}
            if t1 and t2:
                similarity += len(t1 & t2) / len(t1 | t2) * 0.5
                
        # Check condition similarity
        if 'conditions' in p1 and 'conditions' in p2:
            c1 = set(p1['conditions']) if isinstance(p1['conditions'], list) else {p1['conditions']}
            c2 = set(p2['conditions']) if isinstance(p2['conditions'], list) else {p2['conditions']}
            if c1 and c2:
                similarity += len(c1 & c2) / len(c1 | c2) * 0.3
                
        # Check structural similarity
        if p1.get('type') == p2.get('type'):
            similarity += 0.2
            
        return similarity
        
    def update_pattern_performance(self, pattern_id: str, success: bool, context: Dict = None):
        """Update pattern performance metrics"""
        self.pattern_usage_count[pattern_id] += 1
        
        if success:
            current_rate = self.pattern_success_rate[pattern_id]
            count = self.pattern_usage_count[pattern_id]
            # Update running average
            self.pattern_success_rate[pattern_id] = (
                (current_rate * (count - 1) + 1.0) / count
            )
        else:
            current_rate = self.pattern_success_rate[pattern_id]
            count = self.pattern_usage_count[pattern_id]
            self.pattern_success_rate[pattern_id] = (
                (current_rate * (count - 1)) / count
            )
            
        # Store application context
        if pattern_id in self.patterns and context:
            self.patterns[pattern_id]['applications'].append({
                'success': success,
                'context': context,
                'timestamp': datetime.now().isoformat()
            })
            
    def get_best_patterns(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get best performing patterns"""
        pattern_scores = []
        
        for pid in self.patterns:
            if self.pattern_usage_count[pid] > 0:
                success_rate = self.pattern_success_rate[pid]
                usage_factor = min(1.0, self.pattern_usage_count[pid] / 10)
                score = success_rate * 0.7 + usage_factor * 0.3
                pattern_scores.append((pid, score))
                
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores[:n]

class TransferLearner:
    """Implements transfer learning between similar puzzles"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.transfer_history = []
        self.domain_mappings = {}
        
    def learn_from_puzzle(self, puzzle: Dict, solution: Dict):
        """Extract transferable knowledge from solved puzzle"""
        puzzle_id = puzzle.get('id', 'unknown')
        
        # Extract features
        features = self._extract_features(puzzle)
        
        # Extract transformation rules
        rules = self._extract_rules(puzzle, solution)
        
        # Store knowledge
        self.knowledge_base[puzzle_id] = {
            'features': features,
            'rules': rules,
            'solution_type': solution.get('type', 'unknown'),
            'confidence': solution.get('confidence', 0.5)
        }
        
        # Identify domain
        domain = self._identify_domain(features)
        if domain not in self.domain_mappings:
            self.domain_mappings[domain] = []
        self.domain_mappings[domain].append(puzzle_id)
        
    def _extract_features(self, puzzle: Dict) -> Dict:
        """Extract features from puzzle"""
        features = {
            'grid_size': None,
            'color_count': 0,
            'has_symmetry': False,
            'has_patterns': False,
            'transformation_type': None
        }
        
        if 'train' in puzzle and puzzle['train']:
            example = puzzle['train'][0]
            if 'input' in example:
                input_grid = np.array(example['input'])
                features['grid_size'] = input_grid.shape
                features['color_count'] = len(np.unique(input_grid))
                features['has_symmetry'] = self._check_symmetry(input_grid)
                
            if 'output' in example:
                output_grid = np.array(example['output'])
                if 'input' in example:
                    features['transformation_type'] = self._identify_transformation(
                        input_grid, output_grid
                    )
                    
        return features
        
    def _extract_rules(self, puzzle: Dict, solution: Dict) -> List[Dict]:
        """Extract transformation rules"""
        rules = []
        
        # Extract from solution explanation if available
        if 'explanation' in solution:
            explanation = solution['explanation']
            if 'steps' in explanation:
                for step in explanation['steps']:
                    rules.append({
                        'type': step.get('reasoning', 'unknown'),
                        'description': step.get('description', ''),
                        'confidence': 0.8
                    })
                    
        # Extract from pattern analysis
        if 'train' in puzzle:
            for example in puzzle['train']:
                if 'input' in example and 'output' in example:
                    rule = self._analyze_transformation(
                        np.array(example['input']),
                        np.array(example['output'])
                    )
                    if rule:
                        rules.append(rule)
                        
        return rules
        
    def _check_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has symmetry"""
        # Check horizontal symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            return True
        # Check vertical symmetry
        if np.array_equal(grid, np.flipud(grid)):
            return True
        # Check rotational symmetry
        if np.array_equal(grid, np.rot90(grid, 2)):
            return True
        return False
        
    def _identify_transformation(self, input_grid: np.ndarray, 
                                output_grid: np.ndarray) -> str:
        """Identify type of transformation"""
        if input_grid.shape != output_grid.shape:
            return 'size_change'
        elif not np.array_equal(np.unique(input_grid), np.unique(output_grid)):
            return 'color_change'
        elif np.array_equal(input_grid, np.rot90(output_grid, -1)):
            return 'rotation'
        elif np.array_equal(input_grid, np.fliplr(output_grid)):
            return 'reflection'
        else:
            return 'complex'
            
    def _analyze_transformation(self, input_grid: np.ndarray,
                               output_grid: np.ndarray) -> Optional[Dict]:
        """Analyze transformation between grids"""
        # Simplified analysis
        return {
            'type': 'transformation',
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape,
            'preserves_size': input_grid.shape == output_grid.shape,
            'confidence': 0.7
        }
        
    def _identify_domain(self, features: Dict) -> str:
        """Identify puzzle domain"""
        if features['has_symmetry']:
            return 'symmetry_based'
        elif features['transformation_type'] == 'color_change':
            return 'color_based'
        elif features['transformation_type'] == 'size_change':
            return 'size_based'
        elif features['color_count'] <= 3:
            return 'simple_pattern'
        else:
            return 'complex_pattern'
            
    def transfer_knowledge(self, new_puzzle: Dict) -> Dict:
        """Transfer knowledge to new puzzle"""
        # Extract features of new puzzle
        new_features = self._extract_features(new_puzzle)
        new_domain = self._identify_domain(new_features)
        
        # Find similar puzzles
        similar_puzzles = self._find_similar_puzzles(new_features, new_domain)
        
        # Transfer rules
        transferred_rules = []
        for puzzle_id in similar_puzzles:
            if puzzle_id in self.knowledge_base:
                knowledge = self.knowledge_base[puzzle_id]
                for rule in knowledge['rules']:
                    # Adapt rule to new context
                    adapted_rule = self._adapt_rule(rule, new_features)
                    if adapted_rule:
                        transferred_rules.append(adapted_rule)
                        
        # Record transfer
        self.transfer_history.append({
            'target_puzzle': new_puzzle.get('id', 'unknown'),
            'source_puzzles': similar_puzzles,
            'transferred_rules': len(transferred_rules),
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'transferred_rules': transferred_rules,
            'source_puzzles': similar_puzzles,
            'confidence': self._calculate_transfer_confidence(similar_puzzles)
        }
        
    def _find_similar_puzzles(self, features: Dict, domain: str) -> List[str]:
        """Find puzzles with similar features"""
        similar = []
        
        # First check same domain
        if domain in self.domain_mappings:
            candidates = self.domain_mappings[domain]
            
            for puzzle_id in candidates:
                if puzzle_id in self.knowledge_base:
                    stored_features = self.knowledge_base[puzzle_id]['features']
                    similarity = self._calculate_feature_similarity(features, stored_features)
                    if similarity > 0.6:
                        similar.append(puzzle_id)
                        
        return similar[:5]  # Return top 5 similar
        
    def _calculate_feature_similarity(self, f1: Dict, f2: Dict) -> float:
        """Calculate similarity between feature sets"""
        similarity = 0.0
        
        # Check grid size similarity
        if f1['grid_size'] and f2['grid_size']:
            if f1['grid_size'] == f2['grid_size']:
                similarity += 0.3
                
        # Check color count similarity
        if abs(f1['color_count'] - f2['color_count']) <= 1:
            similarity += 0.2
            
        # Check symmetry
        if f1['has_symmetry'] == f2['has_symmetry']:
            similarity += 0.2
            
        # Check transformation type
        if f1['transformation_type'] == f2['transformation_type']:
            similarity += 0.3
            
        return similarity
        
    def _adapt_rule(self, rule: Dict, new_features: Dict) -> Optional[Dict]:
        """Adapt rule to new context"""
        adapted_rule = rule.copy()
        
        # Adjust confidence based on feature match
        if 'preserves_size' in rule and new_features.get('grid_size'):
            # Rule is applicable
            adapted_rule['confidence'] *= 0.9
        else:
            adapted_rule['confidence'] *= 0.7
            
        # Only return if confidence is reasonable
        if adapted_rule['confidence'] > 0.5:
            return adapted_rule
        return None
        
    def _calculate_transfer_confidence(self, similar_puzzles: List[str]) -> float:
        """Calculate confidence in transfer"""
        if not similar_puzzles:
            return 0.1
            
        # Base confidence on number and quality of similar puzzles
        base_confidence = min(0.9, len(similar_puzzles) * 0.2)
        
        # Adjust based on success rate of similar puzzles
        avg_confidence = 0.0
        for puzzle_id in similar_puzzles:
            if puzzle_id in self.knowledge_base:
                avg_confidence += self.knowledge_base[puzzle_id]['confidence']
                
        if similar_puzzles:
            avg_confidence /= len(similar_puzzles)
            
        return base_confidence * avg_confidence

class MetaLearner:
    """Meta-learning system that learns how to learn"""
    
    def __init__(self):
        self.learning_strategies = {}
        self.strategy_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        self.meta_knowledge = {}
        
    def add_learning_strategy(self, strategy_id: str, strategy: Dict):
        """Add a learning strategy"""
        self.learning_strategies[strategy_id] = {
            'strategy': strategy,
            'created_at': datetime.now().isoformat(),
            'applications': []
        }
        
    def select_strategy(self, context: Dict) -> str:
        """Select best learning strategy for context"""
        best_strategy = None
        best_score = 0.0
        
        for strategy_id, strategy_data in self.learning_strategies.items():
            score = self._evaluate_strategy_fit(strategy_data['strategy'], context)
            
            # Adjust score based on past performance
            if self.strategy_performance[strategy_id]['total'] > 0:
                success_rate = (
                    self.strategy_performance[strategy_id]['success'] /
                    self.strategy_performance[strategy_id]['total']
                )
                score *= (0.5 + 0.5 * success_rate)
                
            if score > best_score:
                best_score = score
                best_strategy = strategy_id
                
        return best_strategy or 'default'
        
    def _evaluate_strategy_fit(self, strategy: Dict, context: Dict) -> float:
        """Evaluate how well strategy fits context"""
        fit_score = 0.5  # Base score
        
        # Check if strategy conditions match context
        if 'conditions' in strategy:
            for condition in strategy['conditions']:
                if condition in str(context):
                    fit_score += 0.1
                    
        # Check if strategy type matches problem type
        if 'problem_type' in context and 'applicable_to' in strategy:
            if context['problem_type'] in strategy['applicable_to']:
                fit_score += 0.3
                
        return min(1.0, fit_score)
        
    def update_strategy_performance(self, strategy_id: str, success: bool):
        """Update strategy performance"""
        self.strategy_performance[strategy_id]['total'] += 1
        if success:
            self.strategy_performance[strategy_id]['success'] += 1
            
    def learn_from_experience(self, experience: Dict):
        """Learn from solving experience"""
        # Extract meta-patterns
        if 'successful_strategies' in experience:
            for strategy in experience['successful_strategies']:
                if strategy not in self.meta_knowledge:
                    self.meta_knowledge[strategy] = {
                        'contexts': [],
                        'success_indicators': []
                    }
                self.meta_knowledge[strategy]['contexts'].append(
                    experience.get('context', {})
                )
                
        # Identify new learning strategies
        if 'novel_approach' in experience:
            new_strategy_id = f"learned_{len(self.learning_strategies)}"
            self.add_learning_strategy(new_strategy_id, experience['novel_approach'])
            
    def generate_learning_plan(self, puzzle: Dict) -> Dict:
        """Generate a learning plan for puzzle"""
        plan = {
            'phases': [],
            'strategies': [],
            'estimated_attempts': 0,
            'confidence': 0.0
        }
        
        # Phase 1: Initial analysis
        plan['phases'].append({
            'phase': 'analysis',
            'actions': ['extract_features', 'identify_patterns', 'check_symmetry']
        })
        
        # Phase 2: Strategy selection
        context = {'puzzle_type': 'unknown', 'features': {}}
        selected_strategy = self.select_strategy(context)
        plan['strategies'].append(selected_strategy)
        
        # Phase 3: Learning
        plan['phases'].append({
            'phase': 'learning',
            'actions': ['apply_strategy', 'extract_rules', 'validate_hypothesis']
        })
        
        # Phase 4: Validation
        plan['phases'].append({
            'phase': 'validation',
            'actions': ['test_solution', 'verify_consistency', 'calculate_confidence']
        })
        
        # Estimate attempts based on complexity
        plan['estimated_attempts'] = 3  # Base estimate
        
        # Calculate confidence
        if selected_strategy in self.strategy_performance:
            if self.strategy_performance[selected_strategy]['total'] > 0:
                plan['confidence'] = (
                    self.strategy_performance[selected_strategy]['success'] /
                    self.strategy_performance[selected_strategy]['total']
                )
        else:
            plan['confidence'] = 0.5
            
        return plan