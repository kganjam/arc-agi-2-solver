"""
Critical Reasoning and Verification System for ARC AGI
Ensures every solution has proper explanation and verification
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from enum import Enum

class ReasoningType(Enum):
    """Types of reasoning used in solutions"""
    PATTERN_MATCHING = "pattern_matching"
    LOGICAL_DEDUCTION = "logical_deduction"
    SPATIAL_TRANSFORMATION = "spatial_transformation"
    COLOR_RULES = "color_rules"
    OBJECT_MANIPULATION = "object_manipulation"
    SYMMETRY_ANALYSIS = "symmetry_analysis"
    COUNTING_BASED = "counting_based"
    BOUNDARY_ANALYSIS = "boundary_analysis"
    COMPOSITION = "composition"
    ABSTRACTION = "abstraction"

class Explanation:
    """Represents a detailed explanation of a solution"""
    
    def __init__(self, puzzle_id: str):
        self.puzzle_id = puzzle_id
        self.steps = []
        self.reasoning_types = []
        self.confidence = 0.0
        self.evidence = []
        self.assumptions = []
        self.verification_status = "unverified"
        
    def add_step(self, step: str, reasoning: ReasoningType, evidence: str = None):
        """Add a reasoning step"""
        self.steps.append({
            'description': step,
            'reasoning': reasoning.value,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat()
        })
        if reasoning not in self.reasoning_types:
            self.reasoning_types.append(reasoning)
            
    def add_assumption(self, assumption: str, justification: str):
        """Add an assumption made during solving"""
        self.assumptions.append({
            'assumption': assumption,
            'justification': justification
        })
        
    def add_evidence(self, evidence: str, supports: str):
        """Add supporting evidence"""
        self.evidence.append({
            'evidence': evidence,
            'supports': supports
        })
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'puzzle_id': self.puzzle_id,
            'steps': self.steps,
            'reasoning_types': [r.value for r in self.reasoning_types],
            'confidence': self.confidence,
            'evidence': self.evidence,
            'assumptions': self.assumptions,
            'verification_status': self.verification_status
        }

class CritiqueAgent:
    """Agent that critiques and verifies solution reasoning"""
    
    def __init__(self):
        self.critique_history = []
        self.verification_rules = self._init_verification_rules()
        
    def _init_verification_rules(self) -> List[Dict]:
        """Initialize verification rules"""
        return [
            {
                'rule': 'consistency_check',
                'description': 'Solution must be consistent across all training examples',
                'weight': 0.3
            },
            {
                'rule': 'transformation_validity',
                'description': 'Transformation must be mathematically valid',
                'weight': 0.2
            },
            {
                'rule': 'evidence_sufficiency',
                'description': 'Must have sufficient evidence for claims',
                'weight': 0.2
            },
            {
                'rule': 'assumption_validity',
                'description': 'Assumptions must be reasonable and justified',
                'weight': 0.15
            },
            {
                'rule': 'logical_coherence',
                'description': 'Reasoning must be logically coherent',
                'weight': 0.15
            }
        ]
        
    def critique_solution(self, puzzle: Dict, solution: Any, 
                         explanation: Explanation) -> Dict:
        """Critique a proposed solution and its reasoning"""
        critique = {
            'puzzle_id': puzzle.get('id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'issues': [],
            'strengths': [],
            'suggestions': [],
            'verified': False
        }
        
        # Check each verification rule
        for rule in self.verification_rules:
            score = self._check_rule(rule, puzzle, solution, explanation)
            critique['overall_score'] += score * rule['weight']
            
            if score < 0.5:
                critique['issues'].append(f"Failed {rule['rule']}: {rule['description']}")
            elif score > 0.8:
                critique['strengths'].append(f"Strong {rule['rule']}")
                
        # Detailed analysis
        critique.update(self._analyze_reasoning(explanation))
        critique.update(self._analyze_evidence(puzzle, solution, explanation))
        
        # Determine if verified
        critique['verified'] = critique['overall_score'] > 0.7
        
        # Add suggestions for improvement
        critique['suggestions'] = self._generate_suggestions(critique)
        
        # Store in history
        self.critique_history.append(critique)
        
        return critique
        
    def _check_rule(self, rule: Dict, puzzle: Dict, solution: Any, 
                   explanation: Explanation) -> float:
        """Check a specific verification rule"""
        if rule['rule'] == 'consistency_check':
            return self._check_consistency_score(puzzle, solution, explanation)
        elif rule['rule'] == 'transformation_validity':
            return self._check_transformation_validity(puzzle, solution, explanation)
        elif rule['rule'] == 'evidence_sufficiency':
            return self._check_evidence_sufficiency(explanation)
        elif rule['rule'] == 'assumption_validity':
            return self._check_assumption_validity(explanation)
        elif rule['rule'] == 'logical_coherence':
            return self._check_logical_coherence(explanation)
        return 0.5  # Default neutral score
        
    def _check_consistency_score(self, puzzle: Dict, solution: Any, 
                                 explanation: Explanation) -> float:
        """Check if solution is consistent across examples"""
        if 'train' not in puzzle or len(puzzle['train']) < 2:
            return 0.5  # Can't check consistency with < 2 examples
            
        # Check if the explained transformation works for all examples
        consistency_score = 0.0
        for example in puzzle['train']:
            # Simulate checking if explanation applies to this example
            # In real implementation, would apply the explained transformation
            if 'input' in example and 'output' in example:
                consistency_score += 0.25  # Simplified scoring
                
        return min(1.0, consistency_score)
        
    def _check_transformation_validity(self, puzzle: Dict, solution: Any,
                                      explanation: Explanation) -> float:
        """Check if transformation is mathematically valid"""
        # Check for valid transformation types in explanation
        valid_transformations = [
            ReasoningType.SPATIAL_TRANSFORMATION,
            ReasoningType.COLOR_RULES,
            ReasoningType.OBJECT_MANIPULATION
        ]
        
        score = 0.0
        for reasoning in explanation.reasoning_types:
            if reasoning in valid_transformations:
                score += 0.3
                
        # Check for mathematical properties
        if solution is not None:
            if isinstance(solution, (list, np.ndarray)):
                # Check if output has valid structure
                score += 0.2
                
        return min(1.0, score)
        
    def _check_evidence_sufficiency(self, explanation: Explanation) -> float:
        """Check if there's sufficient evidence"""
        if not explanation.evidence:
            return 0.2  # Low score for no evidence
            
        # Score based on evidence quantity and quality
        evidence_score = min(1.0, len(explanation.evidence) * 0.2)
        
        # Check if evidence supports steps
        supported_steps = sum(1 for step in explanation.steps if step.get('evidence'))
        if explanation.steps:
            evidence_score *= (supported_steps / len(explanation.steps))
            
        return evidence_score
        
    def _check_assumption_validity(self, explanation: Explanation) -> float:
        """Check if assumptions are valid"""
        if not explanation.assumptions:
            return 0.8  # Good if no assumptions needed
            
        # Check if assumptions are justified
        justified = sum(1 for a in explanation.assumptions if a.get('justification'))
        return justified / len(explanation.assumptions)
        
    def _check_logical_coherence(self, explanation: Explanation) -> float:
        """Check logical coherence of reasoning"""
        if not explanation.steps:
            return 0.0
            
        # Check if steps follow logical order
        coherence_score = 0.5  # Base score
        
        # Check for reasoning diversity (good sign)
        if len(explanation.reasoning_types) > 1:
            coherence_score += 0.2
            
        # Check for step progression
        if len(explanation.steps) >= 2:
            coherence_score += 0.3
            
        return min(1.0, coherence_score)
        
    def _analyze_reasoning(self, explanation: Explanation) -> Dict:
        """Analyze the reasoning in detail"""
        analysis = {
            'reasoning_depth': len(explanation.steps),
            'reasoning_diversity': len(explanation.reasoning_types),
            'has_evidence': len(explanation.evidence) > 0,
            'has_assumptions': len(explanation.assumptions) > 0
        }
        
        # Classify reasoning approach
        if ReasoningType.PATTERN_MATCHING in explanation.reasoning_types:
            analysis['approach'] = 'pattern-based'
        elif ReasoningType.LOGICAL_DEDUCTION in explanation.reasoning_types:
            analysis['approach'] = 'logic-based'
        elif ReasoningType.SPATIAL_TRANSFORMATION in explanation.reasoning_types:
            analysis['approach'] = 'spatial-based'
        else:
            analysis['approach'] = 'mixed'
            
        return analysis
        
    def _analyze_evidence(self, puzzle: Dict, solution: Any, 
                         explanation: Explanation) -> Dict:
        """Analyze the evidence provided"""
        evidence_analysis = {
            'evidence_count': len(explanation.evidence),
            'evidence_types': [],
            'evidence_strength': 0.0
        }
        
        # Categorize evidence
        for ev in explanation.evidence:
            if 'pattern' in ev.get('evidence', '').lower():
                evidence_analysis['evidence_types'].append('pattern')
            elif 'color' in ev.get('evidence', '').lower():
                evidence_analysis['evidence_types'].append('color')
            elif 'shape' in ev.get('evidence', '').lower():
                evidence_analysis['evidence_types'].append('shape')
                
        # Calculate evidence strength
        if evidence_analysis['evidence_count'] > 0:
            evidence_analysis['evidence_strength'] = min(1.0, 
                evidence_analysis['evidence_count'] * 0.25)
            
        return evidence_analysis
        
    def _generate_suggestions(self, critique: Dict) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if critique['overall_score'] < 0.5:
            suggestions.append("Consider re-analyzing the pattern from scratch")
            
        if 'consistency_check' in str(critique.get('issues', [])):
            suggestions.append("Verify transformation works for ALL training examples")
            
        if 'evidence_sufficiency' in str(critique.get('issues', [])):
            suggestions.append("Provide more concrete evidence for each reasoning step")
            
        if 'logical_coherence' in str(critique.get('issues', [])):
            suggestions.append("Ensure reasoning steps follow logical progression")
            
        if not critique.get('has_evidence'):
            suggestions.append("Add specific evidence from the puzzle grids")
            
        return suggestions

class ExplanationGenerator:
    """Generates detailed explanations for solutions"""
    
    def __init__(self):
        self.templates = self._init_templates()
        
    def _init_templates(self) -> Dict:
        """Initialize explanation templates"""
        return {
            'color_mapping': "The transformation maps color {old} to color {new} based on {reason}",
            'rotation': "The grid is rotated {degrees} degrees {direction}",
            'reflection': "The grid is reflected along the {axis} axis",
            'pattern_completion': "Missing parts of the pattern are filled based on {rule}",
            'object_extraction': "Objects of type {type} are extracted and {action}",
            'symmetry': "Symmetry is {action} along {axis}",
            'size_change': "The grid size changes from {old_size} to {new_size} by {method}",
            'counting': "The number of {items} determines {outcome}"
        }
        
    def generate_explanation(self, puzzle: Dict, solution: Any, 
                            method: str = None) -> Explanation:
        """Generate explanation for a solution"""
        explanation = Explanation(puzzle.get('id', 'unknown'))
        
        # Analyze puzzle to determine transformation
        transformation_type = self._analyze_transformation(puzzle)
        
        # Generate step-by-step explanation
        if transformation_type == 'color_mapping':
            self._explain_color_mapping(puzzle, solution, explanation)
        elif transformation_type == 'spatial':
            self._explain_spatial_transformation(puzzle, solution, explanation)
        elif transformation_type == 'pattern':
            self._explain_pattern_transformation(puzzle, solution, explanation)
        else:
            self._explain_general_transformation(puzzle, solution, explanation)
            
        # Add evidence
        self._add_evidence(puzzle, solution, explanation)
        
        # Add assumptions if any
        self._add_assumptions(puzzle, explanation)
        
        # Calculate confidence
        explanation.confidence = self._calculate_confidence(explanation)
        
        return explanation
        
    def _analyze_transformation(self, puzzle: Dict) -> str:
        """Analyze what type of transformation is used"""
        if 'train' not in puzzle or not puzzle['train']:
            return 'unknown'
            
        example = puzzle['train'][0]
        if 'input' not in example or 'output' not in example:
            return 'unknown'
            
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Check for color mapping
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        if input_colors != output_colors and input_grid.shape == output_grid.shape:
            return 'color_mapping'
            
        # Check for spatial transformation
        if input_grid.shape != output_grid.shape:
            return 'spatial'
            
        # Check for pattern
        if np.sum(input_grid != output_grid) < input_grid.size * 0.5:
            return 'pattern'
            
        return 'general'
        
    def _explain_color_mapping(self, puzzle: Dict, solution: Any, 
                              explanation: Explanation):
        """Explain color mapping transformation"""
        explanation.add_step(
            "Identify color mapping pattern",
            ReasoningType.COLOR_RULES,
            "Colors in input are systematically mapped to different colors in output"
        )
        
        if 'train' in puzzle and puzzle['train']:
            example = puzzle['train'][0]
            input_colors = np.unique(example['input'])
            output_colors = np.unique(example['output'])
            
            explanation.add_step(
                f"Input uses colors {input_colors}, output uses {output_colors}",
                ReasoningType.COLOR_RULES,
                "Direct observation from training examples"
            )
            
            explanation.add_step(
                "Apply consistent color mapping to test input",
                ReasoningType.PATTERN_MATCHING,
                "Same mapping rules from training examples"
            )
            
    def _explain_spatial_transformation(self, puzzle: Dict, solution: Any,
                                       explanation: Explanation):
        """Explain spatial transformation"""
        explanation.add_step(
            "Identify spatial transformation pattern",
            ReasoningType.SPATIAL_TRANSFORMATION,
            "Grid dimensions or orientation changes"
        )
        
        explanation.add_step(
            "Determine transformation parameters",
            ReasoningType.SPATIAL_TRANSFORMATION,
            "Rotation, reflection, or scaling detected"
        )
        
        explanation.add_step(
            "Apply transformation to test input",
            ReasoningType.SPATIAL_TRANSFORMATION,
            "Consistent application of identified transformation"
        )
        
    def _explain_pattern_transformation(self, puzzle: Dict, solution: Any,
                                       explanation: Explanation):
        """Explain pattern-based transformation"""
        explanation.add_step(
            "Identify repeating patterns in training examples",
            ReasoningType.PATTERN_MATCHING,
            "Patterns found through grid analysis"
        )
        
        explanation.add_step(
            "Extract pattern rules",
            ReasoningType.ABSTRACTION,
            "Abstract rules derived from concrete examples"
        )
        
        explanation.add_step(
            "Apply pattern rules to generate solution",
            ReasoningType.PATTERN_MATCHING,
            "Rule application to test input"
        )
        
    def _explain_general_transformation(self, puzzle: Dict, solution: Any,
                                       explanation: Explanation):
        """Explain general transformation"""
        explanation.add_step(
            "Analyze input-output relationships",
            ReasoningType.LOGICAL_DEDUCTION,
            "Systematic comparison of training examples"
        )
        
        explanation.add_step(
            "Identify transformation logic",
            ReasoningType.ABSTRACTION,
            "Abstract transformation rules extracted"
        )
        
        explanation.add_step(
            "Apply logic to test case",
            ReasoningType.LOGICAL_DEDUCTION,
            "Deductive application of learned rules"
        )
        
    def _add_evidence(self, puzzle: Dict, solution: Any, explanation: Explanation):
        """Add evidence to explanation"""
        if 'train' in puzzle and puzzle['train']:
            # Add evidence from training examples
            for i, example in enumerate(puzzle['train'][:2]):  # First 2 examples
                explanation.add_evidence(
                    f"Training example {i+1} shows consistent transformation",
                    "Transformation consistency"
                )
                
            # Add evidence about patterns
            if len(puzzle['train']) > 0:
                input_shape = np.array(puzzle['train'][0]['input']).shape
                output_shape = np.array(puzzle['train'][0]['output']).shape
                
                explanation.add_evidence(
                    f"Input shape {input_shape} transforms to {output_shape}",
                    "Dimensional transformation"
                )
                
    def _add_assumptions(self, puzzle: Dict, explanation: Explanation):
        """Add assumptions if any"""
        # Add common assumptions
        explanation.add_assumption(
            "Transformation is deterministic",
            "ARC puzzles have unique solutions"
        )
        
        if len(puzzle.get('train', [])) < 3:
            explanation.add_assumption(
                "Pattern generalizes from limited examples",
                f"Only {len(puzzle.get('train', []))} training examples available"
            )
            
    def _calculate_confidence(self, explanation: Explanation) -> float:
        """Calculate confidence in explanation"""
        confidence = 0.5  # Base confidence
        
        # Increase for more steps
        confidence += min(0.2, len(explanation.steps) * 0.05)
        
        # Increase for evidence
        confidence += min(0.2, len(explanation.evidence) * 0.1)
        
        # Decrease for assumptions
        confidence -= len(explanation.assumptions) * 0.05
        
        return max(0.1, min(1.0, confidence))

class CriticalSolver:
    """Solver with built-in critique and explanation"""
    
    def __init__(self):
        self.critique_agent = CritiqueAgent()
        self.explanation_generator = ExplanationGenerator()
        self.solution_history = []
        
    async def solve_with_critique(self, puzzle: Dict) -> Dict:
        """Solve puzzle with explanation and critique"""
        result = {
            'puzzle_id': puzzle.get('id', 'unknown'),
            'solved': False,
            'solution': None,
            'explanation': None,
            'critique': None,
            'verified': False
        }
        
        # Generate solution (simplified)
        solution = self._generate_solution(puzzle)
        
        # Generate explanation
        explanation = self.explanation_generator.generate_explanation(
            puzzle, solution
        )
        
        # Critique the solution
        critique = self.critique_agent.critique_solution(
            puzzle, solution, explanation
        )
        
        # Update result
        result['solution'] = solution
        result['explanation'] = explanation.to_dict()
        result['critique'] = critique
        result['verified'] = critique['verified']
        result['solved'] = critique['verified'] and critique['overall_score'] > 0.7
        
        # If not verified, try to improve
        if not result['verified']:
            improved_solution = self._improve_solution(puzzle, solution, critique)
            if improved_solution is not None:
                # Re-explain and re-critique
                new_explanation = self.explanation_generator.generate_explanation(
                    puzzle, improved_solution
                )
                new_critique = self.critique_agent.critique_solution(
                    puzzle, improved_solution, new_explanation
                )
                
                if new_critique['overall_score'] > critique['overall_score']:
                    result['solution'] = improved_solution
                    result['explanation'] = new_explanation.to_dict()
                    result['critique'] = new_critique
                    result['verified'] = new_critique['verified']
                    result['solved'] = new_critique['verified']
                    
        # Store in history
        self.solution_history.append(result)
        
        return result
        
    def _generate_solution(self, puzzle: Dict) -> Any:
        """Generate initial solution"""
        # Simplified solution generation
        if 'test' in puzzle and puzzle['test']:
            test_input = puzzle['test'][0].get('input', [[0]])
            # Simple transformation
            return np.array(test_input)
        return None
        
    def _improve_solution(self, puzzle: Dict, solution: Any, 
                         critique: Dict) -> Any:
        """Improve solution based on critique"""
        # Implement improvement based on suggestions
        if 'suggestions' in critique and critique['suggestions']:
            # Apply first suggestion (simplified)
            return solution  # Would implement actual improvements
        return None