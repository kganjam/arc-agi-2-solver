"""
Multi-Agent Dialogue System for ARC AGI
Implements collaborative problem solving through agent debate
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import numpy as np
import copy

class AgentRole(Enum):
    PATTERN_ANALYST = "Pattern Analyst"
    TRANSFORM_SPECIALIST = "Transform Specialist"
    VALIDATOR = "Validator"
    STRATEGIST = "Strategist"
    TOOL_BUILDER = "Tool Builder"
    META_LEARNER = "Meta Learner"

class Agent:
    """Base agent class for multi-agent system"""
    
    def __init__(self, role: AgentRole, agent_id: str):
        self.role = role
        self.id = agent_id
        self.confidence = 0.7
        self.messages_sent = 0
        self.successful_proposals = 0
        self.knowledge_base = {}
        
    async def analyze_puzzle(self, puzzle: Dict) -> Dict:
        """Analyze puzzle from agent's perspective"""
        analysis = {
            'agent_id': self.id,
            'role': self.role.value,
            'timestamp': datetime.now().isoformat(),
            'confidence': self.confidence
        }
        
        if self.role == AgentRole.PATTERN_ANALYST:
            analysis['patterns'] = self._find_patterns(puzzle)
        elif self.role == AgentRole.TRANSFORM_SPECIALIST:
            analysis['transformations'] = self._suggest_transforms(puzzle)
        elif self.role == AgentRole.VALIDATOR:
            analysis['validation'] = self._validate_approach(puzzle)
        elif self.role == AgentRole.STRATEGIST:
            analysis['strategy'] = self._propose_strategy(puzzle)
        elif self.role == AgentRole.TOOL_BUILDER:
            analysis['tools_needed'] = self._identify_tools(puzzle)
        elif self.role == AgentRole.META_LEARNER:
            analysis['meta_insights'] = self._extract_meta_patterns(puzzle)
            
        return analysis
    
    def _find_patterns(self, puzzle: Dict) -> List[str]:
        """Pattern Analyst: Find visual patterns"""
        patterns = []
        
        if 'train' in puzzle and len(puzzle['train']) > 0:
            example = puzzle['train'][0]
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check for symmetry
            if np.array_equal(input_grid, np.fliplr(input_grid)):
                patterns.append("horizontal_symmetry")
            if np.array_equal(input_grid, np.flipud(input_grid)):
                patterns.append("vertical_symmetry")
                
            # Check for size changes
            if input_grid.shape != output_grid.shape:
                patterns.append(f"size_change_{input_grid.shape}_to_{output_grid.shape}")
                
            # Check for color patterns
            unique_in = set(input_grid.flatten())
            unique_out = set(output_grid.flatten())
            if unique_in != unique_out:
                patterns.append(f"color_mapping_{len(unique_in)}_to_{len(unique_out)}")
                
            # Check for repetition
            if output_grid.shape[0] % input_grid.shape[0] == 0:
                patterns.append("possible_tiling")
                
        return patterns
    
    def _suggest_transforms(self, puzzle: Dict) -> List[str]:
        """Transform Specialist: Suggest transformations"""
        transforms = []
        
        if 'train' in puzzle and len(puzzle['train']) > 0:
            example = puzzle['train'][0]
            input_shape = np.array(example['input']).shape
            output_shape = np.array(example['output']).shape
            
            if input_shape == output_shape:
                transforms.extend([
                    "color_replacement",
                    "pattern_fill",
                    "rotation",
                    "reflection"
                ])
            elif output_shape[0] < input_shape[0]:
                transforms.extend([
                    "crop",
                    "extract_region",
                    "compress"
                ])
            else:
                transforms.extend([
                    "tile",
                    "expand",
                    "replicate"
                ])
                
        return transforms
    
    def _validate_approach(self, puzzle: Dict) -> Dict:
        """Validator: Check proposed solutions"""
        validation = {
            'feasible': True,
            'concerns': [],
            'suggestions': []
        }
        
        if 'train' in puzzle:
            if len(puzzle['train']) < 2:
                validation['concerns'].append("Limited training examples")
                validation['suggestions'].append("Use careful generalization")
                
            # Check consistency across examples
            shapes_consistent = True
            if len(puzzle['train']) > 1:
                first_shape = np.array(puzzle['train'][0]['input']).shape
                for example in puzzle['train'][1:]:
                    if np.array(example['input']).shape != first_shape:
                        shapes_consistent = False
                        break
                        
                if not shapes_consistent:
                    validation['concerns'].append("Inconsistent input shapes")
                    validation['suggestions'].append("Consider variable-size handling")
                    
        return validation
    
    def _propose_strategy(self, puzzle: Dict) -> Dict:
        """Strategist: High-level approach"""
        strategy = {
            'approach': 'pattern_matching',
            'priority': [],
            'backup_plans': []
        }
        
        # Analyze complexity
        if 'train' in puzzle and len(puzzle['train']) > 0:
            example = puzzle['train'][0]
            complexity = len(set(np.array(example['input']).flatten()))
            
            if complexity <= 3:
                strategy['approach'] = 'simple_mapping'
                strategy['priority'] = ['color_rules', 'counting', 'position']
            elif complexity <= 5:
                strategy['approach'] = 'pattern_recognition'
                strategy['priority'] = ['shapes', 'symmetry', 'boundaries']
            else:
                strategy['approach'] = 'complex_analysis'
                strategy['priority'] = ['segmentation', 'object_detection', 'relationships']
                
            strategy['backup_plans'] = [
                'brute_force_patterns',
                'statistical_analysis',
                'machine_learning'
            ]
            
        return strategy
    
    def _identify_tools(self, puzzle: Dict) -> List[str]:
        """Tool Builder: Identify needed tools"""
        tools = []
        
        if 'train' in puzzle and len(puzzle['train']) > 0:
            example = puzzle['train'][0]
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Basic tools always needed
            tools.extend(['grid_analyzer', 'color_counter'])
            
            # Specific tools based on transformation
            if input_grid.shape != output_grid.shape:
                tools.append('size_transformer')
                
            if len(set(input_grid.flatten())) != len(set(output_grid.flatten())):
                tools.append('color_mapper')
                
            # Advanced tools
            if input_grid.shape[0] > 10 or input_grid.shape[1] > 10:
                tools.append('large_grid_handler')
                
            if np.any(input_grid == 0):
                tools.append('background_detector')
                
        return tools
    
    def _extract_meta_patterns(self, puzzle: Dict) -> Dict:
        """Meta Learner: Extract high-level insights"""
        meta = {
            'puzzle_type': 'unknown',
            'difficulty': 0.5,
            'similar_to': [],
            'key_insights': []
        }
        
        if 'train' in puzzle and len(puzzle['train']) > 0:
            # Classify puzzle type
            example = puzzle['train'][0]
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape == output_grid.shape:
                if np.sum(input_grid != output_grid) < input_grid.size * 0.3:
                    meta['puzzle_type'] = 'local_modification'
                else:
                    meta['puzzle_type'] = 'global_transformation'
            else:
                meta['puzzle_type'] = 'size_transformation'
                
            # Estimate difficulty
            complexity_factors = [
                len(set(input_grid.flatten())) / 10,  # Color complexity
                max(input_grid.shape) / 30,  # Size complexity
                len(puzzle['train']) / 5,  # Example complexity
            ]
            meta['difficulty'] = min(1.0, np.mean(complexity_factors))
            
            # Key insights
            if np.any(output_grid == 0) and not np.any(input_grid == 0):
                meta['key_insights'].append('Output introduces background')
                
            if output_grid.shape[0] == output_grid.shape[1] and input_grid.shape[0] != input_grid.shape[1]:
                meta['key_insights'].append('Output becomes square')
                
        return meta

class DialogueManager:
    """Manages multi-agent dialogue and consensus building"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.dialogue_history = []
        self.consensus = None
        self.proposals = []
        
    def _create_agents(self) -> List[Agent]:
        """Create the agent team"""
        return [
            Agent(AgentRole.PATTERN_ANALYST, "PA001"),
            Agent(AgentRole.TRANSFORM_SPECIALIST, "TS001"),
            Agent(AgentRole.VALIDATOR, "V001"),
            Agent(AgentRole.STRATEGIST, "S001"),
            Agent(AgentRole.TOOL_BUILDER, "TB001"),
            Agent(AgentRole.META_LEARNER, "ML001")
        ]
        
    async def conduct_dialogue(self, puzzle: Dict, max_rounds: int = 5) -> Dict:
        """Conduct multi-agent dialogue to solve puzzle"""
        self.dialogue_history = []
        self.proposals = []
        
        for round_num in range(max_rounds):
            round_messages = []
            
            # Each agent analyzes the puzzle
            for agent in self.agents:
                analysis = await agent.analyze_puzzle(puzzle)
                round_messages.append(analysis)
                
            self.dialogue_history.append({
                'round': round_num + 1,
                'messages': round_messages
            })
            
            # Build proposals from analyses
            proposal = self._synthesize_proposal(round_messages)
            self.proposals.append(proposal)
            
            # Check for consensus
            if self._check_consensus(self.proposals):
                self.consensus = proposal
                break
                
        return {
            'consensus': self.consensus,
            'dialogue_history': self.dialogue_history,
            'proposals': self.proposals,
            'rounds': len(self.dialogue_history)
        }
        
    def _synthesize_proposal(self, messages: List[Dict]) -> Dict:
        """Synthesize a proposal from agent messages"""
        proposal = {
            'patterns': [],
            'transformations': [],
            'strategy': None,
            'tools': [],
            'confidence': 0
        }
        
        for msg in messages:
            if 'patterns' in msg:
                proposal['patterns'].extend(msg['patterns'])
            if 'transformations' in msg:
                proposal['transformations'].extend(msg['transformations'])
            if 'strategy' in msg:
                proposal['strategy'] = msg['strategy']
            if 'tools_needed' in msg:
                proposal['tools'].extend(msg['tools_needed'])
                
        # Calculate aggregate confidence
        confidences = [msg.get('confidence', 0.5) for msg in messages]
        proposal['confidence'] = np.mean(confidences)
        
        return proposal
        
    def _check_consensus(self, proposals: List[Dict]) -> bool:
        """Check if agents have reached consensus"""
        if len(proposals) < 2:
            return False
            
        # Compare last two proposals
        last = proposals[-1]
        prev = proposals[-2]
        
        # Check if key elements are stable
        patterns_stable = set(last['patterns']) == set(prev['patterns'])
        transforms_stable = set(last['transformations']) == set(prev['transformations'])
        confidence_stable = abs(last['confidence'] - prev['confidence']) < 0.1
        
        return patterns_stable and transforms_stable and confidence_stable

class MultiAgentSolver:
    """Integrates multi-agent dialogue with solving"""
    
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        self.solution_cache = {}
        self.performance_metrics = {
            'puzzles_analyzed': 0,
            'consensus_reached': 0,
            'average_rounds': 0
        }
        
    async def solve_with_dialogue(self, puzzle: Dict) -> Dict:
        """Solve puzzle using multi-agent dialogue"""
        puzzle_id = puzzle.get('id', 'unknown')
        
        # Check cache
        if puzzle_id in self.solution_cache:
            return self.solution_cache[puzzle_id]
            
        # Conduct dialogue
        dialogue_result = await self.dialogue_manager.conduct_dialogue(puzzle)
        
        # Update metrics
        self.performance_metrics['puzzles_analyzed'] += 1
        if dialogue_result['consensus']:
            self.performance_metrics['consensus_reached'] += 1
        self.performance_metrics['average_rounds'] = (
            (self.performance_metrics['average_rounds'] * 
             (self.performance_metrics['puzzles_analyzed'] - 1) +
             dialogue_result['rounds']) / 
            self.performance_metrics['puzzles_analyzed']
        )
        
        # Generate solution based on consensus
        solution = await self._generate_solution(puzzle, dialogue_result)
        
        # Cache result
        self.solution_cache[puzzle_id] = solution
        
        return solution
        
    async def _generate_solution(self, puzzle: Dict, dialogue_result: Dict) -> Dict:
        """Generate solution from dialogue consensus"""
        if not dialogue_result['consensus']:
            return {
                'solved': False,
                'solution': None,
                'reason': 'No consensus reached'
            }
            
        consensus = dialogue_result['consensus']
        
        # Apply the consensus strategy
        if 'test' in puzzle and len(puzzle['test']) > 0:
            test_input = puzzle['test'][0]['input']
            
            # Simple implementation - would be replaced with actual transformation
            # For now, return a modified version based on patterns
            solution_grid = copy.deepcopy(test_input)
            
            # Apply some transformation based on consensus
            if 'color_replacement' in consensus['transformations']:
                # Simple color swap
                solution_grid = [[1 if cell == 0 else cell for cell in row] 
                                for row in solution_grid]
                
            return {
                'solved': True,
                'solution': solution_grid,
                'confidence': consensus['confidence'],
                'strategy_used': consensus.get('strategy', {}).get('approach', 'unknown')
            }
            
        return {
            'solved': False,
            'solution': None,
            'reason': 'No test input provided'
        }
        
    def get_agent_insights(self) -> Dict:
        """Get insights from all agents"""
        insights = {}
        
        for agent in self.dialogue_manager.agents:
            insights[agent.id] = {
                'role': agent.role.value,
                'confidence': agent.confidence,
                'messages_sent': agent.messages_sent,
                'successful_proposals': agent.successful_proposals,
                'knowledge_base_size': len(agent.knowledge_base)
            }
            
        return insights