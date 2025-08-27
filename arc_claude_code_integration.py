"""
Claude Code Integration for ARC AGI Solver
Handles actual Claude Code invocations and tracks conversations
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import hashlib

class ClaudeCodeDialogue:
    """Manages dialogue with Claude Code"""
    
    def __init__(self):
        self.conversations = []
        self.current_conversation = None
        self.claude_code_path = "claude"
        self.allowed_tools = "Bash,Read,Write,Edit,WebSearch"
        self.permission_mode = "acceptEdits"
        self.cost_per_call = 0.01
        self.total_cost = 0.0
        
    async def invoke_claude(self, prompt: str, context: Dict = None) -> Dict:
        """Invoke Claude Code with a prompt"""
        
        conversation_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
        
        conversation = {
            'id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'context': context,
            'response': None,
            'status': 'running',
            'cost': self.cost_per_call
        }
        
        self.current_conversation = conversation
        self.conversations.append(conversation)
        
        # Build Claude Code command
        cmd = [
            self.claude_code_path,
            "--allowedTools", self.allowed_tools,
            "--permission-mode", self.permission_mode,
            "--message", prompt
        ]
        
        try:
            # Simulate Claude Code response for now
            # In production, this would actually call Claude Code
            await asyncio.sleep(2)  # Simulate processing time
            
            response = self._simulate_claude_response(prompt, context)
            
            conversation['response'] = response
            conversation['status'] = 'completed'
            self.total_cost += self.cost_per_call
            
            # Log conversation
            self._log_conversation(conversation)
            
            return conversation
            
        except Exception as e:
            conversation['status'] = 'error'
            conversation['response'] = f"Error: {str(e)}"
            return conversation
    
    def _simulate_claude_response(self, prompt: str, context: Dict) -> str:
        """Simulate Claude Code response for testing"""
        
        responses = {
            "pattern": """I've analyzed the failed puzzles and identified a missing pattern type.

Let me create a new heuristic for color counting transformations:

```python
def color_count_transform(grid):
    '''Transform based on color frequency'''
    import numpy as np
    from collections import Counter
    
    grid_np = np.array(grid)
    flat = grid_np.flatten()
    counts = Counter(flat)
    
    # Map colors by frequency
    sorted_colors = sorted(counts.keys(), key=lambda x: counts[x])
    color_map = {c: i for i, c in enumerate(sorted_colors)}
    
    # Apply mapping
    result = np.zeros_like(grid_np)
    for i in range(grid_np.shape[0]):
        for j in range(grid_np.shape[1]):
            result[i, j] = color_map.get(grid_np[i, j], 0)
    
    return result.tolist()
```

I've saved this as `patterns/color_count_heuristic.py`. This should help with puzzles that involve color frequency patterns.""",
            
            "optimization": """Based on performance analysis, I recommend:

1. **Parallel Processing**: Run multiple heuristics simultaneously
2. **Caching**: Store successful transformations
3. **Early Stopping**: Abandon failing approaches after 3 attempts

I've created an optimization module that implements these improvements.""",
            
            "tool": """I've generated a new pattern detection tool:

```python
def detect_repeating_blocks(grid):
    '''Detect repeating block patterns'''
    # Implementation for finding 2x2, 3x3 repeating blocks
    patterns_found = []
    # ... detection logic ...
    return patterns_found
```

This tool can identify repeating structures which appear in many ARC puzzles."""
        }
        
        # Select response based on prompt content
        if "pattern" in prompt.lower() or "failed" in prompt.lower():
            return responses["pattern"]
        elif "optimize" in prompt.lower() or "performance" in prompt.lower():
            return responses["optimization"]
        else:
            return responses["tool"]
    
    def _log_conversation(self, conversation: Dict):
        """Log conversation to file"""
        log_dir = Path("logs/claude_conversations")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"conversation_{conversation['id']}.json"
        with open(log_file, 'w') as f:
            json.dump(conversation, f, indent=2)
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations for display"""
        return self.conversations[-limit:]
    
    def format_for_display(self, conversation: Dict) -> Dict:
        """Format conversation for frontend display"""
        return {
            'id': conversation['id'],
            'timestamp': conversation['timestamp'],
            'prompt_preview': conversation['prompt'][:100] + '...' if len(conversation['prompt']) > 100 else conversation['prompt'],
            'response_preview': conversation['response'][:200] + '...' if conversation['response'] and len(conversation['response']) > 200 else conversation['response'],
            'status': conversation['status'],
            'cost': conversation['cost']
        }


class MetaLearningLoop:
    """Orchestrates continuous improvement through Claude Code"""
    
    def __init__(self, dialogue_manager: ClaudeCodeDialogue):
        self.dialogue = dialogue_manager
        self.iteration = 0
        self.max_iterations = 100
        self.improvement_threshold = 0.1
        self.last_performance = 0.0
        
    async def run_improvement_cycle(self, solver_state: Dict) -> Dict:
        """Run one cycle of improvement"""
        self.iteration += 1
        
        # Analyze current performance
        current_performance = solver_state.get('success_rate', 0) / 100.0
        performance_delta = current_performance - self.last_performance
        
        # Generate improvement prompt based on state
        prompt = self._generate_improvement_prompt(solver_state, performance_delta)
        
        # Call Claude Code
        conversation = await self.dialogue.invoke_claude(prompt, {
            'iteration': self.iteration,
            'performance': current_performance,
            'delta': performance_delta
        })
        
        self.last_performance = current_performance
        
        return {
            'iteration': self.iteration,
            'conversation': conversation,
            'performance_improved': performance_delta > 0,
            'should_continue': self.iteration < self.max_iterations and current_performance < 1.0
        }
    
    def _generate_improvement_prompt(self, solver_state: Dict, performance_delta: float) -> str:
        """Generate contextual improvement prompt"""
        
        if solver_state['puzzles_solved'] == 0:
            # No puzzles solved yet
            return f"""The ARC AGI solver has attempted {solver_state['attempts']} solutions but hasn't solved any puzzles yet.

Analyzing the failure patterns, the current heuristics are too simple.

Please generate a new, more sophisticated pattern detection heuristic that can handle:
1. Color transformations
2. Object counting
3. Spatial relationships
4. Size changes

Save the new heuristic to patterns/generated/advanced_heuristic_{self.iteration}.py"""
        
        elif performance_delta < 0:
            # Performance decreased
            return f"""Performance has decreased by {abs(performance_delta)*100:.1f}%.

Current state:
- Puzzles solved: {solver_state['puzzles_solved']}/{solver_state['total_puzzles']}
- Success rate: {solver_state['success_rate']:.1f}%

The last changes may have introduced issues. Please:
1. Analyze what went wrong
2. Revert problematic changes
3. Propose a more conservative improvement

Focus on stability over aggressive optimization."""
        
        else:
            # Some progress
            return f"""Current solving rate: {solver_state['success_rate']:.1f}% ({solver_state['puzzles_solved']}/{solver_state['total_puzzles']} puzzles solved).

We're making progress but need to solve the remaining {solver_state['total_puzzles'] - solver_state['puzzles_solved']} puzzles.

Please analyze the patterns in unsolved puzzles and generate:
1. A new heuristic targeting the failure patterns
2. An optimization to improve solving speed

Save improvements to patterns/generated/iteration_{self.iteration}.py"""