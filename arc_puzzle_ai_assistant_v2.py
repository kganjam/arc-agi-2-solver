#!/usr/bin/env python3
"""
Enhanced AI Assistant for ARC Puzzle Editor - Version 2
Now with actual function execution and Claude Code integration
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import boto3
import os
from arc_unified_heuristics import unified_heuristics as heuristics_manager
from arc_verification_oracle import verification_oracle

@dataclass
class GridCommand:
    """Represents a parsed grid command"""
    action: str  # 'set_color', 'resize', 'copy', 'fill', 'clear'
    params: Dict[str, Any]
    
class SmartPuzzleAIAssistant:
    """Enhanced AI Assistant with actual function execution"""
    
    # Color name to number mapping
    COLOR_MAP = {
        'black': 0,
        'blue': 1, 
        'red': 2,
        'green': 3,
        'yellow': 4,
        'gray': 5, 'grey': 5,
        'pink': 6, 'magenta': 6,
        'orange': 7,
        'light blue': 8, 'cyan': 8,
        'brown': 9, 'maroon': 9
    }
    
    SYSTEM_PROMPT = """You are an advanced AI assistant for solving ARC (Abstraction and Reasoning Corpus) puzzles.

IMPORTANT INSTRUCTIONS:
1. When asked to execute functions, ALWAYS call them and return real results
2. When analyzing puzzles, use the analyze_pattern and find_transformation functions
3. When asked about heuristics, use get_all_heuristics or get_heuristics_stats
4. When asked to solve, use multiple functions in sequence
5. Be proactive - don't just describe what you would do, actually do it

FUNCTION CALLING FORMAT:
To call a function, use this exact format:
<function_call>
{
  "name": "function_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
</function_call>

You can call multiple functions in sequence. After each function call, I will execute it and give you the result, then you can continue with more analysis or functions.

AVAILABLE CAPABILITIES:
- Grid manipulation: set cells, resize, copy, fill, clear
- Heuristics: retrieve, rank, apply, create, test
- Tools: get, apply, create
- Verification: submit and verify solutions
- Pattern analysis: analyze patterns and find transformations
- Claude Code: generate new solving code when needed

When solving puzzles:
1. First analyze the pattern
2. Search for relevant heuristics
3. Apply promising heuristics
4. Test the solution
5. If needed, create new heuristics or generate code

Be concise but thorough. Execute functions to get real data, don't guess."""
    
    def __init__(self):
        self.current_puzzle = None
        self.output_grid = None
        self.heuristics = []
        self.tools = []
        self.last_applied_heuristic = None
        self.conversation_history = []
        
        # Initialize Bedrock client
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            )
            self.bedrock_available = True
            self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        except Exception as e:
            print(f"Bedrock initialization failed: {e}")
            self.bedrock_available = False
    
    def set_puzzle(self, puzzle: Dict):
        """Set the current puzzle context"""
        self.current_puzzle = puzzle
        # Initialize output grid from test input
        if puzzle and 'test' in puzzle and puzzle['test']:
            test_input = puzzle['test'][0]['input']
            self.output_grid = [row[:] for row in test_input]
    
    def set_output_grid(self, grid: List[List[int]]):
        """Set the output grid"""
        self.output_grid = grid
    
    def set_heuristics(self, heuristics: List[Dict]):
        """Set available heuristics"""
        self.heuristics = heuristics
    
    def set_tools(self, tools: List[Dict]):
        """Set available tools"""
        self.tools = tools
    
    def process_command(self, user_input: str) -> Dict:
        """Process user command with enhanced AI"""
        if not self.bedrock_available:
            # Fallback to basic processing
            return self._basic_process(user_input)
        
        try:
            # Use enhanced Bedrock processing with function execution
            response = self._smart_bedrock_query(user_input)
            return response
        except Exception as e:
            print(f"AI processing error: {e}")
            return {
                "error": str(e),
                "message": "An error occurred processing your request"
            }
    
    def _smart_bedrock_query(self, user_input: str) -> Dict:
        """Enhanced Bedrock query with function execution loop"""
        
        # Build context
        context = self._build_context()
        
        # Start with just the current request
        messages = [
            {
                "role": "user", 
                "content": f"{context}\n\nUser request: {user_input}"
            }
        ]
        
        # Process with function calling loop
        max_iterations = 5
        final_response = ""
        function_results = []
        
        for iteration in range(max_iterations):
            # Query Bedrock
            response_text = self._call_bedrock(messages)
            
            # Check for function calls
            function_calls = self._extract_function_calls(response_text)
            
            if not function_calls:
                # No more functions to call, this is the final response
                final_response = response_text
                break
            
            # Execute functions
            for func_call in function_calls:
                result = self.execute_function(func_call['name'], func_call.get('parameters', {}))
                function_results.append({
                    'function': func_call['name'],
                    'result': result
                })
                
                # Add function result to conversation
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "user",
                    "content": f"Function {func_call['name']} returned: {json.dumps(result)}"
                })
            
            # Continue the loop to let AI process the function results
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": final_response
        })
        
        # Prepare response
        response = {
            "message": final_response,
            "function_results": function_results
        }
        
        # Include updated output grid if modified
        if self.output_grid:
            response["output_grid"] = self.output_grid
        
        return response
    
    def _call_bedrock(self, messages: List[Dict]) -> str:
        """Call Bedrock API with messages"""
        
        # Ensure alternating roles
        clean_messages = []
        last_role = None
        for msg in messages:
            if msg["role"] != last_role:
                clean_messages.append(msg)
                last_role = msg["role"]
        
        # Add system prompt to messages
        messages_with_system = clean_messages
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": messages_with_system,
            "system": self.SYSTEM_PROMPT,
            "temperature": 0.7,
            "top_p": 0.9,
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('content', [{}])[0].get('text', '')
    
    def _format_messages_for_claude(self, messages: List[Dict]) -> str:
        """Format messages for Claude prompt"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                continue  # System message handled separately
            elif msg["role"] == "user":
                prompt += f"\n\nHuman: {msg['content']}"
            elif msg["role"] == "assistant":
                prompt += f"\n\nAssistant: {msg['content']}"
        prompt += "\n\nAssistant:"
        return prompt
    
    def _extract_function_calls(self, text: str) -> List[Dict]:
        """Extract function calls from AI response"""
        function_calls = []
        
        # Look for function call blocks
        pattern = r'<function_call>(.*?)</function_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                func_data = json.loads(match.strip())
                function_calls.append(func_data)
            except json.JSONDecodeError:
                # Try to parse as simple function name
                if '{' not in match:
                    function_calls.append({"name": match.strip(), "parameters": {}})
        
        return function_calls
    
    def _build_context(self) -> str:
        """Build comprehensive context for the AI"""
        context = "CURRENT CONTEXT:\n"
        
        if self.current_puzzle:
            context += f"Puzzle ID: {self.current_puzzle.get('id', 'unknown')}\n"
            
            if 'train' in self.current_puzzle:
                context += f"Training examples: {len(self.current_puzzle['train'])}\n"
                
                # Analyze first example
                if self.current_puzzle['train']:
                    example = self.current_puzzle['train'][0]
                    input_shape = (len(example['input']), len(example['input'][0]))
                    output_shape = (len(example['output']), len(example['output'][0]))
                    context += f"First example: Input {input_shape} → Output {output_shape}\n"
            
            if 'test' in self.current_puzzle and self.current_puzzle['test']:
                test_input = self.current_puzzle['test'][0]['input']
                context += f"Test input size: {len(test_input)}x{len(test_input[0])}\n"
        
        if self.output_grid:
            context += f"Current output grid size: {len(self.output_grid)}x{len(self.output_grid[0])}\n"
        
        context += f"\nAvailable heuristics: {len(heuristics_manager.heuristics)}\n"
        
        return context
    
    def execute_function(self, function_name: str, parameters: Dict) -> Dict:
        """Execute a function call from the AI"""
        try:
            # Grid manipulation functions
            if function_name == "set_output_cell":
                row = parameters.get('row', 0)
                col = parameters.get('col', 0)
                color = parameters.get('color', 0)
                if self.output_grid and 0 <= row < len(self.output_grid) and 0 <= col < len(self.output_grid[0]):
                    self.output_grid[row][col] = color
                    return {"success": True, "message": f"Set cell ({row},{col}) to color {color}"}
                return {"error": "Invalid cell coordinates"}
            
            elif function_name == "resize_output_grid":
                width = parameters.get('width', 3)
                height = parameters.get('height', 3)
                new_grid = [[0 for _ in range(width)] for _ in range(height)]
                if self.output_grid:
                    for i in range(min(height, len(self.output_grid))):
                        for j in range(min(width, len(self.output_grid[0]))):
                            new_grid[i][j] = self.output_grid[i][j]
                self.output_grid = new_grid
                return {"success": True, "message": f"Resized to {height}x{width}"}
            
            elif function_name == "copy_from_input":
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    self.output_grid = [row[:] for row in test_input]
                    return {"success": True, "message": "Copied input to output"}
                return {"error": "No input to copy"}
            
            elif function_name == "clear_output":
                if self.output_grid:
                    for i in range(len(self.output_grid)):
                        for j in range(len(self.output_grid[0])):
                            self.output_grid[i][j] = 0
                    return {"success": True, "message": "Cleared output grid"}
                return {"error": "No output grid"}
            
            # Pattern analysis
            elif function_name == "analyze_pattern":
                return self._analyze_pattern_detailed()
            
            elif function_name == "find_transformation":
                return self._find_transformation()
            
            # Heuristics functions
            elif function_name == "get_all_heuristics":
                all_h = heuristics_manager.get_all_heuristics()
                return {
                    "count": len(all_h),
                    "heuristics": [{"id": h["id"], "name": h["name"]} for h in all_h[:10]]
                }
            
            elif function_name == "get_heuristics_stats":
                stats = heuristics_manager.get_statistics()
                return stats
            
            elif function_name == "search_heuristics":
                query = parameters.get('query', '')
                results = heuristics_manager.search_heuristics(query)
                return {
                    "count": len(results),
                    "results": [{"id": h.id, "name": h.name} for h in results[:5]]
                }
            
            elif function_name == "apply_heuristic":
                h_id = parameters.get('heuristic_id')
                if h_id:
                    puzzle_data = self._get_puzzle_data()
                    result = heuristics_manager.apply_heuristic(h_id, puzzle_data)
                    self.last_applied_heuristic = h_id
                    return result
                return {"error": "No heuristic_id provided"}
            
            elif function_name == "create_heuristic":
                h = heuristics_manager.create_heuristic(
                    name=parameters.get('name', 'New Heuristic'),
                    description=parameters.get('description', ''),
                    pattern_type=parameters.get('pattern_type', 'unknown'),
                    conditions=parameters.get('conditions', []),
                    transformations=parameters.get('transformations', []),
                    complexity=parameters.get('complexity', 1),
                    tags=parameters.get('tags', [])
                )
                return {
                    "success": True,
                    "heuristic_id": h.id,
                    "message": f"Created heuristic: {h.name}"
                }
            
            # Verification
            elif function_name == "submit_solution":
                if not self.current_puzzle or not self.output_grid:
                    return {"error": "No puzzle or solution"}
                
                result = verification_oracle.verify_solution(
                    puzzle_id=self.current_puzzle.get('id', 'unknown'),
                    submitted_output=self.output_grid
                )
                return result
            
            elif function_name == "verify_solution":
                grid = parameters.get('solution_grid', self.output_grid)
                if not grid:
                    return {"error": "No solution to verify"}
                
                result = verification_oracle.verify_solution(
                    puzzle_id=self.current_puzzle.get('id', 'unknown') if self.current_puzzle else 'unknown',
                    submitted_output=grid
                )
                return result
            
            # Get information
            elif function_name == "get_cell_color":
                grid_type = parameters.get('grid_type', 'output')
                row = parameters.get('row', 0)
                col = parameters.get('col', 0)
                
                if grid_type == 'output' and self.output_grid:
                    if 0 <= row < len(self.output_grid) and 0 <= col < len(self.output_grid[0]):
                        return {"color": self.output_grid[row][col]}
                elif grid_type == 'test' and self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    if 0 <= row < len(test_input) and 0 <= col < len(test_input[0]):
                        return {"color": test_input[row][col]}
                
                return {"error": "Invalid cell or grid type"}
            
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_pattern_detailed(self) -> Dict:
        """Detailed pattern analysis"""
        if not self.current_puzzle or 'train' not in self.current_puzzle:
            return {"error": "No training examples to analyze"}
        
        train = self.current_puzzle['train']
        analysis = {
            "num_examples": len(train),
            "transformations": [],
            "colors": set(),
            "size_changes": []
        }
        
        for i, example in enumerate(train):
            in_grid = example['input']
            out_grid = example['output']
            
            # Size analysis
            in_shape = (len(in_grid), len(in_grid[0]))
            out_shape = (len(out_grid), len(out_grid[0]))
            
            if in_shape != out_shape:
                analysis["size_changes"].append({
                    "example": i+1,
                    "from": in_shape,
                    "to": out_shape
                })
            
            # Color analysis
            in_colors = set(cell for row in in_grid for cell in row)
            out_colors = set(cell for row in out_grid for cell in row)
            analysis["colors"] |= in_colors | out_colors
            
            # Check for simple transformations
            if in_shape == out_shape:
                # Check if it's a color mapping
                color_map = {}
                is_color_mapping = True
                for r in range(len(in_grid)):
                    for c in range(len(in_grid[0])):
                        in_c = in_grid[r][c]
                        out_c = out_grid[r][c]
                        if in_c in color_map:
                            if color_map[in_c] != out_c:
                                is_color_mapping = False
                                break
                        else:
                            color_map[in_c] = out_c
                    if not is_color_mapping:
                        break
                
                if is_color_mapping and color_map:
                    analysis["transformations"].append({
                        "type": "color_mapping",
                        "example": i+1,
                        "mapping": color_map
                    })
        
        analysis["colors"] = list(analysis["colors"])
        return analysis
    
    def _find_transformation(self) -> Dict:
        """Find transformation pattern from examples"""
        analysis = self._analyze_pattern_detailed()
        
        if "error" in analysis:
            return analysis
        
        # Suggest transformation based on analysis
        suggestions = []
        
        if analysis["size_changes"]:
            suggestions.append("Size transformation detected - consider cropping or scaling")
        
        if analysis["transformations"]:
            for trans in analysis["transformations"]:
                if trans["type"] == "color_mapping":
                    suggestions.append(f"Color mapping in example {trans['example']}: {trans['mapping']}")
        
        if not suggestions:
            suggestions.append("Complex transformation - may need custom heuristic")
        
        return {
            "analysis": analysis,
            "suggestions": suggestions
        }
    
    def _get_puzzle_data(self) -> Dict:
        """Get puzzle data for heuristic application"""
        if not self.current_puzzle:
            return {}
        
        # Extract features
        colors = set()
        if 'train' in self.current_puzzle:
            for example in self.current_puzzle['train']:
                for row in example['input']:
                    colors.update(row)
                for row in example['output']:
                    colors.update(row)
        
        return {
            'puzzle_id': self.current_puzzle.get('id', 'unknown'),
            'colors': list(colors),
            'size_change': False,  # Will be determined by analysis
            'train': self.current_puzzle.get('train', []),
            'test': self.current_puzzle.get('test', [])
        }
    
    def _basic_process(self, user_input: str) -> Dict:
        """Basic processing without Bedrock"""
        # Simple pattern matching for common requests
        user_input = user_input.lower().strip()
        
        if "grid size" in user_input or "what size" in user_input:
            if self.output_grid:
                return {
                    "message": f"Output grid size: {len(self.output_grid)}x{len(self.output_grid[0])}"
                }
            return {"message": "No output grid loaded"}
        
        elif "heuristic" in user_input and "count" in user_input:
            count = len(heuristics_manager.heuristics)
            return {"message": f"There are {count} heuristics available"}
        
        elif "analyze" in user_input:
            analysis = self._analyze_pattern_detailed()
            return {"message": json.dumps(analysis, indent=2)}
        
        else:
            return {
                "message": "Basic mode: I can help with grid sizes, heuristic counts, and pattern analysis. For advanced features, Bedrock connection is required."
            }

# Create a wrapper class that's compatible with the existing system
class PuzzleAIAssistant(SmartPuzzleAIAssistant):
    """Wrapper for backward compatibility"""
    pass