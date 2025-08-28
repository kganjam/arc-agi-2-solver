#!/usr/bin/env python3
"""
Enhanced AI Assistant for ARC Puzzle Editor - Version 2
Now with actual function execution and Claude Code integration
"""

import re
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import boto3
import os
import asyncio
from arc_unified_heuristics import unified_heuristics as heuristics_manager
from arc_verification_oracle import verification_oracle
from arc_claude_code_integration import ClaudeCodeDialogue

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
    
    # Progress callback for streaming messages
    progress_callback = None
    
    SYSTEM_PROMPT = """You are an expert ARC puzzle solver. You MUST actively solve puzzles, not just analyze them.

🎯 PRIME DIRECTIVE: When asked to solve a puzzle, you MUST:
1. Immediately use get_input_grid to see the test input
2. Use analyze_pattern to understand the transformation
3. Apply the most likely transformation (don't wait for high confidence)
4. Submit the solution to check if it's correct
5. If wrong, try the next most likely approach
6. Keep trying different approaches until you solve it

⚡ ACTION-FIRST APPROACH:
- When user says "solve", "try", "submit", or asks for a solution → IMMEDIATELY start solving
- Don't just analyze - APPLY transformations and SUBMIT solutions
- Even with 60% confidence, TRY IT! You can always try another approach
- Use check_solution after each attempt to verify correctness

📋 SOLVING WORKFLOW:
1. get_input_grid → See the test input
2. analyze_pattern → Understand the pattern (but don't stop here!)
3. Based on analysis, immediately try:
   - If size changed by factor N → apply_cell_expansion with factor N
   - If colors change → apply_color_mapping
   - If grid scales → apply_grid_scaling
   - If pattern unclear → copy_grid to desired size and modify
4. check_solution → Verify if correct
5. submit_solution → Submit your answer
6. If wrong, try next approach immediately

🔧 KEY FUNCTIONS TO USE:
- get_input_grid: Get test input (ALWAYS start with this)
- analyze_pattern: Quick pattern check (don't overthink)
- apply_cell_expansion: For N×N cell scaling
- apply_grid_scaling: For uniform scaling
- copy_grid: Tile input to any size
- resize_output_grid: Change output dimensions
- check_solution: Verify correctness
- submit_solution: Submit final answer
- detect_pattern_type: Get pattern suggestions
- suggest_next_step: Get action recommendations

🎮 PATTERN SHORTCUTS:
- 2×2 → 6×6? → apply_cell_expansion(factor=3)
- 3×3 → 9×9? → apply_cell_expansion(factor=3) or apply_grid_scaling(factor=3)
- Colors inverted? → apply_color_mapping with inverse map
- Pattern repeats? → copy_grid to larger size

⚠️ IMPORTANT RULES:
- NEVER give up after one attempt
- NEVER just analyze without trying solutions
- ALWAYS submit solutions to test them
- If confidence < 80%, still TRY IT
- Keep attempting until solved or user stops you

FUNCTION FORMAT:
<function_call>
{"name": "function_name", "parameters": {"param": "value"}}
</function_call>

You can chain multiple functions. Be aggressive in trying solutions!"""
    
    def __init__(self, progress_callback=None):
        self.current_puzzle = None
        self.output_grid = None
        self.heuristics = []
        self.tools = []
        self.last_applied_heuristic = None
        self.conversation_history = []
        self.max_history_size = 10  # Limit conversation history to prevent memory issues
        self.progress_callback = progress_callback  # For streaming progress messages
        
        # Initialize Bedrock client
        try:
            # Check for AWS credentials
            if not os.environ.get('AWS_ACCESS_KEY_ID') and not os.environ.get('AWS_PROFILE'):
                print("⚠️ AWS credentials not configured. Bedrock features disabled.")
                print("  To enable Bedrock, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                self.bedrock_available = False
                self.bedrock = None
            else:
                self.bedrock = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                )
                # Skip connection test - just try to use it when needed
                self.bedrock_available = True
                self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
                print("✅ Bedrock client initialized (will test on first use)")
        except Exception as e:
            print(f"⚠️ Bedrock initialization failed: {e}")
            self.bedrock_available = False
            self.bedrock = None
        
        # Initialize Claude Code integration
        self.claude_dialogue = ClaudeCodeDialogue()
        self.claude_dialogue.simulate_mode = False  # Use actual Claude Code
    
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
        
        # Try basic processing first for common requests
        # This avoids Bedrock for simple queries that we can handle directly
        basic_result = self._basic_process(user_input)
        if basic_result and "Basic mode:" not in basic_result.get("message", ""):
            # Basic processing handled the request
            return basic_result
        
        # If Bedrock is not available, return the basic result
        if not self.bedrock_available:
            return basic_result
        
        try:
            # Use enhanced Bedrock processing with function execution
            response = self._smart_bedrock_query(user_input)
            return response
        except Exception as e:
            print(f"AI processing error: {e}")
            # Fall back to basic result if Bedrock fails
            return basic_result if basic_result else {
                "error": str(e),
                "message": "An error occurred processing your request"
            }
    
    def _smart_bedrock_query(self, user_input: str) -> Dict:
        """Enhanced Bedrock query with function execution loop"""
        
        try:
            # Build context
            context = self._build_context()
            
            # Send progress about starting Bedrock query
            self._send_progress("🤖 Starting Bedrock AI query...", "bedrock_start")
            
            # Start with just the current request
            messages = [
                {
                    "role": "user", 
                    "content": f"{context}\n\nUser request: {user_input}"
                }
            ]
            
            # Send the prompt being used
            self._send_progress(f"📝 Prompt: {user_input[:200]}...", "bedrock_prompt")
            
            # Process with function calling loop
            max_iterations = 3  # Reduced to prevent hanging
            final_response = ""
            function_results = []
            
            for iteration in range(max_iterations):
                try:
                    # Send progress about Bedrock thinking
                    self._send_progress(f"🧠 Bedrock thinking (iteration {iteration + 1}/{max_iterations})...", "bedrock_thinking")
                    
                    # Query Bedrock with timeout
                    response_text = self._call_bedrock(messages)
                    
                    # Send Bedrock's response
                    self._send_progress(f"💡 Bedrock response: {response_text[:300]}...", "bedrock_response")
                    
                    # Check for function calls
                    function_calls = self._extract_function_calls(response_text)
                    
                    if not function_calls:
                        # No more functions to call, this is the final response
                        final_response = response_text
                        break
                    
                    # Limit function calls per iteration
                    if len(function_calls) > 3:
                        function_calls = function_calls[:3]
                    
                    # Execute functions
                    for func_call in function_calls:
                        try:
                            result = self.execute_function(func_call['name'], func_call.get('parameters', {}))
                            function_results.append({
                                'function': func_call['name'],
                                'result': result
                            })
                        except Exception as e:
                            # Log function execution error but continue
                            function_results.append({
                                'function': func_call['name'],
                                'result': {'error': str(e)}
                            })
                        
                        # Add function result to conversation
                        messages.append({
                            "role": "assistant",
                            "content": response_text[:1000]  # Limit message size
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Function {func_call['name']} returned: {json.dumps(result)[:500]}"  # Limit result size
                        })
                    
                    # Prevent infinite loops - if we're at max iterations, stop
                    if iteration == max_iterations - 1:
                        final_response = response_text if not final_response else final_response
                        break
                        
                except Exception as e:
                    # Handle Bedrock query errors
                    print(f"Error in Bedrock query iteration {iteration}: {e}")
                    final_response = f"I encountered an error processing your request: {str(e)}"
                    break
            
            # Update conversation history (limit size to prevent memory issues)
            if len(self.conversation_history) > self.max_history_size:
                self.conversation_history = self.conversation_history[-self.max_history_size:]
            
            self.conversation_history.append({
                "role": "user",
                "content": user_input[:500]  # Limit input size
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": final_response[:500] if final_response else "No response generated"  # Limit response size
            })
            
            # Prepare response
            response = {
                "message": final_response if final_response else "I couldn't generate a response",
                "function_results": function_results
            }
            
            # Include updated output grid if modified
            if self.output_grid:
                response["output_grid"] = self.output_grid
            
            return response
            
        except Exception as e:
            # Catch-all error handler
            print(f"Critical error in _smart_bedrock_query: {e}")
            return {
                "message": f"An error occurred: {str(e)}",
                "function_results": [],
                "error": str(e)
            }
    
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
                    self._send_progress(f"📚 Retrieving heuristic: {h_id}", "heuristic_retrieval")
                    puzzle_data = self._get_puzzle_data()
                    self._send_progress(f"🔧 Applying heuristic: {h_id}", "heuristic_application")
                    result = heuristics_manager.apply_heuristic(h_id, puzzle_data)
                    self.last_applied_heuristic = h_id
                    self._send_progress(f"✅ Heuristic {h_id} applied successfully", "heuristic_success")
                    return result
                return {"error": "No heuristic_id provided"}
            
            elif function_name == "create_heuristic":
                name = parameters.get('name', 'New Heuristic')
                self._send_progress(f"🆕 Creating new heuristic: {name}", "heuristic_creation")
                h = heuristics_manager.create_heuristic(
                    name=name,
                    description=parameters.get('description', ''),
                    pattern_type=parameters.get('pattern_type', 'unknown'),
                    conditions=parameters.get('conditions', []),
                    transformations=parameters.get('transformations', []),
                    complexity=parameters.get('complexity', 1),
                    tags=parameters.get('tags', [])
                )
                self._send_progress(f"✅ Heuristic '{h.name}' created with ID: {h.id}", "heuristic_created")
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
            
            # Grid transformation functions
            elif function_name == "apply_cell_expansion":
                scaling_factor = parameters.get('scaling_factor', 3)
                return self._apply_cell_expansion(scaling_factor)
            
            elif function_name == "apply_grid_scaling":
                factor = parameters.get('factor', 2)
                return self._apply_grid_scaling(factor)
            
            elif function_name == "apply_color_mapping":
                color_map = parameters.get('color_map', {})
                return self._apply_color_mapping(color_map)
            
            # Claude Code tool generation
            elif function_name == "generate_tool_with_claude":
                tool_desc = parameters.get('description', '')
                tool_name = parameters.get('name', 'new_tool')
                self._send_progress(f"🛠️ Invoking Claude Code to create tool: {tool_name}", "tool_creation_start")
                self._send_progress(f"📝 Tool description: {tool_desc[:100]}...", "tool_description")
                result = self._generate_tool_with_claude(tool_desc, tool_name)
                if result.get('success'):
                    self._send_progress(f"✅ Tool '{tool_name}' created successfully!", "tool_created")
                else:
                    self._send_progress(f"❌ Failed to create tool: {result.get('message', 'Unknown error')}", "tool_creation_failed")
                return result
            
            elif function_name == "list_generated_tools":
                return self._list_generated_tools()
            
            elif function_name == "invoke_generated_tool":
                tool_name = parameters.get('tool_name')
                tool_params = parameters.get('parameters', {})
                return self._invoke_generated_tool(tool_name, tool_params)
            
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
            
            elif function_name == "detect_pattern_type":
                return self._detect_pattern_type()
            
            elif function_name == "suggest_next_step":
                return self._suggest_next_step()
            
            # Additional grid manipulation functions
            elif function_name == "get_input_grid":
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    return {
                        "success": True,
                        "grid": test_input,
                        "height": len(test_input),
                        "width": len(test_input[0]) if test_input else 0
                    }
                return {"error": "No test input available"}
            
            elif function_name == "get_test_output":
                if self.output_grid:
                    return {
                        "success": True,
                        "grid": self.output_grid,
                        "height": len(self.output_grid),
                        "width": len(self.output_grid[0]) if self.output_grid else 0
                    }
                return {"error": "No output grid available"}
            
            elif function_name == "copy_grid":
                src_top = parameters.get('src_top_left', [0, 0])
                dest_top = parameters.get('dest_top_left', [0, 0])
                dest_bottom = parameters.get('dest_bottom_right')
                
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    src_height = len(test_input)
                    src_width = len(test_input[0]) if test_input else 0
                    
                    # Determine destination size
                    if dest_bottom:
                        dest_height = dest_bottom[0] + 1
                        dest_width = dest_bottom[1] + 1
                    else:
                        dest_height = src_height
                        dest_width = src_width
                    
                    # Create or resize output grid
                    self.output_grid = [[0 for _ in range(dest_width)] for _ in range(dest_height)]
                    
                    # Copy and tile the input pattern
                    for i in range(dest_height):
                        for j in range(dest_width):
                            src_i = i % src_height
                            src_j = j % src_width
                            self.output_grid[i][j] = test_input[src_i][src_j]
                    
                    return {
                        "success": True,
                        "message": f"Copied and tiled {src_height}x{src_width} to {dest_height}x{dest_width}",
                        "grid": self.output_grid
                    }
                return {"error": "No input grid to copy"}
            
            elif function_name == "check_solution":
                if not self.output_grid:
                    return {"error": "No solution to check"}
                
                # Check if we have the expected output to compare
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_data = self.current_puzzle['test'][0]
                    if 'output' in test_data:
                        expected = test_data['output']
                        if self._grids_equal(self.output_grid, expected):
                            return {
                                "success": True,
                                "correct": True,
                                "message": "Solution is correct!"
                            }
                        else:
                            differences = self._find_differences(self.output_grid, expected)
                            return {
                                "success": True,
                                "correct": False,
                                "message": "Solution does not match expected output",
                                "differences": differences
                            }
                    else:
                        # No expected output available, just validate format
                        return {
                            "success": True,
                            "message": "No verification data available for this puzzle",
                            "grid_valid": self._validate_grid_format(self.output_grid)
                        }
                return {"error": "Cannot check solution"}
            
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_pattern_detailed(self) -> Dict:
        """Detailed pattern analysis with enhanced grid scaling detection"""
        if not self.current_puzzle or 'train' not in self.current_puzzle:
            return {"error": "No training examples to analyze"}
        
        train = self.current_puzzle['train']
        analysis = {
            "num_examples": len(train),
            "transformations": [],
            "colors": set(),
            "size_changes": [],
            "scaling_patterns": [],
            "cell_expansion_detected": False,
            "scaling_factor": None
        }
        
        for i, example in enumerate(train):
            in_grid = example['input']
            out_grid = example['output']
            
            # Size analysis
            in_shape = (len(in_grid), len(in_grid[0]))
            out_shape = (len(out_grid), len(out_grid[0]))
            
            if in_shape != out_shape:
                size_change = {
                    "example": i+1,
                    "from": in_shape,
                    "to": out_shape
                }
                
                # Check for scaling relationships
                height_factor = out_shape[0] / in_shape[0] if in_shape[0] > 0 else 0
                width_factor = out_shape[1] / in_shape[1] if in_shape[1] > 0 else 0
                
                if height_factor == width_factor and height_factor == int(height_factor) and height_factor > 1:
                    size_change["scaling_factor"] = int(height_factor)
                    size_change["is_uniform_scaling"] = True
                    analysis["scaling_factor"] = int(height_factor)
                    
                    # Check for cell expansion pattern
                    if self._is_cell_expansion(in_grid, out_grid, int(height_factor)):
                        analysis["cell_expansion_detected"] = True
                        analysis["scaling_patterns"].append({
                            "type": "cell_expansion",
                            "example": i+1,
                            "factor": int(height_factor),
                            "pattern": f"Each {in_shape[0]}x{in_shape[1]} cell becomes {int(height_factor)}x{int(height_factor)} block"
                        })
                else:
                    size_change["is_uniform_scaling"] = False
                    size_change["height_factor"] = height_factor
                    size_change["width_factor"] = width_factor
                
                analysis["size_changes"].append(size_change)
            
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
        """Find transformation pattern from examples with enhanced scaling detection"""
        analysis = self._analyze_pattern_detailed()
        
        if "error" in analysis:
            return analysis
        
        # Suggest transformation based on analysis
        suggestions = []
        recommended_functions = []
        
        # Check for cell expansion pattern first
        if analysis.get("cell_expansion_detected"):
            factor = analysis.get("scaling_factor")
            suggestions.append(f"🎯 CELL EXPANSION DETECTED: Each input cell becomes {factor}x{factor} block")
            suggestions.append(f"Recommended: Use apply_cell_expansion with factor={factor}")
            recommended_functions.append({
                "function": "apply_cell_expansion",
                "parameters": {"scaling_factor": factor},
                "confidence": 0.95
            })
        
        # Check for other size transformations
        elif analysis["size_changes"]:
            for change in analysis["size_changes"]:
                if change.get("is_uniform_scaling"):
                    factor = change["scaling_factor"]
                    suggestions.append(f"Uniform scaling detected: {change['from']} → {change['to']} (factor: {factor})")
                    recommended_functions.append({
                        "function": "apply_grid_scaling", 
                        "parameters": {"factor": factor},
                        "confidence": 0.8
                    })
                else:
                    suggestions.append(f"Non-uniform size change: {change['from']} → {change['to']}")
                    suggestions.append("Consider custom cropping, padding, or asymmetric scaling")
        
        # Check for color transformations
        if analysis["transformations"]:
            for trans in analysis["transformations"]:
                if trans["type"] == "color_mapping":
                    suggestions.append(f"Color mapping in example {trans['example']}: {trans['mapping']}")
                    recommended_functions.append({
                        "function": "apply_color_mapping",
                        "parameters": {"color_map": trans['mapping']},
                        "confidence": 0.9
                    })
        
        # Check for scaling patterns
        if analysis.get("scaling_patterns"):
            for pattern in analysis["scaling_patterns"]:
                suggestions.append(f"📏 {pattern['pattern']} (Example {pattern['example']})")
        
        if not suggestions:
            suggestions.append("Complex transformation - may need custom heuristic or additional analysis")
            suggestions.append("Try examining individual cell transformations or spatial patterns")
        
        return {
            "analysis": analysis,
            "suggestions": suggestions,
            "recommended_functions": recommended_functions,
            "primary_pattern": "cell_expansion" if analysis.get("cell_expansion_detected") else "unknown"
        }
    
    def _is_cell_expansion(self, input_grid: List[List[int]], output_grid: List[List[int]], factor: int) -> bool:
        """Check if output is input with each cell expanded to factorxfactor block"""
        in_height, in_width = len(input_grid), len(input_grid[0])
        out_height, out_width = len(output_grid), len(output_grid[0])
        
        # Check if dimensions match scaling factor
        if out_height != in_height * factor or out_width != in_width * factor:
            return False
        
        # Check if each input cell maps to a factorxfactor block in output
        for in_r in range(in_height):
            for in_c in range(in_width):
                input_color = input_grid[in_r][in_c]
                
                # Check the corresponding factorxfactor block in output
                start_r = in_r * factor
                start_c = in_c * factor
                
                for out_r in range(start_r, start_r + factor):
                    for out_c in range(start_c, start_c + factor):
                        if output_grid[out_r][out_c] != input_color:
                            return False
        
        return True
    
    def _apply_cell_expansion(self, scaling_factor: int) -> Dict:
        """Apply cell expansion transformation to current puzzle"""
        if not self.current_puzzle or 'test' not in self.current_puzzle:
            return {"error": "No test input available"}
        
        test_input = self.current_puzzle['test'][0]['input']
        in_height, in_width = len(test_input), len(test_input[0])
        
        # Create output grid with expanded dimensions
        out_height = in_height * scaling_factor
        out_width = in_width * scaling_factor
        output_grid = [[0 for _ in range(out_width)] for _ in range(out_height)]
        
        # Expand each cell to scaling_factor x scaling_factor block
        for in_r in range(in_height):
            for in_c in range(in_width):
                input_color = test_input[in_r][in_c]
                
                # Fill the corresponding block in output
                start_r = in_r * scaling_factor
                start_c = in_c * scaling_factor
                
                for out_r in range(start_r, start_r + scaling_factor):
                    for out_c in range(start_c, start_c + scaling_factor):
                        output_grid[out_r][out_c] = input_color
        
        # Update the output grid
        self.output_grid = output_grid
        
        return {
            "success": True,
            "message": f"Applied cell expansion with factor {scaling_factor}",
            "input_size": f"{in_height}x{in_width}",
            "output_size": f"{out_height}x{out_width}",
            "transformation": f"Each cell expanded to {scaling_factor}x{scaling_factor} block"
        }
    
    def _apply_grid_scaling(self, factor: int) -> Dict:
        """Apply uniform grid scaling"""
        if not self.current_puzzle or 'test' not in self.current_puzzle:
            return {"error": "No test input available"}
        
        test_input = self.current_puzzle['test'][0]['input']
        in_height, in_width = len(test_input), len(test_input[0])
        
        # For now, implement as cell expansion (can be enhanced for other scaling types)
        return self._apply_cell_expansion(factor)
    
    def _apply_color_mapping(self, color_map: Dict[int, int]) -> Dict:
        """Apply color mapping transformation"""
        if not self.output_grid:
            if not self.current_puzzle or 'test' not in self.current_puzzle:
                return {"error": "No grid to apply color mapping to"}
            # Copy test input first
            test_input = self.current_puzzle['test'][0]['input']
            self.output_grid = [row[:] for row in test_input]
        
        # Apply color mapping
        changes_made = 0
        for r in range(len(self.output_grid)):
            for c in range(len(self.output_grid[0])):
                current_color = self.output_grid[r][c]
                if current_color in color_map:
                    self.output_grid[r][c] = color_map[current_color]
                    changes_made += 1
        
        return {
            "success": True,
            "message": f"Applied color mapping, changed {changes_made} cells",
            "mapping": color_map
        }
    
    def _detect_pattern_type(self) -> Dict:
        """Detect the primary pattern type in the current puzzle"""
        analysis = self._analyze_pattern_detailed()
        
        if "error" in analysis:
            return analysis
        
        pattern_scores = {
            "cell_expansion": 0,
            "color_mapping": 0,
            "spatial_transformation": 0,
            "size_change": 0,
            "unknown": 0
        }
        
        # Score different pattern types
        if analysis.get("cell_expansion_detected"):
            pattern_scores["cell_expansion"] = 10
        
        if analysis.get("scaling_factor"):
            pattern_scores["cell_expansion"] += 5
        
        if analysis.get("transformations"):
            for trans in analysis["transformations"]:
                if trans["type"] == "color_mapping":
                    pattern_scores["color_mapping"] += 8
        
        if analysis.get("size_changes"):
            pattern_scores["size_change"] += 6
        
        # Find highest scoring pattern
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        
        return {
            "primary_pattern": best_pattern[0],
            "confidence": min(best_pattern[1] / 10.0, 1.0),
            "pattern_scores": pattern_scores,
            "analysis_summary": analysis
        }
    
    def _suggest_next_step(self) -> Dict:
        """Suggest the next step based on pattern analysis"""
        pattern_detection = self._detect_pattern_type()
        
        if "error" in pattern_detection:
            return pattern_detection
        
        primary_pattern = pattern_detection["primary_pattern"]
        confidence = pattern_detection["confidence"]
        
        if primary_pattern == "cell_expansion" and confidence > 0.5:  # Lowered threshold for action
            analysis = pattern_detection["analysis_summary"]
            factor = analysis.get("scaling_factor")
            return {
                "suggestion": f"Apply cell expansion with factor {factor}",
                "function_call": {
                    "name": "apply_cell_expansion",
                    "parameters": {"scaling_factor": factor}
                },
                "confidence": confidence,
                "reasoning": f"Detected cell expansion pattern where each input cell becomes {factor}x{factor} block"
            }
        
        elif primary_pattern == "color_mapping" and confidence > 0.4:  # Lowered threshold
            analysis = pattern_detection["analysis_summary"]
            if analysis.get("transformations"):
                color_map = analysis["transformations"][0].get("mapping", {})
                return {
                    "suggestion": "Apply color mapping transformation",
                    "function_call": {
                        "name": "apply_color_mapping",
                        "parameters": {"color_map": color_map}
                    },
                    "confidence": confidence,
                    "reasoning": f"Detected consistent color mapping: {color_map}"
                }
        
        # Even with low confidence, suggest concrete actions
        if "size" in primary_pattern or "scale" in primary_pattern:
            # Try cell expansion anyway
            return {
                "suggestion": "Pattern involves size change - let's try cell expansion",
                "function_call": {
                    "name": "apply_cell_expansion",
                    "parameters": {"factor": 3}  # Common factor
                },
                "confidence": confidence,
                "reasoning": f"Size-related pattern detected, trying common 3x expansion"
            }
        
        # Default to trying the detected pattern anyway
        return {
            "suggestion": f"Let's try {primary_pattern} transformation even with {confidence:.0%} confidence",
            "function_call": {
                "name": "detect_pattern_type",
                "parameters": {}
            },
            "confidence": confidence,
            "reasoning": f"Better to try something than nothing - attempting {primary_pattern}"
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
        user_input_lower = user_input.lower().strip()
        
        # Handle fibonacci requests directly
        if "fibonacci" in user_input_lower:
            import re
            # Try to extract number from various formats
            match = re.search(r'fibonacci\((\d+)\)', user_input_lower)
            if not match:
                match = re.search(r'(\d+)(?:th|st|nd|rd)?\s+fibonacci', user_input_lower)
            if not match:
                match = re.search(r'fibonacci.*?(\d+)', user_input_lower)
            
            if match:
                n = int(match.group(1))
                result = self.execute_function(
                    "invoke_generated_tool",
                    {"tool_name": "fibonacci", "parameters": {"n": n}}
                )
                if result.get('success'):
                    return {
                        "message": f"fibonacci({n}) = {result['result']}",
                        "function_results": [{"function": "fibonacci", "result": result}]
                    }
                else:
                    return {"message": f"Error calculating fibonacci({n}): {result.get('error', 'Unknown error')}"}
            else:
                return {"message": "Please specify which fibonacci number you want (e.g., 'fibonacci(20)' or '20th fibonacci number')"}
        
        elif "grid size" in user_input_lower or "what size" in user_input_lower:
            if self.output_grid:
                return {
                    "message": f"Output grid size: {len(self.output_grid)}x{len(self.output_grid[0])}"
                }
            return {"message": "No output grid loaded"}
        
        elif "heuristic" in user_input_lower and "count" in user_input_lower:
            count = len(heuristics_manager.heuristics)
            return {"message": f"There are {count} heuristics available"}
        
        elif "analyze" in user_input_lower:
            analysis = self._analyze_pattern_detailed()
            return {"message": json.dumps(analysis, indent=2)}
        
        elif any(word in user_input_lower for word in ["solve", "try", "submit", "heuristic", "tool", "pattern"]):
            # Proactive solving attempt
            return self._auto_solve_puzzle()
        
        else:
            return {
                "message": "Basic mode: I can help with grid sizes, heuristic counts, pattern analysis, fibonacci calculations, and basic solving. For advanced features, Bedrock connection is required."
            }
    
    def _generate_tool_with_claude(self, description: str, tool_name: str) -> Dict:
        """Generate a new tool using Claude Code"""
        # Create prompt for Claude Code
        prompt = f"""Create a new Python tool file called 'arc_generated_{tool_name}.py' that:

{description}

The tool should:
1. Be a reusable component for the ARC AGI system
2. Include proper docstrings and type hints
3. Have error handling for edge cases
4. Include test functions to verify it works
5. Follow Python best practices

Make it integrate well with the existing ARC AGI puzzle solving system."""
        
        # Run asynchronously in a safe way
        try:
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # We're inside an async context, can't use run_until_complete
                # Return a simple message for now
                return {
                    "success": False,
                    "message": "Cannot generate tools while handling async requests. Please try from a direct script."
                }
            except RuntimeError:
                # No event loop running, we can create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    self.claude_dialogue.invoke_claude(prompt)
                )
                
                if result['status'] == 'completed':
                    # Try to import the generated tool
                    from pathlib import Path
                    tool_file = Path(f"arc_generated_{tool_name}.py")
                    
                    if tool_file.exists():
                        return {
                            "success": True,
                            "message": f"Tool '{tool_name}' generated successfully",
                            "file_path": str(tool_file),
                            "response": result.get('response', '')[:500]
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Tool generation completed but file not found",
                            "response": result.get('response', '')
                        }
                else:
                    return {
                        "success": False,
                        "message": f"Tool generation failed: {result.get('response', 'Unknown error')}"
                    }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error generating tool: {str(e)}"
            }
        finally:
            # Only close the loop if we created it
            if 'loop' in locals() and loop:
                try:
                    loop.close()
                except:
                    pass
    
    def _list_generated_tools(self) -> Dict:
        """List all generated tools"""
        from pathlib import Path
        import glob
        
        # Find all generated tool files
        tool_files = glob.glob("arc_generated_*.py")
        tools = []
        
        for file in tool_files:
            tool_name = file.replace("arc_generated_", "").replace(".py", "")
            tools.append({
                "name": tool_name,
                "file": file,
                "path": str(Path(file).absolute())
            })
        
        return {
            "tools": tools,
            "count": len(tools),
            "message": f"Found {len(tools)} generated tools"
        }
    
    def _invoke_generated_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Invoke a generated tool"""
        from pathlib import Path
        import importlib.util
        
        tool_file = Path(f"arc_generated_{tool_name}.py")
        
        if not tool_file.exists():
            # Check if it's the Fibonacci tool
            if tool_name == "fibonacci":
                tool_file = Path("arc_fibonacci_tool.py")
            else:
                return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            # Import the tool module
            spec = importlib.util.spec_from_file_location(f"arc_generated_{tool_name}", tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to find and execute the main function or class
            if hasattr(module, 'execute'):
                result = module.execute(**parameters)
            elif hasattr(module, 'main'):
                result = module.main(**parameters)
            elif hasattr(module, tool_name):
                func = getattr(module, tool_name)
                result = func(**parameters)
            else:
                # Look for the first callable
                for name, obj in module.__dict__.items():
                    if callable(obj) and not name.startswith('_'):
                        result = obj(**parameters)
                        break
                else:
                    return {"error": f"No executable function found in tool '{tool_name}'"}
            
            return {
                "success": True,
                "result": str(result),
                "tool": tool_name
            }
        except Exception as e:
            return {
                "error": f"Error executing tool '{tool_name}': {str(e)}"
            }

# Create a wrapper class that's compatible with the existing system
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are equal"""
        if not grid1 or not grid2:
            return False
        if len(grid1) != len(grid2):
            return False
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def _find_differences(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict:
        """Find differences between two grids"""
        differences = {
            "size_mismatch": False,
            "different_cells": []
        }
        
        if not grid1 or not grid2:
            differences["error"] = "One or both grids are empty"
            return differences
            
        if len(grid1) != len(grid2) or (grid1 and grid2 and len(grid1[0]) != len(grid2[0])):
            differences["size_mismatch"] = True
            differences["grid1_size"] = f"{len(grid1)}x{len(grid1[0]) if grid1 else 0}"
            differences["grid2_size"] = f"{len(grid2)}x{len(grid2[0]) if grid2 else 0}"
        else:
            for i in range(len(grid1)):
                for j in range(len(grid1[i])):
                    if grid1[i][j] != grid2[i][j]:
                        differences["different_cells"].append({
                            "position": [i, j],
                            "expected": grid2[i][j],
                            "actual": grid1[i][j]
                        })
        
        differences["total_differences"] = len(differences["different_cells"])
        return differences
    
    def _validate_grid_format(self, grid: List[List[int]]) -> bool:
        """Validate that grid is properly formatted"""
        if not grid or not isinstance(grid, list):
            return False
        
        width = len(grid[0]) if grid else 0
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                return False
            for cell in row:
                if not isinstance(cell, int) or cell < 0 or cell > 9:
                    return False
        
        return True
    
    def _send_progress(self, message: str, message_type: str = "info"):
        """Send progress message to callback if available"""
        if self.progress_callback:
            try:
                self.progress_callback({
                    "type": "progress",
                    "message": message,
                    "message_type": message_type,
                    "timestamp": json.dumps(time.time())
                })
            except:
                pass  # Ignore callback errors
    
    def _generate_theory(self, approach: str, params: Dict, analysis: Dict) -> str:
        """Generate a theory about why this approach might work"""
        theory_parts = []
        
        # Base theory on approach type
        if "cell_expansion" in approach:
            factor = params.get('factor', 1)
            theory_parts.append(f"Theory: The pattern shows each input cell expanding to {factor}x{factor} blocks.")
            if analysis.get('cell_expansion_detected'):
                theory_parts.append(f"Evidence: Cell expansion pattern confirmed in training examples.")
        elif "color_mapping" in approach:
            theory_parts.append("Theory: Colors are being systematically transformed.")
            if params.get('color_map'):
                theory_parts.append(f"Mapping: {params['color_map']}")
        elif "grid_scaling" in approach:
            theory_parts.append("Theory: The entire grid is being scaled uniformly.")
        elif "copy_grid" in approach:
            theory_parts.append("Theory: The input pattern is being tiled or repeated.")
        
        # Add confidence based on analysis
        if analysis.get('scaling_factor'):
            theory_parts.append(f"Confidence: High - scaling factor {analysis['scaling_factor']} detected")
        else:
            theory_parts.append("Confidence: Exploratory - testing hypothesis")
        
        return " ".join(theory_parts)
    
    def _auto_solve_puzzle(self) -> Dict:
        """Intelligently solve puzzles using Bedrock AI to generate creative approaches"""
        
        if not self.current_puzzle or 'test' not in self.current_puzzle:
            return {"message": "No puzzle loaded. Please load a puzzle first."}
        
        # Get test input
        test_input = self.current_puzzle['test'][0]['input']
        input_h, input_w = len(test_input), len(test_input[0]) if test_input else 0
        
        # Initialize tracking
        attempts = []
        tried_approaches = set()  # Track what we've tried to avoid repeats
        message_parts = ["🧠 Using AI to intelligently solve puzzle (up to 100 attempts)...\n"]
        
        # Send initial progress
        self._send_progress("🚀 Starting intelligent puzzle solving with Bedrock AI...", "start")
        self._send_progress(f"📊 Analyzing puzzle {self.current_puzzle.get('id', 'unknown')}...", "info")
        
        # Initial analysis
        analysis = self._analyze_pattern_detailed()
        self._send_progress(f"✅ Initial analysis complete: {len(analysis.get('transformations', []))} patterns found", "success")
        
        # Build solving context for Bedrock
        solving_context = {
            "puzzle_id": self.current_puzzle.get('id', 'unknown'),
            "input_size": f"{input_h}x{input_w}",
            "analysis": analysis,
            "attempts": [],
            "solving_history": []
        }
        
        # Maximum attempts
        MAX_ATTEMPTS = 100
        attempt_count = 0
        
        def generate_approach_key(method, params):
            """Generate unique key for approach tracking"""
            return f"{method}:{json.dumps(params, sort_keys=True)}"
        
        def try_approach(method, params, description):
            """Try an approach and check if it solves the puzzle"""
            nonlocal attempt_count, message_parts, attempts, tried_approaches, solving_context
            
            # Check if we've tried this exact approach before
            approach_key = generate_approach_key(method, params)
            if approach_key in tried_approaches:
                return False
            
            tried_approaches.add(approach_key)
            attempt_count += 1
            
            # Generate theory for this approach
            theory = self._generate_theory(method, params, analysis)
            
            # Send progress about what we're trying
            self._send_progress(f"🔄 Attempt #{attempt_count}: {description}", "attempt")
            self._send_progress(f"💭 {theory}", "theory")
            
            # Execute the approach
            result = self.execute_function(method, params)
            
            # Send progress about transformation result  
            if result.get('success'):
                self._send_progress(f"✅ Transformation applied successfully", "success")
            else:
                self._send_progress(f"⚠️ Transformation failed: {result.get('error', 'Unknown error')}", "warning")
            
            attempt_record = {
                "attempt": attempt_count,
                "method": method,
                "params": params,
                "description": description,
                "theory": theory,
                "success": result.get('success', False)
            }
            
            if result.get('success'):
                # Check if solution is correct
                self._send_progress("🔍 Checking solution correctness...", "info")
                check_result = self.execute_function("check_solution", {})
                attempt_record['correct'] = check_result.get('correct', False)
                
                if check_result.get('correct'):
                    self._send_progress(f"🎉 SOLVED! Puzzle solved on attempt #{attempt_count}!", "solved")
                    message_parts.append(f"\n🎉 SOLVED on attempt #{attempt_count}!")
                    message_parts.append(f"Method: {description}")
                    message_parts.append(f"Theory: {theory}")
                    attempts.append(attempt_record)
                    solving_context['attempts'].append(attempt_record)
                    return True
                else:
                    # Analyze why it failed
                    failure_reason = "Unknown failure"
                    if check_result.get('differences'):
                        diff = check_result['differences']
                        if diff.get('size_mismatch'):
                            failure_reason = f"Size mismatch: expected {diff.get('grid2_size')}, got {diff.get('grid1_size')}"
                        elif diff.get('different_cells'):
                            num_diff = len(diff['different_cells'])
                            failure_reason = f"{num_diff} cells differ from expected"
                    attempt_record['failure_reason'] = failure_reason
                    self._send_progress(f"❌ Solution incorrect: {failure_reason}", "failure")
                    message_parts.append(f"Attempt #{attempt_count}: {failure_reason}")
            else:
                attempt_record['error'] = result.get('error', 'Unknown error')
            
            attempts.append(attempt_record)
            solving_context['attempts'].append(attempt_record)
            return False
        
        # First, try the most obvious pattern from initial analysis
        if analysis.get('size_changes'):
            for size_change in analysis['size_changes'][:1]:  # Try just the first one
                factor = size_change.get('scaling_factor', 1)
                if try_approach("apply_cell_expansion", {"factor": factor}, 
                               f"Initial detected pattern: cell expansion {factor}x"):
                    return self._success_response(message_parts, attempts, "cell_expansion")
        
        # Now use Bedrock AI to generate creative approaches
        message_parts.append("\n🤖 Using AI to generate intelligent approaches...")
        self._send_progress("🤖 Engaging Bedrock AI for creative problem solving...", "bedrock")
        
        # Main solving loop - let Bedrock think about what to try next
        while attempt_count < MAX_ATTEMPTS:
            # Build prompt for Bedrock with full context
            bedrock_prompt = f"""I'm trying to solve an ARC puzzle. Here's the context:

Puzzle Analysis:
- Input size: {input_h}x{input_w}
- Pattern analysis: {json.dumps(analysis, indent=2)}

Attempts so far ({attempt_count} total):
{json.dumps(solving_context['attempts'][-10:], indent=2)}  # Show last 10 attempts

Based on what's been tried and what failed, suggest a NEW approach to try. 
If sizes mismatched, try different scaling factors.
If cells differed, try color transformations or spatial operations.

If existing functions don't work, suggest creating a NEW TOOL by calling:
generate_tool_with_claude(description, name)

Be creative! Consider:
- Different scaling factors (1-20)
- Color mappings (inversions, shifts, swaps)
- Grid copying to different sizes
- Combinations of transformations
- Creating new pattern recognition tools
- Developing custom transformation functions

Available functions:
- apply_cell_expansion(factor): Expand each cell to NxN block
- apply_grid_scaling(factor): Scale entire grid  
- apply_color_mapping(color_map): Map colors
- copy_grid(src_top_left, dest_top_left, dest_bottom_right): Copy to different size
- generate_tool_with_claude(description, name): Create a new tool
- clear_output(): Clear the output grid
- submit_solution(): Submit current output as solution

Provide a theory for why your suggested approach will work.
Think step by step and be creative!"""
            
            # Send Bedrock prompt to progress
            self._send_progress(f"📝 Bedrock Query #{attempt_count + 1}: Asking AI for creative ideas...", "bedrock_query")
            self._send_progress(f"Query: {bedrock_prompt[:300]}...", "bedrock_prompt")
            
            # Query Bedrock for next approach
            if self.bedrock_available:
                try:
                    bedrock_response = self._smart_bedrock_query(bedrock_prompt)
                    
                    # Show Bedrock's response
                    if bedrock_response.get('message'):
                        self._send_progress(f"💡 Bedrock Response: {bedrock_response['message'][:500]}", "bedrock_response")
                    
                    # Extract function calls from response
                    if bedrock_response.get('function_results'):
                        for func_result in bedrock_response['function_results']:
                            func_name = func_result.get('function', '')
                            
                            # Handle tool generation suggestions
                            if func_name == 'generate_tool_with_claude':
                                tool_desc = func_result.get('result', {}).get('description', '')
                                tool_name = func_result.get('result', {}).get('name', f'tool_{attempt_count}')
                                self._send_progress(f"🛠️ Bedrock suggests creating new tool: {tool_name}", "tool_suggestion")
                                self._send_progress(f"Tool description: {tool_desc[:200]}", "tool_desc")
                                
                                # Try to generate the tool
                                tool_result = self.execute_function("generate_tool_with_claude", {
                                    "description": tool_desc,
                                    "name": tool_name
                                })
                                
                                if tool_result.get('success'):
                                    self._send_progress(f"✅ New tool '{tool_name}' created successfully!", "tool_created")
                                    # Try to use the new tool
                                    if try_approach("invoke_generated_tool", {"tool_name": tool_name, "parameters": {}},
                                                  f"AI created tool: {tool_name}"):
                                        return self._success_response(message_parts, attempts, "ai_generated_tool")
                            
                            elif func_name == 'apply_cell_expansion':
                                # Bedrock suggested cell expansion
                                factor = func_result.get('result', {}).get('scaling_factor', attempt_count % 10 + 1)
                                if try_approach("apply_cell_expansion", {"factor": factor},
                                              f"AI suggested: cell expansion {factor}x"):
                                    return self._success_response(message_parts, attempts, "ai_cell_expansion")
                            
                            elif func_name == 'submit_solution':
                                # AI wants to submit current solution
                                self._send_progress("📤 AI submitting current output as solution...", "submit")
                                submit_result = self.execute_function("submit_solution", {})
                                if submit_result.get('is_correct'):
                                    return self._success_response(message_parts, attempts, "ai_submit")
                            
                            # Add more function handlers as needed
                    
                    # If no function calls, try to extract suggestions from message
                    ai_message = bedrock_response.get('message', '')
                    if 'scaling' in ai_message.lower() or 'expansion' in ai_message.lower():
                        # Try a new scaling factor based on AI suggestion
                        factor = (attempt_count % 20) + 1
                        if try_approach("apply_cell_expansion", {"factor": factor},
                                      f"AI inspired: scaling {factor}x"):
                            return self._success_response(message_parts, attempts, "ai_scaling")
                    
                except Exception as e:
                    print(f"Bedrock query failed: {e}")
                    # Fall back to systematic exploration
            
            # Fallback: Try systematic approaches if Bedrock isn't helping
            if attempt_count < 30:
                # Try various scaling factors
                factor = (attempt_count % 10) + 1
                if try_approach("apply_cell_expansion", {"factor": factor},
                              f"Systematic: cell expansion {factor}x"):
                    return self._success_response(message_parts, attempts, "systematic_scaling")
                    
            elif attempt_count < 50:
                # Try color transformations
                input_colors = set()
                for row in test_input:
                    input_colors.update(row)
                    
                shift = (attempt_count - 30) % 10
                if try_approach("apply_color_mapping", 
                              {"color_map": {c: (c + shift) % 10 for c in input_colors}},
                              f"Color shift +{shift}"):
                    return self._success_response(message_parts, attempts, "color_transformation")
                    
            elif attempt_count < 80:
                # Try copying to different grid sizes
                size = 3 + ((attempt_count - 50) % 8)
                if try_approach("copy_grid", {
                    "src_top_left": [0, 0],
                    "dest_top_left": [0, 0], 
                    "dest_bottom_right": [size-1, size-1]
                }, f"Copy to {size}x{size} grid"):
                    return self._success_response(message_parts, attempts, "grid_copy")
            else:
                # Extended scaling attempts
                factor = 10 + ((attempt_count - 80) % 10)
                if try_approach("apply_cell_expansion", {"factor": factor},
                              f"Extended scaling {factor}x"):
                    return self._success_response(message_parts, attempts, "extended_scaling")
            
            # Prevent infinite loops
            if attempt_count >= MAX_ATTEMPTS:
                break
        
        # If we've exhausted all attempts
        message_parts.append(f"\n❌ Exhausted {attempt_count} attempts without solving.")
        message_parts.append("The puzzle requires a more complex transformation.")
        message_parts.append("\nAttempt summary:")
        message_parts.append(f"  • Scaling attempts: {len([a for a in attempts if 'scaling' in a.get('description', '').lower()])}")
        message_parts.append(f"  • Color attempts: {len([a for a in attempts if 'color' in a.get('description', '').lower()])}")
        message_parts.append(f"  • Other attempts: {len([a for a in attempts if 'scaling' not in a.get('description', '').lower() and 'color' not in a.get('description', '').lower()])}")
        
        return {
            "message": "\n".join(message_parts),
            "output_grid": self.output_grid,
            "solved": False,
            "attempts": attempts,
            "total_attempts": attempt_count,
            "analysis": analysis,
            "exhausted": True
        }
    
    def _success_response(self, message_parts, attempts, method):
        """Generate successful response"""
        return {
            "message": "\n".join(message_parts),
            "output_grid": self.output_grid,
            "solved": True,
            "method": method,
            "attempts": attempts,
            "total_attempts": len(attempts)
        }

class PuzzleAIAssistant(SmartPuzzleAIAssistant):
    """Wrapper for backward compatibility"""
    pass