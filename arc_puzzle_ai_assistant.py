#!/usr/bin/env python3
"""
AI Assistant for ARC Puzzle Editor
Interprets natural language commands and provides puzzle analysis
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
    
class PuzzleAIAssistant:
    """AI Assistant for puzzle interaction with function calling"""
    
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
    
    def __init__(self):
        self.current_puzzle = None
        self.output_grid = None
        self.heuristics = []  # Store available heuristics
        self.tools = []  # Store available tools
        self.last_applied_heuristic = None  # Track last heuristic for verification feedback
        # Initialize Bedrock client
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            )
            self.bedrock_available = True
        except Exception as e:
            print(f"Warning: Bedrock not available: {e}")
            self.bedrock_available = False
        
    def set_puzzle(self, puzzle: Dict):
        """Set the current puzzle context"""
        self.current_puzzle = puzzle
        # Initialize output grid from test input
        if puzzle and 'test' in puzzle and puzzle['test']:
            test_input = puzzle['test'][0]['input']
            self.output_grid = [row[:] for row in test_input]
    
    def set_output_grid(self, grid: List[List[int]]):
        """Update the output grid state"""
        self.output_grid = grid
    
    def set_heuristics(self, heuristics: List[Dict]):
        """Set available heuristics"""
        self.heuristics = heuristics
    
    def set_tools(self, tools: List[Dict]):
        """Set available tools"""
        self.tools = tools
            
    def process_command(self, user_input: str) -> Dict:
        """Process a natural language command"""
        # If Bedrock is available, use it for more intelligent processing
        if self.bedrock_available:
            try:
                response = self._query_bedrock(user_input)
                if response:
                    return response
            except Exception as e:
                print(f"Bedrock query failed: {e}")
        
        # Fall back to local processing
        user_input = user_input.lower().strip()
        
        # Try to parse as a command
        command = self.parse_command(user_input)
        
        if command:
            result = self.execute_command(command)
            return {
                'type': 'command',
                'command': command.action,
                'result': result,
                'message': self.format_command_result(command, result)
            }
        
        # Try to answer as a question
        answer = self.answer_question(user_input)
        if answer:
            return {
                'type': 'answer',
                'message': answer
            }
            
        # Provide general help
        return {
            'type': 'help',
            'message': self.get_help_message()
        }
        
    def parse_command(self, text: str) -> Optional[GridCommand]:
        """Parse natural language into a grid command"""
        
        # Pattern: "make/set square/cell X,Y color"
        color_pattern = r'(?:make|set|color|paint|change)\s+(?:square|cell|position)?\s*(\d+)\s*,\s*(\d+)\s+(?:to\s+)?(\w+(?:\s+\w+)?)'
        match = re.search(color_pattern, text)
        if match:
            row, col, color_name = match.groups()
            color_num = self._parse_color(color_name)
            if color_num is not None:
                return GridCommand('set_color', {
                    'row': int(row),
                    'col': int(col),
                    'color': color_num
                })
        
        # Pattern: "resize/change output to/grid WxH"
        resize_pattern = r'(?:resize|change|set)\s+(?:the\s+)?(?:output|grid)\s+(?:to\s+)?(\d+)\s*[x×]\s*(\d+)'
        match = re.search(resize_pattern, text)
        if match:
            width, height = match.groups()
            return GridCommand('resize', {
                'width': int(width),
                'height': int(height)
            })
            
        # Pattern: "copy from input"
        if 'copy' in text and 'input' in text:
            return GridCommand('copy', {'source': 'input'})
            
        # Pattern: "clear/reset output"
        if any(word in text for word in ['clear', 'reset']) and 'output' in text:
            return GridCommand('clear', {})
            
        # Pattern: "fill region/area"
        fill_pattern = r'fill\s+(?:from\s+)?(\d+),(\d+)\s+(?:to\s+)?(\d+),(\d+)\s+(?:with\s+)?(\w+)'
        match = re.search(fill_pattern, text)
        if match:
            r1, c1, r2, c2, color_name = match.groups()
            color_num = self._parse_color(color_name)
            if color_num is not None:
                return GridCommand('fill', {
                    'start': (int(r1), int(c1)),
                    'end': (int(r2), int(c2)),
                    'color': color_num
                })
                
        # Pattern: "make X,Y same as input X,Y"
        copy_cell_pattern = r'make\s+(?:square|cell)?\s*(\d+),(\d+)\s+(?:the\s+)?same\s+(?:as|color)\s+(?:input\s+)?(?:at\s+)?(\d+),(\d+)'
        match = re.search(copy_cell_pattern, text)
        if match:
            out_r, out_c, in_r, in_c = match.groups()
            return GridCommand('copy_cell', {
                'output': (int(out_r), int(out_c)),
                'input': (int(in_r), int(in_c))
            })
            
        return None
        
    def _parse_color(self, color_text: str) -> Optional[int]:
        """Parse color name or number to color index"""
        color_text = color_text.strip().lower()
        
        # Try as number first
        try:
            color_num = int(color_text)
            if 0 <= color_num <= 9:
                return color_num
        except ValueError:
            pass
            
        # Try as color name
        return self.COLOR_MAP.get(color_text)
        
    def execute_command(self, command: GridCommand) -> Dict:
        """Execute a parsed command"""
        if not self.output_grid:
            return {'success': False, 'error': 'No puzzle loaded'}
            
        try:
            if command.action == 'set_color':
                row, col = command.params['row'], command.params['col']
                color = command.params['color']
                
                if 0 <= row < len(self.output_grid) and 0 <= col < len(self.output_grid[0]):
                    self.output_grid[row][col] = color
                    return {'success': True, 'grid': self.output_grid}
                else:
                    return {'success': False, 'error': f'Position ({row},{col}) out of bounds'}
                    
            elif command.action == 'resize':
                width = command.params['width']
                height = command.params['height']
                
                new_grid = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        if i < len(self.output_grid) and j < len(self.output_grid[0]):
                            row.append(self.output_grid[i][j])
                        else:
                            row.append(0)
                    new_grid.append(row)
                    
                self.output_grid = new_grid
                return {'success': True, 'grid': self.output_grid}
                
            elif command.action == 'copy':
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    self.output_grid = [row[:] for row in test_input]
                    return {'success': True, 'grid': self.output_grid}
                return {'success': False, 'error': 'No input to copy from'}
                
            elif command.action == 'clear':
                for i in range(len(self.output_grid)):
                    for j in range(len(self.output_grid[0])):
                        self.output_grid[i][j] = 0
                return {'success': True, 'grid': self.output_grid}
                
            elif command.action == 'fill':
                start = command.params['start']
                end = command.params['end']
                color = command.params['color']
                
                for i in range(min(start[0], end[0]), min(max(start[0], end[0]) + 1, len(self.output_grid))):
                    for j in range(min(start[1], end[1]), min(max(start[1], end[1]) + 1, len(self.output_grid[0]))):
                        self.output_grid[i][j] = color
                        
                return {'success': True, 'grid': self.output_grid}
                
            elif command.action == 'copy_cell':
                out_pos = command.params['output']
                in_pos = command.params['input']
                
                if self.current_puzzle and 'test' in self.current_puzzle:
                    test_input = self.current_puzzle['test'][0]['input']
                    if (0 <= in_pos[0] < len(test_input) and 
                        0 <= in_pos[1] < len(test_input[0]) and
                        0 <= out_pos[0] < len(self.output_grid) and
                        0 <= out_pos[1] < len(self.output_grid[0])):
                        
                        color = test_input[in_pos[0]][in_pos[1]]
                        self.output_grid[out_pos[0]][out_pos[1]] = color
                        return {'success': True, 'grid': self.output_grid}
                        
                return {'success': False, 'error': 'Invalid positions'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
        return {'success': False, 'error': 'Unknown command'}
        
    def format_command_result(self, command: GridCommand, result: Dict) -> str:
        """Format command result as human-readable message"""
        if result.get('success'):
            if command.action == 'set_color':
                return f"✓ Set cell ({command.params['row']},{command.params['col']}) to color {command.params['color']}"
            elif command.action == 'resize':
                return f"✓ Resized grid to {command.params['width']}×{command.params['height']}"
            elif command.action == 'copy':
                return "✓ Copied input to output"
            elif command.action == 'clear':
                return "✓ Cleared output grid"
            elif command.action == 'fill':
                return f"✓ Filled region with color {command.params['color']}"
            elif command.action == 'copy_cell':
                return f"✓ Copied color from input {command.params['input']} to output {command.params['output']}"
        else:
            return f"✗ Failed: {result.get('error', 'Unknown error')}"
            
    def answer_question(self, question: str) -> Optional[str]:
        """Answer questions about the puzzle"""
        if not self.current_puzzle:
            return "No puzzle loaded. Please load a puzzle first."
            
        question = question.lower()
        
        # Grid size questions
        if 'grid size' in question or 'dimensions' in question:
            if 'input' in question:
                # Check which example
                example_num = self._extract_number(question)
                if example_num is not None and 'train' in self.current_puzzle:
                    if 0 < example_num <= len(self.current_puzzle['train']):
                        grid = self.current_puzzle['train'][example_num-1]['input']
                        return f"Input {example_num} has size {len(grid)}×{len(grid[0])}"
                        
                # Test input
                if 'test' in question and 'test' in self.current_puzzle:
                    grid = self.current_puzzle['test'][0]['input']
                    return f"Test input has size {len(grid)}×{len(grid[0])}"
                    
            elif 'output' in question:
                example_num = self._extract_number(question)
                if example_num is not None and 'train' in self.current_puzzle:
                    if 0 < example_num <= len(self.current_puzzle['train']):
                        grid = self.current_puzzle['train'][example_num-1]['output']
                        return f"Output {example_num} has size {len(grid)}×{len(grid[0])}"
                        
        # Pattern questions
        if 'pattern' in question:
            return self.analyze_pattern()
            
        # Color questions
        if 'color' in question:
            pos_match = re.search(r'(\d+),(\d+)', question)
            if pos_match:
                row, col = map(int, pos_match.groups())
                return self.get_cell_color_info(row, col, question)
                
        # Training example count
        if 'how many' in question and ('example' in question or 'training' in question):
            if 'train' in self.current_puzzle:
                return f"There are {len(self.current_puzzle['train'])} training examples"
                
        return None
        
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract a number from text"""
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])  # Use last number found
        return None
        
    def get_cell_color_info(self, row: int, col: int, context: str) -> str:
        """Get color information for a specific cell"""
        if 'input' in context:
            if 'test' in self.current_puzzle:
                grid = self.current_puzzle['test'][0]['input']
                if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                    color = grid[row][col]
                    color_name = self.get_color_name(color)
                    return f"Input cell ({row},{col}) is {color_name} (color {color})"
                    
        elif 'output' in context:
            if self.output_grid:
                if 0 <= row < len(self.output_grid) and 0 <= col < len(self.output_grid[0]):
                    color = self.output_grid[row][col]
                    color_name = self.get_color_name(color)
                    return f"Output cell ({row},{col}) is {color_name} (color {color})"
                    
        return f"Cell ({row},{col}) is out of bounds"
        
    def get_color_name(self, color_num: int) -> str:
        """Get color name from number"""
        color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'pink', 'orange', 'light blue', 'brown']
        if 0 <= color_num <= 9:
            return color_names[color_num]
        return f"color {color_num}"
        
    def analyze_pattern(self) -> str:
        """Analyze the puzzle pattern"""
        if not self.current_puzzle or 'train' not in self.current_puzzle:
            return "No training examples to analyze"
            
        train = self.current_puzzle['train']
        
        # Check grid size changes
        size_changes = []
        for i, example in enumerate(train):
            in_shape = (len(example['input']), len(example['input'][0]))
            out_shape = (len(example['output']), len(example['output'][0]))
            if in_shape != out_shape:
                size_changes.append(f"Example {i+1}: {in_shape} → {out_shape}")
                
        if size_changes:
            return f"Grid size changes detected:\n" + "\n".join(size_changes)
            
        # Check for color mappings
        color_mappings = {}
        for example in train:
            for i in range(len(example['input'])):
                for j in range(len(example['input'][0])):
                    in_color = example['input'][i][j]
                    out_color = example['output'][i][j]
                    if in_color != out_color:
                        if in_color not in color_mappings:
                            color_mappings[in_color] = set()
                        color_mappings[in_color].add(out_color)
                        
        if color_mappings:
            mappings = []
            for in_c, out_cs in color_mappings.items():
                out_list = list(out_cs)
                if len(out_list) == 1:
                    mappings.append(f"{self.get_color_name(in_c)} → {self.get_color_name(out_list[0])}")
            if mappings:
                return "Color transformations detected:\n" + "\n".join(mappings)
                
        return "Pattern analysis: No simple transformations detected. The puzzle may involve complex spatial transformations."
        
    def get_help_message(self) -> str:
        """Get help message with example commands"""
        return """I can help you edit the output grid and answer questions about the puzzle!

**Example commands:**
• "Make square 3,3 red" - Set a cell color
• "Make cell 5,3 the same color as input 5,3" - Copy color from input
• "Resize output to 5x5" - Change grid size
• "Copy from input" - Copy entire input to output
• "Clear output" - Reset output to black
• "Fill from 0,0 to 2,2 with blue" - Fill a region

**Example questions:**
• "What is the grid size for input 2?" - Get dimensions
• "What color is input cell 3,4?" - Check cell color
• "How many training examples are there?" - Count examples
• "Analyze the pattern" - Pattern analysis

Try a command or ask a question!"""
    
    def _query_bedrock(self, user_input: str) -> Optional[Dict]:
        """Query AWS Bedrock for intelligent response with full context"""
        if not self.bedrock_available:
            return None
        
        # Prepare comprehensive context
        context = self._build_context()
        functions = self._get_available_functions()
        
        prompt = f"""{context}

Available Functions:
{json.dumps(functions, indent=2)}

User command: {user_input}

You can analyze the puzzle, suggest patterns, modify the output grid, retrieve/create heuristics and tools.
If you need to perform an action, specify which function to call with parameters.
Provide helpful analysis and insights about the puzzle patterns.
        """
        
        try:
            # Use Claude via Bedrock
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 200,
                "temperature": 0.7,
                "top_p": 0.9,
            })
            
            response = self.bedrock.invoke_model(
                body=body,
                modelId="anthropic.claude-v2",
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            ai_response = response_body.get('completion', '').strip()
            
            # Also try to extract any commands from the response
            command = self.parse_command(user_input.lower())
            if command:
                result = self.execute_command(command)
                return {
                    'type': 'command_with_explanation',
                    'command': command.action,
                    'result': result,
                    'message': ai_response + "\n\n" + self.format_command_result(command, result)
                }
            
            return {
                'type': 'ai_response',
                'message': ai_response
            }
            
        except Exception as e:
            print(f"Bedrock query error: {e}")
            return None
    
    def _build_context(self) -> str:
        """Build comprehensive context for AI"""
        context = "You are an AI assistant helping solve ARC AGI puzzles.\n\n"
        
        if self.current_puzzle:
            # Puzzle information
            context += f"Current Puzzle: {self.current_puzzle.get('id', 'Unknown')}\n"
            context += f"Training Examples: {len(self.current_puzzle.get('train', []))}\n"
            
            # Training examples details
            for i, example in enumerate(self.current_puzzle.get('train', [])):
                inp = example.get('input', [])
                out = example.get('output', [])
                context += f"\nExample {i+1}:\n"
                context += f"  Input: {len(inp)}x{len(inp[0]) if inp else 0} grid\n"
                context += f"  Output: {len(out)}x{len(out[0]) if out else 0} grid\n"
                
                # Analyze colors used
                input_colors = set(cell for row in inp for cell in row)
                output_colors = set(cell for row in out for cell in row)
                context += f"  Input colors: {sorted(input_colors)}\n"
                context += f"  Output colors: {sorted(output_colors)}\n"
            
            # Test input details
            if 'test' in self.current_puzzle and self.current_puzzle['test']:
                test_input = self.current_puzzle['test'][0].get('input', [])
                context += f"\nTest Input: {len(test_input)}x{len(test_input[0]) if test_input else 0} grid\n"
                test_colors = set(cell for row in test_input for cell in row)
                context += f"Test colors: {sorted(test_colors)}\n"
            
            # Current output grid state
            if self.output_grid:
                context += f"\nCurrent Output Grid: {len(self.output_grid)}x{len(self.output_grid[0])} grid\n"
                output_colors = set(cell for row in self.output_grid for cell in row)
                context += f"Output colors: {sorted(output_colors)}\n"
        
        # Available heuristics
        if self.heuristics:
            context += f"\nAvailable Heuristics: {len(self.heuristics)}\n"
            for h in self.heuristics[:3]:  # Show first 3
                context += f"  - {h.get('name', 'Unknown')}: {h.get('description', '')}\n"
        
        # Available tools
        if self.tools:
            context += f"\nAvailable Tools: {len(self.tools)}\n"
            for t in self.tools[:3]:  # Show first 3
                context += f"  - {t.get('name', 'Unknown')}: {t.get('description', '')}\n"
        
        return context
    
    def _get_available_functions(self) -> List[Dict]:
        """Get list of functions the AI can call"""
        return [
            {
                "name": "get_cell_color",
                "description": "Get the color of a specific cell",
                "parameters": {"grid_type": "input|output|test", "example_index": "int", "row": "int", "col": "int"}
            },
            {
                "name": "set_output_cell",
                "description": "Set the color of a cell in the output grid",
                "parameters": {"row": "int", "col": "int", "color": "int (0-9)"}
            },
            {
                "name": "resize_output_grid",
                "description": "Resize the output grid",
                "parameters": {"width": "int", "height": "int"}
            },
            {
                "name": "copy_from_input",
                "description": "Copy the test input to output grid",
                "parameters": {}
            },
            {
                "name": "clear_output",
                "description": "Clear the output grid (set all to black/0)",
                "parameters": {}
            },
            {
                "name": "analyze_pattern",
                "description": "Analyze patterns in the training examples",
                "parameters": {}
            },
            {
                "name": "get_all_heuristics",
                "description": "Get all heuristics from knowledge base",
                "parameters": {}
            },
            {
                "name": "get_relevant_heuristics",
                "description": "Get heuristics relevant to current puzzle",
                "parameters": {}
            },
            {
                "name": "rank_heuristics",
                "description": "Rank heuristics by effectiveness for this puzzle",
                "parameters": {"heuristic_ids": "list of heuristic IDs to rank"}
            },
            {
                "name": "apply_heuristic",
                "description": "Apply a specific heuristic to the puzzle",
                "parameters": {"heuristic_id": "string"}
            },
            {
                "name": "test_heuristic",
                "description": "Test a heuristic against training examples",
                "parameters": {"heuristic_id": "string"}
            },
            {
                "name": "create_heuristic",
                "description": "Create a new heuristic and add to knowledge base",
                "parameters": {
                    "name": "string",
                    "description": "string",
                    "pattern_type": "color_mapping|symmetry|size_transform|object_based|pattern_completion",
                    "conditions": "list of conditions when to apply",
                    "transformations": "list of transformations to apply",
                    "complexity": "int (1-5)",
                    "tags": "list of tags"
                }
            },
            {
                "name": "search_heuristics",
                "description": "Search heuristics by name, description, or tags",
                "parameters": {"query": "string"}
            },
            {
                "name": "get_heuristics_stats",
                "description": "Get statistics about heuristics knowledge base",
                "parameters": {}
            },
            {
                "name": "get_tools",
                "description": "Get list of available tools",
                "parameters": {}
            },
            {
                "name": "apply_tool",
                "description": "Apply a specific tool to the puzzle",
                "parameters": {"tool_id": "string", "parameters": "dict"}
            },
            {
                "name": "create_tool",
                "description": "Create a new tool",
                "parameters": {"name": "string", "description": "string", "code": "string"}
            },
            {
                "name": "find_transformation",
                "description": "Find the transformation pattern from training examples",
                "parameters": {}
            },
            {
                "name": "apply_transformation",
                "description": "Apply a transformation to the output grid",
                "parameters": {"transformation_type": "string", "parameters": "dict"}
            },
            {
                "name": "submit_solution",
                "description": "Submit current output grid for verification",
                "parameters": {}
            },
            {
                "name": "verify_solution",
                "description": "Verify if a specific solution is correct",
                "parameters": {"solution_grid": "2D array of integers (optional, uses current output if not provided)"}
            },
            {
                "name": "get_verification_stats",
                "description": "Get verification statistics for current puzzle",
                "parameters": {}
            },
            {
                "name": "check_if_solved",
                "description": "Check if current puzzle has been solved",
                "parameters": {}
            }
        ]
    
    def execute_function(self, function_name: str, parameters: Dict) -> Dict:
        """Execute a function call from the AI"""
        try:
            if function_name == "get_cell_color":
                return self._get_cell_color(**parameters)
            elif function_name == "set_output_cell":
                return self._set_output_cell(**parameters)
            elif function_name == "resize_output_grid":
                return self._resize_output_grid(**parameters)
            elif function_name == "copy_from_input":
                return self._copy_from_input()
            elif function_name == "clear_output":
                return self._clear_output()
            elif function_name == "analyze_pattern":
                return self._analyze_pattern()
            elif function_name == "get_all_heuristics":
                return {"heuristics": heuristics_manager.get_all_heuristics()}
            elif function_name == "get_relevant_heuristics":
                return self._get_relevant_heuristics()
            elif function_name == "rank_heuristics":
                return self._rank_heuristics(**parameters)
            elif function_name == "apply_heuristic":
                return self._apply_heuristic(**parameters)
            elif function_name == "test_heuristic":
                return self._test_heuristic(**parameters)
            elif function_name == "create_heuristic":
                return self._create_heuristic(**parameters)
            elif function_name == "search_heuristics":
                return {"heuristics": heuristics_manager.search_heuristics(parameters.get("query", ""))}
            elif function_name == "get_heuristics_stats":
                return heuristics_manager.get_statistics()
            elif function_name == "get_tools":
                return {"tools": self.tools}
            elif function_name == "apply_tool":
                return self._apply_tool(**parameters)
            elif function_name == "create_tool":
                return self._create_tool(**parameters)
            elif function_name == "find_transformation":
                return self._find_transformation()
            elif function_name == "apply_transformation":
                return self._apply_transformation(**parameters)
            elif function_name == "submit_solution":
                return self._submit_solution()
            elif function_name == "verify_solution":
                return self._verify_solution(**parameters)
            elif function_name == "get_verification_stats":
                return self._get_verification_stats()
            elif function_name == "check_if_solved":
                return self._check_if_solved()
            else:
                return {"error": f"Unknown function: {function_name}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_cell_color(self, grid_type: str, example_index: int, row: int, col: int) -> Dict:
        """Get color of a specific cell"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        try:
            if grid_type == "output":
                if self.output_grid:
                    return {"color": self.output_grid[row][col]}
            elif grid_type == "test":
                test_input = self.current_puzzle['test'][0]['input']
                return {"color": test_input[row][col]}
            elif grid_type == "input":
                example = self.current_puzzle['train'][example_index]
                return {"color": example['input'][row][col]}
            else:
                return {"error": "Invalid grid type"}
        except (IndexError, KeyError) as e:
            return {"error": f"Invalid cell coordinates: {e}"}
    
    def _set_output_cell(self, row: int, col: int, color: int) -> Dict:
        """Set color of output cell"""
        if not self.output_grid:
            return {"error": "No output grid"}
        
        try:
            self.output_grid[row][col] = color
            return {"success": True, "message": f"Set cell ({row},{col}) to color {color}"}
        except IndexError:
            return {"error": "Invalid cell coordinates"}
    
    def _resize_output_grid(self, width: int, height: int) -> Dict:
        """Resize the output grid"""
        new_grid = [[0 for _ in range(width)] for _ in range(height)]
        
        # Copy existing data
        if self.output_grid:
            for i in range(min(height, len(self.output_grid))):
                for j in range(min(width, len(self.output_grid[0]))):
                    new_grid[i][j] = self.output_grid[i][j]
        
        self.output_grid = new_grid
        return {"success": True, "message": f"Resized output to {height}x{width}"}
    
    def _copy_from_input(self) -> Dict:
        """Copy test input to output"""
        if not self.current_puzzle or 'test' not in self.current_puzzle:
            return {"error": "No test input available"}
        
        test_input = self.current_puzzle['test'][0]['input']
        self.output_grid = [row[:] for row in test_input]
        return {"success": True, "message": "Copied test input to output"}
    
    def _clear_output(self) -> Dict:
        """Clear output grid"""
        if not self.output_grid:
            return {"error": "No output grid"}
        
        for i in range(len(self.output_grid)):
            for j in range(len(self.output_grid[0])):
                self.output_grid[i][j] = 0
        
        return {"success": True, "message": "Cleared output grid"}
    
    def _analyze_pattern(self) -> Dict:
        """Analyze patterns in the puzzle"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        analysis = {
            "patterns": [],
            "transformations": [],
            "observations": []
        }
        
        # Analyze training examples
        for i, example in enumerate(self.current_puzzle.get('train', [])):
            inp = example['input']
            out = example['output']
            
            # Check size changes
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                analysis["observations"].append(f"Example {i+1}: Size change from {len(inp)}x{len(inp[0])} to {len(out)}x{len(out[0])}")
            
            # Check color mappings
            input_colors = set(cell for row in inp for cell in row)
            output_colors = set(cell for row in out for cell in row)
            
            if input_colors != output_colors:
                analysis["observations"].append(f"Example {i+1}: Color set changed from {input_colors} to {output_colors}")
        
        return analysis
    
    def _get_relevant_heuristics(self) -> Dict:
        """Get heuristics relevant to current puzzle"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        # Extract puzzle features
        features = self._extract_puzzle_features()
        
        # Get relevant heuristics
        relevant = heuristics_manager.get_relevant_heuristics(features)
        
        return {
            "heuristics": relevant,
            "total": len(relevant),
            "puzzle_features": features
        }
    
    def _rank_heuristics(self, heuristic_ids: List[str]) -> Dict:
        """Rank heuristics by effectiveness"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        puzzle_data = self._get_puzzle_data()
        ranked = heuristics_manager.rank_heuristics(heuristic_ids, puzzle_data)
        
        return {
            "ranked_heuristics": ranked,
            "top_recommendation": ranked[0] if ranked else None
        }
    
    def _apply_heuristic(self, heuristic_id: str) -> Dict:
        """Apply a heuristic to the puzzle"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        # Track the heuristic being applied
        self.last_applied_heuristic = heuristic_id
        
        puzzle_data = self._get_puzzle_data()
        result = heuristics_manager.apply_heuristic(heuristic_id, puzzle_data)
        
        # If heuristic suggests specific transformations, try to apply them
        if result.get("applied") and result.get("confidence", 0) > 0.5:
            # Update output grid based on suggested transformations
            # This is a simplified implementation
            if "color_replace" in result.get("transformations_suggested", []):
                self._apply_color_mapping()
            elif "mirror_horizontal" in result.get("transformations_suggested", []):
                self._apply_mirror_horizontal()
            elif "crop_to_content" in result.get("transformations_suggested", []):
                self._apply_crop()
        
        return result
    
    def _test_heuristic(self, heuristic_id: str) -> Dict:
        """Test a heuristic against training examples"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        test_results = []
        for i, example in enumerate(self.current_puzzle.get('train', [])):
            puzzle_data = {
                "puzzle_id": self.current_puzzle.get('id'),
                "input": example['input'],
                "expected_output": example['output']
            }
            
            result = heuristics_manager.test_heuristic(
                heuristic_id, 
                puzzle_data,
                example['output']
            )
            
            test_results.append({
                "example": i + 1,
                "success": result.get("success", False),
                "analysis": result.get("analysis", "")
            })
        
        success_rate = sum(1 for r in test_results if r["success"]) / len(test_results) if test_results else 0
        
        return {
            "heuristic_id": heuristic_id,
            "test_results": test_results,
            "overall_success_rate": success_rate,
            "recommendation": "Use this heuristic" if success_rate > 0.5 else "Try a different heuristic"
        }
    
    def _create_heuristic(self, name: str, description: str, pattern_type: str, 
                         conditions: List[str], transformations: List[str], 
                         complexity: int = 1, tags: List[str] = None) -> Dict:
        """Create a new heuristic and add to knowledge base"""
        heuristic = heuristics_manager.create_heuristic(
            name=name,
            description=description,
            pattern_type=pattern_type,
            conditions=conditions,
            transformations=transformations,
            complexity=complexity,
            tags=tags
        )
        
        # Return a success response
        return {
            "success": True,
            "heuristic_id": heuristic.id,
            "message": f"Added new heuristic: {name}",
            "heuristic": {
                "id": heuristic.id,
                "name": heuristic.name,
                "description": heuristic.description,
                "pattern_type": heuristic.pattern_type,
                "conditions": heuristic.conditions,
                "transformations": heuristic.transformations,
                "complexity": heuristic.complexity,
                "tags": heuristic.tags
            }
        }
    
    def _extract_puzzle_features(self) -> Dict:
        """Extract features from current puzzle for heuristic matching"""
        if not self.current_puzzle:
            return {}
        
        features = {
            "conditions": [],
            "pattern_type": None,
            "complexity": 1,
            "size_change": False,
            "colors": set()
        }
        
        # Analyze training examples
        for example in self.current_puzzle.get('train', []):
            inp = example.get('input', [])
            out = example.get('output', [])
            
            # Check size changes
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                features["size_change"] = True
                features["conditions"].append("size_transform")
            else:
                features["conditions"].append("same_grid_size")
            
            # Collect colors
            for row in inp:
                features["colors"].update(row)
            for row in out:
                features["colors"].update(row)
            
            # Check for patterns
            if self._has_symmetry(inp) or self._has_symmetry(out):
                features["conditions"].append("mirror_pattern")
                features["pattern_type"] = "symmetry"
            
            if self._has_repeating_pattern(inp):
                features["conditions"].append("repeating_pattern")
            
            if self._count_objects(inp) > 0:
                features["conditions"].append("discrete_objects")
        
        # Estimate complexity
        features["complexity"] = min(5, max(1, len(features["colors"]) - 2))
        
        return features
    
    def _get_puzzle_data(self) -> Dict:
        """Get puzzle data for heuristic processing"""
        if not self.current_puzzle:
            return {}
        
        return {
            "puzzle_id": self.current_puzzle.get('id'),
            "train": self.current_puzzle.get('train', []),
            "test": self.current_puzzle.get('test', []),
            "colors": list(self._extract_puzzle_features().get("colors", [])),
            "size_change": self._extract_puzzle_features().get("size_change", False)
        }
    
    def _has_symmetry(self, grid: List[List[int]]) -> bool:
        """Check if grid has symmetry"""
        if not grid:
            return False
        
        # Check horizontal symmetry
        for i in range(len(grid) // 2):
            if grid[i] != grid[-(i+1)]:
                break
        else:
            return True
        
        # Check vertical symmetry
        for row in grid:
            for j in range(len(row) // 2):
                if row[j] != row[-(j+1)]:
                    break
            else:
                return True
        
        return False
    
    def _has_repeating_pattern(self, grid: List[List[int]]) -> bool:
        """Check if grid has repeating patterns"""
        if not grid or len(grid) < 2:
            return False
        
        # Simple check for row repetition
        for i in range(len(grid) - 1):
            if grid[i] == grid[i + 1]:
                return True
        
        return False
    
    def _count_objects(self, grid: List[List[int]]) -> int:
        """Count discrete objects in grid"""
        # Simplified object counting
        if not grid:
            return 0
        
        visited = set()
        count = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) not in visited and grid[i][j] != 0:
                    # Found new object, do simple flood fill
                    self._mark_object(grid, i, j, visited)
                    count += 1
        
        return count
    
    def _mark_object(self, grid: List[List[int]], i: int, j: int, visited: set):
        """Mark connected cells as visited (simplified flood fill)"""
        if (i, j) in visited or i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
            return
        if grid[i][j] == 0:
            return
        
        visited.add((i, j))
        
        # Check neighbors
        self._mark_object(grid, i+1, j, visited)
        self._mark_object(grid, i-1, j, visited)
        self._mark_object(grid, i, j+1, visited)
        self._mark_object(grid, i, j-1, visited)
    
    def _apply_color_mapping(self):
        """Apply color mapping transformation"""
        # Simplified color mapping
        if self.output_grid and self.current_puzzle:
            # This would implement actual color mapping logic
            pass
    
    def _apply_mirror_horizontal(self):
        """Apply horizontal mirror transformation"""
        if self.output_grid:
            self.output_grid = self.output_grid[::-1]
    
    def _apply_crop(self):
        """Apply cropping to content"""
        if self.output_grid:
            # Find bounding box of non-zero elements
            min_row, max_row = len(self.output_grid), -1
            min_col, max_col = len(self.output_grid[0]) if self.output_grid else 0, -1
            
            for i, row in enumerate(self.output_grid):
                for j, cell in enumerate(row):
                    if cell != 0:
                        min_row = min(min_row, i)
                        max_row = max(max_row, i)
                        min_col = min(min_col, j)
                        max_col = max(max_col, j)
            
            if max_row >= 0 and max_col >= 0:
                # Crop to bounding box
                cropped = []
                for i in range(min_row, max_row + 1):
                    cropped.append(self.output_grid[i][min_col:max_col + 1])
                self.output_grid = cropped
    
    def _apply_tool(self, tool_id: str, parameters: Dict) -> Dict:
        """Apply a tool to the puzzle"""
        # This would integrate with the tools system
        return {"message": f"Would apply tool {tool_id} with parameters {parameters}"}
    
    def _create_tool(self, name: str, description: str, code: str) -> Dict:
        """Create a new tool"""
        new_tool = {
            "id": f"tool_{len(self.tools)}",
            "name": name,
            "description": description,
            "code": code
        }
        self.tools.append(new_tool)
        return {"success": True, "tool": new_tool}
    
    def _find_transformation(self) -> Dict:
        """Find transformation pattern"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        # Analyze transformations
        transformations = []
        
        # This would implement pattern detection logic
        return {"transformations": transformations}
    
    def _apply_transformation(self, transformation_type: str, parameters: Dict) -> Dict:
        """Apply a transformation"""
        # This would implement various transformations
        return {"message": f"Would apply {transformation_type} transformation"}
    
    def _submit_solution(self) -> Dict:
        """Submit current output grid for verification"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        if not self.output_grid:
            return {"error": "No output grid to submit"}
        
        puzzle_id = self.current_puzzle.get('id', 'unknown')
        
        # Get expected output from test data (if available)
        # NOTE: In production, this would be secured server-side
        expected_output = None
        if 'test' in self.current_puzzle and self.current_puzzle['test']:
            # For training/evaluation puzzles, we might have the answer
            # But we don't expose it directly to the AI
            test_data = self.current_puzzle['test'][0]
            if 'output' in test_data:
                expected_output = test_data['output']
        
        # Verify the solution
        result = verification_oracle.verify_solution(
            puzzle_id=puzzle_id,
            submitted_output=self.output_grid,
            expected_output=expected_output
        )
        
        # Update heuristics based on result if we were using one
        if hasattr(self, 'last_applied_heuristic') and self.last_applied_heuristic:
            if result.get('correct'):
                heuristics_manager.update_heuristic_success(
                    self.last_applied_heuristic, 
                    True, 
                    puzzle_id
                )
        
        return {
            "submission_result": result,
            "puzzle_id": puzzle_id,
            "message": self._format_verification_message(result)
        }
    
    def _verify_solution(self, solution_grid: List[List[int]] = None) -> Dict:
        """Verify a specific solution or current output"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        grid_to_verify = solution_grid if solution_grid else self.output_grid
        
        if not grid_to_verify:
            return {"error": "No solution grid to verify"}
        
        puzzle_id = self.current_puzzle.get('id', 'unknown')
        
        # Get expected output (secured)
        expected_output = None
        if 'test' in self.current_puzzle and self.current_puzzle['test']:
            test_data = self.current_puzzle['test'][0]
            if 'output' in test_data:
                expected_output = test_data['output']
        
        # Verify the solution
        result = verification_oracle.verify_solution(
            puzzle_id=puzzle_id,
            submitted_output=grid_to_verify,
            expected_output=expected_output
        )
        
        return {
            "verification": result,
            "message": self._format_verification_message(result)
        }
    
    def _get_verification_stats(self) -> Dict:
        """Get verification statistics for current puzzle"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        puzzle_id = self.current_puzzle.get('id', 'unknown')
        stats = verification_oracle.get_verification_stats(puzzle_id)
        
        status_msg = 'SOLVED' if stats['solved'] else f"{stats['attempts_remaining']} attempts remaining"
        return {
            "stats": stats,
            "message": f"Puzzle {puzzle_id}: {stats['attempts']} attempts, {status_msg}"
        }
    
    def _check_if_solved(self) -> Dict:
        """Check if current puzzle has been solved"""
        if not self.current_puzzle:
            return {"error": "No puzzle loaded"}
        
        puzzle_id = self.current_puzzle.get('id', 'unknown')
        is_solved = verification_oracle.check_if_solved(puzzle_id)
        
        return {
            "puzzle_id": puzzle_id,
            "solved": is_solved,
            "message": f"Puzzle {puzzle_id} is {'SOLVED' if is_solved else 'not solved yet'}"
        }
    
    def _format_verification_message(self, result: Dict) -> str:
        """Format verification result into readable message"""
        if result.get('error'):
            return f"Verification error: {result['error']}"
        
        if result.get('correct'):
            return f"🎉 Correct! {result.get('feedback', 'Solution verified successfully!')}"
        
        if result.get('verified') is False:
            msg = result.get('feedback', 'Solution is incorrect')
            if result.get('accuracy'):
                msg += f" (Accuracy: {result['accuracy']*100:.1f}%)"
            if result.get('hint'):
                msg += f"\nHint: {result['hint']}"
            if result.get('attempts_remaining'):
                msg += f"\nAttempts remaining: {result['attempts_remaining']}"
            return msg
        
        return result.get('message', 'Verification status unknown')