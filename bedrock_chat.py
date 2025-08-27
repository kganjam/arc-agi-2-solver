"""
Bedrock Chat Integration for ARC Puzzle Solving
Connects to AWS Bedrock for AI-powered puzzle analysis
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, Optional

class BedrockChat:
    """Chat interface for AWS Bedrock"""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        
    def send_message(self, message: str, context: Optional[str] = None) -> str:
        """Send message to Bedrock and get response"""
        try:
            # Prepare the request payload
            system_prompt = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. 
You help analyze patterns, suggest transformations, and guide users through solving abstract reasoning challenges.
Focus on identifying visual patterns, transformations, and logical rules in grid-based puzzles."""
            
            if context:
                system_prompt += f"\n\nCurrent puzzle context: {context}"
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            }
            
            # Prepare the request
            url = f"{self.base_url}/model/{self.model_id}/invoke"
            headers = {
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            
            # Send request
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    return result.get('content', [{}])[0].get('text', 'No response received')
                else:
                    return f"Error: HTTP {response.status}"
                    
        except Exception as e:
            return f"Error connecting to Bedrock: {str(e)}"
    
    def analyze_puzzle(self, puzzle_data: Dict) -> str:
        """Analyze an ARC puzzle using Bedrock"""
        # Format puzzle data for analysis
        context = self._format_puzzle_context(puzzle_data)
        
        message = """Please analyze this ARC puzzle. Look for:
1. Visual patterns in the input/output examples
2. Transformation rules that convert input to output
3. Suggestions for solving the test cases
4. Any symmetries, color changes, or geometric transformations

Provide a clear, step-by-step analysis."""
        
        return self.send_message(message, context)
    
    def _format_puzzle_context(self, puzzle_data: Dict) -> str:
        """Format puzzle data for context"""
        context = "ARC Puzzle Data:\n"
        
        if 'train' in puzzle_data:
            context += f"Training Examples: {len(puzzle_data['train'])}\n"
            for i, example in enumerate(puzzle_data['train'][:2]):  # Limit to first 2 examples
                context += f"\nExample {i+1}:\n"
                context += f"Input: {example.get('input', [])}\n"
                context += f"Output: {example.get('output', [])}\n"
        
        if 'test' in puzzle_data:
            context += f"\nTest Cases: {len(puzzle_data['test'])}\n"
            for i, test in enumerate(puzzle_data['test'][:1]):  # Limit to first test case
                context += f"Test {i+1} Input: {test.get('input', [])}\n"
        
        return context

class BedrockChatWidget:
    """Chat widget integrated with Bedrock"""
    
    def __init__(self, parent, bearer_token: str):
        self.parent = parent
        self.bedrock_chat = BedrockChat(bearer_token)
        self.current_puzzle = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the chat UI"""
        import tkinter as tk
        from tkinter import scrolledtext
        
        # Chat frame
        self.chat_frame = tk.LabelFrame(self.parent, text="AI Assistant (Bedrock)", padx=5, pady=5)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Messages display
        self.messages_text = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            height=15, 
            state=tk.DISABLED,
            font=('Consolas', 9)
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Configure text tags
        self.messages_text.tag_configure("user", foreground="blue", font=('Consolas', 9, 'bold'))
        self.messages_text.tag_configure("assistant", foreground="darkgreen", font=('Consolas', 9))
        self.messages_text.tag_configure("system", foreground="gray", font=('Consolas', 9, 'italic'))
        
        # Input area
        input_frame = tk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X)
        
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Consolas', 9))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)
        
        # Buttons
        button_frame = tk.Frame(input_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Button(button_frame, text="Send", command=self._send_message).pack(fill=tk.X, pady=1)
        tk.Button(button_frame, text="Analyze\nPuzzle", command=self._analyze_puzzle).pack(fill=tk.X, pady=1)
        tk.Button(button_frame, text="Clear", command=self._clear_chat).pack(fill=tk.X, pady=1)
        
        # Add welcome message
        self._add_message("system", "AI Assistant connected! Ask me about ARC puzzles or load a puzzle for analysis.")
    
    def _on_enter(self, event):
        """Handle Enter key"""
        if not (event.state & 0x1):  # Not Shift+Enter
            self._send_message()
            return "break"
    
    def _on_shift_enter(self, event):
        """Handle Shift+Enter for newline"""
        return None
    
    def _send_message(self):
        """Send user message"""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        
        self.input_text.delete("1.0", tk.END)
        self._add_message("user", message)
        
        # Get response from Bedrock
        self._add_message("system", "Thinking...")
        self.parent.update()
        
        context = None
        if self.current_puzzle:
            context = self._format_puzzle_context(self.current_puzzle)
        
        response = self.bedrock_chat.send_message(message, context)
        
        # Remove "Thinking..." message
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("end-3l", "end-1l")
        self.messages_text.config(state=tk.DISABLED)
        
        self._add_message("assistant", response)
    
    def _analyze_puzzle(self):
        """Analyze current puzzle"""
        if not self.current_puzzle:
            self._add_message("system", "No puzzle loaded. Please select a puzzle first.")
            return
        
        self._add_message("system", "Analyzing puzzle...")
        self.parent.update()
        
        response = self.bedrock_chat.analyze_puzzle(self.current_puzzle)
        
        # Remove "Analyzing..." message
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("end-3l", "end-1l")
        self.messages_text.config(state=tk.DISABLED)
        
        self._add_message("assistant", response)
    
    def _clear_chat(self):
        """Clear chat history"""
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("1.0", tk.END)
        self.messages_text.config(state=tk.DISABLED)
        self._add_message("system", "Chat cleared. How can I help with ARC puzzles?")
    
    def _add_message(self, sender: str, message: str):
        """Add message to chat display"""
        import datetime
        
        self.messages_text.config(state=tk.NORMAL)
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        header = f"[{timestamp}] {sender.title()}: "
        
        self.messages_text.insert(tk.END, header, sender)
        self.messages_text.insert(tk.END, message + "\n\n")
        
        self.messages_text.config(state=tk.DISABLED)
        self.messages_text.see(tk.END)
    
    def set_current_puzzle(self, puzzle_data: Dict):
        """Set the current puzzle for context"""
        self.current_puzzle = puzzle_data
        self._add_message("system", f"Puzzle loaded with {len(puzzle_data.get('train', []))} training examples.")
    
    def _format_puzzle_context(self, puzzle_data: Dict) -> str:
        """Format puzzle for context"""
        return self.bedrock_chat._format_puzzle_context(puzzle_data)
