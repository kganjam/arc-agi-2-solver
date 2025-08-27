"""
Interactive Chat Interface for ARC Puzzle Solving
Provides a chat window for user interaction and collaboration
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from typing import Callable, Optional, List, Dict
import threading

class ChatMessage:
    """Represents a single chat message"""
    
    def __init__(self, sender: str, content: str, message_type: str = "text"):
        self.sender = sender
        self.content = content
        self.message_type = message_type  # text, heuristic, solution, error
        self.timestamp = datetime.now()

class ChatInterface(tk.Frame):
    """Interactive chat interface for puzzle solving collaboration"""
    
    def __init__(self, parent, on_message_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.on_message_callback = on_message_callback
        self.message_history: List[ChatMessage] = []
        self.current_challenge = None
        self.setup_ui()
        self.add_system_message("Welcome! I'm here to help you solve ARC puzzles. Load a challenge to get started.")
    
    def setup_ui(self):
        """Setup the chat interface UI"""
        # Chat display area
        self.chat_frame = tk.Frame(self)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Messages display
        self.messages_text = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            height=20, 
            state=tk.DISABLED,
            font=('Consolas', 10)
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different message types
        self.messages_text.tag_configure("user", foreground="blue", font=('Consolas', 10, 'bold'))
        self.messages_text.tag_configure("system", foreground="green", font=('Consolas', 10, 'italic'))
        self.messages_text.tag_configure("heuristic", foreground="purple", font=('Consolas', 10))
        self.messages_text.tag_configure("error", foreground="red", font=('Consolas', 10))
        self.messages_text.tag_configure("solution", foreground="darkgreen", font=('Consolas', 10, 'bold'))
        
        # Input area
        input_frame = tk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Message input
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Consolas', 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Return>", self._on_enter_key)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)
        
        # Send button
        self.send_button = tk.Button(input_frame, text="Send", command=self._send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # Quick actions frame
        actions_frame = tk.Frame(self.chat_frame)
        actions_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Quick action buttons
        tk.Button(actions_frame, text="Analyze Pattern", 
                 command=lambda: self._quick_action("analyze_pattern")).pack(side=tk.LEFT, padx=2)
        tk.Button(actions_frame, text="Suggest Heuristic", 
                 command=lambda: self._quick_action("suggest_heuristic")).pack(side=tk.LEFT, padx=2)
        tk.Button(actions_frame, text="Test Solution", 
                 command=lambda: self._quick_action("test_solution")).pack(side=tk.LEFT, padx=2)
        tk.Button(actions_frame, text="Clear Chat", 
                 command=self._clear_chat).pack(side=tk.RIGHT, padx=2)
    
    def _on_enter_key(self, event):
        """Handle Enter key press"""
        if event.state & 0x1:  # Shift is pressed
            return "break"  # Allow newline
        else:
            self._send_message()
            return "break"  # Prevent newline
    
    def _on_shift_enter(self, event):
        """Handle Shift+Enter for newline"""
        return None  # Allow default behavior (newline)
    
    def _send_message(self):
        """Send user message"""
        content = self.input_text.get("1.0", tk.END).strip()
        if content:
            self.add_user_message(content)
            self.input_text.delete("1.0", tk.END)
            
            if self.on_message_callback:
                # Process message in background to avoid UI blocking
                threading.Thread(target=self._process_message, args=(content,), daemon=True).start()
    
    def _process_message(self, content: str):
        """Process user message and generate response"""
        try:
            if self.on_message_callback:
                response = self.on_message_callback(content, self.current_challenge)
                if response:
                    self.add_system_message(response)
        except Exception as e:
            self.add_error_message(f"Error processing message: {str(e)}")
    
    def _quick_action(self, action: str):
        """Handle quick action buttons"""
        actions = {
            "analyze_pattern": "Please analyze the pattern in the current puzzle and suggest transformation rules.",
            "suggest_heuristic": "Can you suggest heuristics that might apply to this puzzle?",
            "test_solution": "Let's test potential solutions for this puzzle."
        }
        
        if action in actions:
            self.add_user_message(actions[action])
            if self.on_message_callback:
                threading.Thread(target=self._process_message, args=(actions[action],), daemon=True).start()
    
    def _clear_chat(self):
        """Clear chat history"""
        self.message_history.clear()
        self.messages_text.config(state=tk.NORMAL)
        self.messages_text.delete("1.0", tk.END)
        self.messages_text.config(state=tk.DISABLED)
        self.add_system_message("Chat cleared. How can I help you with the current puzzle?")
    
    def add_user_message(self, content: str):
        """Add a user message to the chat"""
        message = ChatMessage("User", content, "text")
        self.message_history.append(message)
        self._display_message(message, "user")
    
    def add_system_message(self, content: str):
        """Add a system message to the chat"""
        message = ChatMessage("System", content, "text")
        self.message_history.append(message)
        self._display_message(message, "system")
    
    def add_heuristic_message(self, content: str):
        """Add a heuristic-related message"""
        message = ChatMessage("System", content, "heuristic")
        self.message_history.append(message)
        self._display_message(message, "heuristic")
    
    def add_solution_message(self, content: str):
        """Add a solution-related message"""
        message = ChatMessage("System", content, "solution")
        self.message_history.append(message)
        self._display_message(message, "solution")
    
    def add_error_message(self, content: str):
        """Add an error message"""
        message = ChatMessage("System", content, "error")
        self.message_history.append(message)
        self._display_message(message, "error")
    
    def _display_message(self, message: ChatMessage, tag: str):
        """Display a message in the chat window"""
        self.messages_text.config(state=tk.NORMAL)
        
        # Add timestamp and sender
        timestamp = message.timestamp.strftime("%H:%M:%S")
        header = f"[{timestamp}] {message.sender}: "
        
        self.messages_text.insert(tk.END, header, tag)
        self.messages_text.insert(tk.END, message.content + "\n\n")
        
        self.messages_text.config(state=tk.DISABLED)
        self.messages_text.see(tk.END)
    
    def set_current_challenge(self, challenge):
        """Set the current challenge for context"""
        self.current_challenge = challenge
        if challenge:
            self.add_system_message(f"Loaded challenge: {challenge.id}. What would you like to explore?")
    
    def get_message_history(self) -> List[ChatMessage]:
        """Get the complete message history"""
        return self.message_history.copy()

class PuzzleAnalyzer:
    """Analyzes puzzles and provides insights for chat interface"""
    
    def __init__(self, knowledge_store):
        self.knowledge_store = knowledge_store
    
    def analyze_challenge(self, challenge) -> str:
        """Analyze a challenge and return insights"""
        if not challenge:
            return "No challenge loaded to analyze."
        
        insights = []
        
        # Basic statistics
        train_examples = challenge.get_training_examples()
        test_examples = challenge.get_test_examples()
        
        insights.append(f"Challenge {challenge.id} has {len(train_examples)} training examples and {len(test_examples)} test cases.")
        
        # Grid size analysis
        max_height, max_width = challenge.get_grid_dimensions()
        insights.append(f"Maximum grid size: {max_height}x{max_width}")
        
        # Color analysis
        colors_used = set()
        for example in train_examples:
            for grid_type in ['input', 'output']:
                if grid_type in example:
                    grid = example[grid_type]
                    for row in grid:
                        colors_used.update(row)
        
        insights.append(f"Colors used: {sorted(list(colors_used))}")
        
        # Suggest relevant heuristics
        relevant_heuristics = self.knowledge_store.search_heuristics("transformation")
        if relevant_heuristics:
            insights.append(f"Potentially relevant heuristics: {', '.join([h.name for h in relevant_heuristics[:3]])}")
        
        return "\n".join(insights)
    
    def suggest_heuristics(self, challenge) -> str:
        """Suggest heuristics based on challenge characteristics"""
        if not challenge:
            return "No challenge loaded to analyze."
        
        suggestions = []
        
        # Get all heuristics sorted by success rate
        heuristics = sorted(self.knowledge_store.get_all_heuristics(), 
                          key=lambda h: h.success_rate, reverse=True)
        
        suggestions.append("Top heuristics to try:")
        for i, heuristic in enumerate(heuristics[:5]):
            suggestions.append(f"{i+1}. {heuristic.name} (Success: {heuristic.success_rate:.1%})")
            suggestions.append(f"   {heuristic.description}")
        
        return "\n".join(suggestions)
    
    def process_user_input(self, user_input: str, challenge) -> str:
        """Process user input and generate appropriate response"""
        user_input = user_input.lower().strip()
        
        if "analyze" in user_input or "pattern" in user_input:
            return self.analyze_challenge(challenge)
        elif "heuristic" in user_input or "suggest" in user_input:
            return self.suggest_heuristics(challenge)
        elif "help" in user_input:
            return self._get_help_message()
        else:
            return "I can help you analyze patterns, suggest heuristics, or provide general guidance. What would you like to explore?"
    
    def _get_help_message(self) -> str:
        """Get help message for chat interface"""
        return """Available commands:
- 'analyze' or 'pattern' - Analyze the current puzzle
- 'heuristic' or 'suggest' - Get heuristic suggestions
- 'help' - Show this help message

You can also describe what you're seeing in the puzzle, and I'll try to help you understand it better."""
