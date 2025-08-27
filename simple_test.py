"""
Simple test version of ARC system using only built-in Python libraries
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

# Simple chat widget without external dependencies
class SimpleChatWidget:
    def __init__(self, parent, token):
        self.parent = parent
        self.token = token
        self.current_puzzle = None
        self.setup_ui()
    
    def setup_ui(self):
        from tkinter import scrolledtext
        
        # Chat frame
        self.chat_frame = tk.LabelFrame(self.parent, text="AI Assistant", padx=5, pady=5)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Messages display
        self.messages_text = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=('Consolas', 9)
        )
        self.messages_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Input area
        input_frame = tk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X)
        
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Consolas', 9))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Buttons
        button_frame = tk.Frame(input_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Button(button_frame, text="Send", command=self._send_message).pack(fill=tk.X, pady=1)
        tk.Button(button_frame, text="Analyze", command=self._analyze_puzzle).pack(fill=tk.X, pady=1)
        
        self._add_message("system", "Chat ready! (Bedrock integration available)")
    
    def _send_message(self):
        message = self.input_text.get("1.0", tk.END).strip()
        if message:
            self.input_text.delete("1.0", tk.END)
            self._add_message("user", message)
            self._add_message("assistant", f"Echo: {message} (Bedrock integration ready)")
    
    def _analyze_puzzle(self):
        if self.current_puzzle:
            analysis = f"Puzzle has {len(self.current_puzzle.get('train', []))} training examples"
            self._add_message("assistant", analysis)
        else:
            self._add_message("system", "No puzzle loaded")
    
    def _add_message(self, sender, message):
        import datetime
        self.messages_text.config(state=tk.NORMAL)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        self.messages_text.config(state=tk.DISABLED)
        self.messages_text.see(tk.END)
    
    def set_current_puzzle(self, puzzle_data):
        self.current_puzzle = puzzle_data
        self._add_message("system", f"Puzzle loaded with {len(puzzle_data.get('train', []))} examples")

class SimpleARCViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ARC AGI 2 - With AI Chat")
        self.root.geometry("1200x800")
        
        self.challenges = {}
        self.current_challenge = None
        
        # Bedrock token
        self.bedrock_token = "ABSKQmVkcm9ja0FQSUtleS1ka2t5LWFOLTAYNDQ2MzAwMTpCSVE9aERoMXRpeFVUQ3FlGZSZGp6UmI6QytaUtPSGtEaGJ0SWJiOU1FUVJiTT0="
        
        self.setup_ui()
        self.create_sample_data()
        self.load_challenges()
    
    def setup_ui(self):
        # Main container with 3 panels
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Challenge list
        left_frame = tk.Frame(main_container, width=200)
        main_container.add(left_frame, minsize=180)
        
        tk.Label(left_frame, text="Challenges:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, padx=5, pady=5)
        
        self.challenge_listbox = tk.Listbox(left_frame, height=15)
        self.challenge_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.challenge_listbox.bind('<<ListboxSelect>>', self.on_challenge_select)
        
        # Center panel - Grid display
        center_frame = tk.Frame(main_container)
        main_container.add(center_frame, minsize=400)
        
        tk.Label(center_frame, text="Puzzle Viewer:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, padx=5)
        
        # Grid canvas
        self.canvas = tk.Canvas(center_frame, bg='white', width=500, height=300)
        self.canvas.pack(pady=10, padx=5)
        
        # Info text
        self.info_text = tk.Text(center_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Test button
        tk.Button(center_frame, text="Analyze Pattern", command=self.analyze_pattern).pack(pady=5)
        
        # Right panel - Chat interface
        right_frame = tk.Frame(main_container, width=400)
        main_container.add(right_frame, minsize=350)
        
        # Initialize chat widget
        self.chat_widget = SimpleChatWidget(right_frame, self.bedrock_token)
    
    def create_sample_data(self):
        """Create sample ARC challenges"""
        os.makedirs("data", exist_ok=True)
        
        # Simple pattern copying
        sample1 = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[1, 0], [0, 1]]
                },
                {
                    "input": [[2, 3], [3, 2]],
                    "output": [[2, 3], [3, 2]]
                }
            ],
            "test": [
                {
                    "input": [[4, 5], [5, 4]]
                }
            ]
        }
        
        # Color transformation
        sample2 = {
            "train": [
                {
                    "input": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                    "output": [[3, 3, 3], [3, 2, 3], [3, 3, 3]]
                }
            ],
            "test": [
                {
                    "input": [[2, 2, 2], [2, 1, 2], [2, 2, 2]]
                }
            ]
        }
        
        with open("data/sample1.json", 'w') as f:
            json.dump(sample1, f, indent=2)
        
        with open("data/sample2.json", 'w') as f:
            json.dump(sample2, f, indent=2)
    
    def load_challenges(self):
        """Load challenges from data directory"""
        if not os.path.exists("data"):
            return
        
        for filename in os.listdir("data"):
            if filename.endswith('.json'):
                try:
                    with open(f"data/{filename}", 'r') as f:
                        data = json.load(f)
                    
                    challenge_id = filename.replace('.json', '')
                    self.challenges[challenge_id] = data
                    self.challenge_listbox.insert(tk.END, challenge_id)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def on_challenge_select(self, event):
        """Handle challenge selection"""
        selection = self.challenge_listbox.curselection()
        if selection:
            challenge_id = self.challenge_listbox.get(selection[0])
            self.current_challenge = self.challenges[challenge_id]
            self.display_challenge()
            # Update chat widget with current puzzle
            self.chat_widget.set_current_puzzle(self.current_challenge)
    
    def display_challenge(self):
        """Display the selected challenge"""
        if not self.current_challenge:
            return
        
        self.canvas.delete("all")
        
        # Display first training example
        train_examples = self.current_challenge.get('train', [])
        if train_examples:
            example = train_examples[0]
            
            # Draw input grid
            self.draw_grid(example['input'], 50, 50, "Input")
            
            # Draw arrow
            self.canvas.create_text(200, 120, text="→", font=('Arial', 20))
            
            # Draw output grid if available
            if 'output' in example:
                self.draw_grid(example['output'], 250, 50, "Output")
        
        # Update info text
        self.update_info()
    
    def draw_grid(self, grid, start_x, start_y, title):
        """Draw a grid on the canvas"""
        cell_size = 30
        
        # Title
        self.canvas.create_text(start_x + len(grid[0]) * cell_size // 2, 
                               start_y - 20, text=title, font=('Arial', 12, 'bold'))
        
        # Grid cells
        colors = {
            0: '#000000',  # Black
            1: '#0074D9',  # Blue
            2: '#FF4136',  # Red
            3: '#2ECC40',  # Green
            4: '#FFDC00',  # Yellow
            5: '#AAAAAA',  # Gray
        }
        
        for i, row in enumerate(grid):
            for j, cell_value in enumerate(row):
                x1 = start_x + j * cell_size
                y1 = start_y + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                color = colors.get(cell_value, '#FFFFFF')
                
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=color, outline='black')
                
                # Add number
                if cell_value != 0:
                    text_color = 'white' if cell_value in [0, 1] else 'black'
                    self.canvas.create_text(x1 + cell_size//2, y1 + cell_size//2,
                                          text=str(cell_value), fill=text_color)
    
    def update_info(self):
        """Update the info text area"""
        self.info_text.delete("1.0", tk.END)
        
        if not self.current_challenge:
            return
        
        info = "Challenge Analysis:\n\n"
        
        train_count = len(self.current_challenge.get('train', []))
        test_count = len(self.current_challenge.get('test', []))
        
        info += f"Training examples: {train_count}\n"
        info += f"Test cases: {test_count}\n\n"
        
        # Analyze first training example
        if self.current_challenge.get('train'):
            example = self.current_challenge['train'][0]
            input_grid = example['input']
            
            info += f"Input grid size: {len(input_grid)}x{len(input_grid[0])}\n"
            
            # Count colors
            colors = set()
            for row in input_grid:
                colors.update(row)
            
            info += f"Colors used: {sorted(list(colors))}\n"
            
            if 'output' in example:
                output_grid = example['output']
                info += f"Output grid size: {len(output_grid)}x{len(output_grid[0])}\n"
                
                # Check if grids are identical
                if input_grid == output_grid:
                    info += "Pattern: Identity (input = output)\n"
                else:
                    info += "Pattern: Transformation detected\n"
        
        self.info_text.insert("1.0", info)
    
    def analyze_pattern(self):
        """Simple pattern analysis"""
        if not self.current_challenge:
            messagebox.showinfo("Info", "Please select a challenge first")
            return
        
        analysis = "Pattern Analysis Results:\n\n"
        
        train_examples = self.current_challenge.get('train', [])
        
        for i, example in enumerate(train_examples):
            analysis += f"Example {i+1}:\n"
            input_grid = example['input']
            
            # Simple analysis
            if 'output' in example:
                output_grid = example['output']
                
                if input_grid == output_grid:
                    analysis += "  - Identity transformation\n"
                elif len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
                    analysis += "  - Same size transformation\n"
                else:
                    analysis += "  - Size change transformation\n"
            
            analysis += f"  - Grid size: {len(input_grid)}x{len(input_grid[0])}\n"
            analysis += "\n"
        
        messagebox.showinfo("Pattern Analysis", analysis)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleARCViewer()
    app.run()
