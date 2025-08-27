"""
Grid Display Component for ARC Puzzles
Handles visualization of input/output grids with color coding
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import List, Optional, Callable

class GridDisplay(tk.Frame):
    """Widget for displaying ARC puzzle grids with color coding"""
    
    # ARC color palette (standard colors used in ARC challenges)
    COLORS = {
        0: '#000000',  # Black
        1: '#0074D9',  # Blue
        2: '#FF4136',  # Red
        3: '#2ECC40',  # Green
        4: '#FFDC00',  # Yellow
        5: '#AAAAAA',  # Gray
        6: '#F012BE',  # Fuchsia
        7: '#FF851B',  # Orange
        8: '#7FDBFF',  # Aqua
        9: '#870C25',  # Maroon
    }
    
    def __init__(self, parent, title: str = "Grid", cell_size: int = 30, 
                 on_cell_click: Optional[Callable] = None):
        super().__init__(parent)
        self.title = title
        self.cell_size = cell_size
        self.on_cell_click = on_cell_click
        self.grid_data = None
        self.canvas = None
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        # Title label
        title_label = tk.Label(self, text=self.title, font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Canvas for grid display
        self.canvas = tk.Canvas(self, bg='white', highlightthickness=1, 
                               highlightbackground='gray')
        self.canvas.pack(padx=10, pady=5)
        
        if self.on_cell_click:
            self.canvas.bind("<Button-1>", self._handle_click)
    
    def display_grid(self, grid: List[List[int]]):
        """Display a grid with the given data"""
        if not grid:
            return
        
        self.grid_data = np.array(grid)
        rows, cols = self.grid_data.shape
        
        # Resize canvas
        canvas_width = cols * self.cell_size
        canvas_height = rows * self.cell_size
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Clear previous content
        self.canvas.delete("all")
        
        # Draw grid cells
        for i in range(rows):
            for j in range(cols):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                color_value = self.grid_data[i, j]
                fill_color = self.COLORS.get(color_value, '#FFFFFF')
                
                # Draw cell rectangle
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=fill_color, 
                                           outline='black', 
                                           width=1)
                
                # Add cell value text for clarity
                if color_value != 0:  # Don't show 0 on black background
                    text_color = 'white' if color_value in [0, 1, 9] else 'black'
                    self.canvas.create_text(x1 + self.cell_size//2, 
                                          y1 + self.cell_size//2,
                                          text=str(color_value),
                                          fill=text_color,
                                          font=('Arial', 8))
    
    def _handle_click(self, event):
        """Handle mouse clicks on grid cells"""
        if self.grid_data is None or not self.on_cell_click:
            return
        
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if 0 <= row < self.grid_data.shape[0] and 0 <= col < self.grid_data.shape[1]:
            self.on_cell_click(row, col, self.grid_data[row, col])
    
    def clear(self):
        """Clear the grid display"""
        if self.canvas:
            self.canvas.delete("all")
        self.grid_data = None

class PuzzleViewer(tk.Frame):
    """Complete puzzle viewer with training examples and test cases"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.current_challenge = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the puzzle viewer interface"""
        # Main container with scrollable frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Challenge info
        self.info_frame = tk.Frame(main_frame)
        self.info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.challenge_label = tk.Label(self.info_frame, text="No challenge loaded", 
                                       font=('Arial', 14, 'bold'))
        self.challenge_label.pack()
        
        # Notebook for training/test tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Training examples tab
        self.train_frame = tk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Training Examples")
        
        # Test cases tab
        self.test_frame = tk.Frame(self.notebook)
        self.notebook.add(self.test_frame, text="Test Cases")
        
        # Scrollable containers
        self._setup_scrollable_frame(self.train_frame, "train_canvas")
        self._setup_scrollable_frame(self.test_frame, "test_canvas")
    
    def _setup_scrollable_frame(self, parent, canvas_name):
        """Setup a scrollable frame for examples"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        setattr(self, canvas_name, canvas)
        setattr(self, f"{canvas_name}_frame", scrollable_frame)
    
    def load_challenge(self, challenge):
        """Load and display a challenge"""
        self.current_challenge = challenge
        self.challenge_label.config(text=f"Challenge: {challenge.id}")
        
        # Clear previous content
        for widget in self.train_canvas_frame.winfo_children():
            widget.destroy()
        for widget in self.test_canvas_frame.winfo_children():
            widget.destroy()
        
        # Display training examples
        for i, example in enumerate(challenge.get_training_examples()):
            self._display_example(self.train_canvas_frame, f"Training {i+1}", 
                                example, show_output=True)
        
        # Display test cases
        for i, example in enumerate(challenge.get_test_examples()):
            self._display_example(self.test_canvas_frame, f"Test {i+1}", 
                                example, show_output=False)
    
    def _display_example(self, parent, title, example, show_output=True):
        """Display a single input/output example"""
        example_frame = tk.Frame(parent, relief=tk.RIDGE, borderwidth=2)
        example_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Example title
        title_label = tk.Label(example_frame, text=title, font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Container for input/output grids
        grids_frame = tk.Frame(example_frame)
        grids_frame.pack()
        
        # Input grid
        input_grid = GridDisplay(grids_frame, title="Input", cell_size=25)
        input_grid.pack(side=tk.LEFT, padx=10)
        input_grid.display_grid(example['input'])
        
        # Arrow
        if show_output and 'output' in example:
            arrow_label = tk.Label(grids_frame, text="→", font=('Arial', 20))
            arrow_label.pack(side=tk.LEFT, padx=10)
            
            # Output grid
            output_grid = GridDisplay(grids_frame, title="Output", cell_size=25)
            output_grid.pack(side=tk.LEFT, padx=10)
            output_grid.display_grid(example['output'])
        elif not show_output:
            # For test cases, show a placeholder for solution
            solution_label = tk.Label(grids_frame, text="Solution: ?", 
                                    font=('Arial', 12), fg='red')
            solution_label.pack(side=tk.LEFT, padx=20)
