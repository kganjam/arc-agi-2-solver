"""
Basic test - minimal ARC viewer without complex dependencies
"""

import tkinter as tk
from tkinter import messagebox
import json
import os

class BasicARCTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ARC Basic Test")
        self.root.geometry("600x400")
        
        # Simple UI
        tk.Label(self.root, text="ARC Basic Test", font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Test button
        tk.Button(self.root, text="Test Connection", command=self.test_connection).pack(pady=10)
        
        # Text area for output
        self.output = tk.Text(self.root, height=15, width=70)
        self.output.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.output.insert(tk.END, "Basic ARC test loaded successfully!\n")
        self.output.insert(tk.END, "Click 'Test Connection' to test features.\n\n")
    
    def test_connection(self):
        self.output.insert(tk.END, "Testing basic functionality...\n")
        
        # Test 1: Create sample data
        try:
            os.makedirs("data", exist_ok=True)
            sample = {"test": "data"}
            with open("data/test.json", "w") as f:
                json.dump(sample, f)
            self.output.insert(tk.END, "✓ File operations work\n")
        except Exception as e:
            self.output.insert(tk.END, f"✗ File error: {e}\n")
        
        # Test 2: Basic grid display
        try:
            canvas = tk.Canvas(self.root, width=100, height=100, bg='white')
            canvas.pack_forget()  # Don't actually show it
            self.output.insert(tk.END, "✓ Canvas creation works\n")
        except Exception as e:
            self.output.insert(tk.END, f"✗ Canvas error: {e}\n")
        
        self.output.insert(tk.END, "\nBasic tests completed!\n")
        self.output.see(tk.END)
    
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error running app: {e}")

if __name__ == "__main__":
    try:
        app = BasicARCTest()
        app.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()
