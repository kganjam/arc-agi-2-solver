"""
ARC AGI Application with full dataset support and navigation
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for puzzles
all_puzzles = []
puzzle_index = {}

def load_arc_dataset():
    """Load all ARC puzzles from data directory"""
    global all_puzzles, puzzle_index
    
    data_dir = Path("data/arc_agi")
    puzzles = []
    
    # Load training samples
    training_file = data_dir / "training_sample.json"
    if training_file.exists():
        with open(training_file, 'r') as f:
            training_data = json.load(f)
            for puzzle_id, puzzle_data in training_data.items():
                puzzles.append({
                    "id": puzzle_id,
                    "dataset": "training",
                    "train": puzzle_data.get("train", []),
                    "test": puzzle_data.get("test", [])
                })
    
    # Load evaluation samples
    eval_file = data_dir / "evaluation_sample.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
            for puzzle_id, puzzle_data in eval_data.items():
                puzzles.append({
                    "id": puzzle_id,
                    "dataset": "evaluation",
                    "train": puzzle_data.get("train", []),
                    "test": puzzle_data.get("test", [])
                })
    
    # Add some hardcoded sample puzzles if no data found
    if not puzzles:
        puzzles = [
            {
                "id": "sample_001",
                "dataset": "samples",
                "train": [
                    {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                    {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]}
                ],
                "test": [{"input": [[4, 5], [5, 4]]}]
            },
            {
                "id": "sample_002",
                "dataset": "samples",
                "train": [
                    {"input": [[1, 1, 1], [1, 2, 1], [1, 1, 1]], "output": [[3, 3, 3], [3, 2, 3], [3, 3, 3]]}
                ],
                "test": [{"input": [[2, 2, 2], [2, 1, 2], [2, 2, 2]]}]
            },
            {
                "id": "sample_003",
                "dataset": "samples",
                "train": [
                    {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]}
                ],
                "test": [{"input": [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]}]
            }
        ]
    
    all_puzzles = puzzles
    puzzle_index = {puzzle["id"]: i for i, puzzle in enumerate(puzzles)}
    
    print(f"Loaded {len(all_puzzles)} puzzles")

# Load dataset on startup
load_arc_dataset()

class ChatMessage(BaseModel):
    message: str
    puzzle_number: Optional[int] = None

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI Challenge Solver</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .container { display: flex; height: 100vh; }
        .panel { padding: 20px; border-right: 1px solid #ccc; overflow-y: auto; }
        .puzzle-panel { flex: 1; }
        .chat-panel { width: 400px; display: flex; flex-direction: column; }
        
        /* Navigation controls */
        .nav-controls { 
            display: flex; 
            align-items: center; 
            gap: 10px; 
            padding: 10px; 
            background: #f0f0f0; 
            border-bottom: 1px solid #ccc;
        }
        .nav-btn { 
            padding: 8px 16px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        .nav-btn:hover { background: #0056b3; }
        .nav-btn:disabled { background: #ccc; cursor: not-allowed; }
        .puzzle-number-display { 
            font-weight: bold; 
            padding: 8px 16px; 
            background: white; 
            border: 1px solid #ccc; 
            border-radius: 4px;
        }
        .goto-input { 
            width: 60px; 
            padding: 8px; 
            border: 1px solid #ccc; 
            border-radius: 4px;
        }
        
        /* Grid display */
        .grid { display: inline-block; border: 2px solid #333; margin: 10px; }
        .grid-row { display: flex; }
        .grid-cell { 
            width: 30px; 
            height: 30px; 
            border: 1px solid #666; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: bold; 
        }
        
        /* Chat interface */
        .chat-messages { 
            flex: 1; 
            overflow-y: auto; 
            border: 1px solid #ccc; 
            padding: 10px; 
            background: #f9f9f9; 
        }
        .message { 
            margin: 10px 0; 
            padding: 8px; 
            border-radius: 4px; 
        }
        .user { background: #e3f2fd; }
        .assistant { background: #f1f8e9; }
        .chat-input-container { padding: 10px; }
        .chat-input { 
            width: 100%; 
            height: 60px; 
            margin-bottom: 10px; 
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .send-btn { 
            width: 100%;
            padding: 10px 20px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        .send-btn:hover { background: #0056b3; }
        
        .example-container { margin: 20px 0; }
        .example-title { font-weight: bold; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel puzzle-panel">
            <div class="nav-controls">
                <button class="nav-btn" onclick="previousPuzzle()">← Previous</button>
                <span class="puzzle-number-display">
                    Puzzle <span id="current-number">1</span> of <span id="total-number">0</span>
                </span>
                <button class="nav-btn" onclick="nextPuzzle()">Next →</button>
                <span style="margin-left: 20px;">Go to:</span>
                <input type="number" class="goto-input" id="goto-input" min="1" onkeypress="if(event.key==='Enter') gotoPuzzle()">
                <button class="nav-btn" onclick="gotoPuzzle()">Go</button>
            </div>
            
            <div id="puzzle-content">
                <h3 id="puzzle-id">Loading...</h3>
                <div id="puzzle-examples"></div>
            </div>
        </div>
        
        <div class="panel chat-panel">
            <h3>AI Assistant</h3>
            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">Ready to help! I can analyze the current puzzle and provide insights.</div>
            </div>
            <div class="chat-input-container">
                <textarea class="chat-input" id="chat-input" placeholder="Ask about the puzzle... (Press Enter to send)"></textarea>
                <button class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let allPuzzles = [];
        let currentPuzzleIndex = 0;
        
        const colors = {
            0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40', 4: '#FFDC00',
            5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B', 8: '#7FDBFF', 9: '#870C25'
        };
        
        async function loadPuzzles() {
            try {
                const response = await fetch('/api/puzzles');
                const data = await response.json();
                allPuzzles = data.puzzles;
                document.getElementById('total-number').textContent = allPuzzles.length;
                displayPuzzle(0);
            } catch (error) {
                console.error('Error loading puzzles:', error);
            }
        }
        
        function displayPuzzle(index) {
            if (index < 0 || index >= allPuzzles.length) return;
            
            currentPuzzleIndex = index;
            const puzzle = allPuzzles[index];
            
            document.getElementById('current-number').textContent = index + 1;
            document.getElementById('puzzle-id').textContent = `Puzzle: ${puzzle.id} (${puzzle.dataset})`;
            
            let html = '';
            
            // Training examples
            puzzle.train.forEach((example, i) => {
                html += `<div class="example-container">`;
                html += `<div class="example-title">Training Example ${i + 1}</div>`;
                html += '<div style="display: flex; align-items: center; gap: 20px;">';
                html += renderGrid(example.input);
                html += '<span style="font-size: 20px;">→</span>';
                if (example.output) {
                    html += renderGrid(example.output);
                }
                html += '</div></div>';
            });
            
            // Test cases
            puzzle.test.forEach((test, i) => {
                html += `<div class="example-container">`;
                html += `<div class="example-title">Test Case ${i + 1}</div>`;
                html += '<div style="display: flex; align-items: center; gap: 20px;">';
                html += renderGrid(test.input);
                html += '<span style="font-size: 20px;">→ ?</span>';
                html += '</div></div>';
            });
            
            document.getElementById('puzzle-examples').innerHTML = html;
        }
        
        function renderGrid(grid) {
            let html = '<div class="grid">';
            grid.forEach(row => {
                html += '<div class="grid-row">';
                row.forEach(cell => {
                    const color = colors[cell] || '#FFFFFF';
                    const textColor = [0, 1, 9].includes(cell) ? '#FFFFFF' : '#000000';
                    html += `<div class="grid-cell" style="background-color: ${color}; color: ${textColor};">${cell}</div>`;
                });
                html += '</div>';
            });
            html += '</div>';
            return html;
        }
        
        function previousPuzzle() {
            if (currentPuzzleIndex > 0) {
                displayPuzzle(currentPuzzleIndex - 1);
            }
        }
        
        function nextPuzzle() {
            if (currentPuzzleIndex < allPuzzles.length - 1) {
                displayPuzzle(currentPuzzleIndex + 1);
            }
        }
        
        function gotoPuzzle() {
            const input = document.getElementById('goto-input');
            const puzzleNum = parseInt(input.value);
            if (puzzleNum >= 1 && puzzleNum <= allPuzzles.length) {
                displayPuzzle(puzzleNum - 1);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        puzzle_number: currentPuzzleIndex
                    })
                });
                const data = await response.json();
                addMessage('assistant', data.response);
            } catch (error) {
                addMessage('assistant', 'Error: Could not process message');
            }
        }
        
        function addMessage(type, content) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.innerHTML = `<strong>${type === 'user' ? 'You' : 'AI'}:</strong> ${content}`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        document.getElementById('chat-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Load puzzles on startup
        loadPuzzles();
    </script>
</body>
</html>
    """)

@app.get("/api/test")
async def test():
    return {"status": "working", "message": "Backend is running"}

@app.get("/api/puzzles")
async def get_puzzles():
    """Return all puzzles"""
    return {"puzzles": all_puzzles}

@app.get("/api/puzzle/{puzzle_index}")
async def get_puzzle(puzzle_index: int):
    """Get a specific puzzle by index"""
    if puzzle_index < 0 or puzzle_index >= len(all_puzzles):
        raise HTTPException(status_code=404, detail="Puzzle not found")
    return all_puzzles[puzzle_index]

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Chat endpoint that can access puzzle data"""
    puzzle_info = ""
    
    if message.puzzle_number is not None and 0 <= message.puzzle_number < len(all_puzzles):
        puzzle = all_puzzles[message.puzzle_number]
        
        # Analyze the puzzle
        train_count = len(puzzle.get("train", []))
        test_count = len(puzzle.get("test", []))
        
        # Get grid sizes
        grid_info = []
        for i, example in enumerate(puzzle.get("train", [])):
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])
            
            input_size = f"{len(input_grid)}x{len(input_grid[0])}" if input_grid else "N/A"
            output_size = f"{len(output_grid)}x{len(output_grid[0])}" if output_grid else "N/A"
            
            grid_info.append(f"Training {i+1}: Input {input_size}, Output {output_size}")
        
        for i, test in enumerate(puzzle.get("test", [])):
            test_grid = test.get("input", [])
            test_size = f"{len(test_grid)}x{len(test_grid[0])}" if test_grid else "N/A"
            grid_info.append(f"Test {i+1}: Input {test_size}")
        
        puzzle_info = f"""
Current puzzle: {puzzle.get('id', 'Unknown')} ({puzzle.get('dataset', 'unknown')} dataset)

Summary:
- {train_count} training example(s)
- {test_count} test case(s)

Grid sizes:
{chr(10).join(grid_info)}
"""
        
        # Check if user is asking about the puzzle
        if any(word in message.message.lower() for word in ["puzzle", "current", "grid", "size", "show", "display", "what"]):
            return {"response": puzzle_info}
    
    # Default response
    default_response = "I can help you analyze ARC puzzles! "
    if message.puzzle_number is not None:
        default_response += puzzle_info
    else:
        default_response += "No puzzle selected. Please navigate to a puzzle first."
    
    default_response += "\n\nTry asking me to describe the current puzzle or analyze patterns!"
    
    return {"response": default_response}

if __name__ == "__main__":
    import uvicorn
    print("Starting ARC AGI Application on http://localhost:8050")
    uvicorn.run(app, host="0.0.0.0", port=8050)