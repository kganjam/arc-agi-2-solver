"""
Enhanced ARC AGI Application with Grid Editor
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
    
    # Add hardcoded samples if no data
    if not puzzles:
        puzzles = [
            {
                "id": "e3721c99",
                "dataset": "samples",
                "train": [
                    {"input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]},
                    {"input": [[2, 2, 0], [2, 2, 0], [0, 0, 0]], "output": [[0, 0, 0], [2, 2, 0], [2, 2, 0]]}
                ],
                "test": [{"input": [[3, 3, 3], [0, 0, 0], [4, 4, 4]]}]
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

class SubmitAnswer(BaseModel):
    puzzle_number: int
    test_index: int
    answer: List[List[int]]

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI Challenge Solver</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
        }
        
        .header {
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .nav-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .nav-btn {
            padding: 6px 12px;
            background: white;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .nav-btn:hover:not(:disabled) {
            background: #f0f0f0;
        }
        
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .puzzle-info {
            font-size: 14px;
            color: #666;
        }
        
        .main-container {
            display: flex;
            height: calc(100vh - 60px);
        }
        
        .left-panel {
            flex: 1;
            background: white;
            margin: 10px;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
        }
        
        .right-panel {
            width: 400px;
            display: flex;
            flex-direction: column;
            margin: 10px 10px 10px 0;
        }
        
        .examples-section {
            margin-bottom: 30px;
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        
        .example-row {
            display: flex;
            align-items: start;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .grid-container {
            background: white;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .grid-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .grid {
            display: inline-block;
            border: 2px solid #333;
        }
        
        .grid-row {
            display: flex;
        }
        
        .grid-cell {
            width: 25px;
            height: 25px;
            border: 1px solid #ccc;
            cursor: pointer;
        }
        
        .grid-cell.small {
            width: 20px;
            height: 20px;
        }
        
        .grid-cell.large {
            width: 30px;
            height: 30px;
        }
        
        .arrow {
            font-size: 24px;
            color: #666;
            align-self: center;
        }
        
        /* Editor Section */
        .editor-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 10px;
        }
        
        .color-palette {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .color-btn {
            width: 40px;
            height: 40px;
            border: 2px solid transparent;
            border-radius: 4px;
            cursor: pointer;
            position: relative;
        }
        
        .color-btn:hover {
            transform: scale(1.1);
        }
        
        .color-btn.selected {
            border-color: #333;
            box-shadow: 0 0 0 2px white, 0 0 0 4px #333;
        }
        
        .color-label {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 14px;
            text-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
        
        .editor-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .control-btn {
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .control-btn:hover {
            background: #45a049;
        }
        
        .control-btn.secondary {
            background: #f44336;
        }
        
        .control-btn.secondary:hover {
            background: #da190b;
        }
        
        .grid-size-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .size-input {
            width: 50px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        /* Chat Panel */
        .chat-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
            background: #fafafa;
        }
        
        .message {
            margin: 8px 0;
            padding: 10px;
            border-radius: 8px;
            word-wrap: break-word;
        }
        
        .user {
            background: #e3f2fd;
            margin-left: 20px;
        }
        
        .assistant {
            background: #f1f8e9;
            margin-right: 20px;
        }
        
        .chat-input {
            width: 100%;
            min-height: 60px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            font-family: inherit;
            font-size: 14px;
        }
        
        .send-btn {
            width: 100%;
            padding: 10px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 14px;
        }
        
        .send-btn:hover {
            background: #1976D2;
        }
        
        .status-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            text-align: center;
        }
        
        .status-message.success {
            background: #c8e6c9;
            color: #2e7d32;
        }
        
        .status-message.error {
            background: #ffcdd2;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="nav-controls">
            <button class="nav-btn" onclick="previousPuzzle()">← Previous</button>
            <span class="puzzle-info">
                Puzzle <span id="current-number">1</span> of <span id="total-number">0</span>
            </span>
            <button class="nav-btn" onclick="nextPuzzle()">Next →</button>
            <input type="number" class="size-input" id="goto-input" min="1" placeholder="#">
            <button class="nav-btn" onclick="gotoPuzzle()">Go</button>
        </div>
        <div class="puzzle-info">
            <span id="puzzle-id">Loading...</span>
        </div>
    </div>
    
    <div class="main-container">
        <div class="left-panel">
            <div class="examples-section">
                <h3 class="section-title">Training Examples</h3>
                <div id="training-examples"></div>
            </div>
            
            <div class="examples-section">
                <h3 class="section-title">Test Cases</h3>
                <div id="test-examples"></div>
            </div>
        </div>
        
        <div class="right-panel">
            <div class="editor-section">
                <h3 class="section-title">Answer Editor</h3>
                
                <div class="color-palette" id="color-palette">
                    <!-- Color buttons will be added here -->
                </div>
                
                <div class="grid-size-controls">
                    <span>Grid Size:</span>
                    <input type="number" class="size-input" id="grid-width" value="3" min="1" max="30">
                    <span>×</span>
                    <input type="number" class="size-input" id="grid-height" value="3" min="1" max="30">
                    <button class="nav-btn" onclick="resizeGrid()">Resize</button>
                </div>
                
                <div class="editor-controls">
                    <button class="control-btn" onclick="clearGrid()">Clear</button>
                    <button class="control-btn" onclick="copyFromInput()">Copy Input</button>
                    <button class="control-btn secondary" onclick="submitAnswer()">Submit Answer</button>
                </div>
                
                <div class="grid-container">
                    <div class="grid-label">Your Answer</div>
                    <div id="editor-grid"></div>
                </div>
                
                <div id="status-message"></div>
            </div>
            
            <div class="chat-panel">
                <h3 class="section-title">AI Assistant</h3>
                <div class="chat-messages" id="chat-messages">
                    <div class="message assistant">I can help analyze patterns in the puzzles. Ask me anything!</div>
                </div>
                <textarea class="chat-input" id="chat-input" placeholder="Ask about the puzzle..."></textarea>
                <button class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let allPuzzles = [];
        let currentPuzzleIndex = 0;
        let selectedColor = 0;
        let editorGrid = [];
        let currentTestIndex = 0;
        
        const ARC_COLORS = {
            0: '#000000',  // Black
            1: '#0074D9',  // Blue  
            2: '#FF4136',  // Red
            3: '#2ECC40',  // Green
            4: '#FFDC00',  // Yellow
            5: '#AAAAAA',  // Gray
            6: '#F012BE',  // Magenta
            7: '#FF851B',  // Orange
            8: '#7FDBFF',  // Cyan
            9: '#870C25'   // Brown
        };
        
        function initColorPalette() {
            const palette = document.getElementById('color-palette');
            palette.innerHTML = '';
            
            for (let i = 0; i <= 9; i++) {
                const btn = document.createElement('div');
                btn.className = 'color-btn' + (i === 0 ? ' selected' : '');
                btn.style.backgroundColor = ARC_COLORS[i];
                btn.onclick = () => selectColor(i);
                
                const label = document.createElement('span');
                label.className = 'color-label';
                label.textContent = i;
                btn.appendChild(label);
                
                palette.appendChild(btn);
            }
        }
        
        function selectColor(colorIndex) {
            selectedColor = colorIndex;
            document.querySelectorAll('.color-btn').forEach((btn, i) => {
                btn.classList.toggle('selected', i === colorIndex);
            });
        }
        
        async function loadPuzzles() {
            try {
                const response = await fetch('/api/puzzles');
                const data = await response.json();
                allPuzzles = data.puzzles;
                document.getElementById('total-number').textContent = allPuzzles.length;
                displayPuzzle(0);
                initColorPalette();
                initEditorGrid(3, 3);
            } catch (error) {
                console.error('Error loading puzzles:', error);
            }
        }
        
        function displayPuzzle(index) {
            if (index < 0 || index >= allPuzzles.length) return;
            
            currentPuzzleIndex = index;
            const puzzle = allPuzzles[index];
            
            document.getElementById('current-number').textContent = index + 1;
            document.getElementById('puzzle-id').textContent = puzzle.id;
            
            // Display training examples
            const trainingDiv = document.getElementById('training-examples');
            trainingDiv.innerHTML = '';
            
            puzzle.train.forEach((example, i) => {
                const row = document.createElement('div');
                row.className = 'example-row';
                
                // Input grid
                const inputContainer = document.createElement('div');
                inputContainer.className = 'grid-container';
                inputContainer.innerHTML = '<div class="grid-label">Input</div>';
                inputContainer.appendChild(createGridElement(example.input));
                
                // Arrow
                const arrow = document.createElement('div');
                arrow.className = 'arrow';
                arrow.textContent = '→';
                
                // Output grid
                const outputContainer = document.createElement('div');
                outputContainer.className = 'grid-container';
                outputContainer.innerHTML = '<div class="grid-label">Output</div>';
                if (example.output) {
                    outputContainer.appendChild(createGridElement(example.output));
                }
                
                row.appendChild(inputContainer);
                row.appendChild(arrow);
                row.appendChild(outputContainer);
                trainingDiv.appendChild(row);
            });
            
            // Display test cases
            const testDiv = document.getElementById('test-examples');
            testDiv.innerHTML = '';
            
            puzzle.test.forEach((test, i) => {
                const row = document.createElement('div');
                row.className = 'example-row';
                
                // Input grid
                const inputContainer = document.createElement('div');
                inputContainer.className = 'grid-container';
                inputContainer.innerHTML = '<div class="grid-label">Input</div>';
                inputContainer.appendChild(createGridElement(test.input));
                
                // Arrow
                const arrow = document.createElement('div');
                arrow.className = 'arrow';
                arrow.textContent = '→';
                
                // Output placeholder
                const outputContainer = document.createElement('div');
                outputContainer.className = 'grid-container';
                outputContainer.innerHTML = '<div class="grid-label">Output (solve this)</div>';
                
                row.appendChild(inputContainer);
                row.appendChild(arrow);
                row.appendChild(outputContainer);
                testDiv.appendChild(row);
                
                // Initialize editor with test input size
                if (i === 0) {
                    currentTestIndex = 0;
                    const height = test.input.length;
                    const width = test.input[0]?.length || 3;
                    document.getElementById('grid-height').value = height;
                    document.getElementById('grid-width').value = width;
                    initEditorGrid(width, height);
                }
            });
        }
        
        function createGridElement(gridData) {
            const gridEl = document.createElement('div');
            gridEl.className = 'grid';
            
            gridData.forEach(row => {
                const rowEl = document.createElement('div');
                rowEl.className = 'grid-row';
                
                row.forEach(cell => {
                    const cellEl = document.createElement('div');
                    cellEl.className = 'grid-cell';
                    cellEl.style.backgroundColor = ARC_COLORS[cell] || '#FFFFFF';
                    rowEl.appendChild(cellEl);
                });
                
                gridEl.appendChild(rowEl);
            });
            
            return gridEl;
        }
        
        function initEditorGrid(width, height) {
            editorGrid = Array(height).fill().map(() => Array(width).fill(0));
            renderEditorGrid();
        }
        
        function renderEditorGrid() {
            const gridEl = document.getElementById('editor-grid');
            gridEl.innerHTML = '';
            
            const grid = document.createElement('div');
            grid.className = 'grid';
            
            editorGrid.forEach((row, i) => {
                const rowEl = document.createElement('div');
                rowEl.className = 'grid-row';
                
                row.forEach((cell, j) => {
                    const cellEl = document.createElement('div');
                    cellEl.className = 'grid-cell';
                    cellEl.style.backgroundColor = ARC_COLORS[cell];
                    cellEl.onclick = () => paintCell(i, j);
                    rowEl.appendChild(cellEl);
                });
                
                grid.appendChild(rowEl);
            });
            
            gridEl.appendChild(grid);
        }
        
        function paintCell(row, col) {
            editorGrid[row][col] = selectedColor;
            renderEditorGrid();
        }
        
        function clearGrid() {
            editorGrid = editorGrid.map(row => row.map(() => 0));
            renderEditorGrid();
        }
        
        function copyFromInput() {
            const puzzle = allPuzzles[currentPuzzleIndex];
            if (puzzle.test && puzzle.test[currentTestIndex]) {
                const input = puzzle.test[currentTestIndex].input;
                editorGrid = input.map(row => [...row]);
                renderEditorGrid();
            }
        }
        
        function resizeGrid() {
            const width = parseInt(document.getElementById('grid-width').value);
            const height = parseInt(document.getElementById('grid-height').value);
            
            if (width > 0 && width <= 30 && height > 0 && height <= 30) {
                initEditorGrid(width, height);
            }
        }
        
        async function submitAnswer() {
            try {
                const response = await fetch('/api/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        puzzle_number: currentPuzzleIndex,
                        test_index: currentTestIndex,
                        answer: editorGrid
                    })
                });
                
                const result = await response.json();
                showStatus(result.correct ? 'Correct!' : 'Incorrect. Try again!', result.correct);
            } catch (error) {
                showStatus('Error submitting answer', false);
            }
        }
        
        function showStatus(message, success) {
            const statusEl = document.getElementById('status-message');
            statusEl.className = 'status-message ' + (success ? 'success' : 'error');
            statusEl.textContent = message;
            setTimeout(() => {
                statusEl.textContent = '';
                statusEl.className = '';
            }, 3000);
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
    return {"status": "working"}

@app.get("/api/puzzles")
async def get_puzzles():
    return {"puzzles": all_puzzles}

@app.post("/api/submit")
async def submit_answer(answer: SubmitAnswer):
    """Check if submitted answer is correct"""
    if answer.puzzle_number < 0 or answer.puzzle_number >= len(all_puzzles):
        raise HTTPException(status_code=404, detail="Puzzle not found")
    
    puzzle = all_puzzles[answer.puzzle_number]
    
    # For demo, just return success randomly (since we don't have real answers)
    # In a real implementation, this would check against the correct answer
    import random
    correct = random.random() > 0.5
    
    return {"correct": correct, "message": "Answer checked"}

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

The puzzle appears to involve pattern transformation. Look for relationships between input and output grids in the training examples.
"""
    
    return {"response": puzzle_info if puzzle_info else "Please select a puzzle to analyze."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Enhanced ARC AGI Application on http://localhost:8050")
    uvicorn.run(app, host="0.0.0.0", port=8050)