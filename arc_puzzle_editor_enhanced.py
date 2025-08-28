#!/usr/bin/env python3
"""
Enhanced ARC Puzzle Editor matching ARC Prize interface
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
from typing import Dict, List, Optional
from arc_puzzle_loader import puzzle_loader

app = FastAPI()

class PuzzleRequest(BaseModel):
    index: Optional[int] = None
    puzzle_id: Optional[str] = None

class SolutionSubmit(BaseModel):
    puzzle_id: str
    solution: List[List[int]]

@app.get("/api/puzzle/current")
async def get_current_puzzle():
    """Get current puzzle"""
    puzzle = puzzle_loader.get_current_puzzle()
    if puzzle:
        position = puzzle_loader.get_current_position()
        return {
            "puzzle": puzzle,
            "position": position,
            "total": puzzle_loader.get_puzzle_count()
        }
    raise HTTPException(status_code=404, detail="No puzzle loaded")

@app.get("/api/puzzle/{puzzle_id}")
async def get_puzzle_by_id(puzzle_id: str):
    """Get puzzle by ID"""
    puzzle = puzzle_loader.get_puzzle_by_id(puzzle_id)
    if puzzle:
        return {"puzzle": puzzle}
    raise HTTPException(status_code=404, detail=f"Puzzle {puzzle_id} not found")

@app.post("/api/puzzle/next")
async def next_puzzle():
    """Go to next puzzle"""
    puzzle = puzzle_loader.next_puzzle()
    if puzzle:
        position = puzzle_loader.get_current_position()
        return {
            "puzzle": puzzle,
            "position": position,
            "total": puzzle_loader.get_puzzle_count()
        }
    raise HTTPException(status_code=404, detail="No more puzzles")

@app.post("/api/puzzle/previous")
async def previous_puzzle():
    """Go to previous puzzle"""
    puzzle = puzzle_loader.previous_puzzle()
    if puzzle:
        position = puzzle_loader.get_current_position()
        return {
            "puzzle": puzzle,
            "position": position,
            "total": puzzle_loader.get_puzzle_count()
        }
    raise HTTPException(status_code=404, detail="No previous puzzle")

@app.post("/api/puzzle/goto/{index}")
async def goto_puzzle(index: int):
    """Go to specific puzzle by index"""
    puzzle = puzzle_loader.get_puzzle(index)
    if puzzle:
        position = puzzle_loader.get_current_position()
        return {
            "puzzle": puzzle,
            "position": position,
            "total": puzzle_loader.get_puzzle_count()
        }
    raise HTTPException(status_code=404, detail=f"Invalid puzzle index: {index}")

def get_editor_html():
    """Get the enhanced puzzle editor HTML"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI Puzzle Editor</title>
    <style>
        * { 
            box-sizing: border-box; 
            margin: 0; 
            padding: 0; 
        }
        
        html, body {
            height: 100%;
            overflow: hidden;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: #000;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
        }
        
        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .title {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }
        
        .subtitle {
            color: #999;
            font-size: 14px;
            margin-top: 5px;
        }
        
        /* Navigation Bar */
        .nav-bar {
            background: #2a2a2a;
            padding: 10px 20px;
            border-bottom: 1px solid #444;
        }
        
        .nav-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .puzzle-info {
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 1;
        }
        
        .puzzle-id {
            color: #999;
            font-size: 14px;
        }
        
        .puzzle-id-value {
            color: #fff;
            font-weight: bold;
            font-family: monospace;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .nav-btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .nav-btn:hover:not(:disabled) {
            background: #555;
        }
        
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .puzzle-counter {
            color: #999;
            font-size: 14px;
            padding: 0 10px;
        }
        
        .puzzle-selector {
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        /* Main Content */
        .main-container {
            width: 100%;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            flex: 1;
            overflow: hidden;
            height: calc(100vh - 120px);
        }
        
        .examples-column {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            height: 100%;
            max-height: calc(100vh - 140px);
        }
        
        .middle-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
            height: 100%;
            max-height: calc(100vh - 140px);
            overflow-y: auto;
            min-height: 0;
        }
        
        .chat-column {
            background: #2a2a2a;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            height: 100%;
            max-height: calc(100vh - 140px);
            min-height: 0;
            overflow: hidden;
        }
        
        .test-section-container {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
        }
        
        .controls-section {
            background: #333;
            border-radius: 8px;
            padding: 15px;
        }
        
        /* AI Chat Section */
        .ai-chat-section {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
        }
        
        .chat-header {
            padding: 15px;
            background: #1f1f1f;
            border-radius: 8px 8px 0 0;
            border-bottom: 1px solid #444;
            flex-shrink: 0;
        }
        
        .chat-header h3 {
            color: #fff;
            margin: 0;
            font-size: 16px;
        }
        
        .chat-messages {
            flex: 1 1 auto;
            padding: 15px;
            overflow-y: auto;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 0;
            max-height: calc(100vh - 300px);
        }
        
        .chat-message {
            padding: 10px;
            border-radius: 6px;
            max-width: 90%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        .chat-message.user {
            background: #444;
            align-self: flex-end;
            color: #fff;
        }
        
        .chat-message.ai {
            background: #333;
            align-self: flex-start;
            color: #ddd;
        }
        
        .chat-message.ai code {
            background: #1a1a1a;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #444;
            display: flex;
            gap: 10px;
            flex-shrink: 0;
        }
        
        .chat-input input {
            flex: 1;
            padding: 8px 12px;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .chat-input button {
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .chat-input button:hover {
            background: #45a049;
        }
        
        /* Section Headers */
        .section-header {
            color: #fff;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Grid Containers */
        .examples-section, .test-section {
            margin-bottom: 30px;
        }
        
        .example-pair, .test-pair {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .grid-container {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
        }
        
        .grid-label {
            color: #999;
            font-size: 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }
        
        .grid-size {
            color: #666;
            font-family: monospace;
        }
        
        .puzzle-grid {
            display: inline-block;
            border: 2px solid #555;
            background: #000;
        }
        
        .grid-row {
            display: flex;
            height: 20px;
        }
        
        .grid-cell {
            width: 20px;
            height: 20px;
            border: 1px solid #333;
            cursor: pointer;
            position: relative;
        }
        
        .grid-cell:hover {
            border-color: #666;
        }
        
        .grid-cell.editable:hover::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border: 2px solid #fff;
            pointer-events: none;
        }
        
        .arrow {
            color: #666;
            font-size: 24px;
        }
        
        /* Color Palette */
        .color-palette {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            margin: 20px 0;
        }
        
        .color-btn {
            width: 40px;
            height: 40px;
            border: 2px solid #444;
            cursor: pointer;
            border-radius: 4px;
            position: relative;
            transition: all 0.2s;
        }
        
        .color-btn:hover {
            transform: scale(1.1);
            border-color: #666;
        }
        
        .color-btn.selected {
            border-color: #fff;
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }
        
        .color-btn::after {
            content: attr(data-color);
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            color: #666;
            font-size: 10px;
        }
        
        /* Control Buttons */
        .control-buttons {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 20px;
        }
        
        .control-btn {
            background: #444;
            color: #fff;
            border: none;
            padding: 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .control-btn:hover {
            background: #555;
        }
        
        .control-btn.primary {
            background: #4CAF50;
        }
        
        .control-btn.primary:hover {
            background: #45a049;
        }
        
        /* Edit Mode Buttons */
        .edit-modes {
            display: flex;
            gap: 5px;
            margin: 15px 0;
        }
        
        .mode-btn {
            flex: 1;
            padding: 8px;
            background: #333;
            color: #999;
            border: 1px solid #444;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .mode-btn:first-child {
            border-radius: 4px 0 0 4px;
        }
        
        .mode-btn:last-child {
            border-radius: 0 4px 4px 0;
        }
        
        .mode-btn.active {
            background: #555;
            color: #fff;
            border-color: #666;
        }
        
        .mode-btn:hover {
            background: #444;
            color: #fff;
        }
        
        /* Grid Size Controls */
        .grid-controls {
            display: flex;
            gap: 10px;
            margin: 15px 0;
        }
        
        .size-input {
            width: 60px;
            padding: 5px;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
        }
        
        /* ARC Color Palette */
        .color-0 { background: #000000; }
        .color-1 { background: #0074D9; }
        .color-2 { background: #FF4136; }
        .color-3 { background: #2ECC40; }
        .color-4 { background: #FFDC00; }
        .color-5 { background: #AAAAAA; }
        .color-6 { background: #F012BE; }
        .color-7 { background: #FF851B; }
        .color-8 { background: #7FDBFF; }
        .color-9 { background: #870C25; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div>
                <div class="title">PLAY</div>
                <div class="subtitle">Try ARC-AGI-1 and 2. Given the examples, identify the pattern, solve the test puzzle.</div>
            </div>
            <div>
                <a href="/dashboard" style="background: #4CAF50; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold; display: inline-block;">
                    📊 Dashboard
                </a>
            </div>
        </div>
    </div>
    
    <div class="nav-bar">
        <div class="nav-content">
            <div class="puzzle-info">
                <span class="puzzle-id">Puzzle ID: <span id="puzzle-id" class="puzzle-id-value">Loading...</span></span>
            </div>
            <div class="nav-buttons">
                <button class="nav-btn" onclick="previousPuzzle()" id="prev-btn">Previous</button>
                <span class="puzzle-counter"><span id="current-index">1</span> of <span id="total-puzzles">0</span></span>
                <button class="nav-btn" onclick="nextPuzzle()" id="next-btn">Next</button>
                <select class="puzzle-selector" onchange="selectPuzzleSet(this.value)">
                    <option>Public Evaluation Set v2 (Hard)</option>
                    <option>Public Training Set v1 (Easy)</option>
                </select>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="examples-column">
            <div class="section-header">Examples</div>
            <div id="training-examples"></div>
        </div>
        
        <div class="middle-column">
            <div class="test-section-container">
                <div class="section-header">Test</div>
                <div id="test-examples"></div>
            </div>
            
            <div class="controls-section">
                <div class="section-header">Controls</div>
                
                <div>
                    <div style="color: #999; font-size: 12px; margin-bottom: 10px;">1. Configure your output grid:</div>
                    <div class="grid-controls">
                        <input type="number" class="size-input" id="output-width" value="3" min="1" max="30">
                        <span style="color: #666;">x</span>
                        <input type="number" class="size-input" id="output-height" value="3" min="1" max="30">
                        <button class="control-btn" onclick="resizeOutput()" style="padding: 5px 10px;">Resize</button>
                    </div>
                </div>
                
                <div>
                    <div style="color: #999; font-size: 12px; margin-bottom: 10px;">2. Edit your output grid cells:</div>
                    <div class="edit-modes">
                        <button class="mode-btn active" onclick="setEditMode('edit')">✏️ Edit</button>
                        <button class="mode-btn" onclick="setEditMode('select')">⬚ Select</button>
                        <button class="mode-btn" onclick="setEditMode('fill')">🪣 Fill</button>
                    </div>
                    
                    <div class="color-palette">
                        <div class="color-btn color-0 selected" data-color="0" onclick="selectColor(0)"></div>
                        <div class="color-btn color-1" data-color="1" onclick="selectColor(1)"></div>
                        <div class="color-btn color-2" data-color="2" onclick="selectColor(2)"></div>
                        <div class="color-btn color-3" data-color="3" onclick="selectColor(3)"></div>
                        <div class="color-btn color-4" data-color="4" onclick="selectColor(4)"></div>
                        <div class="color-btn color-5" data-color="5" onclick="selectColor(5)"></div>
                        <div class="color-btn color-6" data-color="6" onclick="selectColor(6)"></div>
                        <div class="color-btn color-7" data-color="7" onclick="selectColor(7)"></div>
                        <div class="color-btn color-8" data-color="8" onclick="selectColor(8)"></div>
                        <div class="color-btn color-9" data-color="9" onclick="selectColor(9)"></div>
                    </div>
                </div>
                
                <div>
                    <div style="color: #999; font-size: 12px; margin-bottom: 10px;">3. See if your output is correct:</div>
                    <div class="control-buttons">
                        <button class="control-btn" onclick="copyFromInput()">Copy from input</button>
                        <button class="control-btn" onclick="clearOutput()">Clear</button>
                        <button class="control-btn" onclick="resetOutput()">Reset</button>
                        <button class="control-btn primary" onclick="submitSolution()">Submit solution</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chat-column">
            <div class="ai-chat-section">
                <div class="chat-header">
                    <h3>🤖 AI Assistant</h3>
                </div>
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-message ai">
                        Welcome! I can help you edit the output grid and answer questions about the puzzle. Try commands like "make cell 3,3 red" or ask "what is the grid size?"
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Type a command or question..." onkeydown="handleChatKeyPress(event)">
                    <button onclick="sendChatMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentPuzzle = null;
        let currentPosition = [1, 0];  // Changed to array format [current, total]
        let selectedColor = 0;
        let editMode = 'edit';
        let outputGrid = [];
        
        // Command history for AI chat
        let commandHistory = [];
        let historyIndex = -1;
        let currentInput = '';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, starting puzzle load...');
            // Add a small delay to ensure all elements are ready
            setTimeout(() => {
                loadCurrentPuzzle();
            }, 100);
        });
        
        async function loadCurrentPuzzle() {
            try {
                console.log('Starting to load puzzle...');  // Debug log
                document.getElementById('puzzle-id').textContent = 'Loading...';
                
                const response = await fetch('/api/puzzle/current');
                console.log('API response status:', response.status);  // Debug log
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                
                console.log('Loaded puzzle data:', data);  // Debug log
                
                if (!data || !data.puzzle) {
                    throw new Error('No puzzle in response');
                }
                
                currentPuzzle = data.puzzle;
                currentPosition = data.position || [1, 1];
                
                console.log('About to update display...');  // Debug log
                updateDisplay();
                setPuzzleContext();
            } catch (error) {
                console.error('Error loading puzzle:', error);
                // Add error message to page
                document.getElementById('puzzle-id').textContent = 'Error loading';
                // Load a default puzzle if API fails
                loadDefaultPuzzle();
            }
        }
        
        async function nextPuzzle() {
            try {
                const response = await fetch('/api/puzzle/next', {method: 'POST'});
                const data = await response.json();
                
                currentPuzzle = data.puzzle;
                currentPosition = data.position;
                
                updateDisplay();
                setPuzzleContext();
            } catch (error) {
                console.error('Error loading next puzzle:', error);
            }
        }
        
        async function previousPuzzle() {
            try {
                const response = await fetch('/api/puzzle/previous', {method: 'POST'});
                const data = await response.json();
                
                currentPuzzle = data.puzzle;
                currentPosition = data.position;
                
                updateDisplay();
                setPuzzleContext();
            } catch (error) {
                console.error('Error loading previous puzzle:', error);
            }
        }
        
        function updateDisplay() {
            if (!currentPuzzle) {
                console.error('No puzzle to display');
                return;
            }
            
            console.log('Updating display with puzzle:', currentPuzzle.id);  // Debug log
            console.log('Position:', currentPosition);  // Debug log
            
            // Update navigation
            document.getElementById('puzzle-id').textContent = currentPuzzle.id || 'Unknown';
            document.getElementById('current-index').textContent = currentPosition ? currentPosition[0] : 1;
            document.getElementById('total-puzzles').textContent = currentPosition ? currentPosition[1] : 0;
            
            // Update button states
            document.getElementById('prev-btn').disabled = currentPosition && currentPosition[0] <= 1;
            document.getElementById('next-btn').disabled = currentPosition && currentPosition[0] >= currentPosition[1];
            
            // Display training examples
            const trainingContainer = document.getElementById('training-examples');
            trainingContainer.innerHTML = '';
            
            console.log('Training examples:', currentPuzzle.train);  // Debug log
            
            if (currentPuzzle.train && currentPuzzle.train.length > 0) {
                currentPuzzle.train.forEach((example, index) => {
                    console.log(`Creating training example ${index}:`, example);  // Debug log
                    const pairDiv = document.createElement('div');
                    pairDiv.className = 'example-pair';
                    
                    // Input
                    const inputContainer = createGridContainer(
                        `Ex.${index + 1} Input`,
                        example.input,
                        false
                    );
                    
                    // Arrow
                    const arrow = document.createElement('div');
                    arrow.className = 'arrow';
                    arrow.textContent = '→';
                    
                    // Output
                    const outputContainer = createGridContainer(
                        `Ex.${index + 1} Output`,
                        example.output,
                        false
                    );
                    
                    pairDiv.appendChild(inputContainer);
                    pairDiv.appendChild(arrow);
                    pairDiv.appendChild(outputContainer);
                    trainingContainer.appendChild(pairDiv);
                });
            }
            
            // Display test
            const testContainer = document.getElementById('test-examples');
            testContainer.innerHTML = '';
            
            if (currentPuzzle.test && currentPuzzle.test[0]) {
                const testPair = document.createElement('div');
                testPair.className = 'test-pair';
                
                // Test input
                const testInput = createGridContainer(
                    'Input',
                    currentPuzzle.test[0].input,
                    false
                );
                
                // Arrow
                const arrow = document.createElement('div');
                arrow.className = 'arrow';
                arrow.textContent = '→';
                
                // Initialize output grid
                const inputGrid = currentPuzzle.test[0].input;
                outputGrid = inputGrid.map(row => row.map(() => 0));
                
                // Update size inputs
                document.getElementById('output-width').value = inputGrid[0].length;
                document.getElementById('output-height').value = inputGrid.length;
                
                // Test output (editable)
                const testOutput = createGridContainer(
                    'Output',
                    outputGrid,
                    true
                );
                
                testPair.appendChild(testInput);
                testPair.appendChild(arrow);
                testPair.appendChild(testOutput);
                testContainer.appendChild(testPair);
            }
        }
        
        function createGridContainer(label, grid, editable = false) {
            const container = document.createElement('div');
            container.className = 'grid-container';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'grid-label';
            labelDiv.innerHTML = `
                <span>${label}</span>
                <span class="grid-size">(${grid.length}x${grid[0].length})</span>
            `;
            container.appendChild(labelDiv);
            
            const gridDiv = document.createElement('div');
            gridDiv.className = 'puzzle-grid';
            if (editable) gridDiv.id = 'output-grid';
            
            grid.forEach((row, i) => {
                const rowDiv = document.createElement('div');
                rowDiv.className = 'grid-row';
                
                row.forEach((cell, j) => {
                    const cellDiv = document.createElement('div');
                    cellDiv.className = `grid-cell color-${cell}`;
                    if (editable) {
                        cellDiv.classList.add('editable');
                        cellDiv.onclick = () => editCell(i, j);
                    }
                    cellDiv.dataset.row = i;
                    cellDiv.dataset.col = j;
                    rowDiv.appendChild(cellDiv);
                });
                
                gridDiv.appendChild(rowDiv);
            });
            
            container.appendChild(gridDiv);
            return container;
        }
        
        function editCell(row, col) {
            if (editMode === 'edit') {
                outputGrid[row][col] = selectedColor;
                updateOutputGrid();
            } else if (editMode === 'fill') {
                floodFill(row, col, outputGrid[row][col], selectedColor);
                updateOutputGrid();
            }
        }
        
        function floodFill(row, col, targetColor, replaceColor) {
            if (targetColor === replaceColor) return;
            if (outputGrid[row][col] !== targetColor) return;
            
            outputGrid[row][col] = replaceColor;
            
            // Check adjacent cells
            if (row > 0) floodFill(row - 1, col, targetColor, replaceColor);
            if (row < outputGrid.length - 1) floodFill(row + 1, col, targetColor, replaceColor);
            if (col > 0) floodFill(row, col - 1, targetColor, replaceColor);
            if (col < outputGrid[0].length - 1) floodFill(row, col + 1, targetColor, replaceColor);
        }
        
        function updateOutputGrid() {
            const gridDiv = document.getElementById('output-grid');
            if (!gridDiv) return;
            
            gridDiv.innerHTML = '';
            outputGrid.forEach((row, i) => {
                const rowDiv = document.createElement('div');
                rowDiv.className = 'grid-row';
                
                row.forEach((cell, j) => {
                    const cellDiv = document.createElement('div');
                    cellDiv.className = `grid-cell color-${cell} editable`;
                    cellDiv.onclick = () => editCell(i, j);
                    cellDiv.dataset.row = i;
                    cellDiv.dataset.col = j;
                    rowDiv.appendChild(cellDiv);
                });
                
                gridDiv.appendChild(rowDiv);
            });
        }
        
        function selectColor(color) {
            selectedColor = color;
            document.querySelectorAll('.color-btn').forEach(btn => {
                btn.classList.remove('selected');
            });
            document.querySelector(`.color-btn[data-color="${color}"]`).classList.add('selected');
        }
        
        function setEditMode(mode) {
            editMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        function resizeOutput() {
            const width = parseInt(document.getElementById('output-width').value);
            const height = parseInt(document.getElementById('output-height').value);
            
            const newGrid = [];
            for (let i = 0; i < height; i++) {
                const row = [];
                for (let j = 0; j < width; j++) {
                    if (outputGrid[i] && outputGrid[i][j] !== undefined) {
                        row.push(outputGrid[i][j]);
                    } else {
                        row.push(0);
                    }
                }
                newGrid.push(row);
            }
            
            outputGrid = newGrid;
            
            // Update display
            const testContainer = document.getElementById('test-examples');
            const testPair = testContainer.querySelector('.test-pair');
            if (testPair) {
                const outputContainer = testPair.children[2];
                testPair.replaceChild(
                    createGridContainer('Output', outputGrid, true),
                    outputContainer
                );
            }
        }
        
        function copyFromInput() {
            if (currentPuzzle && currentPuzzle.test && currentPuzzle.test[0]) {
                outputGrid = currentPuzzle.test[0].input.map(row => [...row]);
                updateOutputGrid();
            }
        }
        
        function clearOutput() {
            outputGrid = outputGrid.map(row => row.map(() => 0));
            updateOutputGrid();
        }
        
        function resetOutput() {
            if (currentPuzzle && currentPuzzle.test && currentPuzzle.test[0]) {
                const inputGrid = currentPuzzle.test[0].input;
                outputGrid = inputGrid.map(row => row.map(() => 0));
                updateOutputGrid();
            }
        }
        
        async function submitSolution() {
            if (!currentPuzzle || !outputGrid) {
                alert('No puzzle loaded or no solution provided');
                return;
            }
            
            const response = await fetch('/api/submit-solution', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    puzzle_id: currentPuzzle.id,
                    solution: outputGrid
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                alert(`✅ Correct! Solution verified!\n\nAccuracy: ${(result.accuracy * 100).toFixed(1)}%`);
            } else {
                let message = `❌ Incorrect solution\n\n`;
                if (result.accuracy !== undefined) {
                    message += `Accuracy: ${(result.accuracy * 100).toFixed(1)}%\n`;
                }
                if (result.message) {
                    message += `Feedback: ${result.message}\n`;
                }
                if (result.attempts_remaining !== undefined) {
                    message += `Attempts remaining: ${result.attempts_remaining}`;
                }
                alert(message);
            }
        }
        
        function loadDefaultPuzzle() {
            // Default puzzle for when API is not available
            currentPuzzle = {
                id: 'default_001',
                train: [
                    {
                        input: [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                        output: [[3, 0, 0], [0, 2, 0], [0, 0, 1]]
                    }
                ],
                test: [
                    {
                        input: [[4, 0, 0], [0, 5, 0], [0, 0, 6]]
                    }
                ]
            };
            currentPosition = [1, 1];
            updateDisplay();
        }
        
        async function selectPuzzleSet(value) {
            // Switch between puzzle sets
            console.log('Switching to puzzle set:', value);
            
            try {
                const response = await fetch('/api/puzzle/switch-set', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        puzzle_set: value
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.success) {
                    console.log('Successfully switched to:', data.puzzle_set);
                    
                    // Update the current puzzle with the first puzzle from new set
                    if (data.puzzle) {
                        currentPuzzle = data.puzzle;
                        currentPosition = data.position || [1, 1];
                        
                        updateDisplay();
                        setPuzzleContext();
                        
                        // Show a success message
                        const statusDiv = document.createElement('div');
                        statusDiv.style.cssText = 'position:fixed;top:70px;right:20px;background:#4CAF50;color:white;padding:10px 20px;border-radius:4px;z-index:1000';
                        statusDiv.textContent = `Loaded ${value}`;
                        document.body.appendChild(statusDiv);
                        setTimeout(() => statusDiv.remove(), 3000);
                    }
                } else {
                    console.error('Failed to switch puzzle set');
                }
            } catch (error) {
                console.error('Error switching puzzle set:', error);
                alert('Failed to switch puzzle set: ' + error.message);
            }
        }
        
        // AI Chat Functions
        async function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add to command history
            commandHistory.push(message);
            historyIndex = commandHistory.length;
            currentInput = '';
            
            // Add user message to chat
            addChatMessage(message, 'user');
            input.value = '';
            
            // Send to AI
            try {
                const response = await fetch('/api/puzzle/ai-chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        puzzle_id: currentPuzzle ? currentPuzzle.id : null,
                        output_grid: outputGrid
                    })
                });
                
                const result = await response.json();
                
                // Handle updated output grid if present
                if (result.updated_output_grid || result.output_grid) {
                    outputGrid = result.updated_output_grid || result.output_grid;
                    updateOutputGrid();
                }
                
                // Handle function call results
                if (result.function_call && result.function_result) {
                    // Show what function was called
                    let funcMessage = `Executed function: ${result.function_call.name}`;
                    if (result.function_result.message) {
                        funcMessage += ` - ${result.function_result.message}`;
                    }
                    addChatMessage(funcMessage, 'ai');
                }
                
                // Display main message
                if (result.message) {
                    addChatMessage(result.message, 'ai');
                }
                
                // Handle old format for backward compatibility
                if (result.type === 'command' && result.result && result.result.success && result.result.grid) {
                    outputGrid = result.result.grid;
                    updateOutputGrid();
                }
                
            } catch (error) {
                console.error('Error sending chat message:', error);
                addChatMessage('Sorry, there was an error processing your request.', 'ai');
            }
        }
        
        function addChatMessage(message, sender) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${sender}`;
            
            // Convert markdown-style formatting to HTML
            message = message.replace(/•/g, '&bull;');
            message = message.replace(/\\n/g, '<br>');
            message = message.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            
            messageDiv.innerHTML = message;
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function handleChatKeyPress(event) {
            const input = document.getElementById('chat-input');
            
            if (event.key === 'Enter') {
                sendChatMessage();
            } else if (event.key === 'ArrowUp') {
                event.preventDefault();
                
                // Save current input if we're at the end of history
                if (historyIndex === commandHistory.length) {
                    currentInput = input.value;
                }
                
                // Move up in history
                if (historyIndex > 0) {
                    historyIndex--;
                    input.value = commandHistory[historyIndex];
                }
            } else if (event.key === 'ArrowDown') {
                event.preventDefault();
                
                // Move down in history
                if (historyIndex < commandHistory.length) {
                    historyIndex++;
                    
                    if (historyIndex === commandHistory.length) {
                        // We're at the end, restore current input or clear
                        input.value = currentInput;
                    } else {
                        input.value = commandHistory[historyIndex];
                    }
                }
            }
        }
        
        // Set puzzle context when puzzle changes
        async function setPuzzleContext() {
            if (!currentPuzzle) return;
            
            try {
                await fetch('/api/puzzle/set-context', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        puzzle_id: currentPuzzle.id
                    })
                });
            } catch (error) {
                console.error('Error setting puzzle context:', error);
            }
        }
    </script>
</body>
</html>
"""

@app.get("/enhanced-editor")
async def enhanced_editor():
    """Serve the enhanced puzzle editor"""
    return HTMLResponse(get_editor_html())