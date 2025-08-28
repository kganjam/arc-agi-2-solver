"""
Real-time Solving Visualization for ARC AGI
Visual display of the solving process with animations
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

app = FastAPI()

class SolvingVisualizer:
    """Manages real-time visualization of solving process"""
    
    def __init__(self):
        self.active_sessions = {}
        self.solving_history = []
        self.websocket_clients = []
        
    async def visualize_solving_step(self, step_data: Dict):
        """Send solving step to all connected clients"""
        message = {
            'type': 'solving_step',
            'timestamp': datetime.now().isoformat(),
            'data': step_data
        }
        
        # Broadcast to all clients
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except:
                self.websocket_clients.remove(client)
    
    async def animate_transformation(self, input_grid: List, output_grid: List, 
                                   transformation_type: str):
        """Animate transformation from input to output"""
        frames = self.generate_animation_frames(input_grid, output_grid, transformation_type)
        
        for frame in frames:
            await self.visualize_solving_step({
                'frame': frame,
                'transformation': transformation_type
            })
            await asyncio.sleep(0.1)  # Animation speed
    
    def generate_animation_frames(self, input_grid: List, output_grid: List, 
                                 transformation_type: str) -> List:
        """Generate animation frames for transformation"""
        frames = []
        
        if transformation_type == 'color_mapping':
            # Gradual color transition
            for step in range(10):
                frame = self.interpolate_colors(input_grid, output_grid, step / 10)
                frames.append(frame)
                
        elif transformation_type == 'rotation':
            # Rotation animation
            for angle in range(0, 91, 10):
                frame = self.rotate_grid(input_grid, angle)
                frames.append(frame)
                
        elif transformation_type == 'pattern_growth':
            # Growing pattern animation
            frames = self.generate_growth_animation(input_grid, output_grid)
            
        else:
            # Default: fade transition
            frames = [input_grid, output_grid]
            
        return frames
    
    def interpolate_colors(self, grid1: List, grid2: List, t: float) -> List:
        """Interpolate between two color grids"""
        result = []
        for i in range(len(grid1)):
            row = []
            for j in range(len(grid1[0])):
                # Simple interpolation (could be more sophisticated)
                val1 = grid1[i][j]
                val2 = grid2[i][j] if i < len(grid2) and j < len(grid2[0]) else val1
                interpolated = int(val1 * (1 - t) + val2 * t)
                row.append(interpolated)
            result.append(row)
        return result
    
    def rotate_grid(self, grid: List, angle: int) -> List:
        """Rotate grid by angle degrees"""
        # Simplified rotation (90-degree steps)
        if angle >= 90:
            return [list(row) for row in zip(*grid[::-1])]
        return grid
    
    def generate_growth_animation(self, start: List, end: List) -> List:
        """Generate frames showing pattern growth"""
        frames = []
        # Simplified growth animation
        for size in range(1, max(len(end), len(end[0])) + 1):
            frame = [[0 for _ in range(size)] for _ in range(size)]
            for i in range(min(size, len(end))):
                for j in range(min(size, len(end[0]))):
                    if i < len(end) and j < len(end[0]):
                        frame[i][j] = end[i][j]
            frames.append(frame)
        return frames

visualizer = SolvingVisualizer()

@app.websocket("/ws/visualizer")
async def websocket_visualizer(websocket: WebSocket):
    """WebSocket endpoint for real-time visualization"""
    await websocket.accept()
    visualizer.websocket_clients.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except:
        visualizer.websocket_clients.remove(websocket)

@app.get("/visualizer")
async def visualizer_page():
    """Real-time solving visualization page"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI - Real-time Solving Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .visualization-area {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .grid-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 30px 0;
        }
        
        .grid-wrapper {
            text-align: center;
        }
        
        .grid {
            display: inline-block;
            border: 3px solid #333;
            background: white;
            position: relative;
        }
        
        .grid-cell {
            width: 30px;
            height: 30px;
            border: 1px solid #ddd;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .grid-row {
            display: flex;
        }
        
        .arrow {
            font-size: 48px;
            color: #667eea;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.2); }
        }
        
        .transformation-label {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin: 20px 0;
            display: inline-block;
        }
        
        .agent-dialogue {
            background: #f5f5f5;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .agent-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .agent-pattern { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .agent-transform { background: #f3e5f5; border-left: 4px solid #9c27b0; }
        .agent-validator { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .agent-strategist { background: #fff3e0; border-left: 4px solid #ff9800; }
        
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .pattern-discovery {
            background: #fffbf0;
            border: 2px solid gold;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        
        .pattern-discovery::before {
            content: '✨';
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 24px;
            animation: sparkle 1s infinite;
        }
        
        @keyframes sparkle {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .solving-progress {
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            height: 100%;
            width: 0%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        /* Color palette for ARC */
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
        
        .neural-activity {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            color: #0f4c75;
        }
        
        .neuron {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin: 2px;
            background: #0f4c75;
            opacity: 0.3;
            transition: all 0.3s;
        }
        
        .neuron.active {
            background: #00ff00;
            opacity: 1;
            box-shadow: 0 0 10px #00ff00;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Real-time Solving Visualization</h1>
            <p>Watch the AI solve ARC puzzles in real-time</p>
        </div>
        
        <div class="visualization-area">
            <div class="stats-panel">
                <div class="stat-card">
                    <div class="stat-label">Current Puzzle</div>
                    <div class="stat-value" id="current-puzzle">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Solving Time</div>
                    <div class="stat-value" id="solving-time">0.0s</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Attempts</div>
                    <div class="stat-value" id="attempts">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value" id="confidence">0%</div>
                </div>
            </div>
            
            <div class="solving-progress">
                <div class="progress-bar" id="progress">0%</div>
            </div>
            
            <div class="grid-container">
                <div class="grid-wrapper">
                    <h3>Input</h3>
                    <div class="grid" id="input-grid"></div>
                </div>
                <div class="arrow">→</div>
                <div class="grid-wrapper">
                    <h3>Processing</h3>
                    <div class="grid" id="processing-grid"></div>
                </div>
                <div class="arrow">→</div>
                <div class="grid-wrapper">
                    <h3>Output</h3>
                    <div class="grid" id="output-grid"></div>
                </div>
            </div>
            
            <div class="transformation-label" id="transformation">
                Analyzing Pattern...
            </div>
            
            <div class="pattern-discovery" id="pattern-discovery" style="display: none;">
                <h3>🎉 Pattern Discovered!</h3>
                <p id="pattern-description"></p>
            </div>
            
            <div class="agent-dialogue">
                <h3>Multi-Agent Dialogue</h3>
                <div id="dialogue"></div>
            </div>
            
            <div class="neural-activity">
                <h3 style="color: white;">Neural Network Activity</h3>
                <div id="neural-viz"></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let startTime = Date.now();
        let attemptCount = 0;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8052/ws/visualizer');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleVisualizationUpdate(data);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function createGrid(gridData, containerId) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            if (!gridData) return;
            
            gridData.forEach(row => {
                const rowDiv = document.createElement('div');
                rowDiv.className = 'grid-row';
                
                row.forEach(cell => {
                    const cellDiv = document.createElement('div');
                    cellDiv.className = 'grid-cell color-' + cell;
                    rowDiv.appendChild(cellDiv);
                });
                
                container.appendChild(rowDiv);
            });
        }
        
        function handleVisualizationUpdate(data) {
            if (data.type === 'solving_step') {
                const stepData = data.data;
                
                if (stepData.frame) {
                    createGrid(stepData.frame, 'processing-grid');
                }
                
                if (stepData.transformation) {
                    document.getElementById('transformation').textContent = 
                        'Applying: ' + stepData.transformation;
                }
                
                // Update solving time
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                document.getElementById('solving-time').textContent = elapsed + 's';
            }
            
            // Add agent dialogue
            if (data.agent_message) {
                addAgentMessage(data.agent_message);
            }
            
            // Update neural activity
            if (data.neural_activity) {
                updateNeuralViz(data.neural_activity);
            }
        }
        
        function addAgentMessage(message) {
            const dialogue = document.getElementById('dialogue');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'agent-message agent-' + message.agent_type;
            msgDiv.innerHTML = '<strong>' + message.agent + ':</strong> ' + message.text;
            dialogue.appendChild(msgDiv);
            dialogue.scrollTop = dialogue.scrollHeight;
        }
        
        function updateNeuralViz(activity) {
            const viz = document.getElementById('neural-viz');
            viz.innerHTML = '';
            
            for (let i = 0; i < 100; i++) {
                const neuron = document.createElement('span');
                neuron.className = 'neuron';
                if (Math.random() < activity) {
                    neuron.classList.add('active');
                }
                viz.appendChild(neuron);
            }
        }
        
        // Simulate solving process
        function simulateSolving() {
            // Sample puzzle
            const inputGrid = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ];
            
            const outputGrid = [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]
            ];
            
            createGrid(inputGrid, 'input-grid');
            createGrid(outputGrid, 'output-grid');
            
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 10;
                document.getElementById('progress').style.width = progress + '%';
                document.getElementById('progress').textContent = progress + '%';
                document.getElementById('confidence').textContent = progress + '%';
                
                if (progress >= 100) {
                    clearInterval(progressInterval);
                    document.getElementById('pattern-discovery').style.display = 'block';
                    document.getElementById('pattern-description').textContent = 
                        'Cross pattern inversion detected! Colors are inverted where cross pattern exists.';
                }
            }, 500);
            
            // Simulate agent dialogue
            const messages = [
                {agent: 'Pattern Analyst', agent_type: 'pattern', text: 'Detecting cross-shaped pattern in input'},
                {agent: 'Transform Specialist', agent_type: 'transform', text: 'Suggesting color inversion transformation'},
                {agent: 'Validator', agent_type: 'validator', text: 'Confirming pattern consistency across examples'},
                {agent: 'Strategist', agent_type: 'strategist', text: 'Applying learned transformation to test input'}
            ];
            
            messages.forEach((msg, i) => {
                setTimeout(() => addAgentMessage(msg), i * 1000);
            });
            
            // Simulate neural activity
            setInterval(() => {
                updateNeuralViz(Math.random() * 0.5 + 0.3);
            }, 300);
        }
        
        // Connect and start simulation
        connectWebSocket();
        setTimeout(simulateSolving, 1000);
        
        // Update attempt counter
        setInterval(() => {
            attemptCount++;
            document.getElementById('attempts').textContent = attemptCount;
        }, 2000);
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    import uvicorn
    print("Real-time Solving Visualizer")
    print("Open http://localhost:8052/visualizer to view")
    uvicorn.run(app, host="0.0.0.0", port=8052)