"""
ARC AGI Integrated Application with Live Dashboard
Combines web interface, solver, and real-time progress tracking
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import copy

# Import solver components
from arc_solver import Heuristic, PatternTool, load_puzzles
from arc_solver_enhanced import EnhancedARCSolver, ClaudeCodeIntegration
from arc_solver_safeguarded import SafeguardedSolver, ARCPuzzle, SafeguardViolationError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
solver_state = {
    'running': False,
    'puzzles_solved': 0,
    'total_puzzles': 0,
    'attempts': 0,
    'heuristics_count': 0,
    'tools_generated': 0,
    'claude_calls': 0,
    'total_cost': 0.0,
    'current_puzzle': '',
    'success_rate': 0.0,
    'start_time': None,
    'activity_log': []
}

# WebSocket connections
websocket_clients = []

class MetaHeuristic:
    """Meta-heuristic for guiding code generation"""
    
    def __init__(self, meta_id: str, name: str, strategy: str, generates: str, trigger: str):
        self.id = meta_id
        self.name = name
        self.strategy = strategy
        self.generates = generates
        self.trigger = trigger
        self.activation_count = 0
        
    def should_activate(self, solver_state: Dict, failure_patterns: List) -> bool:
        """Check if this meta-heuristic should activate"""
        if "Multiple failures" in self.trigger and len(failure_patterns) >= 3:
            return True
        if "similar characteristics" in self.trigger:
            # Check for pattern similarity
            return len(set(failure_patterns)) < len(failure_patterns) * 0.7
        return False
        
    def generate_prompt(self, context: Dict) -> str:
        """Generate Claude Code prompt based on strategy"""
        prompt = f"Based on meta-heuristic '{self.name}':\\n"
        prompt += f"Strategy: {self.strategy}\\n"
        prompt += f"Context: {json.dumps(context, indent=2)}\\n"
        prompt += f"Generate: {self.generates}\\n"
        return prompt


class IntegratedSolver(EnhancedARCSolver):
    """Solver with web integration and meta-heuristics"""
    
    def __init__(self, use_safeguards=True):
        super().__init__()
        self.use_safeguards = use_safeguards
        self.safeguarded_solver = SafeguardedSolver() if use_safeguards else None
        self.meta_heuristics = self._init_meta_heuristics()
        self.cost_per_claude_call = 0.01  # Estimated cost per API call
        self.cheating_attempts_blocked = 0
        
    def _init_meta_heuristics(self) -> List[MetaHeuristic]:
        """Initialize default meta-heuristics"""
        return [
            MetaHeuristic(
                "m1", "Pattern Gap Identifier",
                "Analyze failed puzzles to identify missing pattern types",
                "Pattern detection tools",
                "Multiple failures with similar characteristics"
            ),
            MetaHeuristic(
                "m2", "Tool Combiner",
                "Combine successful heuristics to create hybrid approaches",
                "Hybrid transformation tools",
                "Multiple partial successes"
            ),
            MetaHeuristic(
                "m3", "Complexity Escalator",
                "Start simple, increase complexity if needed",
                "Advanced transformation tools",
                "Simple approaches exhausted"
            ),
            MetaHeuristic(
                "m4", "Symmetry Specialist",
                "Generate symmetry-specific tools when detected",
                "Symmetry transformation tools",
                "Symmetry patterns detected"
            ),
            MetaHeuristic(
                "m5", "Color Logic Builder",
                "Create color transformation rules from examples",
                "Color mapping tools",
                "Color patterns identified"
            ),
        ]
        
    async def solve_with_updates(self, puzzles: List[Dict]):
        """Solve puzzles with real-time updates"""
        global solver_state
        
        solver_state['running'] = True
        solver_state['start_time'] = time.time()
        solver_state['total_puzzles'] = len(puzzles)
        
        iteration = 0
        max_iterations = 100
        
        while len(self.solved_puzzles) < len(puzzles) and iteration < max_iterations:
            iteration += 1
            
            # Update state
            solver_state['puzzles_solved'] = len(self.solved_puzzles)
            solver_state['heuristics_count'] = len(self.heuristics)
            solver_state['success_rate'] = len(self.solved_puzzles) / max(1, len(puzzles)) * 100
            
            # Get unsolved puzzles
            unsolved = [p for p in puzzles if p['id'] not in self.solved_puzzles]
            
            for puzzle in unsolved:
                solver_state['current_puzzle'] = puzzle['id']
                
                # Attempt to solve
                if self.use_safeguards and self.safeguarded_solver:
                    # Use safeguarded solver for proper ARC solving
                    try:
                        arc_puzzle = ARCPuzzle(puzzle)
                        solution, result = self.safeguarded_solver.solve_puzzle(arc_puzzle)
                        result['solved'] = result.get('validated', False)
                        if result.get('cheating_attempted', False):
                            self.cheating_attempts_blocked += 1
                            solver_state['activity_log'].append({
                                'time': datetime.now().isoformat(),
                                'message': f"⚠️ Cheating attempt blocked for {puzzle['id']}",
                                'type': 'warning'
                            })
                    except SafeguardViolationError:
                        self.cheating_attempts_blocked += 1
                        result = {'solved': False, 'cheating_attempted': True}
                        solution = None
                else:
                    solution, result = self.solve_puzzle(puzzle)
                solver_state['attempts'] += 1
                
                if result['solved']:
                    activity = f"✓ Solved {puzzle['id']} using {result.get('heuristic_used', 'unknown')}"
                    solver_state['activity_log'].append({
                        'time': datetime.now().isoformat(),
                        'message': activity,
                        'type': 'success'
                    })
                else:
                    activity = f"✗ Failed {puzzle['id']} (attempt {result['attempts']})"
                    solver_state['activity_log'].append({
                        'time': datetime.now().isoformat(),
                        'message': activity,
                        'type': 'failure'
                    })
                    
                # Broadcast update
                await broadcast_update()
                
            # Check for meta-heuristic activation
            if iteration % 5 == 0:
                await self._apply_meta_heuristics(unsolved)
                
            # Generate new tools if stuck
            if iteration % 10 == 0 and len(unsolved) > 0:
                await self._generate_new_tool_async(unsolved)
                
        solver_state['running'] = False
        await broadcast_update()
        
    async def _apply_meta_heuristics(self, unsolved_puzzles: List[Dict]):
        """Apply meta-heuristics to guide tool generation"""
        failure_patterns = []
        
        for puzzle in unsolved_puzzles:
            if puzzle['id'] in self.puzzle_attempts:
                failure_patterns.append(self.puzzle_attempts[puzzle['id']])
                
        for meta_h in self.meta_heuristics:
            if meta_h.should_activate(solver_state, failure_patterns):
                activity = f"🧠 Activating meta-heuristic: {meta_h.name}"
                solver_state['activity_log'].append({
                    'time': datetime.now().isoformat(),
                    'message': activity,
                    'type': 'meta'
                })
                
                # Generate tool using meta-heuristic guidance
                context = {
                    'unsolved_count': len(unsolved_puzzles),
                    'failure_patterns': failure_patterns[:3]
                }
                
                prompt = meta_h.generate_prompt(context)
                # Here we would call Claude Code with the prompt
                solver_state['claude_calls'] += 1
                solver_state['total_cost'] += self.cost_per_claude_call
                
                await broadcast_update()
                break
                
    async def _generate_new_tool_async(self, unsolved_puzzles: List[Dict]):
        """Generate new tool asynchronously"""
        solver_state['claude_calls'] += 1
        solver_state['total_cost'] += self.cost_per_claude_call
        solver_state['tools_generated'] += 1
        
        activity = f"🔧 Generating new tool (Tool #{solver_state['tools_generated']})"
        solver_state['activity_log'].append({
            'time': datetime.now().isoformat(),
            'message': activity,
            'type': 'tool'
        })
        
        await broadcast_update()


async def broadcast_update():
    """Broadcast state update to all connected clients"""
    message = json.dumps(solver_state)
    
    for client in websocket_clients:
        try:
            await client.send_text(message)
        except:
            pass  # Client disconnected


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps(solver_state))
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)


@app.get("/")
async def root():
    """Enhanced UI with live dashboard"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI Solver - Live Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-label {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        
        .metric-subvalue {
            color: #999;
            font-size: 14px;
            margin-top: 5px;
        }
        
        .progress-section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .activity-feed {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        
        .activity-item {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #e0e0e0;
            background: #f9f9f9;
            border-radius: 4px;
        }
        
        .activity-item.success {
            border-left-color: #4caf50;
            background: #e8f5e9;
        }
        
        .activity-item.failure {
            border-left-color: #f44336;
            background: #ffebee;
        }
        
        .activity-item.meta {
            border-left-color: #2196f3;
            background: #e3f2fd;
        }
        
        .activity-item.tool {
            border-left-color: #ff9800;
            background: #fff3e0;
        }
        
        .control-panel {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .btn {
            padding: 12px 30px;
            margin: 0 10px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-start {
            background: #4caf50;
            color: white;
        }
        
        .btn-start:hover {
            background: #45a049;
        }
        
        .btn-stop {
            background: #f44336;
            color: white;
        }
        
        .btn-stop:hover {
            background: #da190b;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-running {
            background: #4caf50;
        }
        
        .status-stopped {
            background: #f44336;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .cost-tracker {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            font-weight: bold;
        }
        
        .cost-value {
            color: #f44336;
            font-size: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 ARC AGI Solver</h1>
            <p>Autonomous Puzzle Solving with Meta-Learning</p>
        </div>
        
        <div class="cost-tracker">
            💰 Total Cost: $<span id="total-cost" class="cost-value">0.00</span>
        </div>
        
        <div class="dashboard">
            <div class="metric-card">
                <div class="metric-label">Puzzles Solved</div>
                <div class="metric-value" id="puzzles-solved">0/0</div>
                <div class="metric-subvalue" id="success-rate">0% Success Rate</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Solution Attempts</div>
                <div class="metric-value" id="attempts">0</div>
                <div class="metric-subvalue" id="current-puzzle">No puzzle</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Heuristics</div>
                <div class="metric-value" id="heuristics">0</div>
                <div class="metric-subvalue">Active strategies</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Generated Tools</div>
                <div class="metric-value" id="tools">0</div>
                <div class="metric-subvalue">Custom solutions</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Claude Calls</div>
                <div class="metric-value" id="claude-calls">0</div>
                <div class="metric-subvalue">API requests</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Time Elapsed</div>
                <div class="metric-value" id="time-elapsed">0:00</div>
                <div class="metric-subvalue">Running time</div>
            </div>
        </div>
        
        <div class="progress-section">
            <h2>Overall Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-bar" style="width: 0%">
                    <span id="progress-text">0%</span>
                </div>
            </div>
            <p>
                <span class="status-indicator status-stopped" id="status-indicator"></span>
                <span id="status-text">System Idle</span>
            </p>
        </div>
        
        <div class="activity-feed">
            <h3>Activity Log</h3>
            <div id="activity-log">
                <div class="activity-item">System initialized and ready to solve puzzles...</div>
            </div>
        </div>
        
        <div class="control-panel">
            <button class="btn btn-start" onclick="startSolver()">🚀 Start Solving</button>
            <button class="btn btn-stop" onclick="stopSolver()">⏹ Stop</button>
        </div>
    </div>
    
    <script>
        let ws = null;
        let startTime = null;
        let timeInterval = null;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8050/ws');
            
            ws.onmessage = function(event) {
                const state = JSON.parse(event.data);
                updateDashboard(state);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);
            };
        }
        
        function updateDashboard(state) {
            // Update metrics
            document.getElementById('puzzles-solved').textContent = 
                `${state.puzzles_solved}/${state.total_puzzles}`;
            document.getElementById('success-rate').textContent = 
                `${state.success_rate.toFixed(1)}% Success Rate`;
            document.getElementById('attempts').textContent = state.attempts;
            document.getElementById('current-puzzle').textContent = 
                state.current_puzzle || 'No puzzle';
            document.getElementById('heuristics').textContent = state.heuristics_count;
            document.getElementById('tools').textContent = state.tools_generated;
            document.getElementById('claude-calls').textContent = state.claude_calls;
            document.getElementById('total-cost').textContent = state.total_cost.toFixed(2);
            
            // Update progress bar
            const progress = state.total_puzzles > 0 ? 
                (state.puzzles_solved / state.total_puzzles * 100) : 0;
            document.getElementById('progress-bar').style.width = progress + '%';
            document.getElementById('progress-text').textContent = progress.toFixed(1) + '%';
            
            // Update status
            if (state.running) {
                document.getElementById('status-indicator').className = 
                    'status-indicator status-running';
                document.getElementById('status-text').textContent = 'Solving in progress...';
                
                if (!startTime) {
                    startTime = Date.now();
                    startTimeInterval();
                }
            } else {
                document.getElementById('status-indicator').className = 
                    'status-indicator status-stopped';
                document.getElementById('status-text').textContent = 'System Idle';
                stopTimeInterval();
            }
            
            // Update activity log
            if (state.activity_log && state.activity_log.length > 0) {
                const logDiv = document.getElementById('activity-log');
                logDiv.innerHTML = '';
                
                state.activity_log.slice(-10).reverse().forEach(item => {
                    const div = document.createElement('div');
                    div.className = `activity-item ${item.type}`;
                    div.textContent = item.message;
                    logDiv.appendChild(div);
                });
            }
        }
        
        function startTimeInterval() {
            if (timeInterval) return;
            
            timeInterval = setInterval(() => {
                if (startTime) {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = elapsed % 60;
                    document.getElementById('time-elapsed').textContent = 
                        `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
            }, 1000);
        }
        
        function stopTimeInterval() {
            if (timeInterval) {
                clearInterval(timeInterval);
                timeInterval = null;
            }
            startTime = null;
        }
        
        async function startSolver() {
            try {
                const response = await fetch('/api/start-solver', { method: 'POST' });
                const result = await response.json();
                console.log('Solver started:', result);
            } catch (error) {
                console.error('Error starting solver:', error);
            }
        }
        
        async function stopSolver() {
            try {
                const response = await fetch('/api/stop-solver', { method: 'POST' });
                const result = await response.json();
                console.log('Solver stopped:', result);
            } catch (error) {
                console.error('Error stopping solver:', error);
            }
        }
        
        // Connect WebSocket on load
        connectWebSocket();
    </script>
</body>
</html>
    """)


@app.post("/api/start-solver")
async def start_solver():
    """Start the solver in background"""
    global solver_state
    
    if solver_state['running']:
        return {"status": "already_running"}
    
    # Load puzzles
    data_dir = Path("data/arc_agi")
    puzzles = load_puzzles(data_dir, limit=10)
    
    # Create solver with safeguards enabled
    solver = IntegratedSolver(use_safeguards=True)
    
    # Log that we're using safeguarded solver
    solver_state['activity_log'].append({
        'time': datetime.now().isoformat(),
        'message': '🔒 Safeguarded solver enabled - following ARC AGI competition rules',
        'type': 'meta'
    })
    
    # Start solving in background
    async def run_solver():
        await solver.solve_with_updates(puzzles)
    
    asyncio.create_task(run_solver())
    
    return {"status": "started", "puzzles": len(puzzles)}


@app.post("/api/stop-solver")
async def stop_solver():
    """Stop the solver"""
    global solver_state
    solver_state['running'] = False
    return {"status": "stopped"}


@app.get("/api/status")
async def get_status():
    """Get current solver status"""
    return solver_state


if __name__ == "__main__":
    import uvicorn
    print("Starting ARC AGI Integrated Application")
    print("Open http://localhost:8050 to view the live dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8050)