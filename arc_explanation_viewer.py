"""
Interactive Explanation Viewer for ARC AGI Solutions
Provides detailed, step-by-step explanations of puzzle solutions
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json
from typing import Dict, List
from datetime import datetime

app = FastAPI()

@app.get("/explanation-viewer")
async def explanation_viewer():
    """Interactive explanation viewer page"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI - Interactive Explanation Viewer</title>
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
        
        .explanation-container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .puzzle-selector {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .explanation-steps {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .step-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 20px;
            position: relative;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .step-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .step-card.active {
            background: linear-gradient(135deg, #f5f7ff 0%, #f0f2ff 100%);
            border-color: #667eea;
        }
        
        .step-number {
            position: absolute;
            left: -15px;
            top: 20px;
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .step-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding-left: 20px;
        }
        
        .step-description {
            color: #666;
            line-height: 1.6;
            padding-left: 20px;
        }
        
        .reasoning-type {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin: 10px 0 10px 20px;
        }
        
        .evidence-box {
            background: #f5fff5;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 15px 0 15px 20px;
            border-radius: 5px;
        }
        
        .evidence-title {
            font-weight: bold;
            color: #4caf50;
            margin-bottom: 5px;
        }
        
        .visual-demo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
        }
        
        .grid-mini {
            display: inline-block;
            border: 2px solid #333;
        }
        
        .grid-cell-mini {
            width: 20px;
            height: 20px;
            border: 1px solid #ddd;
            display: inline-block;
        }
        
        .grid-row-mini {
            display: flex;
        }
        
        .critique-section {
            background: #fff8e1;
            border: 2px solid #ffc107;
            border-radius: 15px;
            padding: 20px;
            margin: 30px 0;
        }
        
        .critique-score {
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
            text-align: center;
            margin: 20px 0;
        }
        
        .critique-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .critique-item {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .critique-item.pass {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .critique-item.fail {
            background: #ffebee;
            color: #c62828;
        }
        
        .assumptions-list {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .confidence-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
            height: 100%;
            transition: width 0.5s ease;
        }
        
        .interactive-controls {
            position: fixed;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        
        .control-btn {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: none;
            border-radius: 8px;
            background: #667eea;
            color: white;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .control-btn:hover {
            background: #764ba2;
        }
        
        /* ARC colors */
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
        
        .pattern-highlight {
            animation: highlight 2s infinite;
        }
        
        @keyframes highlight {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📖 Interactive Explanation Viewer</h1>
            <p>Understand how the AI solves ARC puzzles step-by-step</p>
        </div>
        
        <div class="explanation-container">
            <div class="puzzle-selector">
                <h3>Select Puzzle</h3>
                <select id="puzzle-select" onchange="loadExplanation(this.value)" style="padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <option value="puzzle1">Puzzle 1: Color Inversion</option>
                    <option value="puzzle2">Puzzle 2: Pattern Rotation</option>
                    <option value="puzzle3">Puzzle 3: Object Extraction</option>
                    <option value="puzzle4">Puzzle 4: Symmetry Detection</option>
                </select>
            </div>
            
            <div class="explanation-steps" id="steps">
                <div class="step-card active" onclick="selectStep(1)">
                    <div class="step-number">1</div>
                    <div class="step-title">Initial Pattern Analysis</div>
                    <div class="reasoning-type">PATTERN_MATCHING</div>
                    <div class="step-description">
                        The system begins by analyzing the training examples to identify common patterns.
                        Multiple agents collaborate to examine the input-output relationships.
                    </div>
                    <div class="evidence-box">
                        <div class="evidence-title">Evidence</div>
                        <div>Training example 1 shows a cross pattern being inverted</div>
                        <div>All examples maintain the same grid dimensions</div>
                        <div>Color mapping is consistent across examples</div>
                    </div>
                    <div class="visual-demo">
                        <div>
                            <p style="text-align: center; margin-bottom: 10px;">Input</p>
                            <div class="grid-mini" id="demo-input-1"></div>
                        </div>
                        <div style="font-size: 24px;">→</div>
                        <div>
                            <p style="text-align: center; margin-bottom: 10px;">Output</p>
                            <div class="grid-mini" id="demo-output-1"></div>
                        </div>
                    </div>
                </div>
                
                <div class="step-card" onclick="selectStep(2)">
                    <div class="step-number">2</div>
                    <div class="step-title">Transformation Rule Extraction</div>
                    <div class="reasoning-type">LOGICAL_DEDUCTION</div>
                    <div class="step-description">
                        Based on the patterns identified, the system deduces the transformation rules
                        that convert inputs to outputs consistently across all training examples.
                    </div>
                    <div class="evidence-box">
                        <div class="evidence-title">Extracted Rules</div>
                        <div>• Where cross pattern exists: invert colors (0→1, 1→0)</div>
                        <div>• Where no pattern: maintain original color</div>
                        <div>• Transformation is position-dependent</div>
                    </div>
                </div>
                
                <div class="step-card" onclick="selectStep(3)">
                    <div class="step-number">3</div>
                    <div class="step-title">Validation Against Training Data</div>
                    <div class="reasoning-type">VERIFICATION</div>
                    <div class="step-description">
                        The extracted rules are validated by applying them to all training inputs
                        and comparing the results with expected outputs.
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 95%"></div>
                    </div>
                    <p style="text-align: center; color: #666;">Confidence: 95%</p>
                </div>
                
                <div class="step-card" onclick="selectStep(4)">
                    <div class="step-number">4</div>
                    <div class="step-title">Apply to Test Input</div>
                    <div class="reasoning-type">PATTERN_APPLICATION</div>
                    <div class="step-description">
                        The validated transformation rules are applied to the test input to generate
                        the final solution.
                    </div>
                    <div class="visual-demo">
                        <div>
                            <p style="text-align: center; margin-bottom: 10px;">Test Input</p>
                            <div class="grid-mini" id="test-input"></div>
                        </div>
                        <div style="font-size: 24px;">→</div>
                        <div>
                            <p style="text-align: center; margin-bottom: 10px;">Solution</p>
                            <div class="grid-mini" id="test-solution"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="critique-section">
                <h3>Solution Critique</h3>
                <div class="critique-score">87%</div>
                <p style="text-align: center; color: #666;">Overall Verification Score</p>
                
                <div class="critique-details">
                    <div class="critique-item pass">
                        ✓ Consistency Check
                    </div>
                    <div class="critique-item pass">
                        ✓ Transformation Valid
                    </div>
                    <div class="critique-item pass">
                        ✓ Evidence Sufficient
                    </div>
                    <div class="critique-item pass">
                        ✓ Logical Coherence
                    </div>
                    <div class="critique-item fail">
                        ✗ Full Generalization
                    </div>
                </div>
                
                <div class="assumptions-list">
                    <h4>Assumptions Made</h4>
                    <ul>
                        <li>Pattern is deterministic and consistent</li>
                        <li>Transformation generalizes from 3 training examples</li>
                        <li>No hidden rules beyond observed patterns</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="interactive-controls">
            <h4>Controls</h4>
            <button class="control-btn" onclick="playAnimation()">▶ Play Animation</button>
            <button class="control-btn" onclick="showAllSteps()">Show All Steps</button>
            <button class="control-btn" onclick="showCritique()">View Critique</button>
            <button class="control-btn" onclick="exportExplanation()">Export PDF</button>
        </div>
    </div>
    
    <script>
        let currentStep = 1;
        
        function createMiniGrid(data, containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            container.innerHTML = '';
            data.forEach(row => {
                const rowDiv = document.createElement('div');
                rowDiv.className = 'grid-row-mini';
                row.forEach(cell => {
                    const cellDiv = document.createElement('div');
                    cellDiv.className = 'grid-cell-mini color-' + cell;
                    rowDiv.appendChild(cellDiv);
                });
                container.appendChild(rowDiv);
            });
        }
        
        function selectStep(stepNum) {
            currentStep = stepNum;
            document.querySelectorAll('.step-card').forEach((card, index) => {
                if (index + 1 === stepNum) {
                    card.classList.add('active');
                } else {
                    card.classList.remove('active');
                }
            });
        }
        
        function loadExplanation(puzzleId) {
            // Load different explanation based on selected puzzle
            console.log('Loading explanation for:', puzzleId);
            
            // Update grids with sample data
            const sampleGrid = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ];
            
            const invertedGrid = [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]
            ];
            
            createMiniGrid(sampleGrid, 'demo-input-1');
            createMiniGrid(invertedGrid, 'demo-output-1');
            createMiniGrid(sampleGrid, 'test-input');
            createMiniGrid(invertedGrid, 'test-solution');
        }
        
        function playAnimation() {
            let step = 1;
            const interval = setInterval(() => {
                selectStep(step);
                step++;
                if (step > 4) {
                    clearInterval(interval);
                }
            }, 1500);
        }
        
        function showAllSteps() {
            document.querySelectorAll('.step-card').forEach(card => {
                card.classList.add('active');
            });
        }
        
        function showCritique() {
            document.querySelector('.critique-section').scrollIntoView({
                behavior: 'smooth'
            });
        }
        
        function exportExplanation() {
            alert('Exporting explanation to PDF...');
        }
        
        // Initialize with sample data
        loadExplanation('puzzle1');
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    import uvicorn
    print("Interactive Explanation Viewer")
    print("Open http://localhost:8053/explanation-viewer to view")
    uvicorn.run(app, host="0.0.0.0", port=8053)