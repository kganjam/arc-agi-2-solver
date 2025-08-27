"""
Simple test server to verify the application works
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>ARC AGI Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { display: flex; gap: 20px; }
        .panel { border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        .challenges { width: 200px; }
        .puzzle { flex: 1; }
        .chat { width: 300px; }
        .challenge-item { padding: 10px; margin: 5px 0; background: #f0f0f0; cursor: pointer; border-radius: 4px; }
        .challenge-item:hover { background: #e0e0e0; }
        .challenge-item.active { background: #007bff; color: white; }
        .grid { display: inline-block; border: 2px solid #333; margin: 10px; }
        .grid-row { display: flex; }
        .grid-cell { width: 30px; height: 30px; border: 1px solid #666; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .chat-messages { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
        .message { margin: 10px 0; padding: 8px; border-radius: 4px; }
        .user { background: #e3f2fd; }
        .assistant { background: #f1f8e9; }
        .chat-input { width: 100%; height: 60px; margin-bottom: 10px; }
        .send-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>ARC AGI Challenge Solver</h1>
    <div class="container">
        <div class="panel challenges">
            <h3>Challenges</h3>
            <div class="challenge-item" onclick="selectChallenge('sample1')">sample1</div>
            <div class="challenge-item" onclick="selectChallenge('sample2')">sample2</div>
        </div>
        
        <div class="panel puzzle">
            <h3>Puzzle Viewer</h3>
            <div id="puzzle-content">Select a challenge to view</div>
        </div>
        
        <div class="panel chat">
            <h3>AI Assistant</h3>
            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">AI Assistant ready! Select a puzzle and ask questions.</div>
            </div>
            <textarea class="chat-input" id="chat-input" placeholder="Ask about puzzle patterns... (Press Enter to send)"></textarea>
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const challenges = {
            sample1: {
                id: 'sample1',
                train: [
                    { input: [[1, 0], [0, 1]], output: [[1, 0], [0, 1]] }
                ],
                test: [
                    { input: [[4, 5], [5, 4]] }
                ]
            },
            sample2: {
                id: 'sample2',
                train: [
                    { input: [[1, 1, 1], [1, 2, 1], [1, 1, 1]], output: [[3, 3, 3], [3, 2, 3], [3, 3, 3]] }
                ],
                test: [
                    { input: [[2, 2, 2], [2, 1, 2], [2, 2, 2]] }
                ]
            }
        };
        
        const colors = {
            0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40', 4: '#FFDC00',
            5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B', 8: '#7FDBFF', 9: '#870C25'
        };
        
        let currentChallenge = null;
        
        function selectChallenge(challengeId) {
            // Update UI
            document.querySelectorAll('.challenge-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.classList.add('active');
            
            currentChallenge = challengeId;
            const challenge = challenges[challengeId];
            
            let html = `<h4>${challenge.id}</h4>`;
            
            // Training examples
            challenge.train.forEach((example, i) => {
                html += `<h5>Training Example ${i + 1}</h5>`;
                html += '<div style="display: flex; align-items: center; gap: 20px;">';
                html += renderGrid(example.input);
                html += '<span style="font-size: 20px;">→</span>';
                if (example.output) {
                    html += renderGrid(example.output);
                }
                html += '</div>';
            });
            
            // Test cases
            challenge.test.forEach((test, i) => {
                html += `<h5>Test Case ${i + 1}</h5>`;
                html += '<div style="display: flex; align-items: center; gap: 20px;">';
                html += renderGrid(test.input);
                html += '<span style="font-size: 20px;">→ ?</span>';
                html += '</div>';
            });
            
            document.getElementById('puzzle-content').innerHTML = html;
            addMessage('system', `Loaded puzzle: ${challengeId}`);
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
        
        function addMessage(type, content) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.innerHTML = `<strong>${type === 'user' ? 'You' : type === 'assistant' ? 'AI' : 'System'}:</strong> ${content}`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            // Simulate AI response
            setTimeout(() => {
                let response = 'I can help analyze ARC puzzles! ';
                if (currentChallenge) {
                    const challenge = challenges[currentChallenge];
                    response += `Looking at ${currentChallenge}, I see ${challenge.train.length} training example(s). `;
                    if (message.toLowerCase().includes('pattern')) {
                        response += 'The pattern appears to involve color transformations or spatial relationships.';
                    } else if (message.toLowerCase().includes('solve')) {
                        response += 'To solve this, look for consistent rules between input and output grids.';
                    } else {
                        response += 'What specific aspect would you like me to analyze?';
                    }
                } else {
                    response += 'Please select a challenge first to analyze.';
                }
                addMessage('assistant', response);
            }, 1000);
        }
        
        // Enter key support
        document.getElementById('chat-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """)

@app.get("/api/test")
async def test():
    return {"status": "working", "message": "Backend is running"}

if __name__ == "__main__":
    print("Starting test server on http://localhost:8050")
    uvicorn.run(app, host="0.0.0.0", port=8050)
