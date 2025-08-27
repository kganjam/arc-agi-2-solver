"""
FastAPI Backend for ARC AGI Challenge Solving System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
import urllib.request
import urllib.parse

app = FastAPI(title="ARC AGI Challenge Solver", version="1.0.0")

# Enable CORS for Vue.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (only if directory exists)
import os
if os.path.exists("frontend/dist"):
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

# Data models
class ARCExample(BaseModel):
    input: List[List[int]]
    output: Optional[List[List[int]]] = None

class ARCChallenge(BaseModel):
    id: str
    train: List[ARCExample]
    test: List[ARCExample]

class ChatMessage(BaseModel):
    message: str
    challenge_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str

# Global storage
challenges: Dict[str, ARCChallenge] = {}
bedrock_token = "ABSKQmVkcm9ja0FQSUtleS1ka2t5LWFOLTAYNDQ2MzAwMTpCSVE9aERoMXRpeFVUQ3FlGZSZGp6UmI6QytaUtPSGtEaGJ0SWJiOU1FUVJiTT0="

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    load_sample_challenges()

def load_sample_challenges():
    """Load sample ARC challenges"""
    os.makedirs("data", exist_ok=True)
    
    # Sample challenge 1
    sample1 = ARCChallenge(
        id="sample1",
        train=[
            ARCExample(input=[[1, 0], [0, 1]], output=[[1, 0], [0, 1]]),
            ARCExample(input=[[2, 3], [3, 2]], output=[[2, 3], [3, 2]])
        ],
        test=[
            ARCExample(input=[[4, 5], [5, 4]])
        ]
    )
    
    # Sample challenge 2
    sample2 = ARCChallenge(
        id="sample2",
        train=[
            ARCExample(
                input=[[1, 1, 1], [1, 2, 1], [1, 1, 1]], 
                output=[[3, 3, 3], [3, 2, 3], [3, 3, 3]]
            )
        ],
        test=[
            ARCExample(input=[[2, 2, 2], [2, 1, 2], [2, 2, 2]])
        ]
    )
    
    challenges["sample1"] = sample1
    challenges["sample2"] = sample2

@app.get("/")
async def read_root():
    """Serve the main application"""
    if os.path.exists("frontend/dist/index.html"):
        return HTMLResponse(open("frontend/dist/index.html").read())
    else:
        return HTMLResponse("""
        <html>
        <head><title>ARC AGI Backend</title></head>
        <body>
            <h1>ARC AGI Backend Running</h1>
            <p>Backend is running on port 8050</p>
            <p>API endpoints:</p>
            <ul>
                <li><a href="/api/challenges">/api/challenges</a> - List challenges</li>
                <li><a href="/docs">/docs</a> - API documentation</li>
            </ul>
        </body>
        </html>
        """)

@app.get("/api/challenges")
async def get_challenges():
    """Get all available challenges"""
    return {"challenges": list(challenges.keys())}

@app.get("/api/challenges/{challenge_id}")
async def get_challenge(challenge_id: str):
    """Get a specific challenge"""
    if challenge_id not in challenges:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenges[challenge_id]

@app.post("/api/chat")
async def chat_with_ai(message: ChatMessage):
    """Chat with Bedrock AI about puzzles"""
    try:
        # Prepare context if challenge is provided
        context = ""
        if message.challenge_id and message.challenge_id in challenges:
            challenge = challenges[message.challenge_id]
            context = f"Current ARC puzzle: {challenge.id} with {len(challenge.train)} training examples"
        
        # Call Bedrock API
        response = await call_bedrock_api(message.message, context)
        
        from datetime import datetime
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def call_bedrock_api(message: str, context: str = "") -> str:
    """Call AWS Bedrock API"""
    try:
        system_prompt = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. 
You help analyze patterns, suggest transformations, and guide users through solving abstract reasoning challenges.
Focus on identifying visual patterns, transformations, and logical rules in grid-based puzzles."""
        
        if context:
            system_prompt += f"\n\nContext: {context}"
        
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }
        
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke"
        headers = {
            'Authorization': f'Bearer {bedrock_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('content', [{}])[0].get('text', 'No response received')
            else:
                return f"Error: HTTP {response.status}"
                
    except Exception as e:
        return f"Error connecting to Bedrock: {str(e)}"

@app.post("/api/analyze/{challenge_id}")
async def analyze_challenge(challenge_id: str):
    """Analyze a specific challenge"""
    if challenge_id not in challenges:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    challenge = challenges[challenge_id]
    
    # Basic analysis
    analysis = {
        "challenge_id": challenge_id,
        "training_examples": len(challenge.train),
        "test_cases": len(challenge.test),
        "patterns": [],
        "suggestions": []
    }
    
    # Analyze first training example
    if challenge.train:
        first_example = challenge.train[0]
        input_grid = first_example.input
        
        analysis["grid_size"] = f"{len(input_grid)}x{len(input_grid[0])}"
        
        # Get unique colors
        colors = set()
        for row in input_grid:
            colors.update(row)
        analysis["colors_used"] = sorted(list(colors))
        
        # Check if input equals output
        if first_example.output:
            if input_grid == first_example.output:
                analysis["patterns"].append("Identity transformation")
            else:
                analysis["patterns"].append("Transformation detected")
        
        analysis["suggestions"] = [
            "Look for color patterns",
            "Check for symmetries",
            "Analyze object shapes"
        ]
    
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
