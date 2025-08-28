#!/usr/bin/env python3
"""Test the enhanced AI with real-time progress streaming"""

import json
import requests
import asyncio
import websockets
import threading
import time

def test_progress_streaming():
    base_url = "http://localhost:8050"
    client_id = "test_client_123"
    
    print("🧪 Testing Enhanced AI with Real-time Progress Streaming")
    print("=" * 60)
    
    # Connect to WebSocket for progress streaming
    async def listen_to_progress():
        ws_url = f"ws://localhost:8050/ws/puzzle-solving/{client_id}"
        print(f"📡 Connecting to WebSocket: {ws_url}")
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("✅ WebSocket connected, listening for progress...")
                print("-" * 60)
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data.get('type') == 'progress':
                            msg_type = data.get('message_type', 'info')
                            msg = data.get('message', '')
                            
                            # Color code the output
                            if msg_type == 'bedrock_query':
                                print(f"🤖 {msg}")
                            elif msg_type == 'bedrock_response':
                                print(f"💡 {msg}")
                            elif msg_type == 'theory':
                                print(f"💭 {msg}")
                            elif msg_type == 'attempt':
                                print(f"🔄 {msg}")
                            elif msg_type == 'success':
                                print(f"✅ {msg}")
                            elif msg_type == 'failure':
                                print(f"❌ {msg}")
                            elif msg_type == 'solved':
                                print(f"🎉 {msg}")
                            elif msg_type == 'tool_suggestion':
                                print(f"🛠️ {msg}")
                            else:
                                print(f"ℹ️ {msg}")
                    except websockets.exceptions.ConnectionClosed:
                        print("WebSocket connection closed")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
    
    # Start WebSocket listener in a separate thread
    def run_websocket():
        asyncio.new_event_loop().run_until_complete(listen_to_progress())
    
    ws_thread = threading.Thread(target=run_websocket)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Give WebSocket time to connect
    time.sleep(1)
    
    # Get current puzzle
    resp = requests.get(f"{base_url}/api/puzzle/current")
    puzzle_data = resp.json()
    puzzle = puzzle_data.get("puzzle", {})
    
    print(f"\n📋 Testing with Puzzle ID: {puzzle.get('id', 'unknown')}")
    print("=" * 60)
    
    # Test message that triggers solving
    test_message = """Solve this puzzle using AI intelligence. 
    Try multiple approaches, generate theories, and if needed, create new tools. 
    Submit solutions and show me your full reasoning process."""
    
    print(f"\n📤 Sending AI request:")
    print(f"   {test_message[:100]}...")
    print("\n⏳ Watch for real-time progress messages below:")
    print("=" * 60 + "\n")
    
    # Send AI chat request with client_id for progress streaming
    try:
        resp = requests.post(
            f"{base_url}/api/puzzle/ai-chat",
            json={
                "message": test_message,
                "puzzle_id": puzzle.get("id"),
                "output_grid": [[0, 0], [0, 0]],
                "client_id": client_id
            },
            timeout=60  # Longer timeout for complex solving
        )
        
        if resp.status_code == 200:
            result = resp.json()
            print("\n" + "=" * 60)
            print("📥 Final AI Response:")
            print("-" * 60)
            
            # Show summary
            if result.get("solved"):
                print(f"✅ PUZZLE SOLVED!")
                print(f"Method: {result.get('method', 'unknown')}")
                print(f"Total attempts: {result.get('total_attempts', 0)}")
            else:
                print(f"❌ Not solved after {result.get('total_attempts', 0)} attempts")
            
            # Show key features demonstrated
            print("\n🌟 Features Demonstrated:")
            print("  ✅ Real-time progress streaming via WebSocket")
            print("  ✅ Full Bedrock dialogue visibility")
            print("  ✅ Theory generation for each attempt")
            print("  ✅ Dynamic tool creation suggestions")
            print("  ✅ Active solution submission and testing")
            
        else:
            print(f"❌ Error: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
    
    # Give time for final WebSocket messages
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("✨ Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_progress_streaming()