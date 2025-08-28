#!/usr/bin/env python3
"""
Comprehensive test suite for AI puzzle-solving capabilities
Tests all major features including progress streaming, Bedrock integration,
tool creation, heuristics, and Claude Code invocation
"""

import json
import requests
import asyncio
import websockets
import threading
import time
import sys
from typing import Dict, List, Any

class ComprehensiveAITester:
    def __init__(self):
        self.base_url = "http://localhost:8050"
        self.client_id = f"test_client_{int(time.time())}"
        self.progress_messages = []
        self.test_results = {}
        
    def print_header(self, text: str):
        """Print formatted test header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
        
    def print_subheader(self, text: str):
        """Print formatted test subheader"""
        print(f"\n>>> {text}")
        print("-"*60)
        
    async def listen_to_progress(self):
        """WebSocket listener for progress messages"""
        ws_url = f"ws://localhost:8050/ws/puzzle-solving/{self.client_id}"
        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get('type') == 'progress':
                            self.progress_messages.append(data)
                            self.display_progress(data)
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"WebSocket error: {e}")
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
    
    def display_progress(self, data: Dict):
        """Display progress message with appropriate formatting"""
        msg_type = data.get('message_type', 'info')
        msg = data.get('message', '')
        
        # Color codes for terminal
        COLORS = {
            'bedrock_query': '\033[36m',  # Cyan
            'bedrock_response': '\033[35m',  # Magenta
            'bedrock_prompt': '\033[90m',  # Gray
            'theory': '\033[33m',  # Yellow
            'attempt': '\033[34m',  # Blue
            'success': '\033[32m',  # Green
            'failure': '\033[31m',  # Red
            'solved': '\033[92m',  # Bright Green
            'tool_suggestion': '\033[36m',  # Cyan
            'tool_created': '\033[32m',  # Green
            'info': '\033[0m',  # Default
        }
        
        color = COLORS.get(msg_type, COLORS['info'])
        reset = '\033[0m'
        
        # Format based on message type
        if msg_type == 'bedrock_prompt':
            # Show truncated prompt
            print(f"{color}📝 Bedrock Prompt: {msg[:200]}...{reset}")
        else:
            print(f"{color}{msg}{reset}")
            
    def start_websocket_listener(self):
        """Start WebSocket listener in background thread"""
        def run_ws():
            asyncio.new_event_loop().run_until_complete(self.listen_to_progress())
        
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        time.sleep(1)  # Give WebSocket time to connect
        
    def test_puzzle_solving_with_progress(self) -> Dict:
        """Test 1: Basic puzzle solving with progress streaming"""
        self.print_subheader("Test 1: Basic Puzzle Solving with Progress Streaming")
        
        # Get current puzzle
        resp = requests.get(f"{self.base_url}/api/puzzle/current")
        puzzle_data = resp.json()
        puzzle = puzzle_data.get("puzzle", {})
        
        print(f"Testing with puzzle: {puzzle.get('id', 'unknown')}")
        
        # Clear progress messages
        self.progress_messages = []
        
        # Send solving request
        message = "Solve this puzzle step by step. Show me your reasoning."
        print(f"Sending: {message}")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": puzzle.get("id"),
                "output_grid": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "client_id": self.client_id
            },
            timeout=30
        )
        
        result = resp.json() if resp.status_code == 200 else {"error": resp.text}
        
        # Check progress messages
        progress_count = len(self.progress_messages)
        print(f"\n✓ Received {progress_count} progress messages")
        
        # Analyze progress message types
        message_types = {}
        for msg in self.progress_messages:
            msg_type = msg.get('message_type', 'unknown')
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        print("Progress message breakdown:")
        for msg_type, count in message_types.items():
            print(f"  - {msg_type}: {count}")
        
        return {
            "success": progress_count > 0,
            "progress_count": progress_count,
            "message_types": message_types,
            "response": result
        }
    
    def test_bedrock_dialogue(self) -> Dict:
        """Test 2: Full Bedrock dialogue visibility"""
        self.print_subheader("Test 2: Bedrock Dialogue and Multi-turn Reasoning")
        
        self.progress_messages = []
        
        message = """
        Analyze this puzzle using your Bedrock AI capabilities.
        Try multiple approaches and show me your full reasoning process.
        Generate theories about the pattern and test them.
        """
        
        print(f"Sending complex reasoning request...")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": "test_puzzle",
                "output_grid": [[1, 2], [3, 4]],
                "client_id": self.client_id
            },
            timeout=45
        )
        
        # Look for Bedrock-specific messages
        bedrock_messages = [
            msg for msg in self.progress_messages 
            if 'bedrock' in msg.get('message_type', '').lower()
        ]
        
        theory_messages = [
            msg for msg in self.progress_messages 
            if msg.get('message_type') == 'theory'
        ]
        
        print(f"\n✓ Bedrock dialogue messages: {len(bedrock_messages)}")
        print(f"✓ Theory generation messages: {len(theory_messages)}")
        
        return {
            "success": len(bedrock_messages) > 0,
            "bedrock_count": len(bedrock_messages),
            "theory_count": len(theory_messages),
            "total_messages": len(self.progress_messages)
        }
    
    def test_tool_creation(self) -> Dict:
        """Test 3: Dynamic tool creation via Bedrock"""
        self.print_subheader("Test 3: Dynamic Tool Creation")
        
        self.progress_messages = []
        
        message = """
        This puzzle requires a special pattern detector.
        Create a new tool to detect diagonal patterns and use it to solve the puzzle.
        Show me the tool creation process.
        """
        
        print(f"Requesting tool creation...")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": "test_puzzle",
                "output_grid": [[0, 0], [0, 0]],
                "client_id": self.client_id
            },
            timeout=60
        )
        
        # Look for tool-related messages
        tool_messages = [
            msg for msg in self.progress_messages 
            if 'tool' in msg.get('message_type', '').lower()
        ]
        
        print(f"\n✓ Tool-related messages: {len(tool_messages)}")
        
        return {
            "success": len(tool_messages) > 0,
            "tool_message_count": len(tool_messages),
            "response": resp.json() if resp.status_code == 200 else {"error": resp.text}
        }
    
    def test_heuristic_usage(self) -> Dict:
        """Test 4: Heuristic retrieval and usage"""
        self.print_subheader("Test 4: Heuristic Retrieval and Usage")
        
        self.progress_messages = []
        
        message = """
        Use your available heuristics to solve this puzzle.
        Show me which heuristics you're applying and why.
        Try multiple heuristics if needed.
        """
        
        print(f"Testing heuristic usage...")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": "test_puzzle",
                "output_grid": [[0, 1], [1, 0]],
                "client_id": self.client_id
            },
            timeout=30
        )
        
        result = resp.json() if resp.status_code == 200 else {"error": resp.text}
        
        # Check for heuristic application in function results
        function_results = result.get('function_results', [])
        heuristic_calls = [
            fr for fr in function_results 
            if 'heuristic' in fr.get('function', '').lower()
        ]
        
        print(f"\n✓ Heuristic function calls: {len(heuristic_calls)}")
        
        return {
            "success": len(heuristic_calls) > 0 or 'heuristic' in str(result).lower(),
            "heuristic_calls": len(heuristic_calls),
            "response": result
        }
    
    def test_claude_code_invocation(self) -> Dict:
        """Test 5: Claude Code self-update invocation"""
        self.print_subheader("Test 5: Claude Code Self-Update")
        
        self.progress_messages = []
        
        message = """
        I need you to improve your own capabilities.
        Use Claude Code to update your pattern recognition system.
        Add a new feature to detect rotational symmetry in puzzles.
        """
        
        print(f"Testing Claude Code invocation for self-improvement...")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": "test_puzzle",
                "output_grid": [[0, 0], [0, 0]],
                "client_id": self.client_id
            },
            timeout=45
        )
        
        result = resp.json() if resp.status_code == 200 else {"error": resp.text}
        
        # Check for Claude Code related activity
        claude_mentioned = 'claude' in str(result).lower()
        
        print(f"\n✓ Claude Code mentioned: {claude_mentioned}")
        
        return {
            "success": claude_mentioned,
            "response": result
        }
    
    def test_multi_attempt_solving(self) -> Dict:
        """Test 6: Multiple solving attempts with different approaches"""
        self.print_subheader("Test 6: Multi-Attempt Solving")
        
        self.progress_messages = []
        
        message = """
        Try to solve this puzzle.
        If your first approach doesn't work, try at least 5 different approaches.
        Show me each attempt and why you think it might work.
        """
        
        print(f"Testing multiple solving attempts...")
        
        resp = requests.post(
            f"{self.base_url}/api/puzzle/ai-chat",
            json={
                "message": message,
                "puzzle_id": "test_puzzle",
                "output_grid": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "client_id": self.client_id
            },
            timeout=60
        )
        
        # Count attempt messages
        attempt_messages = [
            msg for msg in self.progress_messages 
            if msg.get('message_type') == 'attempt'
        ]
        
        print(f"\n✓ Attempt messages: {len(attempt_messages)}")
        
        return {
            "success": len(attempt_messages) >= 3,
            "attempt_count": len(attempt_messages),
            "total_progress": len(self.progress_messages)
        }
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        self.print_header("🧪 COMPREHENSIVE AI TESTING SUITE")
        print("Testing all AI capabilities including progress streaming,")
        print("Bedrock integration, tool creation, and more...")
        
        # Start WebSocket listener
        print("\n📡 Starting WebSocket listener...")
        self.start_websocket_listener()
        
        # Run tests
        tests = [
            self.test_puzzle_solving_with_progress,
            self.test_bedrock_dialogue,
            self.test_tool_creation,
            self.test_heuristic_usage,
            self.test_claude_code_invocation,
            self.test_multi_attempt_solving
        ]
        
        results = []
        for i, test in enumerate(tests, 1):
            try:
                result = test()
                results.append(result)
                status = "✅ PASSED" if result.get('success') else "❌ FAILED"
                print(f"\nTest {i} Result: {status}")
                time.sleep(2)  # Give time between tests
            except Exception as e:
                print(f"\nTest {i} Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Summary
        self.print_header("📊 TEST RESULTS SUMMARY")
        
        passed = sum(1 for r in results if r.get('success'))
        total = len(results)
        
        print(f"\nTests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # Detailed breakdown
        print("\n📋 Detailed Results:")
        test_names = [
            "Progress Streaming",
            "Bedrock Dialogue",
            "Tool Creation",
            "Heuristic Usage",
            "Claude Code Invocation",
            "Multi-Attempt Solving"
        ]
        
        for i, (name, result) in enumerate(zip(test_names, results)):
            status = "✅" if result.get('success') else "❌"
            print(f"  {status} {name}")
            if not result.get('success'):
                error = result.get('error', 'Test criteria not met')
                print(f"     └─ Issue: {error}")
        
        # Feature verification
        print("\n🔍 Feature Verification:")
        total_progress = sum(len(self.progress_messages) for _ in range(1))
        print(f"  • Total progress messages received: {total_progress}")
        print(f"  • WebSocket streaming: {'✅ Working' if total_progress > 0 else '❌ Not working'}")
        print(f"  • Bedrock integration: {'✅ Active' if results[1].get('bedrock_count', 0) > 0 else '⚠️ Limited'}")
        print(f"  • Theory generation: {'✅ Active' if results[1].get('theory_count', 0) > 0 else '⚠️ Limited'}")
        print(f"  • Tool creation: {'✅ Supported' if results[2].get('success') else '⚠️ Limited'}")
        print(f"  • Heuristics: {'✅ Working' if results[3].get('success') else '⚠️ Limited'}")
        
        return passed == total

def main():
    """Main test execution"""
    tester = ComprehensiveAITester()
    
    # Check if backend is running
    try:
        resp = requests.get(f"{tester.base_url}/api/status", timeout=2)
        if resp.status_code != 200:
            print("❌ Backend is not running! Please start it first.")
            sys.exit(1)
    except:
        print("❌ Cannot connect to backend at http://localhost:8050")
        print("Please ensure the backend is running: python start_backend.py")
        sys.exit(1)
    
    # Run tests
    success = tester.run_all_tests()
    
    print("\n" + "="*70)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ SOME TESTS FAILED - Review the results above")
    print("="*70)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()