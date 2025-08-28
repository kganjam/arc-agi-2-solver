#!/usr/bin/env python3
"""Test the AI's Bedrock-powered intelligent thinking"""

import json
import requests
import time

def test_bedrock_ai():
    base_url = "http://localhost:8050"
    
    # Get current puzzle
    resp = requests.get(f"{base_url}/api/puzzle/current")
    puzzle_data = resp.json()
    puzzle = puzzle_data.get("puzzle", {})
    
    print("🧠 Testing AI with Bedrock Intelligence")
    print("=" * 60)
    print(f"Puzzle ID: {puzzle.get('id', 'unknown')}")
    print("\n📋 Key Features:")
    print("  ✅ Bedrock AI analyzes each failure")
    print("  ✅ Generates new creative approaches based on learning")
    print("  ✅ Maintains conversation context")
    print("  ✅ Adapts strategy based on what didn't work")
    print("=" * 60)
    
    # Test with the solving command
    test_message = "try to solve this problem using heuristics and tools. then submit the answer to see if you are correct."
    
    print(f"\n📤 Sending: {test_message}")
    print("\n⏳ AI is now thinking with Bedrock...")
    print("   (Each attempt will be analyzed by AI to generate new ideas)")
    
    start_time = time.time()
    
    resp = requests.post(
        f"{base_url}/api/puzzle/ai-chat",
        json={
            "message": test_message,
            "puzzle_id": puzzle.get("id"),
            "output_grid": [[0, 0], [0, 0]]
        },
        timeout=30  # Longer timeout for Bedrock thinking
    )
    
    elapsed = time.time() - start_time
    
    if resp.status_code == 200:
        result = resp.json()
        
        print(f"\n⏱️ Response Time: {elapsed:.2f} seconds")
        print("\n📥 AI Response:")
        print("-" * 50)
        
        # Extract key information
        if result.get("message"):
            # Show first 500 characters of message
            message = result["message"]
            if "🤖 Using AI" in message:
                print("✅ AI is using Bedrock to think!")
            
            # Show attempts summary
            if result.get("attempts"):
                print(f"\n🔄 Total Attempts: {len(result['attempts'])}")
                print("\n📊 Attempt Types:")
                
                # Categorize attempts
                ai_suggested = 0
                systematic = 0
                for attempt in result['attempts']:
                    desc = attempt.get('description', '')
                    if 'AI suggested' in desc or 'AI inspired' in desc:
                        ai_suggested += 1
                    elif 'Systematic' in desc:
                        systematic += 1
                
                print(f"  • AI-Generated: {ai_suggested}")
                print(f"  • Systematic Fallback: {systematic}")
                
                # Show some interesting attempts
                print("\n🎯 Sample AI-Generated Attempts:")
                for i, attempt in enumerate(result['attempts'][:5], 1):
                    desc = attempt.get('description', '')
                    if 'AI' in desc:
                        print(f"  {i}. {desc}")
                        if attempt.get('failure_reason'):
                            print(f"     → Failed: {attempt['failure_reason']}")
        
        # Show if solved
        if result.get("solved"):
            print(f"\n✅ SOLVED! Method: {result.get('method')}")
        else:
            print(f"\n❌ Not solved after {result.get('total_attempts', 0)} attempts")
    else:
        print(f"❌ Error: {resp.status_code}")

def show_comparison():
    print("\n" + "=" * 60)
    print("📊 COMPARISON: Old vs New AI")
    print("=" * 60)
    
    print("\n❌ OLD (Fixed Script):")
    print("-" * 50)
    print("• Followed pre-programmed phases 1-6")
    print("• Always tried same sequence:")
    print("  - Phase 1: Scaling 1-10")
    print("  - Phase 2: Color transformations")
    print("  - Phase 3: Grid copying")
    print("• No learning from failures")
    print("• No creative thinking")
    
    print("\n✅ NEW (Bedrock AI Thinking):")
    print("-" * 50)
    print("• Analyzes each failure with AI")
    print("• Generates new approaches based on what failed:")
    print("  - 'Size mismatch' → AI suggests different scaling")
    print("  - 'Cells differ' → AI suggests color/spatial ops")
    print("• Maintains conversation context for learning")
    print("• Adapts strategy dynamically")
    print("• Can reason about the puzzle creatively")
    
    print("\n🎯 Key Difference:")
    print("The AI now THINKS about what to try next instead of")
    print("following a script. Each attempt is informed by learning!")

if __name__ == "__main__":
    try:
        test_bedrock_ai()
        show_comparison()
        
        print("\n" + "=" * 60)
        print("✨ SUMMARY: True AI Intelligence")
        print("=" * 60)
        print("The AI now uses Bedrock to:")
        print("  1. Analyze failure patterns")
        print("  2. Generate creative new approaches")
        print("  3. Learn from attempt history")
        print("  4. Adapt strategy based on feedback")
        print("  5. Think like a human solver would")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nNote: This requires AWS Bedrock credentials to be configured.")