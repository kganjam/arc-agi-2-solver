#!/usr/bin/env python3
"""
ARC AGI Watchdog System
Ensures continuous operation and self-improvement
"""

import time
import subprocess
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

class ARCWatchdog:
    """Watchdog for continuous self-improvement"""
    
    def __init__(self):
        self.running = True
        self.solver_process = None
        self.log_dir = Path("logs/watchdog")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.log_dir / "status.json"
        self.restart_count = 0
        self.last_improvement = datetime.now()
        self.improvement_interval = timedelta(minutes=30)
        
    def log(self, message, level="INFO"):
        """Log watchdog events"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        # Write to log file
        log_file = self.log_dir / f"watchdog_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a') as f:
            f.write(log_entry + "\n")
            
    def update_status(self, status_data):
        """Update status file"""
        status_data['timestamp'] = datetime.now().isoformat()
        status_data['restart_count'] = self.restart_count
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
            
    def check_solver_health(self):
        """Check if solver is running and healthy"""
        if self.solver_process:
            # Check if process is alive
            if self.solver_process.poll() is not None:
                self.log("Solver process died", "ERROR")
                return False
                
            # Check CPU usage (shouldn't be stuck)
            try:
                process = psutil.Process(self.solver_process.pid)
                cpu_percent = process.cpu_percent(interval=1)
                
                if cpu_percent < 1:
                    self.log(f"Solver appears stuck (CPU: {cpu_percent}%)", "WARNING")
                    # Don't kill yet, might be waiting for Claude Code
                    
            except psutil.NoSuchProcess:
                self.log("Solver process not found", "ERROR")
                return False
                
        else:
            self.log("No solver process", "WARNING")
            return False
            
        return True
        
    def start_solver(self):
        """Start the solver process"""
        self.log("Starting solver process")
        
        try:
            # Start the integrated app with solver
            self.solver_process = subprocess.Popen(
                [sys.executable, "arc_integrated_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.log(f"Solver started with PID: {self.solver_process.pid}")
            self.update_status({
                'solver_pid': self.solver_process.pid,
                'status': 'running',
                'start_time': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.log(f"Failed to start solver: {e}", "ERROR")
            return False
            
    def trigger_improvement(self):
        """Trigger self-improvement cycle"""
        self.log("Triggering self-improvement cycle")
        
        # Read current performance
        try:
            results_file = Path("enhanced_solver_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                solve_rate = results.get('solved_count', 0) / max(1, results.get('total_puzzles', 1))
                
                if solve_rate < 1.0:
                    # Generate improvement prompt
                    prompt = f"""
                    Current solve rate: {solve_rate:.1%}
                    Puzzles: {results.get('solved_count')}/{results.get('total_puzzles')}
                    
                    Analyze failures and generate:
                    1. New heuristic for unsolved patterns
                    2. Optimization for slow-solving puzzles
                    
                    Save improvements to patterns/improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py
                    """
                    
                    # Call Claude Code
                    self.call_claude_code(prompt)
                    
                else:
                    self.log("Perfect solve rate achieved!", "SUCCESS")
                    
        except Exception as e:
            self.log(f"Error reading performance: {e}", "ERROR")
            
        self.last_improvement = datetime.now()
        
    def call_claude_code(self, prompt):
        """Call Claude Code for improvements"""
        self.log("Calling Claude Code for improvements")
        
        cmd = [
            "claude",
            "--allowedTools", "Bash,Read,WebSearch,Fetch",
            "--permission-mode", "acceptEdits",
            "--message", prompt
        ]
        
        try:
            # Log the conversation
            conversation_log = Path("logs/claude_conversations") / f"improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            conversation_log.parent.mkdir(parents=True, exist_ok=True)
            
            # In production, this would actually call Claude Code
            # For now, log the attempt
            with open(conversation_log, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt,
                    'status': 'simulated'
                }, f, indent=2)
                
            self.log(f"Claude Code conversation logged to: {conversation_log}")
            
        except Exception as e:
            self.log(f"Claude Code call failed: {e}", "ERROR")
            
    def monitor_loop(self):
        """Main monitoring loop"""
        self.log("Watchdog monitoring started")
        
        while self.running:
            try:
                # Check solver health
                if not self.check_solver_health():
                    self.log("Solver unhealthy, restarting", "WARNING")
                    self.restart_count += 1
                    
                    if self.solver_process:
                        self.solver_process.terminate()
                        time.sleep(2)
                        
                    self.start_solver()
                    
                # Check if improvement needed
                time_since_improvement = datetime.now() - self.last_improvement
                if time_since_improvement > self.improvement_interval:
                    self.trigger_improvement()
                    
                # Update status
                self.update_status({
                    'status': 'monitoring',
                    'solver_healthy': self.check_solver_health(),
                    'last_improvement': self.last_improvement.isoformat()
                })
                
                # Sleep before next check
                time.sleep(10)
                
            except KeyboardInterrupt:
                self.log("Received shutdown signal")
                self.running = False
                
            except Exception as e:
                self.log(f"Monitor error: {e}", "ERROR")
                time.sleep(5)
                
    def commit_improvements(self):
        """Periodically commit improvements to git"""
        self.log("Committing improvements to git")
        
        try:
            # Add all changes
            subprocess.run(["git", "add", "-A"], check=False)
            
            # Get status for commit message
            status = "Continuous improvement checkpoint"
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                    if 'solver_healthy' in status_data:
                        status = f"Watchdog: Solver {'healthy' if status_data['solver_healthy'] else 'restarted'}"
                        
            # Commit
            subprocess.run([
                "git", "commit", "-m", 
                f"{status}\nRestart count: {self.restart_count}\nTimestamp: {datetime.now().isoformat()}"
            ], check=False)
            
            # Push
            subprocess.run(["git", "push"], check=False)
            
            self.log("Changes committed and pushed")
            
        except Exception as e:
            self.log(f"Git commit failed: {e}", "WARNING")
            
    def run(self):
        """Main watchdog execution"""
        print("="*70)
        print("ARC AGI Watchdog System")
        print("Ensuring Continuous Self-Improvement")
        print("="*70)
        
        # Start solver
        if not self.start_solver():
            self.log("Failed to start solver, exiting", "ERROR")
            return
            
        # Start commit thread
        def commit_loop():
            while self.running:
                time.sleep(600)  # Commit every 10 minutes
                self.commit_improvements()
                
        commit_thread = threading.Thread(target=commit_loop, daemon=True)
        commit_thread.start()
        
        # Run monitoring loop
        try:
            self.monitor_loop()
        finally:
            self.log("Watchdog shutting down")
            
            if self.solver_process:
                self.solver_process.terminate()
                
            # Final commit
            self.commit_improvements()
            
            self.log("Watchdog stopped")


def main():
    """Entry point"""
    watchdog = ARCWatchdog()
    
    # Check for daemon mode
    if "--daemon" in sys.argv:
        # Run as background daemon
        import daemon
        with daemon.DaemonContext():
            watchdog.run()
    else:
        # Run in foreground
        watchdog.run()


if __name__ == "__main__":
    main()