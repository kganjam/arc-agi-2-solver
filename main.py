"""
ARC AGI 2 Challenge Solving System - Main Orchestrator
Starts backend, frontend, and launches browser with comprehensive logging
"""

import subprocess
import os
import sys
import time
import threading
import logging
import platform
import urllib.request
from datetime import datetime
from pathlib import Path

class ARCOrchestrator:
    """Main orchestrator for ARC puzzle solving system"""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging for orchestrator and processes"""
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup main orchestrator logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_timestamp = timestamp
        log_file = self.logs_dir / f"orchestrator_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("ARCOrchestrator")
        self.logger.info("ARC AGI System Orchestrator initialized")
        self.logger.info(f"Logging to: {log_file}")
        
        # Setup log files for backend and frontend
        self.backend_log_file = self.logs_dir / f"backend_{timestamp}.log"
        self.frontend_log_file = self.logs_dir / f"frontend_{timestamp}.log"
        
    def is_wsl(self):
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def get_edge_command(self):
        """Get appropriate Edge command for platform"""
        system = platform.system().lower()
        
        if system == "windows":
            # Try different Edge paths on Windows
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            ]
            for path in edge_paths:
                if os.path.exists(path):
                    return [path, "--new-window"]
            return ["msedge", "--new-window"]  # Fallback
        
        elif system == "linux" and self.is_wsl():
            # WSL - use Windows Edge via cmd.exe
            return ["cmd.exe", "/c", "start", "msedge", "--new-window"]
        
        else:
            # Linux native - try different Edge variants
            edge_commands = ["microsoft-edge", "microsoft-edge-stable", "edge"]
            for cmd in edge_commands:
                try:
                    subprocess.run(["which", cmd], check=True, capture_output=True)
                    return [cmd, "--new-window"]
                except:
                    continue
            return ["xdg-open"]  # Fallback to default browser
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        self.logger.info("Starting FastAPI backend...")
        
        try:
            with open(self.backend_log_file, 'w') as log_file:
                self.backend_process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8050"],
                    cwd="backend",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            self.logger.info(f"Backend started with PID: {self.backend_process.pid}")
            self.logger.info(f"Backend logs: {self.backend_log_file}")
            
            # Wait for backend to start
            time.sleep(3)
            
            # Test backend
            if self.test_backend():
                self.logger.info("Backend is responding")
                return True
            else:
                self.logger.error("Backend failed to start properly")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False
    
    def test_backend(self):
        """Test if backend is responding"""
        try:
            import urllib.request
            import urllib.error
            
            # Test backend health
            req = urllib.request.Request("http://localhost:8050/api/test")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except:
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        self.logger.info("Starting frontend server...")
        
        try:
            # Use the simple test server instead of complex Vue setup
            with open(self.frontend_log_file, 'w') as log_file:
                self.frontend_process = subprocess.Popen(
                    [sys.executable, "test_simple.py"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
            
            self.logger.info(f"Frontend started with PID: {self.frontend_process.pid}")
            self.logger.info(f"Frontend logs: {self.frontend_log_file}")
            
            # Wait for frontend to start
            time.sleep(2)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False
    
    def launch_browser(self):
        """Launch Microsoft Edge with the application URL"""
        url = "http://localhost:8050"
        self.logger.info(f"Launching browser: {url}")
        
        try:
            edge_cmd = self.get_edge_command()
            
            if "xdg-open" in edge_cmd:
                # Linux fallback - just open URL
                subprocess.Popen(edge_cmd + [url])
            else:
                # Windows/WSL - use Edge with new window
                subprocess.Popen(edge_cmd + [url])
            
            self.logger.info("Browser launched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch browser: {e}")
            self.logger.info(f"Please manually open: {url}")
            return False
    
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        while True:
            try:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    self.logger.error("Backend process died, restarting...")
                    self.start_backend()
                
                # Check frontend
                if self.frontend_process and self.frontend_process.poll() is not None:
                    self.logger.error("Frontend process died, restarting...")
                    self.start_frontend()
                
                time.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(5)
    
    def cleanup(self):
        """Clean up processes on exit"""
        self.logger.info("Shutting down ARC AGI System...")
        
        if self.backend_process:
            self.logger.info("Terminating backend process...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.logger.info("Terminating frontend process...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        self.logger.info("Cleanup complete")
    
    def run(self):
        """Main execution method"""
        try:
            # Start backend
            if not self.start_backend():
                self.logger.error("Failed to start backend, exiting")
                return False
            
            # Start frontend
            if not self.start_frontend():
                self.logger.error("Failed to start frontend, exiting")
                return False
            
            # Launch browser
            time.sleep(2)  # Give servers time to fully start
            self.launch_browser()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            self.logger.info("ARC AGI System is running!")
            self.logger.info("Application available at: http://localhost:8050")
            self.logger.info("Press Ctrl+C to stop")
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            return False
        
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    print("="*60)
    print("ARC AGI 2 Challenge Solving System")
    print("FastAPI + Vue.js Web Application")
    print("="*60)
    
    orchestrator = ARCOrchestrator()
    success = orchestrator.run()
    
    if success:
        print("\nSystem started successfully!")
    else:
        print("\nSystem failed to start properly.")
        sys.exit(1)

if __name__ == "__main__":
    main()
