#!/usr/bin/env python3
"""
ARC AGI Application Launcher
Activates virtual environment, starts backend/frontend, and launches Edge browser
Works on Windows, Linux, and WSL
"""

import subprocess
import sys
import os
import time
import signal
import platform
import threading
import logging
import urllib.request
from datetime import datetime
from pathlib import Path

class ARCLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.base_dir = Path(__file__).parent.absolute()
        self.venv_dir = self.base_dir / "venv"
        self.running = True
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the launcher"""
        # Create logs directory
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"launcher_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("ARCLauncher")
        self.logger.info("ARC AGI Launcher initialized")
        
        # Setup log files for backend
        self.backend_log_file = self.logs_dir / f"backend_{timestamp}.log"
        
    def print_header(self, text):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print('='*60)
        
    def is_wsl(self):
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
            
    def get_python_executable(self):
        """Get the correct Python executable from venv"""
        if platform.system() == "Windows":
            python_exe = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_dir / "bin" / "python"
            
        # If venv doesn't exist, use current Python
        if not python_exe.exists():
            return sys.executable
            
        return str(python_exe)
        
    def activate_venv(self):
        """Ensure virtual environment is active"""
        self.print_header("Virtual Environment")
        
        # Check if venv exists
        if not self.venv_dir.exists():
            print(f"✗ Virtual environment not found at {self.venv_dir}")
            print("  Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            print("✓ Virtual environment created")
            
            # Install requirements
            print("\n📦 Installing dependencies in venv...")
            python_exe = self.get_python_executable()
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✓ Dependencies installed")
        else:
            print(f"✓ Virtual environment found at {self.venv_dir}")
            
        # Return the Python executable to use
        return self.get_python_executable()
        
    def test_backend(self):
        """Test if backend is responding"""
        try:
            req = urllib.request.Request("http://localhost:8050/api/test")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except:
            return False
        
    def start_backend(self, python_exe):
        """Start the FastAPI backend"""
        self.print_header("Starting Backend")
        self.logger.info("Starting backend server...")
        
        try:
            # Use the enhanced ARC app as backend
            print("Starting backend server on port 8050...")
            
            # Write to log file
            with open(self.backend_log_file, 'w') as log_file:
                self.backend_process = subprocess.Popen(
                    [python_exe, "arc_app_enhanced.py"],
                    cwd=str(self.base_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            self.logger.info(f"Backend started with PID: {self.backend_process.pid}")
            self.logger.info(f"Backend logs: {self.backend_log_file}")
            
            # Wait for backend to start
            time.sleep(3)
            
            # Check if process is running and test endpoint
            if self.backend_process.poll() is None:
                if self.test_backend():
                    print("✓ Backend started successfully")
                    print("  URL: http://localhost:8050")
                    self.logger.info("Backend is responding")
                    return True
                else:
                    print("✗ Backend started but not responding")
                    self.logger.error("Backend not responding to health check")
                    return False
            else:
                print("✗ Backend failed to start")
                self.logger.error("Backend process died")
                return False
                
        except Exception as e:
            print(f"✗ Failed to start backend: {e}")
            self.logger.error(f"Failed to start backend: {e}")
            return False
            
    def start_frontend(self, python_exe):
        """Start the frontend (if separate from backend)"""
        # In this implementation, frontend is served by the backend
        # This method is kept for future expansion
        print("✓ Frontend served by backend at http://localhost:8050")
        return True
        
    def launch_edge(self):
        """Launch Microsoft Edge with --new-window"""
        self.print_header("Launching Browser")
        
        url = "http://localhost:8050"
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows - try different Edge paths
                edge_paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
                ]
                edge_cmd = None
                for path in edge_paths:
                    if os.path.exists(path):
                        edge_cmd = [path, "--new-window", url]
                        break
                        
                if not edge_cmd:
                    # Try using just 'msedge' command
                    edge_cmd = ["msedge", "--new-window", url]
                    
            elif self.is_wsl():
                # WSL - use Windows Edge via cmd.exe
                print("  Detected WSL environment")
                edge_cmd = ["cmd.exe", "/c", "start", "msedge", "--new-window", url]
                
            else:
                # Linux native - try different Edge variants
                edge_commands = ["microsoft-edge"] # "microsoft-edge-stable", "microsoft-edge-dev"
                edge_cmd = None
                for cmd in edge_commands:
                    try:
                        result = subprocess.run(["which", cmd], capture_output=True, text=True)
                        if result.returncode == 0:
                            edge_cmd = [cmd, "--new-window", url]
                            break
                    except:
                        continue
                        
                if not edge_cmd:
                    # Fallback to xdg-open
                    print("  Edge not found, using default browser")
                    edge_cmd = ["xdg-open", url]
                    
            # Launch the browser
            print(f"  Launching: {' '.join(edge_cmd)}")
            subprocess.Popen(edge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓ Browser launched")
            return True
            
        except Exception as e:
            print(f"✗ Failed to launch browser: {e}")
            print(f"  Please manually open: {url}")
            return False
            
    def monitor_processes(self):
        """Monitor backend process and restart if needed"""
        while self.running:
            try:
                if self.backend_process and self.backend_process.poll() is not None:
                    print("\n⚠️  Backend process died, restarting...")
                    python_exe = self.get_python_executable()
                    self.start_backend(python_exe)
                    
                time.sleep(5)
            except:
                break
                
    def cleanup(self):
        """Clean up processes on exit"""
        print("\n\nShutting down...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                
        print("✓ Cleanup complete")
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        self.cleanup()
        sys.exit(0)
        
    def run(self):
        """Main execution"""
        print("="*60)
        print("  ARC AGI Challenge Solver")
        print("  Application Launcher")
        print("="*60)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Activate virtual environment
            python_exe = self.activate_venv()
            
            # Start backend
            if not self.start_backend(python_exe):
                print("\n✗ Failed to start application")
                return False
                
            # Start frontend (served by backend in this case)
            self.start_frontend(python_exe)
            
            # Launch browser after a short delay
            time.sleep(1)
            self.launch_edge()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            # Success message
            self.print_header("Application Running")
            print("  URL: http://localhost:8050")
            print("  Press Ctrl+C to stop")
            print()
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return False
        finally:
            self.cleanup()
            
        return True

def main():
    """Entry point"""
    launcher = ARCLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main()