#!/usr/bin/env python3
"""
Simple test script to verify the ARC AGI application works
"""

import sys
import time
import subprocess
import urllib.request
import json
from pathlib import Path

class ApplicationTester:
    def __init__(self):
        self.backend_process = None
        self.test_results = []
        
    def print_header(self, text):
        print(f"\n{'='*60}")
        print(f"  {text}")
        print('='*60)
        
    def test_dependencies(self):
        """Check if required dependencies are installed"""
        self.print_header("Testing Dependencies")
        
        try:
            import fastapi
            print("✓ FastAPI installed")
            self.test_results.append(("FastAPI", True))
        except ImportError:
            print("✗ FastAPI not installed")
            self.test_results.append(("FastAPI", False))
            
        try:
            import uvicorn
            print("✓ Uvicorn installed")
            self.test_results.append(("Uvicorn", True))
        except ImportError:
            print("✗ Uvicorn not installed")
            self.test_results.append(("Uvicorn", False))
            
        try:
            import pydantic
            print("✓ Pydantic installed")
            self.test_results.append(("Pydantic", True))
        except ImportError:
            print("✗ Pydantic not installed")
            self.test_results.append(("Pydantic", False))
            
        return all(result[1] for result in self.test_results)
        
    def test_backend_startup(self):
        """Test if backend starts correctly"""
        self.print_header("Testing Backend Startup")
        
        try:
            # Start backend in test mode
            print("Starting backend server...")
            self.backend_process = subprocess.Popen(
                [sys.executable, "test_simple.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            time.sleep(3)
            
            # Check if process is still running
            if self.backend_process.poll() is None:
                print("✓ Backend process started")
                self.test_results.append(("Backend startup", True))
                return True
            else:
                print("✗ Backend process failed to start")
                stdout, stderr = self.backend_process.communicate()
                print(f"Error: {stderr}")
                self.test_results.append(("Backend startup", False))
                return False
                
        except Exception as e:
            print(f"✗ Failed to start backend: {e}")
            self.test_results.append(("Backend startup", False))
            return False
            
    def test_api_endpoints(self):
        """Test API endpoints"""
        self.print_header("Testing API Endpoints")
        
        base_url = "http://localhost:8050"
        
        # Test root endpoint
        try:
            req = urllib.request.Request(f"{base_url}/")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print("✓ Root endpoint responding")
                    self.test_results.append(("Root endpoint", True))
                else:
                    print(f"✗ Root endpoint returned {response.status}")
                    self.test_results.append(("Root endpoint", False))
        except Exception as e:
            print(f"✗ Root endpoint failed: {e}")
            self.test_results.append(("Root endpoint", False))
            
        # Test API test endpoint
        try:
            req = urllib.request.Request(f"{base_url}/api/test")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get('status') == 'working':
                        print("✓ API test endpoint working")
                        self.test_results.append(("API test", True))
                    else:
                        print("✗ API test endpoint returned unexpected data")
                        self.test_results.append(("API test", False))
                else:
                    print(f"✗ API test endpoint returned {response.status}")
                    self.test_results.append(("API test", False))
        except Exception as e:
            print(f"✗ API test endpoint failed: {e}")
            self.test_results.append(("API test", False))
            
    def test_frontend_content(self):
        """Test if frontend content is served correctly"""
        self.print_header("Testing Frontend Content")
        
        try:
            req = urllib.request.Request("http://localhost:8050/")
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8')
                
                # Check for key elements
                checks = [
                    ("ARC AGI" in content, "Page title"),
                    ("puzzle" in content.lower(), "Puzzle viewer"),
                    ("chat" in content.lower(), "Chat interface"),
                    ("challenge" in content.lower(), "Challenge selector")
                ]
                
                for check, name in checks:
                    if check:
                        print(f"✓ {name} found")
                        self.test_results.append((name, True))
                    else:
                        print(f"✗ {name} not found")
                        self.test_results.append((name, False))
                        
        except Exception as e:
            print(f"✗ Failed to test frontend: {e}")
            self.test_results.append(("Frontend content", False))
            
    def cleanup(self):
        """Clean up test processes"""
        if self.backend_process:
            print("\nStopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        print(f"\nResults: {passed}/{total} tests passed")
        print("\nDetailed Results:")
        for name, result in self.test_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")
            
        if passed == total:
            print("\n🎉 All tests passed! The application is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
            return False
            
    def run_tests(self):
        """Run all tests"""
        print("="*60)
        print("  ARC AGI Application Test Suite")
        print("="*60)
        
        try:
            # Check dependencies
            if not self.test_dependencies():
                print("\n⚠️  Missing dependencies. Installing...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                print("Dependencies installed. Please run the test again.")
                return False
                
            # Start backend
            if not self.test_backend_startup():
                return False
                
            # Test endpoints
            self.test_api_endpoints()
            
            # Test frontend
            self.test_frontend_content()
            
            # Print summary
            return self.print_summary()
            
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    tester = ApplicationTester()
    success = tester.run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()