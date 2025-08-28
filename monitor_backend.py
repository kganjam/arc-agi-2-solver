#!/usr/bin/env python3
"""
Monitor backend stability
"""

import time
import psutil
import requests
import sys

def check_backend_health():
    """Check if backend is healthy"""
    try:
        response = requests.get("http://localhost:8050/api/status", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_backend_process():
    """Find the backend process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('start_backend.py' in arg or 'arc_integrated_app.py' in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor():
    """Monitor backend stability"""
    print("Monitoring backend stability...")
    print("-" * 50)
    
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds
    max_checks = 20  # Monitor for 100 seconds
    
    proc = get_backend_process()
    if not proc:
        print("❌ Backend process not found!")
        return False
    
    initial_pid = proc.pid
    print(f"✓ Backend process found: PID {initial_pid}")
    
    restarts = 0
    checks = 0
    
    while checks < max_checks:
        checks += 1
        
        # Check if process is still running
        current_proc = get_backend_process()
        if not current_proc:
            print(f"❌ Backend process died after {time.time() - start_time:.1f} seconds!")
            return False
        
        if current_proc.pid != initial_pid:
            restarts += 1
            print(f"⚠️ Backend restarted! New PID: {current_proc.pid} (restart #{restarts})")
            initial_pid = current_proc.pid
        
        # Check health
        if check_backend_health():
            print(f"✓ Check {checks}/{max_checks}: Backend healthy (PID {current_proc.pid})")
        else:
            print(f"❌ Check {checks}/{max_checks}: Backend not responding!")
        
        # Check memory usage
        try:
            memory_mb = current_proc.memory_info().rss / 1024 / 1024
            cpu_percent = current_proc.cpu_percent(interval=1)
            print(f"  Memory: {memory_mb:.1f} MB, CPU: {cpu_percent:.1f}%")
            
            if memory_mb > 500:
                print(f"  ⚠️ High memory usage: {memory_mb:.1f} MB")
        except:
            pass
        
        time.sleep(check_interval)
    
    elapsed = time.time() - start_time
    print("-" * 50)
    
    if restarts > 0:
        print(f"❌ Backend restarted {restarts} times in {elapsed:.1f} seconds")
        return False
    else:
        print(f"✅ Backend stable for {elapsed:.1f} seconds with no restarts!")
        return True

if __name__ == "__main__":
    success = monitor()
    sys.exit(0 if success else 1)