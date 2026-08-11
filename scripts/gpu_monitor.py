#!/usr/bin/env python3
"""
Monitor GPU usage during prediction execution.
"""
import subprocess
import time
import sys
import signal
import threading


class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.gpu_utilization = []
        self.gpu_memory = []
        self.running = False
        self.thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Query GPU utilization and memory usage
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) == 2:
                                util = float(parts[0].strip())
                                mem = float(parts[1].strip())
                                self.gpu_utilization.append(util)
                                self.gpu_memory.append(mem)
            except Exception:
                pass
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if not self.gpu_utilization:
            return 0.0, 0.0, 0.0, 0.0
        
        avg_util = sum(self.gpu_utilization) / len(self.gpu_utilization)
        max_util = max(self.gpu_utilization)
        avg_mem = sum(self.gpu_memory) / len(self.gpu_memory)
        max_mem = max(self.gpu_memory)
        
        return avg_util, max_util, avg_mem, max_mem


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gpu_monitor.py <command> [args...]")
        sys.exit(1)
    
    # Start monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    
    # Run the command
    command = sys.argv[1:]
    try:
        result = subprocess.run(command)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        exit_code = 1
    
    # Stop monitoring and get stats
    avg_util, max_util, avg_mem, max_mem = monitor.stop()
    
    # Output stats in CSV format: avg_gpu_util,max_gpu_util,avg_gpu_mem,max_gpu_mem
    print(f"{avg_util:.2f},{max_util:.2f},{avg_mem:.2f},{max_mem:.2f}")
    
    sys.exit(exit_code)
