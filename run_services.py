import subprocess
import sys
import os
import time
import signal
import atexit

# Flag to indicate we're running both services
os.environ["RUNNING_BOTH_SERVICES"] = "1"

def start_services():
    """Start both Flask and FastAPI services in separate processes"""
    print("Starting Flask web application on port 5000...")
    flask_process = subprocess.Popen([sys.executable, "app.py"])
    
    print("Starting FastAPI service on port 8000...")
    fastapi_process = subprocess.Popen([sys.executable, "api.py"])
    
    # Function to terminate processes on exit
    def cleanup():
        print("\nShutting down services...")
        if flask_process.poll() is None:
            print("Terminating Flask service...")
            flask_process.terminate()
            flask_process.wait(timeout=5)
        
        if fastapi_process.poll() is None:
            print("Terminating FastAPI service...")
            fastapi_process.terminate()
            fastapi_process.wait(timeout=5)
        
        print("All services terminated.")
    
    # Register the cleanup function
    atexit.register(cleanup)
    
    # Handle keyboard interrupt (Ctrl+C)
    def signal_handler(sig, frame):
        print("\nReceived termination signal...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\nServices running! Press Ctrl+C to stop.")
    print(f"- Web Interface: http://localhost:5000")
    print(f"- API Documentation: http://localhost:8000/docs")
    
    # Keep the main process running
    try:
        while True:
            time.sleep(1)
            # Check if either process has terminated
            if flask_process.poll() is not None:
                print("Flask process has terminated unexpectedly!")
                sys.exit(1)
            if fastapi_process.poll() is not None:
                print("FastAPI process has terminated unexpectedly!")
                sys.exit(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    start_services() 