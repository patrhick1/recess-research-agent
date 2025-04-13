from datetime import datetime
import sys
import traceback
import os

def log_message(message, log_file=None):
    """Log a message to a file and update status"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Always write to global log file
        global_log_path = "logs/report_generation.log"
        with open(global_log_path, "a", encoding='utf-8') as f:
            f.write(log_entry)
            f.flush()
        
        # Use session-specific log file if provided
        if log_file:
            try:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(log_entry)
                    f.flush()
            except Exception as f_err:
                print(f"ERROR writing to specific log file {log_file}: {str(f_err)}", file=sys.stderr, flush=True)
        
        # Also print to console
        print(message, flush=True)
    except Exception as e:
        print(f"ERROR LOGGING: {str(e)}", file=sys.stderr, flush=True)
        print(f"Attempted to log: {message}", file=sys.stderr, flush=True)
        traceback.print_exc() 