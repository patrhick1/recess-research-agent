from datetime import datetime

def log_message(message, log_file=None):
    """Log a message to a file and update status"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    # Use session-specific log file if provided, otherwise use default
    log_path = log_file if log_file else "logs/report_generation.log"
    
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(log_entry)
    
    # Also print to console
    print(message) 