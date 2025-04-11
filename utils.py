from datetime import datetime

def log_message(message):
    """Log a message to a file and update status"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open("logs/report_generation.log", "a") as f:
        f.write(log_entry)
    
    # Also print to console
    print(message) 