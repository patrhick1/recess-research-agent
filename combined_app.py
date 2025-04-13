import os
import asyncio
import threading
import json
import time
import uuid
from datetime import datetime, timedelta
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
from utils import log_message
from starlette.middleware.wsgi import WSGIMiddleware
from flask import Flask, render_template, request, jsonify, send_file, session
import sys
import glob

from working_agent import (
    generate_report_plan, 
    parallelize_section_writing, 
    section_builder_subagent, 
    format_completed_sections, 
    parallelize_final_section_writing, 
    write_final_sections, 
    compile_final_report,
    StateGraph, 
    START, 
    END, 
    ReportState, 
    ReportStateInput,
    ReportStateOutput,
    set_log_file  # Import the set_log_file function
)
import working_agent
# Create Flask app
flask_app = Flask(__name__)
flask_app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Create FastAPI app
fastapi_app = FastAPI(
    title="Deep Researcher API",
    description="API for generating market analysis reports for companies",
    version="1.0.0"
)

# Global variables to track report generation
REPORTS_DIR = "reports"
user_status = {}  # Dictionary to store status for each session/request ID
REPORT_RETENTION_DAYS = 7  # Number of days to keep reports before deletion

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Create log file directory
os.makedirs("logs", exist_ok=True)

# FastAPI models
class ReportRequest(BaseModel):
    """Model for report generation request"""
    company_name: str
    time_period: str

class ReportResponse(BaseModel):
    """Model for report generation response"""
    request_id: str
    status: str
    message: str

class ReportStatus(BaseModel):
    """Model for report status response"""
    request_id: str
    is_generating: bool
    progress: float
    message: str
    company_name: str
    time_period: str
    report_file: Optional[str] = None
    error: Optional[str] = None

# Shared functionality
async def initialize_agent() -> StateGraph:
    """Initialize and compile the agent state graph with proper parallelization"""
    # Create main report builder graph
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)

    # Add nodes
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    # Connect nodes with proper conditional edges for parallelization
    builder.add_edge(START, "generate_report_plan")
    builder.add_conditional_edges("generate_report_plan", 
                                 parallelize_section_writing,
                                 ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", 
                                 parallelize_final_section_writing,
                                 ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    return builder.compile()

async def generate_report_async(company_name, time_period, session_id):
    """Generate report asynchronously using the working_agent.py code"""
    global user_status
    
    try:
        user_status[session_id]["is_generating"] = True
        user_status[session_id]["progress"] = 0.05
        user_status[session_id]["message"] = "Starting report generation..."
        
        # Create a session-specific log file
        log_file = f"logs/report_generation_{session_id}.log"
        
        # Ensure log file exists and is writable - Start with a clean file
        try:
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting report generation for {company_name} ({time_period})\n")
        except Exception as e:
            print(f"Error creating log file {log_file}: {e}")
            # Use a fallback log file
            log_file = f"logs/report_{session_id}.log"
            with open(log_file, "w", encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting report generation for {company_name} ({time_period})\n")
        
        log_message(f"Starting report generation for {company_name} ({time_period})", log_file)
        
        # Set up the log file in working_agent
        try:
            # Set the log file directly using the imported function
            set_log_file(log_file)
            log_message("Set log file in working_agent module", log_file)
            
            # Make sure the global working_agent module also has the log file set
            if 'working_agent' in sys.modules:
                sys.modules['working_agent'].CURRENT_LOG_FILE = log_file
                log_message("Set log file in working_agent module (sys.modules)", log_file)
        except Exception as e:
            log_message(f"Error setting up log file in working_agent: {str(e)}", log_file)
            import traceback
            log_message(traceback.format_exc(), log_file)
        
        # Initialize and compile the agent
        log_message("Initializing agent", log_file)
        reporter_agent = await initialize_agent()
        
        user_status[session_id]["progress"] = 0.1
        log_message("Agent compiled successfully", log_file)
        
        # Initial state
        topic = f"{company_name} {time_period} performance"
        state = {
            "company_name": company_name,
            "time_period": time_period,
            "topic": topic,
            "sections": [],
            "completed_sections": [],
            "report_sections_from_research": "",
            "final_report": "",
            "filename": "",
            "config": {
                "research_type": "Business",
                "target_audience": "Marketing Executives",
                "writing_style": "Professional"
            },
            "log_file": log_file,  # Add log file to state
            "session_id": session_id  # Add session ID to state
        }
        
        # Create progress monitoring task
        def update_progress():
            """Update progress based on log messages"""
            progress_steps = {
                "--- Reading Report Structure from File ---": 0.15,
                "--- Generating Search Queries for Section:": 0.25,
                "--- Searching Web for Queries Completed ---": 0.35,
                "--- Collecting Evidence for Section:": 0.45,
                "--- Mapping Evidence to Predefined Subsections for Section:": 0.55,
                "--- Writing Paragraphs for Section:": 0.65,
                "--- Synthesizing Subsections for Section:": 0.75,
                "--- Synthesizing Section:": 0.85,
                "--- Formatting Completed Sections ---": 0.9,
                "--- Writing Final Section:": 0.95,
                "--- Compiling Final Report ---": 0.98,
            }
            
            last_position = 0
            while (session_id in user_status and 
                   user_status[session_id]["is_generating"] and 
                   user_status[session_id]["progress"] < 1.0):
                try:
                    # Check if log file exists
                    if not os.path.exists(log_file):
                        print(f"Log file {log_file} does not exist")
                        time.sleep(1)
                        continue
                    
                    # Check if thread should stop (session cancelled)
                    if not user_status[session_id]["is_generating"]:
                        print(f"Report generation for session {session_id} has been cancelled")
                        break
                        
                    # Read only new log entries since last check
                    with open(log_file, "r") as f:
                        f.seek(last_position)
                        new_log_content = f.read()
                        last_position = f.tell()
                    
                    if new_log_content:
                        # Check each progress step in order
                        for step, progress_value in progress_steps.items():
                            if step in new_log_content and user_status[session_id]["progress"] < progress_value:
                                user_status[session_id]["progress"] = progress_value
                                user_status[session_id]["message"] = f"Step: {step.replace('---', '').strip()}"
                                print(f"Progress updated to {progress_value}: {step} for session {session_id}")
                
                except Exception as e:
                    print(f"Error reading log file for session {session_id}: {e}")
                    if "socket" in str(e).lower():
                        print("Application is reloading - stopping progress monitoring")
                        break
                
                time.sleep(1)
        
        # Start progress monitoring in a thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Run the report generation
        try:
            log_message("Starting agent execution", log_file)
            log_message(f"Initial state: {json.dumps({k: v for k, v in state.items() if k != 'log_file'}, indent=2)}", log_file)
            
            try:
                # Regenerate a clean filename for this specific company and time period
                clean_company = company_name.replace(" ", "_")
                clean_period = time_period.replace(" ", "_")
                expected_filename = f"{clean_company}_{clean_period}_market_analysis.md"
                
                # Check if we need to cancel before proceeding with agent execution
                if not user_status[session_id]["is_generating"]:
                    log_message("Report generation was cancelled before agent execution", log_file)
                    user_status[session_id]["message"] = "Report generation cancelled"
                    return False
                
                result = await reporter_agent.ainvoke(state)
                log_message(f"Agent execution completed successfully", log_file)
            except Exception as agent_exec_error:
                import traceback
                error_trace = traceback.format_exc()
                log_message(f"CRITICAL ERROR during agent.ainvoke: {str(agent_exec_error)}", log_file)
                log_message(f"Error trace: {error_trace}", log_file)
                user_status[session_id]["error"] = f"Agent execution error: {str(agent_exec_error)}"
                user_status[session_id]["progress"] = 1.0
                user_status[session_id]["message"] = "Report generation failed - execution error"
                return False
            
            # Process result
            report_filename = result.get("filename", "")
            if not report_filename:
                # If filename not in result, construct it using the same pattern as in working_agent.py
                report_filename = expected_filename
                log_message(f"Filename not found in result, using generated filename: {report_filename}", log_file)
            
            # Check if there's a mismatch or if the file has an incorrect company name
            if clean_company.lower() not in report_filename.lower():
                log_message(f"Warning: Generated filename ({report_filename}) doesn't match expected company name. Renaming to {expected_filename}", log_file)
                
                # If file exists with the wrong name, rename it
                if os.path.exists(report_filename):
                    try:
                        os.rename(report_filename, expected_filename)
                        report_filename = expected_filename
                        log_message(f"Successfully renamed file to {expected_filename}", log_file)
                    except Exception as rename_error:
                        log_message(f"Error renaming file: {str(rename_error)}", log_file)
                else:
                    # Just use the expected filename if original doesn't exist
                    report_filename = expected_filename

            user_status[session_id]["report_file"] = report_filename
            log_message(f"Report generated successfully: {report_filename}", log_file)
            
            # Copy the report to the reports directory with a timestamp and session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_copy = f"{REPORTS_DIR}/{timestamp}_{session_id}_{report_filename}"
            log_message(f"Copying report to: {report_copy}", log_file)
            
            # Check if file exists before copying
            if os.path.exists(report_filename):
                with open(report_filename, "r", encoding='utf-8') as src, open(report_copy, "w", encoding='utf-8') as dst:
                    content = src.read()
                    dst.write(content)
                    log_message(f"Report copied successfully, {len(content)} characters", log_file)
                user_status[session_id]["report_file"] = report_copy
                user_status[session_id]["progress"] = 1.0
                user_status[session_id]["message"] = "Report generation complete"
            else:
                error_msg = f"Report file not found: {report_filename}"
                log_message(error_msg, log_file)
                log_message(f"Current directory contents: {os.listdir('.')}", log_file)
                user_status[session_id]["error"] = error_msg
                user_status[session_id]["progress"] = 1.0
                user_status[session_id]["message"] = "Report generation failed - file not found"
            
            return True
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"Error during agent execution: {str(e)}"
            log_message(error_msg, log_file)
            log_message(f"Error trace: {error_trace}", log_file)
            user_status[session_id]["error"] = error_msg
            user_status[session_id]["progress"] = 1.0
            user_status[session_id]["message"] = "Report generation failed - execution error"
            return False
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"Error setting up report generation: {str(e)}"
        log_message(error_msg, log_file if 'log_file' in locals() else None)
        log_message(f"Error trace: {error_trace}", log_file if 'log_file' in locals() else None)
        user_status[session_id]["error"] = error_msg
        user_status[session_id]["progress"] = 1.0
        user_status[session_id]["message"] = "Report generation failed - setup error"
        return False
    finally:
        if session_id in user_status:
            user_status[session_id]["is_generating"] = False

def start_report_generation(company_name, time_period, session_id):
    """Start the report generation process in a background thread"""
    global user_status
    
    # Initialize session status
    user_status[session_id] = {
        "is_generating": True,
        "progress": 0,
        "message": "Initializing...",
        "report_file": None,
        "error": None,
        "company_name": company_name,
        "time_period": time_period
    }
    
    # Create a session-specific log file
    log_file = f"logs/report_generation_{session_id}.log"
    
    # Clear the log file before starting a new report
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("")
    except Exception as e:
        print(f"Error clearing log file for session {session_id}: {e}")
    
    # Define a wrapper function to run the async function
    def run_async_report():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_report_async(company_name, time_period, session_id))
        except Exception as e:
            print(f"Error in report generation thread for session {session_id}: {e}")
            if session_id in user_status:
                user_status[session_id]["error"] = str(e)
                user_status[session_id]["is_generating"] = False
        finally:
            loop.close()
    
    # Start generation in a thread
    thread = threading.Thread(target=run_async_report)
    thread.daemon = True
    thread.start()
    
    return True

# Flask routes
@flask_app.route('/')
def index():
    """Render the main page"""
    # Create a session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@flask_app.route('/generate', methods=['POST'])
def generate():
    """Start generating a report based on form input"""
    company_name = request.form.get('company_name', '')
    time_period = request.form.get('time_period', '')
    
    if not company_name or not time_period:
        return jsonify({
            "success": False,
            "message": "Please provide both company name and time period",
            "error_code": "MISSING_PARAMETERS"
        }), 400
    
    # Get session_id from form data first, then query param, then Flask session
    session_id = request.form.get('session_id') or request.args.get('session_id')
    
    if not session_id:
        # If no session ID provided, create a new one
        session_id = str(uuid.uuid4())
        # Store in Flask session as fallback
        session['session_id'] = session_id
    
    # Start report generation
    success = start_report_generation(company_name, time_period, session_id)
    
    return jsonify({
        "success": success,
        "message": "Report generation started",
        "session_id": session_id
    })

@flask_app.route('/status')
def get_status():
    """Get the current status of report generation for the session"""
    # Get session ID from query parameter first, then fallback to Flask session
    session_id = request.args.get('session_id')
    
    if not session_id:
        # Fallback to Flask session, but return a warning
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({
                "success": False,
                "message": "Missing session ID parameter",
                "error_code": "MISSING_SESSION_ID",
                "is_generating": False,
                "progress": 0,
                "report_file": None,
                "error": "Session ID is required to check status"
            }), 400
        
        # Return a warning that using Flask session is deprecated
        return jsonify({
            "success": True,
            "warning": "Using Flask session ID is deprecated. Please include session_id parameter explicitly.",
            "is_generating": False,
            "progress": 0,
            "message": "Using deprecated session mechanism",
            "report_file": None,
            "error": None,
            "company_name": "",
            "time_period": "",
            "session_id": session_id
        })
    
    if session_id not in user_status:
        # Check if there are any reports for this session ID in the reports directory
        session_reports = glob.glob(os.path.join(REPORTS_DIR, f"*_{session_id}_*"))
        
        if session_reports:
            # Found reports for this session
            latest_report = max(session_reports, key=os.path.getmtime)
            report_filename = os.path.basename(latest_report)
            
            # Extract information if possible
            parts = report_filename.split('_')
            company_name = ""
            time_period = ""
            
            if len(parts) > 3:
                # Try to get company name and time period from filename
                company_time = '_'.join(parts[3:]).replace('_market_analysis.md', '')
                if '_' in company_time:
                    company_parts = company_time.split('_')
                    company_name = ' '.join(company_parts[:-1])
                    time_period = company_parts[-1].replace('_', ' ')
            
            return jsonify({
                "success": True,
                "is_generating": False,
                "progress": 1.0,
                "message": "Report already generated",
                "report_file": latest_report,
                "error": None,
                "company_name": company_name,
                "time_period": time_period,
                "status": "completed"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Invalid or expired session ID",
                "error_code": "INVALID_SESSION_ID",
                "is_generating": False,
                "progress": 0,
                "report_file": None,
                "error": "Session not found or expired"
            }), 404
    
    return jsonify(user_status[session_id])

@flask_app.route('/download/<path:filename>')
def download_report(filename):
    """Download the generated report"""
    # Get session ID from query parameter first, then fallback to Flask session
    session_id = request.args.get('session_id') or session.get('session_id', '')
    
    if not session_id:
        return jsonify({
            "success": False,
            "message": "Missing session ID parameter",
            "error_code": "MISSING_SESSION_ID"
        }), 400
    
    # Check if the file path is absolute or relative
    if os.path.isabs(filename):
        file_path = filename
    else:
        file_path = os.path.join(os.getcwd(), filename)
    
    if os.path.exists(file_path):
        # Mark the status as completed instead of resetting
        if session_id in user_status:
            user_status[session_id]["is_generating"] = False
            user_status[session_id]["progress"] = 1.0
            user_status[session_id]["message"] = "Report generation complete"
            # Keep the report_file field to allow subsequent downloads
            # Add completed_at timestamp
            user_status[session_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_status[session_id]["status"] = "completed"
        
        # Append to log file instead of clearing it
        try:
            log_file = f"logs/report_generation_{session_id}.log"
            if os.path.exists(log_file):
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Report downloaded by user\n")
        except Exception as e:
            print(f"Error updating log file for session {session_id}: {e}")
        
        return send_file(file_path, as_attachment=True)
    else:
        # Try looking in the reports directory if not found
        reports_path = os.path.join(os.getcwd(), REPORTS_DIR, os.path.basename(filename))
        if os.path.exists(reports_path):
            # Mark the status as completed instead of resetting
            if session_id in user_status:
                user_status[session_id]["is_generating"] = False
                user_status[session_id]["progress"] = 1.0
                user_status[session_id]["message"] = "Report generation complete"
                # Keep the report_file field to allow subsequent downloads
                # Add completed_at timestamp
                user_status[session_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_status[session_id]["status"] = "completed"
            
            # Append to log file instead of clearing it
            try:
                log_file = f"logs/report_generation_{session_id}.log"
                if os.path.exists(log_file):
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Report downloaded by user\n")
            except Exception as e:
                print(f"Error updating log file for session {session_id}: {e}")
            
            return send_file(reports_path, as_attachment=True)
        
        return jsonify({
            "success": False,
            "message": f"File not found: {filename}. Please check if the report was generated successfully.",
            "error_code": "FILE_NOT_FOUND"
        }), 404

# Add a route to cancel an ongoing report generation
@flask_app.route('/cancel', methods=['POST'])
def cancel_report():
    """Cancel an ongoing report generation"""
    # Get session ID from query parameter first, then fallback to Flask session
    session_id = request.args.get('session_id')
    
    if not session_id:
        # Fallback to Flask session with warning
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({
                "success": False,
                "message": "Missing session ID parameter",
                "error_code": "MISSING_SESSION_ID"
            }), 400
        
        # Return a warning that using Flask session is deprecated
        return jsonify({
            "success": True,
            "warning": "Using Flask session ID is deprecated. Please include session_id parameter explicitly.",
            "message": "Using deprecated session mechanism"
        })
    
    if session_id not in user_status:
        return jsonify({
            "success": False,
            "message": "No active report generation to cancel. Invalid or expired session ID.",
            "error_code": "INVALID_SESSION_ID"
        }), 404
    
    # Update status to mark as not generating
    if user_status[session_id]["is_generating"]:
        user_status[session_id]["is_generating"] = False
        user_status[session_id]["message"] = "Report generation cancelled by user"
        user_status[session_id]["status"] = "cancelled"
        
        # Append to the log file instead of clearing
        try:
            log_file = f"logs/report_generation_{session_id}.log"
            if os.path.exists(log_file):
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Report generation cancelled by user\n")
        except Exception as e:
            print(f"Error updating log file for session {session_id}: {e}")
    
    return jsonify({
        "success": True,
        "message": "Report generation cancelled"
    })

# Add a route to validate session IDs
@flask_app.route('/validate-session')
def validate_session():
    """Validate a session ID and return its status"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({
            "valid": False,
            "message": "Missing session ID",
            "error_code": "MISSING_SESSION_ID"
        })
    
    if session_id not in user_status:
        # Check if there are any reports for this session ID in the reports directory
        session_reports = glob.glob(os.path.join(REPORTS_DIR, f"*_{session_id}_*"))
        
        if session_reports:
            # Found reports for this session, it's valid but not active
            return jsonify({
                "valid": True,
                "status": "completed",
                "message": "Session has completed reports but is not active"
            })
        else:
            return jsonify({
                "valid": False, 
                "message": "Session not found or expired",
                "error_code": "INVALID_SESSION_ID"
            })
    
    # Session exists in user_status
    status = "active" if user_status[session_id].get("is_generating", False) else "completed"
    return jsonify({
        "valid": True,
        "status": status,
        "data": {
            "is_generating": user_status[session_id].get("is_generating", False),
            "progress": user_status[session_id].get("progress", 0),
            "company_name": user_status[session_id].get("company_name", ""),
            "time_period": user_status[session_id].get("time_period", "")
        }
    })

# FastAPI routes
def start_report_generation_task(company_name: str, time_period: str, request_id: str):
    """Start the report generation task (for background tasks)"""
    asyncio.run(generate_report_async(company_name, time_period, request_id))

@fastapi_app.post("/api/generate-report", response_model=ReportResponse)
async def api_generate_report(report_request: ReportRequest, background_tasks: BackgroundTasks):
    """Start generating a report based on request data"""
    company_name = report_request.company_name
    time_period = report_request.time_period
    
    if not company_name or not time_period:
        raise HTTPException(status_code=400, detail="Please provide both company name and time period")
    
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Initialize request status
    user_status[request_id] = {
        "is_generating": True,
        "progress": 0,
        "message": "Initializing...",
        "report_file": None,
        "error": None,
        "company_name": company_name,
        "time_period": time_period
    }
    
    # Create a request-specific log file
    log_file = f"logs/report_generation_{request_id}.log"
    
    # Clear the log file before starting a new report
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("")
    except Exception as e:
        print(f"Error clearing log file for request {request_id}: {e}")
    
    # Add task to background tasks
    background_tasks.add_task(start_report_generation_task, company_name, time_period, request_id)
    
    return ReportResponse(
        request_id=request_id,
        status="processing",
        message="Report generation has been started. Use the /api/report-status endpoint to check progress."
    )

@fastapi_app.get("/api/report-status/{request_id}", response_model=ReportStatus)
async def api_get_report_status(request_id: str):
    """Get the current status of report generation for the given request ID"""
    if not request_id or request_id not in user_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    return ReportStatus(
        request_id=request_id,
        **user_status[request_id]
    )

@fastapi_app.get("/api/download-report/{request_id}")
async def api_download_report(request_id: str):
    """Download the generated report"""
    if not request_id or request_id not in user_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    status = user_status[request_id]
    if not status["report_file"]:
        if status["error"]:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {status['error']}")
        else:
            raise HTTPException(status_code=404, detail="Report file not found or not yet generated")
    
    report_file = status["report_file"]
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report file not found on server")
    
    return FileResponse(
        path=report_file,
        filename=os.path.basename(report_file),
        media_type="text/markdown"
    )

@fastapi_app.delete("/api/report/{request_id}")
async def api_delete_report(request_id: str):
    """Delete a report and its associated data"""
    if not request_id or request_id not in user_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    status = user_status[request_id]
    
    # Delete the report file if it exists
    if status["report_file"] and os.path.exists(status["report_file"]):
        try:
            os.remove(status["report_file"])
        except Exception as e:
            print(f"Error deleting report file: {e}")
    
    # Delete the log file if it exists
    log_file = f"logs/report_generation_{request_id}.log"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"Error deleting log file: {e}")
    
    # Remove the request status
    user_status.pop(request_id, None)
    
    return {"detail": "Report and associated data deleted successfully"}

# Add API route to cancel a report
@fastapi_app.post("/api/cancel-report/{request_id}")
async def api_cancel_report(request_id: str):
    """Cancel an ongoing report generation"""
    if not request_id or request_id not in user_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    # Update status to mark as not generating
    if user_status[request_id]["is_generating"]:
        user_status[request_id]["is_generating"] = False
        user_status[request_id]["message"] = "Report generation cancelled by user"
        
        # Update the log file
        try:
            log_file = f"logs/report_generation_{request_id}.log"
            if os.path.exists(log_file):
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write("\n[SYSTEM] Report generation cancelled by user\n")
        except Exception as e:
            print(f"Error updating log file for request {request_id}: {e}")
    
    return {"detail": "Report generation cancelled"}

# Create report cleanup task
def cleanup_old_reports():
    """Delete reports older than REPORT_RETENTION_DAYS days"""
    try:
        log_message("Starting report cleanup task")
        
        # Get current time
        now = datetime.now()
        retention_period = timedelta(days=REPORT_RETENTION_DAYS)
        
        # List all files in reports directory
        report_files = glob.glob(os.path.join(REPORTS_DIR, "*"))
        
        # Track deleted files for logging
        deleted_count = 0
        
        for file_path in report_files:
            try:
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                    
                # Get file creation/modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Check if file is older than retention period
                if now - file_time > retention_period:
                    # Try to extract session ID from filename to clean up status
                    filename = os.path.basename(file_path)
                    parts = filename.split('_')
                    
                    # Expected format: YYYYMMDD_HHMMSS_session-id_filename.md
                    if len(parts) > 2:
                        # Try to find session ID in user_status
                        session_id = parts[2]
                        if session_id in user_status:
                            user_status.pop(session_id, None)
                            log_message(f"Cleaned up status for session {session_id}")
                            
                            # Also clean up log file
                            log_file = f"logs/report_generation_{session_id}.log"
                            if os.path.exists(log_file):
                                os.remove(log_file)
                    
                    # Delete the report file
                    os.remove(file_path)
                    deleted_count += 1
                    log_message(f"Deleted old report: {file_path}")
            
            except Exception as e:
                log_message(f"Error processing file {file_path}: {str(e)}")
                
        log_message(f"Report cleanup completed. Deleted {deleted_count} old reports.")
    except Exception as e:
        log_message(f"Error in cleanup task: {str(e)}")

# Start periodic cleanup task
def start_cleanup_task():
    """Start the periodic report cleanup task"""
    def run_cleanup():
        while True:
            try:
                cleanup_old_reports()
            except Exception as e:
                print(f"Error in cleanup task: {e}")
            
            # Sleep for 24 hours before next cleanup
            time.sleep(24 * 60 * 60)
    
    # Start cleanup in a background thread
    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    print("Started periodic report cleanup task")

# Add API documentation path
@fastapi_app.get("/api/docs", include_in_schema=False)
async def api_docs_redirect():
    """Redirect to the API documentation"""
    return Response(
        status_code=307,  # Temporary redirect
        headers={"Location": "/docs"}
    )

# Mount Flask app to FastAPI
fastapi_app.mount("/", WSGIMiddleware(flask_app))

# Serve static files
fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")

# Start cleanup task on application startup
start_cleanup_task()

if __name__ == "__main__":
    # Get port from environment variable for Replit compatibility
    port = int(os.environ.get('PORT', 8000))
    # Run with host='0.0.0.0' for Replit
    uvicorn.run("combined_app:fastapi_app", host="0.0.0.0", port=port, reload=True) 