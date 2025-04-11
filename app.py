from flask import Flask, render_template, request, jsonify, send_file
import os
import asyncio
import threading
import json
import time
from datetime import datetime
from utils import log_message
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
    ReportStateOutput
)

app = Flask(__name__)

# Global variables to track report generation
REPORTS_DIR = "reports"
status = {
    "is_generating": False,
    "progress": 0,
    "message": "",
    "report_file": None,
    "error": None,
    "company_name": "",
    "time_period": ""
}

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Create log file directory
os.makedirs("logs", exist_ok=True)

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

async def generate_report_async(company_name, time_period):
    """Generate report asynchronously using the working_agent.py code"""
    global status
    
    try:
        status["is_generating"] = True
        status["progress"] = 0.05
        status["message"] = "Starting report generation..."
        
        log_message(f"Starting report generation for {company_name} ({time_period})")
        
        # Initialize and compile the agent
        log_message("Initializing agent")
        reporter_agent = await initialize_agent()
        
        status["progress"] = 0.1
        log_message("Agent compiled successfully")
        
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
            }
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
            while status["is_generating"] and status["progress"] < 1.0:
                try:
                    # Read only new log entries since last check
                    with open("logs/report_generation.log", "r") as f:
                        f.seek(last_position)
                        new_log_content = f.read()
                        last_position = f.tell()
                    
                    if new_log_content:
                        # Check each progress step in order
                        for step, progress_value in progress_steps.items():
                            if step in new_log_content and status["progress"] < progress_value:
                                status["progress"] = progress_value
                                status["message"] = f"Step: {step.replace('---', '').strip()}"
                                print(f"Progress updated to {progress_value}: {step}")
                
                except Exception as e:
                    print(f"Error reading log file: {e}")
                    # If we get a socket error, it means Flask is reloading
                    if "socket" in str(e).lower():
                        print("Flask is reloading - stopping progress monitoring")
                        break
                
                time.sleep(1)
        
        # Start progress monitoring in a thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Run the report generation
        try:
            log_message("Starting agent execution")
            result = await reporter_agent.ainvoke(state)
            
            # Process result
            report_filename = result.get("filename", "")
            if not report_filename:
                # If filename not in result, construct it using the same pattern as in working_agent.py
                clean_company = company_name.replace(" ", "_")
                clean_period = time_period.replace(" ", "_")
                report_filename = f"{clean_company}_{clean_period}_market_analysis.md"

            status["report_file"] = report_filename
            log_message(f"Report generated successfully: {report_filename}")
            
            # Copy the report to the reports directory with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_copy = f"{REPORTS_DIR}/{timestamp}_{report_filename}"
            
            # Check if file exists before copying
            if os.path.exists(report_filename):
                with open(report_filename, "r", encoding='utf-8') as src, open(report_copy, "w", encoding='utf-8') as dst:
                    dst.write(src.read())
                status["report_file"] = report_copy
                status["progress"] = 1.0
                status["message"] = "Report generation complete"
            else:
                error_msg = f"Report file not found: {report_filename}"
                log_message(error_msg)
                status["error"] = error_msg
                status["progress"] = 1.0
                status["message"] = "Report generation failed - file not found"
            
            return True
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            log_message(error_msg)
            status["error"] = error_msg
            return False
            
    except Exception as e:
        error_msg = f"Error setting up report generation: {str(e)}"
        log_message(error_msg)
        status["error"] = error_msg
        return False
    finally:
        status["is_generating"] = False

def start_report_generation(company_name, time_period):
    """Start the report generation process in a background thread"""
    global status
    
    # Reset status
    status = {
        "is_generating": True,
        "progress": 0,
        "message": "Initializing...",
        "report_file": None,
        "error": None,
        "company_name": company_name,
        "time_period": time_period
    }
    
    # Clear the log file before starting a new report
    try:
        with open("logs/report_generation.log", "w", encoding='utf-8') as f:
            f.write("")
    except Exception as e:
        print(f"Error clearing log file: {e}")
    
    # Define a wrapper function to run the async function
    def run_async_report():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_report_async(company_name, time_period))
        except Exception as e:
            print(f"Error in report generation thread: {e}")
            status["error"] = str(e)
            status["is_generating"] = False
        finally:
            loop.close()
    
    # Start generation in a thread
    thread = threading.Thread(target=run_async_report)
    thread.daemon = True
    thread.start()
    
    return True

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Start generating a report based on form input"""
    company_name = request.form.get('company_name', '')
    time_period = request.form.get('time_period', '')
    
    if not company_name or not time_period:
        return jsonify({
            "success": False,
            "message": "Please provide both company name and time period"
        }), 400
    
    # Start report generation
    success = start_report_generation(company_name, time_period)
    
    return jsonify({
        "success": success,
        "message": "Report generation started"
    })

@app.route('/status')
def get_status():
    """Get the current status of report generation"""
    return jsonify(status)

@app.route('/download/<path:filename>')
def download_report(filename):
    """Download the generated report"""
    global status  # Move global declaration to the top
    
    # Check if the file path is absolute or relative
    if os.path.isabs(filename):
        file_path = filename
    else:
        file_path = os.path.join(os.getcwd(), filename)
    
    if os.path.exists(file_path):
        # Reset the status and clear the log file after successful download
        status = {
            "is_generating": False,
            "progress": 0,
            "message": "",
            "report_file": None,
            "error": None,
            "company_name": "",
            "time_period": ""
        }
        
        # Clear the log file
        try:
            with open("logs/report_generation.log", "w") as f:
                f.write("")
        except Exception as e:
            print(f"Error clearing log file: {e}")
        
        return send_file(file_path, as_attachment=True)
    else:
        # Try looking in the reports directory if not found
        reports_path = os.path.join(os.getcwd(), REPORTS_DIR, os.path.basename(filename))
        if os.path.exists(reports_path):
            # Reset the status and clear the log file after successful download
            status = {
                "is_generating": False,
                "progress": 0,
                "message": "",
                "report_file": None,
                "error": None,
                "company_name": "",
                "time_period": ""
            }
            
            # Clear the log file
            try:
                with open("logs/report_generation.log", "w") as f:
                    f.write("")
            except Exception as e:
                print(f"Error clearing log file: {e}")
            
            return send_file(reports_path, as_attachment=True)
        
        return jsonify({
            "success": False,
            "message": f"File not found: {filename}. Please check if the report was generated successfully."
        }), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000) 