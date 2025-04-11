import os
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
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

app = FastAPI(
    title="Deep Researcher API",
    description="API for generating market analysis reports for companies",
    version="1.0.0"
)

# Global variables to track report generation
REPORTS_DIR = "reports"
# Dictionary to store status for each request ID
request_status = {}

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Create log file directory
os.makedirs("logs", exist_ok=True)

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

async def generate_report_async(company_name: str, time_period: str, request_id: str):
    """Generate report asynchronously using the working_agent.py code"""
    global request_status
    
    try:
        request_status[request_id]["is_generating"] = True
        request_status[request_id]["progress"] = 0.05
        request_status[request_id]["message"] = "Starting report generation..."
        
        # Create a request-specific log file
        log_file = f"logs/report_generation_{request_id}.log"
        log_message(f"Starting report generation for {company_name} ({time_period})", log_file)
        
        # Initialize and compile the agent
        log_message("Initializing agent", log_file)
        reporter_agent = await initialize_agent()
        
        request_status[request_id]["progress"] = 0.1
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
            }
        }
        
        # Run the report generation
        try:
            log_message("Starting agent execution", log_file)
            result = await reporter_agent.ainvoke(state)
            
            # Process result
            report_filename = result.get("filename", "")
            if not report_filename:
                # If filename not in result, construct it using the same pattern as in working_agent.py
                clean_company = company_name.replace(" ", "_")
                clean_period = time_period.replace(" ", "_")
                report_filename = f"{clean_company}_{clean_period}_market_analysis.md"

            # Copy the report to the reports directory with a timestamp and request ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_copy = f"{REPORTS_DIR}/{timestamp}_{request_id}_{report_filename}"
            
            # Check if file exists before copying
            if os.path.exists(report_filename):
                with open(report_filename, "r", encoding='utf-8') as src, open(report_copy, "w", encoding='utf-8') as dst:
                    dst.write(src.read())
                request_status[request_id]["report_file"] = report_copy
                request_status[request_id]["progress"] = 1.0
                request_status[request_id]["message"] = "Report generation complete"
                log_message(f"Report generated successfully: {report_copy}", log_file)
            else:
                error_msg = f"Report file not found: {report_filename}"
                log_message(error_msg, log_file)
                request_status[request_id]["error"] = error_msg
                request_status[request_id]["progress"] = 1.0
                request_status[request_id]["message"] = "Report generation failed - file not found"
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            log_message(error_msg, log_file)
            request_status[request_id]["error"] = error_msg
            request_status[request_id]["progress"] = 1.0
            request_status[request_id]["message"] = "Report generation failed"
            
    except Exception as e:
        error_msg = f"Error setting up report generation: {str(e)}"
        log_file = f"logs/report_generation_{request_id}.log" if request_id else "logs/report_generation.log"
        log_message(error_msg, log_file)
        if request_id in request_status:
            request_status[request_id]["error"] = error_msg
            request_status[request_id]["progress"] = 1.0
            request_status[request_id]["message"] = "Report generation failed"
    finally:
        if request_id in request_status:
            request_status[request_id]["is_generating"] = False

def start_report_generation_task(company_name: str, time_period: str, request_id: str):
    """Start the report generation task (for background tasks)"""
    asyncio.run(generate_report_async(company_name, time_period, request_id))

@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(report_request: ReportRequest, background_tasks: BackgroundTasks):
    """Start generating a report based on request data"""
    company_name = report_request.company_name
    time_period = report_request.time_period
    
    if not company_name or not time_period:
        raise HTTPException(status_code=400, detail="Please provide both company name and time period")
    
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Initialize request status
    request_status[request_id] = {
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
        message="Report generation has been started. Use the /report-status endpoint to check progress."
    )

@app.get("/report-status/{request_id}", response_model=ReportStatus)
async def get_report_status(request_id: str):
    """Get the current status of report generation for the given request ID"""
    if not request_id or request_id not in request_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    return ReportStatus(
        request_id=request_id,
        **request_status[request_id]
    )

@app.get("/download-report/{request_id}")
async def download_report(request_id: str):
    """Download the generated report"""
    if not request_id or request_id not in request_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    status = request_status[request_id]
    if not status["report_file"]:
        if status["error"]:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {status['error']}")
        else:
            raise HTTPException(status_code=404, detail="Report file not found or not yet generated")
    
    report_file = status["report_file"]
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report file not found on server")
    
    # Clean up status after download (optional)
    # request_status.pop(request_id, None)
    
    return FileResponse(
        path=report_file,
        filename=os.path.basename(report_file),
        media_type="text/markdown"
    )

@app.delete("/report/{request_id}")
async def delete_report(request_id: str):
    """Delete a report and its associated data"""
    if not request_id or request_id not in request_status:
        raise HTTPException(status_code=404, detail="Report request not found")
    
    status = request_status[request_id]
    
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
    request_status.pop(request_id, None)
    
    return {"detail": "Report and associated data deleted successfully"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 