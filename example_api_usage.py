#!/usr/bin/env python3
"""
Deep Researcher API Example Usage
This example script demonstrates how to use the Deep Researcher API to generate,
check status, and download market analysis reports.
"""

import requests
import json
import time
import argparse
import sys

def generate_report(company_name, time_period, base_url):
    """Generate a market analysis report for a specific company and time period."""
    endpoint = f"{base_url}/api/generate-report"
    
    payload = {
        "company_name": company_name,
        "time_period": time_period
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        request_id = result.get("request_id")
        print(f"Report generation started with request ID: {request_id}")
        return request_id, result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, response.json() if response.text else {"error": "Unknown error"}


def check_report_status(request_id, base_url):
    """Check the status of a report generation request."""
    endpoint = f"{base_url}/api/report-status/{request_id}"
    
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        status = response.json()
        print(f"Report status: {status['message']}")
        print(f"Progress: {status['progress'] * 100:.1f}%")
        if status.get("error"):
            print(f"Error: {status['error']}")
        return status
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def download_report(request_id, output_file, base_url):
    """Download a generated report."""
    endpoint = f"{base_url}/api/download-report/{request_id}"
    
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        # If output_file not specified, use the filename from Content-Disposition
        if not output_file:
            content_disposition = response.headers.get('Content-Disposition', '')
            if 'filename=' in content_disposition:
                output_file = content_disposition.split('filename=')[1].strip('"')
            else:
                output_file = f"report_{request_id}.md"
        
        # Write the report content to file
        with open(output_file, 'wb') as f:
            f.write(response.content)
            
        print(f"Report downloaded successfully to {output_file}")
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False


def delete_report(request_id, base_url):
    """Delete a report and its associated data."""
    endpoint = f"{base_url}/api/report/{request_id}"
    
    response = requests.delete(endpoint)
    
    if response.status_code == 200:
        print(f"Report with ID {request_id} deleted successfully")
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False


def full_report_workflow(company_name, time_period, base_url):
    """Complete workflow to generate, monitor, and download a report."""
    # Step 1: Generate report
    print(f"Generating report for {company_name} during {time_period}...")
    request_id, response = generate_report(company_name, time_period, base_url)
    
    if not request_id:
        print("Failed to start report generation")
        return None
    
    # Step 2: Poll for status until complete
    try:
        while True:
            status = check_report_status(request_id, base_url)
            
            if not status:
                print("Error checking status")
                return None
            
            # Check for errors
            if status.get("error"):
                print(f"Error generating report: {status['error']}")
                return None
                
            # Check if complete
            if status.get("progress") == 1.0:
                print("Report generation complete!")
                break
                
            # Progress information
            print(f"Progress: {status['progress'] * 100:.1f}% - {status['message']}")
            
            # Wait before checking again
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return None
    
    # Step 3: Download the report
    output_file = f"{company_name}_{time_period}_report.md".replace(" ", "_")
    success = download_report(request_id, output_file, base_url)
    
    if success:
        print(f"Report workflow completed successfully. Report saved to {output_file}")
        return output_file
    else:
        print("Failed to download report")
        return None


def main():
    """Command line interface for the Deep Researcher API example."""
    parser = argparse.ArgumentParser(description="Deep Researcher API Example Usage")
    parser.add_argument("--company", "-c", type=str, required=True, help="Company name to analyze")
    parser.add_argument("--period", "-p", type=str, required=True, help="Time period for analysis (e.g., '2023', 'Q1 2023')")
    parser.add_argument("--url", "-u", type=str, default="http://localhost:8000", help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--output", "-o", type=str, help="Output file name (optional)")
    parser.add_argument("--request-id", "-r", type=str, help="Use existing request ID instead of generating a new report")
    parser.add_argument("--check-only", action="store_true", help="Only check status of existing request (requires --request-id)")
    parser.add_argument("--download-only", action="store_true", help="Only download report (requires --request-id)")
    parser.add_argument("--delete", action="store_true", help="Delete report data (requires --request-id)")
    
    args = parser.parse_args()
    
    # Validate args
    if args.check_only or args.download_only or args.delete:
        if not args.request_id:
            print("Error: --request-id is required with --check-only, --download-only, or --delete")
            sys.exit(1)
    
    # Handle different operation modes
    if args.delete and args.request_id:
        delete_report(args.request_id, args.url)
        return
    
    if args.check_only and args.request_id:
        check_report_status(args.request_id, args.url)
        return
        
    if args.download_only and args.request_id:
        output_file = args.output or f"report_{args.request_id}.md"
        download_report(args.request_id, output_file, args.url)
        return
    
    if args.request_id:
        # Use existing request ID to continue the workflow
        print(f"Continuing with existing request ID: {args.request_id}")
        status = check_report_status(args.request_id, args.url)
        
        if status and status.get("progress") == 1.0:
            # Report is ready for download
            output_file = args.output or f"{args.company}_{args.period}_report.md".replace(" ", "_")
            download_report(args.request_id, output_file, args.url)
        else:
            print("Report is still being generated. Run with --check-only to monitor progress.")
    else:
        # Run the full workflow
        output_file = full_report_workflow(args.company, args.period, args.url)
        if output_file:
            print(f"Report generation workflow completed successfully.")


if __name__ == "__main__":
    main() 