# Deep Researcher API Usage Guide with Python

This guide demonstrates how to interact with the Deep Researcher API using Python's `requests` library.

## Prerequisites

- Python 3.6 or higher
- `requests` library

```python
pip install requests
```

## API Endpoints

- Generate Report: `/api/generate-report`
- Check Status: `/api/report-status/{request_id}`
- Download Report: `/api/download-report/{request_id}`
- Delete Report: `/api/report/{request_id}`

## 1. Generating a Report

```python
import requests
import json

def generate_report(company_name, time_period, base_url="http://localhost:8000"):
    """
    Generate a market analysis report for a specific company and time period.
    
    Args:
        company_name (str): Name of the company to analyze
        time_period (str): Time period for analysis (e.g., "Q2 2023", "2020-2023")
        base_url (str): Base URL of the API
        
    Returns:
        tuple: (request_id, response_dict)
    """
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

# Example usage
request_id, response = generate_report("Apple", "2022-2023")
print(response)
```

## 2. Checking Report Status

```python
def check_report_status(request_id, base_url="http://localhost:8000"):
    """
    Check the status of a report generation request.
    
    Args:
        request_id (str): The ID of the report generation request
        base_url (str): Base URL of the API
        
    Returns:
        dict: Status information
    """
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

# Example usage
status = check_report_status(request_id)
print(status)
```

## 3. Downloading a Report

```python
def download_report(request_id, output_file=None, base_url="http://localhost:8000"):
    """
    Download a generated report.
    
    Args:
        request_id (str): The ID of the report generation request
        output_file (str, optional): Path to save the downloaded report
        base_url (str): Base URL of the API
        
    Returns:
        bool: Success status
    """
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

# Example usage
download_report(request_id, "company_report.md")
```

## 4. Deleting a Report

```python
def delete_report(request_id, base_url="http://localhost:8000"):
    """
    Delete a report and its associated data.
    
    Args:
        request_id (str): The ID of the report to delete
        base_url (str): Base URL of the API
        
    Returns:
        bool: Success status
    """
    endpoint = f"{base_url}/api/report/{request_id}"
    
    response = requests.delete(endpoint)
    
    if response.status_code == 200:
        print(f"Report with ID {request_id} deleted successfully")
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False

# Example usage
delete_report(request_id)
```

## 5. Complete Workflow Example

Here's a complete example of generating a report, polling for status, and downloading it when ready:

```python
import requests
import json
import time

def full_report_workflow(company_name, time_period, base_url="http://localhost:8000"):
    """
    Complete workflow to generate, monitor, and download a report.
    
    Args:
        company_name (str): Name of the company to analyze
        time_period (str): Time period for analysis
        base_url (str): Base URL of the API
        
    Returns:
        str: Path to the downloaded report, or None if failed
    """
    # Step 1: Generate report
    print(f"Generating report for {company_name} during {time_period}...")
    request_id, response = generate_report(company_name, time_period, base_url)
    
    if not request_id:
        print("Failed to start report generation")
        return None
    
    # Step 2: Poll for status until complete
    status = None
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
        print("Process interrupted by user")
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

# Example usage
report_path = full_report_workflow("Microsoft", "2023")
```

## Error Handling

The API returns appropriate HTTP status codes:

- 200: Success
- 400: Bad request (e.g., missing parameters)
- 404: Report or resource not found
- 500: Server error during report generation

Always check the response status code and handle errors appropriately in your application.

## Tips for Using the API

1. **Large Reports**: Report generation can take several minutes depending on the company and time period.
2. **Polling Frequency**: When checking status, consider using a polling interval of 5-10 seconds to avoid overwhelming the server.
3. **API Base URL**: Adjust the `base_url` parameter based on your deployment environment.
4. **Error Handling**: Always implement proper error handling for production applications. 