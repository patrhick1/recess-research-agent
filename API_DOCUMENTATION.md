# Deep Researcher API Documentation

The Deep Researcher API provides programmatic access to the report generation functionality, allowing you to integrate market analysis reports into your own applications.

## Base URL

When running locally:
```
http://localhost:8000
```

## API Endpoints

### Generate Report

Starts the generation of a new market analysis report.

**Endpoint**: `POST /generate-report`

**Request Body**:
```json
{
  "company_name": "Apple Inc.",
  "time_period": "Q4 2023"
}
```

**Response**:
```json
{
  "request_id": "7f4b3a12-ec29-4c41-a832-1b01e6df9e72",
  "status": "processing",
  "message": "Report generation has been started. Use the /report-status endpoint to check progress."
}
```

**Response Status Codes**:
- `200 OK`: Request was successful
- `400 Bad Request`: Missing required parameters

### Check Report Status

Get the current status of a report generation request.

**Endpoint**: `GET /report-status/{request_id}`

**Parameters**:
- `request_id`: The ID returned from the generate-report endpoint

**Response**:
```json
{
  "request_id": "7f4b3a12-ec29-4c41-a832-1b01e6df9e72",
  "is_generating": true,
  "progress": 0.45,
  "message": "Step: Collecting Evidence for Section",
  "company_name": "Apple Inc.",
  "time_period": "Q4 2023",
  "report_file": null,
  "error": null
}
```

When complete:
```json
{
  "request_id": "7f4b3a12-ec29-4c41-a832-1b01e6df9e72",
  "is_generating": false,
  "progress": 1.0,
  "message": "Report generation complete",
  "company_name": "Apple Inc.",
  "time_period": "Q4 2023",
  "report_file": "reports/20250411_123456_7f4b3a12-ec29-4c41-a832-1b01e6df9e72_Apple_Inc._Q4_2023_market_analysis.md",
  "error": null
}
```

**Response Status Codes**:
- `200 OK`: Request was successful
- `404 Not Found`: Request ID not found

### Download Report

Download the generated report file.

**Endpoint**: `GET /download-report/{request_id}`

**Parameters**:
- `request_id`: The ID returned from the generate-report endpoint

**Response**:
- A markdown file containing the generated report

**Response Status Codes**:
- `200 OK`: File returned successfully
- `404 Not Found`: Request ID or file not found
- `500 Internal Server Error`: Report generation failed

### Delete Report

Delete a report and its associated data.

**Endpoint**: `DELETE /report/{request_id}`

**Parameters**:
- `request_id`: The ID of the report to delete

**Response**:
```json
{
  "detail": "Report and associated data deleted successfully"
}
```

**Response Status Codes**:
- `200 OK`: Report deleted successfully
- `404 Not Found`: Request ID not found

## Usage Examples

### Using cURL

#### Generate a Report

```bash
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Tesla", "time_period": "Q1 2024"}'
```

#### Check Report Status

```bash
curl -X GET http://localhost:8000/report-status/7f4b3a12-ec29-4c41-a832-1b01e6df9e72
```

#### Download Report

```bash
curl -X GET http://localhost:8000/download-report/7f4b3a12-ec29-4c41-a832-1b01e6df9e72 --output tesla_report.md
```

### Using Python

```python
import requests
import time

# Generate a report
response = requests.post(
    "http://localhost:8000/generate-report",
    json={"company_name": "Netflix", "time_period": "Q4 2023"}
)
data = response.json()
request_id = data["request_id"]
print(f"Started report generation with ID: {request_id}")

# Poll for status until complete
while True:
    status_response = requests.get(f"http://localhost:8000/report-status/{request_id}")
    status = status_response.json()
    
    print(f"Progress: {status['progress']*100:.1f}% - {status['message']}")
    
    if status["progress"] >= 1.0 or not status["is_generating"]:
        break
    
    time.sleep(10)  # Check every 10 seconds

# Download the report if available
if status["report_file"]:
    print(f"Downloading report...")
    report_response = requests.get(f"http://localhost:8000/download-report/{request_id}")
    
    with open("netflix_report.md", "wb") as f:
        f.write(report_response.content)
    
    print(f"Report saved to netflix_report.md")
else:
    print(f"Error: {status['error']}")
```

## Interactive Documentation

When running the FastAPI server locally, you can access interactive API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These interactive pages allow you to explore the API and make test requests directly from your browser. 