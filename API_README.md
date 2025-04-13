# Deep Researcher API Guide

This directory contains documentation and examples for using the Deep Researcher API to generate market analysis reports programmatically.

## Contents

- [API_USAGE_EXAMPLES.md](API_USAGE_EXAMPLES.md) - Detailed documentation with Python code examples
- [example_api_usage.py](example_api_usage.py) - Ready-to-use command line script for API interaction

## Quick Start

### Prerequisites

- Python 3.6 or higher
- `requests` library (`pip install requests`)

### Using the Example Script

The `example_api_usage.py` script provides a command-line interface for interacting with the API:

```bash
# Generate a report and follow the entire workflow 
python example_api_usage.py --company "Apple" --period "2023"

# Specify a custom API endpoint
python example_api_usage.py --company "Microsoft" --period "Q2 2023" --url "https://your-api-endpoint.com"

# Check status of an existing report
python example_api_usage.py --company "Google" --period "2023" --request-id "1234-5678-9012" --check-only

# Download a completed report
python example_api_usage.py --company "Tesla" --period "2022-2023" --request-id "1234-5678-9012" --download-only --output "tesla_report.md"

# Delete a report and associated data
python example_api_usage.py --company "Amazon" --period "2023" --request-id "1234-5678-9012" --delete
```

### Script Help

```bash
python example_api_usage.py --help
```

## API Endpoints

- `POST /api/generate-report` - Start report generation
- `GET /api/report-status/{request_id}` - Check report status
- `GET /api/download-report/{request_id}` - Download completed report
- `DELETE /api/report/{request_id}` - Delete report data

## Report Generation Process

1. **Initiate Report Generation**: Send a request with company name and time period
2. **Monitor Progress**: Poll the status endpoint to track report generation
3. **Download Report**: Once the report is complete, download the generated markdown file

## Advanced Usage

For more detailed examples and integration patterns, see [API_USAGE_EXAMPLES.md](API_USAGE_EXAMPLES.md). 