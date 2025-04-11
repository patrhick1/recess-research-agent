# Deep Researcher v2

An AI-powered research tool that generates comprehensive market analysis reports for companies.

## Features

- Generates detailed market analysis reports for any company and time period
- Automatically researches using multiple web sources
- Parallelized processing for faster report generation
- Clean and intuitive web interface
- Progress tracking during report generation

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Deep-Researcher-v2
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to http://127.0.0.1:5000/

3. Enter a company name and time period, then click "Generate Report"

4. Wait for the report to be generated and download it when complete

## Project Structure

- `app.py`: Main Flask application
- `working_agent.py`: Agent logic for report generation
- `utils.py`: Utility functions
- `templates/`: HTML templates
- `static/`: Static assets (CSS, JS)
- `reports/`: Generated reports (created at runtime)
- `logs/`: Log files (created at runtime)

## License

[MIT](LICENSE) 