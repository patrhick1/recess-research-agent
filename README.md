# Deep Researcher v2

An AI-powered research tool that generates comprehensive market analysis reports for companies.

## Features

- Generates detailed market analysis reports for any company and time period
- Automatically researches using multiple web sources
- Parallelized processing for faster report generation
- Clean and intuitive web interface
- Progress tracking during report generation
- Admin dashboard for monitoring and managing reports
- Secure authentication system for administrative access

## Installation

1. Clone this repository:
```bash
git clone https://github.com/patrhick1/recess-research-agent.git
cd recess-research-agent
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
python combined_app.py
```

2. Open your browser and navigate to http://localhost:8000/

3. Log in with the default admin credentials:
   - Username: admin
   - Password: change_this_password_immediately

4. Generate reports through the admin dashboard or use the "Generate New Report" button

5. Monitor report generation progress and download completed reports

## Admin Features

- Secure login system with password protection
- Dashboard displaying all report generation sessions
- Real-time progress tracking for ongoing reports
- Session filtering and sorting options
- Detailed session view with logs
- Report download functionality
- Session management and deletion

## Project Structure

- `combined_app.py`: Main FastAPI/Flask combined application
- `working_agent.py`: Agent logic for report generation
- `utils.py`: Utility functions
- `templates/`: HTML templates for UI
- `static/`: Static assets (CSS, JS)
- `reports/`: Generated reports (created at runtime)
- `logs/`: Log files (created at runtime)
- `prompts/`: Prompt templates for the AI assistant

## Security Considerations

- Default admin credentials should be changed immediately in production
- Environment variables should be used for sensitive credentials
- All routes are protected with authentication
- Consider implementing HTTPS in production

## License

[MIT](LICENSE) 