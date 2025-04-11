# Deep Researcher v2 - Technical Documentation

## System Overview

Deep Researcher is an AI-powered market analysis tool that generates comprehensive business reports for companies based on real-time web research. The system uses a sophisticated agent-based architecture to parallelize research tasks, intelligently gather evidence, and synthesize polished, professional reports.

## Architecture & Workflow Diagram

```
┌───────────────────────────────────────────────────┐
│                  Web Interface                    │
│                  (Flask + HTML)                   │
└───────────────────┬───────────────────────────────┘
                    │ HTTP Requests
                    ▼
┌───────────────────────────────────────────────────┐
│                Flask Application                  │
│   - Session Management                            │
│   - Status Tracking                               │
│   - Report File Handling                          │
└───────────────────┬───────────────────────────────┘
                    │ Async Thread
                    ▼
┌───────────────────────────────────────────────────┐
│             Report Generation Agent               │
│                 (LangGraph FSM)                   │
└───┬───────────────┬────────────────┬──────────────┘
    │               │                │
    ▼               ▼                ▼
┌─────────┐    ┌──────────┐    ┌──────────────┐
│ Generate │    │ Research │    │    Final     │
│  Report  │    │ Sections │    │  Synthesis   │
│   Plan   │    │(Parallel)│    │& Compilation │
└─────────┘    └──────────┘    └──────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │   Web Search & Evidence  │
         │   Collection (Parallel)  │
         └──────────────────────────┘
```

### Detailed Workflow

1. **User Input**: User enters company name and time period via web interface
2. **Session Creation**: Flask assigns unique session ID to track user's request
3. **Report Generation Initialization**:
   - Thread created to handle async report generation
   - State graph initialized with nodes for each processing stage
4. **Report Planning**:
   - Generates report structure based on predefined templates
   - Identifies key research sections needed
5. **Parallel Section Research**:
   - For each section:
     - Generates search queries based on section topic
     - Executes web searches to gather relevant information
     - Collects and organizes evidence points
     - Maps evidence to appropriate subsections
     - Writes detailed paragraphs based on evidence
     - Synthesizes subsections into cohesive content
6. **Report Compilation**:
   - Formats all sections according to template
   - Writes executive summary and conclusion sections
   - Compiles final report with proper formatting
7. **Delivery**:
   - Saves report as markdown file
   - Makes file available for download to user

## Stack Details

### Backend Technologies
- **Python 3.x**: Core programming language
- **Flask**: Web framework for handling HTTP requests and serving the application
- **Asyncio**: For asynchronous programming and concurrent operations
- **Threading**: For parallel execution of agent tasks
- **LangGraph**: State machine implementation for agent orchestration
- **LangChain**: Framework for building LLM-powered applications
- **AI Models**:
  - Various LLM APIs for natural language understanding and generation
  - Web search integrations for research capabilities

### Frontend Technologies
- **HTML/CSS**: Basic structure and styling
- **JavaScript**: Dynamic client-side behavior
- **Session-based Authentication**: For multi-user support

### Storage
- **File-based storage**: For reports and logs
- **In-memory state**: For tracking report generation progress

## Input and Output

### Inputs
- **Company Name**: Name of the company to research (e.g., "Apple Inc.", "Nike", "Tesla")
- **Time Period**: Specific time frame for analysis (e.g., "Q4 2023", "FY 2023", "Last 5 years")

### Outputs
- **Markdown Report File**: Comprehensive market analysis report including:
  - Executive Summary
  - Company Overview
  - Financial Performance Analysis
  - Product and Market Analysis
  - Competitive Landscape
  - Strategic Recommendations
  - Sources and References

## Use Case Example

### Use Case: Quarterly Market Analysis for Danone

**Input:**
- Company Name: "Danone"
- Time Period: "Q4 2024"

**Process Flow:**
1. User enters "Danone" and "Q4 2024" into the web interface
2. System generates a unique session ID and initializes report generation
3. Report planning agent creates a structure for Danone's market analysis
4. Research agents execute parallel web searches for:
   - Danone's financial performance in Q4 2024
   - Product portfolio and category performance
   - Market trends affecting Danone's sectors
   - Competitive actions and market share data
   - Strategic initiatives announced by Danone
5. Evidence collection agents gather and organize facts from search results
6. Section writing agents synthesize research into coherent sections
7. Final compilation agent produces the complete report
8. User receives notification when the report is ready
9. User downloads the "Danone_Q4_2024_market_analysis.md" file

**Output:** A comprehensive 10-15 page market analysis report on Danone's Q4 2024 performance, including financial analysis, product category insights, competitive positioning, and strategic recommendations.

## System Components

### 1. Flask Web Application (`app.py`)
- Handles HTTP requests, session management, and report delivery
- Manages multi-user concurrent report generation
- Tracks and reports progress to users

### 2. Agent Orchestration (`working_agent.py`)
- Implements the state machine for report generation flow
- Manages parallel execution of research and writing tasks
- Contains specialized agents for different aspects of report creation

### 3. Utility Functions (`utils.py`)
- Provides logging and other helper functions
- Manages session-specific log files

### 4. Prompt Templates (`prompts/`)
- `report_structure.txt`: Defines the structure and sections of reports
- `recess_info.txt`: Contains guidance for research methodology

## Scalability and Multi-User Support

The system is designed to support multiple concurrent users through:
- Session-based user tracking with unique UUIDs
- Isolated data storage per session
- Independent processing threads for each report
- Session-specific log files
- Resource cleanup after report completion

## Future Enhancement Opportunities

1. **Authentication System**: Add user accounts for persistent report storage
2. **Database Integration**: Replace file-based storage with database for scalability
3. **Report Templates**: Allow customizable report templates for different industries
4. **PDF Export**: Add capability to export reports in PDF format
5. **Dashboard**: Create an admin dashboard to monitor system usage and performance
6. **API Access**: Develop a REST API for programmatic access to report generation
7. **Custom Research Sources**: Allow users to specify preferred information sources

## Conclusion

Deep Researcher v2 combines modern web technologies with advanced AI capabilities to deliver automated yet high-quality market analysis reports. The system's parallelized architecture allows it to efficiently research, analyze, and synthesize information from multiple sources simultaneously, producing comprehensive reports in a fraction of the time required for manual research. 