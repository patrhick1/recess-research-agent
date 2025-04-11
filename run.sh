#!/bin/bash
echo "Setting up environment..."

# Create necessary directories
mkdir -p reports
mkdir -p logs

# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py 