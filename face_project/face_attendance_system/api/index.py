"""
Vercel Serverless Entry Point
-----------------------------
Wraps the Flask application for deployment on Vercel serverless functions.

This module is the entry point that Vercel invokes for every HTTP request.
It creates the Flask app and handles all routing through Vercel's Python runtime.
"""

import sys
import os

# Add project root to Python path so all imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the Flask application
from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "production"))

# Export app for Vercel's Python runtime
# Vercel uses this 'app' variable to handle HTTP requests
