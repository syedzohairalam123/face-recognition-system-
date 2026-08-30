#!/usr/bin/env python3
"""
Face Recognition Attendance System
-----------------------------------
Entry point for the Flask application.

Usage:
    python run.py              # Run in development mode
    python run.py --production # Run in production mode
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Initialize and run the Flask application."""
    from app import create_app

    config_name = os.environ.get("FLASK_ENV", "development")

    if "--production" in sys.argv:
        config_name = "production"
    elif "--testing" in sys.argv:
        config_name = "testing"

    app = create_app(config_name)

    print(f"\n{'='*60}")
    print(f"  Face Recognition Attendance System")
    print(f"  Environment: {config_name}")
    print(f"  URL: http://127.0.0.1:5000")
    print(f"{'='*60}\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=(config_name == "development"),
    )


# Create app for Vercel deployment
from app import create_app
app = create_app(os.environ.get("FLASK_ENV", "production"))


if __name__ == "__main__":
    main()
