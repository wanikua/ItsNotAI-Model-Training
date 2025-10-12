#!/usr/bin/env python3
"""
Run the AI Image Detection Dashboard with proper configuration to avoid torch import issues.
"""

import os
import sys
import subprocess

def run_dashboard():
    """Run the Streamlit dashboard with proper configuration."""
    
    # Set environment variables to avoid torch issues with Streamlit watcher
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_CLIENT_SHOW_ERROR_DETAILS"] = "true"
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Command to run Streamlit
    cmd = [
        sys.executable, 
        "-m", "streamlit", 
        "run", 
        "src/mlflow/dashboard.py",
        "--server.fileWatcherType=none",
        "--server.headless=true",
        "--client.showErrorDetails=true",
        "--runner.magicEnabled=false",
        "--logger.level=warning"
    ]
    
    print("Starting AI Image Detection Dashboard...")
    print("Navigate to the URL shown below to access the dashboard.")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(run_dashboard())
