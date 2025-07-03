import subprocess
import time
import sys
import os

def run_flask():
    print("Starting Flask server...")
    subprocess.Popen(['python', 'auth.py'], cwd=os.getcwd())

def run_streamlit():
    print("Starting Streamlit server...")
    subprocess.Popen(['streamlit', 'run', 'app2.py', '--server.port', '8501'], cwd=os.getcwd())

if __name__ == '__main__':
    # Ensure Conda environment is activated (inform user if manual activation needed)
    print("Ensure Conda environment 'url_detector_env' is activated before running this script.")
    print("Run: conda activate url_detector_env")
    
    # Start Flask server first
    run_flask()
    print("Flask server running at http://localhost:5000")
    print("Please log in to continue...")

    # Wait for successful login by checking for a flag file
    login_flag = "login_success.flag"
    while not os.path.exists(login_flag):
        time.sleep(1)  # Poll every second

    # Once login is successful, start Streamlit
    run_streamlit()
    
    print("Both servers are running.")
    print("Flask: http://localhost:5000")
    print("Streamlit: http://localhost:8501")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)  # Keep script running
    except KeyboardInterrupt:
        print("Stopping servers...")
        # Clean up the flag file
        if os.path.exists(login_flag):
            os.remove(login_flag)
        # Note: Subprocesses will need manual termination or system restart