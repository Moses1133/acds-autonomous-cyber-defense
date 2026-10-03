"""
Launch the ACDS Streamlit dashboard.

Usage: python scripts\run_dashboard.py
"""
import subprocess, sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app_path = os.path.join(root, "src", "dashboard", "app.py")

print(f"Launching Streamlit app: {app_path}")
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
