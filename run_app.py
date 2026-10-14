"""Launcher for Freight Tiger Streamlit Dashboard."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "src" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching Freight Tiger Streamlit Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)
