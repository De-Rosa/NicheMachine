import sys
import subprocess
from pathlib import Path
import argparse
import os
import signal

project_root = Path(__file__).parent
src_dir = project_root / "src"

def main():
    parser = argparse.ArgumentParser(description="Main orchestrator for NicheMachine.")
    parser.add_argument("--preload", action="store_true", help="Run the preload step.")
    parser.add_argument("--rank", action="store_true", help="Run ranking step.")
    parser.add_argument("--run-server", action="store_true", help="Run web server and bridge.")
    args = parser.parse_args()

    web_proc = None
    bridge_proc = None

    try:
        # Step 1: Preload
        if args.preload:
            from src.cli.preload import preload
            preload()

        # Step 2: Ranking
        if args.rank:
            from src.data_prep import rank_tracks_by_viral_probability
            ranked_df = rank_tracks_by_viral_probability("preload.csv")

        # Step 3: Web server and bridge
        if args.run_server:
            print("Starting web server...")
            web_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=src_dir / "web",
                preexec_fn=os.setsid  # start in a new process group
            )

            print("Starting bridge_serial.py...")
            bridge_proc = subprocess.Popen(
                [sys.executable, src_dir / "bridge_serial.py"],
                preexec_fn=os.setsid  # start in a new process group
            )

            print("Server and bridge started. Press Ctrl+C to stop.")

            # Wait until either process exits
            while True:
                web_proc.poll()
                bridge_proc.poll()
                # Sleep briefly to avoid busy loop
                import time
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Shutting down...")

        # Kill the entire process groups
        if web_proc:
            os.killpg(os.getpgid(web_proc.pid), signal.SIGTERM)
        if bridge_proc:
            os.killpg(os.getpgid(bridge_proc.pid), signal.SIGTERM)

        print("All subprocesses terminated. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()

