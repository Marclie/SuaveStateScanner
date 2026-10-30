#!/usr/bin/env python3
"""
Suave State Scanner - Modern GUI Entrypoint (Single-Node Dash).

Usage:
    python suave_gui_runner.py
    python suave_gui_runner.py --infile input.csv --num-states 9 --config suave.json
"""

import argparse
import sys
import os

from suave_gui.app import create_app
from suave_state_scanner import SuaveStateScanner


def main():
    parser = argparse.ArgumentParser(
        description="Suave State Scanner - Modern Interactive Plotting & Reordering Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--infile",
        default="input.csv" if os.path.exists("input.csv") else None,
        help="Initial dataset CSV path to load (optional)",
    )
    parser.add_argument(
        "-n", "--num-states",
        type=int,
        default=9,
        help="Number of electronic states per reaction coordinate point",
    )
    parser.add_argument(
        "-c", "--config",
        default="suave.json" if os.path.exists("suave.json") else None,
        help="Configuration JSON file path (optional)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to bind the local Dash web server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash development server in debug mode",
    )

    args = parser.parse_args()

    # Instantiate the backend scanner instance directly if an input file is provided
    scanner = None
    if args.infile and os.path.exists(args.infile):
        print(f"[SuaveGUI] Initializing backend SuaveStateScanner from '{args.infile}'...")
        scanner = SuaveStateScanner(args.infile, "output.csv", args.num_states, args.config)

    print(f"[SuaveGUI] Starting Suave State Scanner GUI on http://{args.host}:{args.port}/")
    app = create_app(
        suave_instance=scanner,
        default_dataset_path=args.infile,
        default_num_states=args.num_states,
        default_config_path=args.config,
    )

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
