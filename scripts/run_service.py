#!/usr/bin/env python3
"""
VPS Service Runner

This script starts the VPS service for visual positioning.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.service import create_app

def main():
    parser = argparse.ArgumentParser(description='Start VPS service')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the service to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind the service to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create and run the Flask app
    app = create_app(args.config)
    
    print(f"Starting VPS service on {args.host}:{args.port}")
    print(f"Configuration file: {args.config}")
    print("Press Ctrl+C to stop the service")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 