#!/usr/bin/env python3
"""
AI Training Memory System - Main Entry Point

Usage:
    python main.py              # Run example
    python main.py cli          # Run CLI interface
    python main.py demo         # Run all demos
    python main.py test         # Run tests
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_example():
    """Run simple example."""
    from example import main as example_main
    example_main()


def run_cli():
    """Run CLI interface."""
    from src.interface.cli import main as cli_main
    cli_main()


def run_demo():
    """Run comprehensive demos."""
    from src.interface.demo import main as demo_main
    demo_main()


def run_tests():
    """Run test suite."""
    import subprocess
    subprocess.run(['pytest', 'tests/', '-v'])


def show_help():
    """Show help message."""
    print("""
AI Training Memory System - Main Entry Point

Usage:
    python main.py              # Run example
    python main.py example      # Run example (explicit)
    python main.py cli          # Run CLI interface
    python main.py demo         # Run all demos
    python main.py test         # Run tests
    python main.py help         # Show this help

Description:
    A complete memory-augmented AI system that enables continuous learning
    through vector database storage and semantic retrieval.

Features:
    - Vector-based memory storage (384D embeddings)
    - Duplicate detection with MD5 hashing
    - Semantic similarity search
    - Hybrid solving (rule-based + neural network)
    - Performance tracking and visualization
    - Save/load functionality

Documentation:
    - README.md           : Overview and quick start
    - docs/API.md         : Complete API reference
    - docs/ARCHITECTURE.md: System architecture
    - docs/TUTORIAL.md    : Step-by-step tutorial

Examples:
    # Run interactive demo
    python main.py demo

    # Start CLI interface
    python main.py cli

    # Run tests
    python main.py test

For more information, visit: github.com/nickyjbn/NickyMajorProject
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default to example
        run_example()
    else:
        command = sys.argv[1].lower()
        
        if command in ['example', 'ex']:
            run_example()
        elif command in ['cli', 'interface']:
            run_cli()
        elif command in ['demo', 'demos']:
            run_demo()
        elif command in ['test', 'tests']:
            run_tests()
        elif command in ['help', '-h', '--help']:
            show_help()
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for usage information")
            sys.exit(1)


if __name__ == "__main__":
    main()
