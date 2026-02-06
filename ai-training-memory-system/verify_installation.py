#!/usr/bin/env python3
"""
Installation verification script for AI Training Memory System.
Run this to check if the system is properly installed.
"""

import sys


def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_imports():
    """Check if all required modules can be imported."""
    required = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('sentence_transformers', 'sentence-transformers'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_good = True
    for module, package in required:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            all_good = False
    
    return all_good


def check_project_structure():
    """Check if project structure is correct."""
    import os
    from pathlib import Path
    
    required_dirs = [
        'src/core',
        'src/solvers',
        'src/utils',
        'src/interface',
        'tests',
        'docs',
        'notebooks',
        'data'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ not found")
            all_good = False
    
    return all_good


def check_core_imports():
    """Check if core modules can be imported."""
    try:
        from src.core.document import SimpleDocument
        print("✅ SimpleDocument")
        
        from src.solvers.rule_based import solve_with_rules
        print("✅ Rule-based solver")
        
        from src.utils.text_processing import normalize_text
        print("✅ Text processing")
        
        return True
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("AI Training Memory System - Installation Verification")
    print("="*60)
    print()
    
    print("Checking Python version...")
    python_ok = check_python_version()
    print()
    
    print("Checking project structure...")
    structure_ok = check_project_structure()
    print()
    
    print("Checking core modules...")
    core_ok = check_core_imports()
    print()
    
    print("Checking dependencies...")
    deps_ok = check_imports()
    print()
    
    print("="*60)
    if python_ok and structure_ok and core_ok and deps_ok:
        print("✅ All checks passed! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run example: python example.py")
        print("  2. Run demos: python main.py demo")
        print("  3. Start CLI: python main.py cli")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("To install dependencies:")
        print("  pip install -r requirements.txt")
    print("="*60)


if __name__ == "__main__":
    main()
