#!/usr/bin/env python3
"""
Setup script to create the required directory structure and missing files.
Run this script to quickly set up the Enhanced ML Pipeline.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure."""
    
    # Define the directory structure
    directories = [
        'core',
        'database', 
        'analysis',
        'ui',
        'documentation',
        'utils',
        'models'
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_contents = {
        'core/__init__.py': '''"""Core ML pipeline components."""
from .pipeline import EnhancedMLPipeline
from .data_loader import DataLoader
__all__ = ['EnhancedMLPipeline', 'DataLoader']''',
        
        'database/__init__.py': '''"""Database and experiment tracking components."""
from .experiment_db import ExperimentDatabase
__all__ = ['ExperimentDatabase']''',
        
        'analysis/__init__.py': '''"""Data analysis and visualization components."""
from .data_analyzer import SmartDataAnalyzer
__all__ = ['SmartDataAnalyzer']''',
        
        'ui/__init__.py': '''"""User interface components."""
from .terminal_ui import EnhancedTerminalUI
__all__ = ['EnhancedTerminalUI']''',
        
        'documentation/__init__.py': '''"""Documentation and help system components."""
from .parameter_docs import ParameterDocumentation
__all__ = ['ParameterDocumentation']''',
        
        'utils/__init__.py': '''"""Utility functions and helpers."""
from .validation import (
    validate_environment,
    validate_file_path,
    validate_dataframe,
    validate_target_column,
    validate_data_quality
)
__all__ = [
    'validate_environment',
    'validate_file_path', 
    'validate_dataframe',
    'validate_target_column',
    'validate_data_quality'
]''',
        
        'models/__init__.py': '''"""Machine learning model implementations."""
__all__ = []'''
    }
    
    # Create __init__.py files
    for file_path, content in init_contents.items():
        Path(file_path).write_text(content)
        print(f"✅ Created file: {file_path}")

def main():
    """Main setup function."""
    print("🚀 Setting up Enhanced ML Pipeline directory structure...")
    print("=" * 60)
    
    create_directory_structure()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy the provided code files to their respective directories")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the pipeline: python main.py")
    print("\n📁 Directory structure created:")
    
    # Show the created structure
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    main()