#!/usr/bin/env python3
"""
Setup script for Enhanced ML Pipeline
Creates all necessary files and directories
"""

import os
from pathlib import Path

def create_file(filepath, content):
    """Create a file with given content."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created: {filepath}")

def setup_ml_pipeline():
    """Set up the complete ML pipeline structure."""
    
    print("🚀 Setting up Enhanced ML Pipeline v2.0...")
    
    # Create __init__.py files for all packages
    init_files = [
        "core/__init__.py",
        "database/__init__.py", 
        "analysis/__init__.py",
        "ui/__init__.py",
        "documentation/__init__.py",
        "utils/__init__.py",
        "models/__init__.py"
    ]
    
    for init_file in init_files:
        create_file(init_file, '# Package initialization\n')
    
    # Create a simple requirements.txt
    requirements = """# Enhanced ML Pipeline Requirements
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Optional but recommended
rich>=13.0.0
xgboost>=1.6.0
openpyxl>=3.0.0
"""
    
    create_file("requirements.txt", requirements)
    
    # Create a simple test data file
    test_data = """id,feature1,feature2,feature3,target
1,1.2,3.4,0,0
2,2.1,4.5,1,1
3,3.0,2.1,0,0
4,1.8,5.2,1,1
5,2.5,3.8,0,1
6,3.2,4.1,1,0
7,1.9,2.9,0,0
8,2.8,4.7,1,1
9,3.1,3.5,0,1
10,2.2,4.2,1,0
"""
    
    create_file("sample_data.csv", test_data)
    
    # Create a README
    readme = """# Enhanced ML Pipeline v2.0

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. Try with sample data:
   - Training file: `sample_data.csv`
   - Target column: `target`

## Features

- Automated data loading and preprocessing
- Multiple ML algorithms
- Model comparison and evaluation
- Experiment tracking
- Submission file generation

## Supported Models

- K-Nearest Neighbors (KNN)
- Decision Tree
- Linear/Logistic Regression
- Random Forest
- XGBoost (if installed)

## File Structure

```
enhanced_ml_pipeline/
├── main.py                 # Main entry point
├── config.py              # Configuration
├── core/                  # Core pipeline logic
├── models/                # Model implementations
├── database/              # Experiment tracking
├── analysis/              # Data analysis
├── ui/                    # User interface
├── utils/                 # Utilities
└── sample_data.csv        # Sample dataset
```

Happy machine learning! 🚀
"""
    
    create_file("README.md", readme)
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the pipeline: python main.py")
    print("3. Try with sample_data.csv")
    print("\n🎯 Your Enhanced ML Pipeline is ready!")

if __name__ == "__main__":
    setup_ml_pipeline()