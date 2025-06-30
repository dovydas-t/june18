# Enhanced ML Pipeline v2.0

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
