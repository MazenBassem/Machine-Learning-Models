# GitHub Copilot Instructions

## Repository Overview
This is a comprehensive collection of machine learning models and Jupyter notebooks for educational and research purposes.

## Code Style and Conventions

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable names that describe the data or model being worked with
- Add docstrings to all functions and classes
- Use type hints where appropriate

### Jupyter Notebooks
- Keep notebooks well-organized with clear section headers
- Include markdown cells to explain the purpose of each section
- Add comments to explain complex algorithms or data transformations
- Clear all outputs before committing (optional, but recommended for cleaner diffs)

## Documentation Requirements
- All machine learning models should include:
  - Model description and purpose
  - Dataset information (source, size, features)
  - Hyperparameters and their rationale
  - Performance metrics and evaluation results
  - Usage examples

## Machine Learning Best Practices
- Always split data into training, validation, and test sets
- Set random seeds for reproducibility
- Document preprocessing steps clearly
- Include model evaluation metrics (accuracy, precision, recall, F1-score, etc.)
- Visualize results where appropriate (confusion matrices, learning curves, etc.)

## Dependencies
- Use `requirements.txt` or `environment.yml` to specify dependencies
- Pin dependency versions for reproducibility
- Document any special installation steps

## Testing
- Include unit tests for utility functions and data processing pipelines
- Validate model outputs with sample data
- Test edge cases in data preprocessing

## File Organization
- Organize models by category or algorithm type
- Use descriptive file names
- Include a README.md in subdirectories to explain the contents

## General Guidelines
- Keep code modular and reusable
- Avoid hardcoded paths; use relative paths or configuration files
- Include error handling for data loading and processing
- Comment on any non-obvious algorithmic choices
