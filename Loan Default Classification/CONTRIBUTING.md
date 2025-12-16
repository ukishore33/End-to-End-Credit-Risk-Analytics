# Contributing to Loan Eligibility Prediction

First off, thank you for considering contributing to this project! It's people like you that make this project better.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct (see CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if relevant**
- **Provide your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding style** used throughout the project
3. **Write clear commit messages**
4. **Include tests** for any new functionality
5. **Update documentation** as needed
6. **Ensure all tests pass** before submitting

## Development Setup

1. Clone your fork of the repository:
```bash
git clone https://github.com/your-username/Loan-Eligibility-Prediction-BFSI.git
cd Loan-Eligibility-Prediction-BFSI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused

### Code Formatting

We use the following tools for code quality:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Testing

- Write unit tests for all new functions
- Maintain test coverage above 80%
- Run tests before submitting PR:

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

Example:
```
Add feature to predict loan eligibility

- Implement RandomForest classifier
- Add feature engineering pipeline
- Update documentation

Closes #123
```

## Project Structure

When adding new code, follow the project structure:

- `src/data/` - Data loading and preprocessing
- `src/features/` - Feature engineering
- `src/models/` - Model training and prediction
- `src/visualization/` - Plotting and visualization
- `src/utils/` - Utility functions
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `notebooks/` - Jupyter notebooks for exploration
- `docs/` - Documentation

## Documentation

- Update README.md if you change functionality
- Add docstrings to all functions and classes
- Update docs/ folder for major changes
- Include code examples in documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this error is raised
    """
    pass
```

## Review Process

1. All submissions require review before merging
2. Maintainers will review your PR within a few days
3. Address any feedback promptly
4. Once approved, a maintainer will merge your PR

## Questions?

Feel free to open an issue with the label "question" if you have any questions about contributing.

## Recognition

Contributors will be recognized in the project README and release notes.

Thank you for contributing! 🎉
