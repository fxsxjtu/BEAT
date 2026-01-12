# Contributing to BEAT

We welcome contributions to the BEAT project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, PyTorch version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an issue with:
- A clear description of the enhancement
- Use cases and motivation
- Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update documentation if needed

3. **Test your changes**
   - Ensure existing tests pass
   - Add tests for new functionality
   - Test on at least one of the supported datasets

4. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure CI tests pass

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and modular

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BEAT.git
cd BEAT

# Create virtual environment
conda create -n beat-dev python=3.8
conda activate beat-dev

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

## Documentation

- Update README.md if you add new features
- Add docstrings to all public functions/classes
- Update examples if the API changes

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to BEAT!
