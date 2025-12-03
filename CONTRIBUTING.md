# Contributing to Siren AI v2

First off, thank you for considering contributing to Siren AI v2! It's people like you that make this project a success in wildlife conservation efforts.

## 🌟 Welcome!

Siren AI v2 is an open-source project aimed at reducing human-elephant conflict through intelligent technology. We welcome contributions of all kinds: code, documentation, bug reports, feature requests, and more.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When reporting a bug, include:**

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Code samples** (if applicable)
- **Screenshots** (if applicable)

**Template:**

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.8.10]
- Siren AI version: [e.g. v2.0.1]

**Additional Context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**When suggesting an enhancement:**

- **Use a clear title**
- **Provide detailed description**
- **Explain why this would be useful**
- **Include examples** (if possible)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Submit a pull request**

## 🛠️ Development Setup

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/siren-ai-v2.git
cd siren-ai-v2

# Add upstream remote
git remote add upstream https://github.com/originalowner/siren-ai-v2.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_sarsa_engine.py -v

# Run with markers
pytest tests/ -m "not slow"
```

### Code Quality Checks

```bash
# Format code
black .

# Sort imports
isort .

# Lint
flake8 .
pylint src/

# Type checking
mypy src/
```

## 🎨 Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort

### Example

```python
"""
Module docstring explaining purpose.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from config import SARSA_CONFIG
from safety_security import SafetyWrapper


class ExampleClass:
    """
    Class docstring with clear description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    """
    
    def __init__(self, param1: int, param2: str) -> None:
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self, input_value: float) -> Optional[List[float]]:
        """
        Method docstring with Args, Returns, Raises.
        
        Args:
            input_value: Description
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: When input is invalid
        """
        if input_value < 0:
            raise ValueError("Input must be non-negative")
        
        result = [input_value * 2]
        return result
```

### Documentation Style

- Use **Google-style docstrings**
- Include **type hints**
- Document all **public APIs**
- Add **usage examples** where helpful

## 📝 Commit Messages

We follow the **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(sarsa): add eligibility trace decay parameter

Add configurable decay rate for eligibility traces to improve
learning stability in sparse reward scenarios.

Closes #42

fix(safety): correct aggression override logic

The aggression flag was not properly triggering M3 escalation
when cooldown was active. Fixed priority order.

Fixes #58

docs(readme): update installation instructions

Add section for ESP32-S3 firmware installation and clarify
Python version requirements.

test(safety): add scenario tests for lockout mechanism

Add 5 new test cases covering edge cases in lockout timer
activation and expiry.
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass

## Related Issues
Closes #(issue_number)
```

### Review Process

1. **Automated checks** must pass
2. **At least one approval** required
3. **All comments** must be resolved
4. **No merge conflicts**

### After Approval

- Maintainer will merge your PR
- Your contribution will be acknowledged
- Branch will be deleted

## 🧪 Testing Guidelines

### Test Coverage Requirements

- **Minimum 80% overall coverage**
- **100% for safety-critical code**
- **All new features must have tests**

### Test Structure

```python
import pytest
from sarsa_engine import SARSALambdaAgent

class TestSARSAAgent:
    """Test suite for SARSA agent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return SARSALambdaAgent(num_actions=4, seed=42)
    
    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.num_actions == 4
        assert len(agent.q_table) == 0
    
    def test_action_selection(self, agent):
        """Test action selection mechanism."""
        state = "test_state"
        action = agent.choose_action(state)
        assert 0 <= action < 4
```

### Running Specific Tests

```bash
# Run by marker
pytest -m "safety"

# Run by keyword
pytest -k "aggression"

# Run failed tests only
pytest --lf

# Run in parallel
pytest -n auto
```

## 📚 Documentation

### Code Documentation

- **All functions** need docstrings
- **Complex logic** needs inline comments
- **Type hints** for function signatures

### Project Documentation

Update relevant docs in `docs/` directory:

- **API changes** → Update API_REFERENCE.md
- **New features** → Update appropriate guide
- **Configuration** → Update CONFIG.md

### Building Docs Locally

```bash
cd docs/
make html
# Open _build/html/index.html
```

## 🎯 Areas for Contribution

### High Priority

- **Bug fixes** in safety mechanisms
- **Test coverage** improvements
- **Documentation** enhancements
- **Performance** optimizations

### Medium Priority

- **New deterrent strategies**
- **Additional visualizations**
- **Configuration presets**
- **Edge device optimizations**

### Low Priority

- **Code refactoring**
- **Style improvements**
- **Example notebooks**
- **Tutorial content**

## 🏆 Recognition

Contributors will be:

- **Listed** in CONTRIBUTORS.md
- **Mentioned** in release notes
- **Thanked** in project documentation

## 💬 Questions?

- **GitHub Discussions**: For general questions
- **Issues**: For bug reports and feature requests
- **Email**: For private inquiries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to wildlife conservation! 🦣🌍**
