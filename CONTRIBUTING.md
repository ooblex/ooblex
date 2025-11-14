# Contributing to Ooblex

Thank you for your interest in contributing to Ooblex! This document provides guidelines and instructions for contributing.

---

## 🎯 Our Philosophy

**Ooblex focuses on:**
- Ultra-low-latency real-time video processing
- Simple, clean, tested code
- Production-ready infrastructure
- No unnecessary complexity ("no AI slop")

**Before adding features, ask:**
1. Does it serve the core purpose (low-latency video AI)?
2. Is it actually needed, or just "nice to have"?
3. Can it be tested?
4. Is it documented?

**We prefer:**
- ✅ Simple solutions over complex ones
- ✅ Tested code over untested code
- ✅ Clear documentation over assumptions
- ✅ Performance over features

---

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ooblex.git
cd ooblex
git remote add upstream https://github.com/ooblex/ooblex.git
```

### 2. Set Up Development Environment
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Start services
docker compose up -d redis rabbitmq

# Run tests
pytest
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

---

## 📝 What We Need

### High Priority
- 🐛 **Bug fixes** - Fix issues, improve stability
- 📚 **Documentation** - Improve docs, add examples
- 🧪 **Tests** - Add tests, improve coverage
- ⚡ **Performance** - Optimize latency, throughput
- 🔒 **Security** - Fix vulnerabilities, improve security

### Medium Priority
- ✨ **Effects** - New OpenCV-based effects (no models)
- 🎨 **Examples** - Real-world use case examples
- 🐳 **Docker** - Improve Docker configs
- 📊 **Monitoring** - Better metrics, dashboards

### Please Discuss First
- 🎯 **Major features** - Create issue to discuss
- 🏗️ **Architecture changes** - Requires consensus
- 📦 **New dependencies** - Must be justified

---

## 💻 Development Guidelines

### Code Style

**Python:**
- PEP 8 compliant
- Use `black` for formatting
- Use `isort` for imports
- Use `flake8` for linting
- Type hints where helpful

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Run all checks
pre-commit run --all-files
```

**Naming:**
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

**Comments:**
- Write clear, concise comments
- Explain "why", not "what"
- Document complex algorithms
- Add docstrings to public functions

### Testing Requirements

**All code must have tests!**

```bash
# Run tests
pytest

# Run specific test
pytest tests/unit/test_core_services.py -v

# Run with coverage
pytest --cov=services --cov=code --cov-report=html

# Run benchmarks
pytest tests/benchmarks --benchmark-only
```

**Test types:**
1. **Unit tests** - Test individual functions (no external services)
2. **Integration tests** - Test with Redis/RabbitMQ
3. **E2E tests** - Test complete pipeline
4. **Benchmarks** - Performance validation

**Writing tests:**
```python
def test_my_feature():
    """Test that my feature works correctly"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = my_feature(input_data)
    
    # Assert
    assert result == expected_output
    assert result.is_valid()
```

### Documentation

**Every feature needs docs!**

- Update README.md if user-facing
- Add docstrings to functions
- Create examples in `examples/`
- Update relevant markdown files

**Docstring format:**
```python
def process_frame(frame: np.ndarray, effect: str) -> np.ndarray:
    """
    Apply an effect to a video frame.
    
    Args:
        frame: Input frame as BGR numpy array
        effect: Effect name (e.g., 'cartoon', 'blur')
    
    Returns:
        Processed frame as BGR numpy array
    
    Raises:
        ValueError: If effect is unknown
    
    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> result = process_frame(frame, 'cartoon')
    """
```

---

## 🎨 Adding New Effects

Want to add an image processing effect?

### 1. Add Effect Function
```python
# code/brain_simple.py

def effect_my_effect(image):
    """Your effect description"""
    # Your OpenCV code here
    processed = cv2.someOperation(image)
    return processed
```

### 2. Register Effect
```python
# In apply_effect() function
elif effect_name == "my_effect":
    return effect_my_effect(image)
```

### 3. Add Tests
```python
# tests/unit/test_effects.py

def test_my_effect():
    """Test my_effect works correctly"""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = effect_my_effect(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
```

### 4. Document It
```markdown
# In README.md, add to effects table:
| **My Effect** | ~XX FPS | Description here |
```

### 5. Benchmark It
```python
# tests/benchmarks/test_performance.py

def test_my_effect_performance(benchmark):
    """Benchmark my_effect"""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = benchmark(effect_my_effect, frame)
```

---

## 🔧 Adding ML Models

Want to integrate an AI model?

### 1. Create Model Worker
```python
# code/my_model_worker.py
# Use tensorthread_shell.py as template
```

### 2. Add Model Documentation
```markdown
# models/my_model/README.md
- Model description
- Download instructions
- Performance benchmarks
- Usage examples
```

### 3. Test Without Model
```python
# Mock the model for testing
def test_my_model_worker():
    with patch('my_model.load') as mock_load:
        mock_load.return_value = MockModel()
        # Test worker logic
```

### 4. Add to models/README.md
```markdown
### My Model
pip install my-framework
# Download instructions
# Example usage
```

---

## 📋 Pull Request Process

### 1. Before Submitting

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Added tests for new code
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear

```bash
# Final check before submitting
black .
isort .
flake8 .
pytest
```

### 2. PR Template

Use this template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manually tested
- [ ] Benchmarks run

## Performance Impact
- Latency: +/- X ms
- Throughput: +/- X FPS
- Memory: +/- X MB

## Documentation
- [ ] README updated
- [ ] Docstrings added
- [ ] Examples added

## Screenshots
(if applicable)
```

### 3. Review Process

**What we look for:**
- Code quality and style
- Test coverage
- Performance impact
- Documentation completeness
- Backwards compatibility

**Timeline:**
- Initial review: 1-3 days
- Follow-up reviews: 1-2 days
- Merge: After approval + CI passes

---

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
1. Run command X
2. Do action Y
3. See error Z

**Expected behavior**
What should have happened

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Ooblex version: [e.g., commit hash or tag]
- Docker version: [e.g., 24.0.0]

**Logs**
```
Paste relevant logs here
```

**Additional context**
Any other relevant information
```

---

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this needed? What problem does it solve?

**Proposed Solution**
How would you implement this?

**Alternatives Considered**
Other approaches you've considered

**Performance Impact**
Expected latency/throughput impact

**Additional Context**
Any other relevant information
```

**Before requesting:**
1. Search existing issues
2. Check if it aligns with core purpose
3. Consider if it can be a plugin/extension

---

## 📊 Performance Guidelines

**All changes should:**
- Maintain sub-400ms end-to-end latency
- Not reduce throughput significantly
- Be benchmarked

**Testing performance:**
```bash
# Benchmark before changes
pytest tests/benchmarks --benchmark-only --benchmark-save=before

# Make changes

# Benchmark after changes
pytest tests/benchmarks --benchmark-only --benchmark-save=after

# Compare
pytest-benchmark compare before after
```

**Optimization priorities:**
1. Algorithmic improvements
2. Reduce allocations
3. Vectorization (NumPy)
4. Caching
5. Parallel processing

---

## 🔒 Security

**Reporting vulnerabilities:**
- DO NOT create public issues
- Email: security@ooblex.com (if available)
- Or use GitHub Security Advisories

**Security checklist:**
- No hardcoded secrets
- Input validation
- SQL injection prevention
- XSS prevention
- CSRF protection
- Secure defaults

---

## 📜 Code of Conduct

**Be respectful:**
- Welcome newcomers
- Be patient with questions
- Provide constructive feedback
- No harassment or discrimination
- Keep discussions professional

**Focus on:**
- Technical merit
- Project goals
- User benefit
- Code quality

---

## 🎓 Learning Resources

**Understanding Ooblex:**
- [README.md](README.md) - Overview
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - Architecture
- [QUICKSTART.md](QUICKSTART.md) - Getting started
- [models/README.md](models/README.md) - Model integration

**Technologies:**
- [OpenCV Documentation](https://docs.opencv.org/)
- [Redis Documentation](https://redis.io/docs/)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials)
- [WebRTC](https://webrtc.org/getting-started/overview)

---

## ❓ Questions?

- **General questions**: [GitHub Discussions](https://github.com/ooblex/ooblex/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/ooblex/ooblex/issues)
- **Feature requests**: [GitHub Issues](https://github.com/ooblex/ooblex/issues)

---

## 🙏 Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- Project README

**Thank you for contributing to Ooblex!**
