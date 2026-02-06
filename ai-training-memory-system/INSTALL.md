# Installation and Setup Guide

## Quick Start

### 1. System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for dependencies and models
- **Operating System**: Windows, macOS, or Linux

### 2. Installation Steps

#### Step 1: Navigate to Project Directory

```bash
cd ai-training-memory-system
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` (for text embeddings)
- `torch` (for neural networks)
- `scikit-learn` (for ML utilities)
- `numpy` (for numerical operations)
- `pandas` (for data handling)
- `matplotlib` (for visualization)
- `pytest` (for testing)

**Note**: First installation may take 5-10 minutes as it downloads the SentenceTransformer model (~100MB).

#### Step 4: Install Package

```bash
pip install -e .
```

#### Step 5: Verify Installation

```bash
python verify_installation.py
```

Expected output:
```
✅ All checks passed! System is ready to use.
```

### 3. First Run

#### Option A: Run Example Script

```bash
python example.py
```

This will demonstrate:
- Basic problem solving
- Duplicate detection
- Different operations
- Semantic similarity
- Training
- Memory statistics

#### Option B: Interactive CLI

```bash
python main.py cli
```

Try these commands:
```
solve What is 5 plus 3?
memory
train
stats
quit
```

#### Option C: Comprehensive Demos

```bash
python main.py demo
```

This runs all 4 demonstration scenarios.

### 4. Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/demonstration.ipynb
```

## Troubleshooting

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: Model Download Fails

**Error**: Can't download SentenceTransformer model

**Solution**:
```bash
# Pre-download model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: Memory Errors

**Error**: Out of memory when running

**Solution**:
```python
# Use smaller memory limit
memory = AITrainingMemory(max_memory_entries=1000)
```

### Issue: Slow Performance

**Solution 1**: Disable neural network
```python
memory = AITrainingMemory(enable_neural_network=False)
```

**Solution 2**: Use faster threshold
```python
memory = AITrainingMemory(similarity_threshold=0.6)
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_memory.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report:
```bash
# Open htmlcov/index.html in browser
```

## Development Setup

### For Contributors

1. Fork the repository
2. Clone your fork
3. Create a branch for your feature
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```
5. Make your changes
6. Run tests: `pytest tests/ -v`
7. Ensure code quality: `pylint src/`
8. Submit a pull request

### Code Style

Follow PEP 8:
```bash
# Check style
pylint src/

# Format code
black src/
```

## Environment Variables

Optional configuration via environment variables:

```bash
# Set model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Set device for PyTorch
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

## Docker (Optional)

Build and run in Docker:

```bash
# Build image
docker build -t ai-memory-system .

# Run container
docker run -it ai-memory-system python example.py
```

## Next Steps

After successful installation:

1. 📖 Read the [Tutorial](docs/TUTORIAL.md)
2. 🔍 Explore the [API Documentation](docs/API.md)
3. 🏗️ Understand the [Architecture](docs/ARCHITECTURE.md)
4. 📓 Try the [Jupyter Notebook](notebooks/demonstration.ipynb)
5. 🧪 Run the [Tests](tests/)

## Getting Help

- 📚 Check documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Read FAQ in [Tutorial](docs/TUTORIAL.md)

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
pip install -e . --upgrade
```

---

**Happy Learning! 🚀**
