# Tutorial: Getting Started with AI Training Memory System

## Complete Step-by-Step Guide

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [CLI Usage](#cli-usage)
6. [Visualization](#visualization)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Introduction

Welcome to the AI Training Memory System tutorial! This guide will walk you through everything you need to know to use the system effectively.

### What You'll Learn

- How to install and set up the system
- Basic problem solving and memory retrieval
- Training the system with examples
- Using the CLI interface
- Creating visualizations
- Best practices for optimal performance

---

## Installation

### Step 1: Prerequisites

Ensure you have Python 3.8 or higher:

```bash
python --version  # Should be 3.8+
```

### Step 2: Clone/Download

Navigate to the project directory:

```bash
cd ai-training-memory-system
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- sentence-transformers (for embeddings)
- torch (for neural networks)
- scikit-learn (for ML utilities)
- numpy, pandas (for data processing)
- matplotlib (for visualization)
- pytest (for testing)

### Step 4: Install Package

```bash
pip install -e .
```

### Step 5: Verify Installation

```python
python -c "from src.core.memory import AITrainingMemory; print('Success!')"
```

---

## Basic Usage

### Example 1: Your First Query

```python
from src.core.memory import AITrainingMemory

# Initialize the memory system
memory = AITrainingMemory()

# Solve a simple problem
result = memory.solve_problem("What is 5 plus 3?")

# Display the result
print(f"Answer: {result['answer']}")
print(f"Explanation: {result['explanation']}")
print(f"Method: {result['method']}")
print(f"Time: {result['time']:.4f} seconds")
```

**Output:**
```
Answer: 8.0
Explanation: Added 5.0 + 3.0 = 8.0
Method: hybrid_rule_primary
Time: 0.1234 seconds
```

### Example 2: Duplicate Detection

```python
# Ask the same question twice
result1 = memory.solve_problem("What is 10 plus 20?")
print(f"First time - Method: {result1['method']}, Time: {result1['time']:.4f}s")

result2 = memory.solve_problem("What is 10 plus 20?")
print(f"Second time - Method: {result2['method']}, Time: {result2['time']:.4f}s")
```

**Output:**
```
First time - Method: hybrid_rule_primary, Time: 0.1123s
Second time - Method: memory_hit, Time: 0.0023s
```

**Observation**: Second query is ~50x faster! ⚡

### Example 3: Semantic Similarity

```python
# Ask similar questions with different wordings
memory.solve_problem("What is 5 plus 3?")

# Try different phrasing
result = memory.solve_problem("Calculate 5 + 3")
print(f"Method: {result['method']}")  # Might be similarity_match

result = memory.solve_problem("Add 5 and 3")
print(f"Method: {result['method']}")  # Might be similarity_match
```

### Example 4: Different Operations

```python
# Test all operations
problems = {
    "addition": "What is 15 plus 25?",
    "subtraction": "Calculate 50 minus 18",
    "multiplication": "What is 7 times 8?",
    "division": "Divide 100 by 4"
}

for op, problem in problems.items():
    result = memory.solve_problem(problem)
    print(f"{op}: {problem} = {result['answer']}")
```

**Output:**
```
addition: What is 15 plus 25? = 40.0
subtraction: Calculate 50 minus 18 = 32.0
multiplication: What is 7 times 8? = 56.0
division: Divide 100 by 4 = 25.0
```

---

## Advanced Features

### Training the System

```python
# Define training examples
training_examples = [
    {
        'problem': 'What is 12 times 12?',
        'answer': 144,
        'operation': 'multiplication'
    },
    {
        'problem': 'Calculate 200 divided by 8',
        'answer': 25,
        'operation': 'division'
    },
    {
        'problem': 'Add 99 and 101',
        'answer': 200,
        'operation': 'addition'
    }
]

# Run training
memory.training_phase(training_examples)

print(f"Training cycles completed: {memory.training_cycles}")
print(f"Total memories: {len(memory.memory_documents)}")
```

### Viewing Memory

```python
# Display comprehensive memory statistics
memory.show_memory()
```

**Output:**
```
============================================================
AI TRAINING MEMORY SYSTEM - MEMORY STATUS
============================================================

📊 MEMORY STATISTICS:
  Total questions stored: 15
  Unique questions: 12
  Repeated questions: 3
  Vector dimension: 384D
  Estimated memory size: 0.06 MB

⚡ PERFORMANCE METRICS:
  Total queries: 15
  Memory hits: 3
  Hit rate: 20.00%
  Average solve time: 0.0542s

🧠 LEARNING PROGRESS:
  Training cycles: 1
  Neural network enabled: True
  Mistake patterns learned: 0

📝 SAMPLE PROBLEMS (last 5):
  • What is 12 times 12?... = 144 [2026-02-06 04:52:22]
  ...
```

### Similarity Search

```python
# Find similar problems
query = "What is 5 plus 3?"
similar_docs = memory.similarity_search(query, k=3, threshold=0.7)

for doc, similarity in similar_docs:
    problem = doc.metadata.get('problem', doc.page_content)
    solution = doc.metadata.get('solution')
    print(f"Similarity: {similarity:.2f} | {problem} = {solution}")
```

### Save and Load

```python
# Save memory to file
memory.save('data/saved_memories/my_session.pkl')
print("Memory saved!")

# Load in new session
new_memory = AITrainingMemory()
new_memory.load('data/saved_memories/my_session.pkl')
print(f"Loaded {len(new_memory.memory_documents)} memories")

# Verify it works
result = new_memory.solve_problem("What is 5 plus 3?")
print(f"Method: {result['method']}")  # Should be memory_hit
```

### Custom Configuration

```python
# Create memory with custom settings
custom_memory = AITrainingMemory(
    similarity_threshold=0.8,      # Stricter matching
    duplicate_check=True,          # Enable duplicate detection
    max_memory_entries=5000,       # Limit memory size
    enable_neural_network=False    # Disable NN for speed
)

# Use as normal
result = custom_memory.solve_problem("What is 2 + 2?")
```

---

## CLI Usage

### Starting the CLI

```bash
cd ai-training-memory-system
python -m src.interface.cli
```

### Interactive Session Example

```
============================================================
AI TRAINING MEMORY SYSTEM
============================================================

Type 'help' for available commands
Type 'quit' or 'exit' to exit

============================================================

💭 > solve What is 5 plus 3?

🔍 Solving: What is 5 plus 3?

✨ Answer: 8.0
📋 Explanation: Added 5.0 + 3.0 = 8.0
⚙️  Method: hybrid_rule_primary
🎯 Confidence: 1.00
⏱️  Time: 0.1234s

💭 > memory

============================================================
AI TRAINING MEMORY SYSTEM - MEMORY STATUS
============================================================
[Memory statistics displayed...]

💭 > train

🎓 Entering training mode...
✅ Training complete!
📊 Total training cycles: 1

💭 > stats

============================================================
PERFORMANCE STATISTICS
============================================================
[Performance report displayed...]

💭 > save my_memory.pkl
✅ Memory saved to data/saved_memories/my_memory.pkl

💭 > quit

👋 Goodbye!
```

### CLI Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `solve <problem>` | Solve a problem | `solve What is 5 + 3?` |
| `train` | Enter training mode | `train` |
| `memory` | Show memory contents | `memory` |
| `stats` | Display statistics | `stats` |
| `save [file]` | Save memory | `save my_mem.pkl` |
| `load <file>` | Load memory | `load my_mem.pkl` |
| `export <format>` | Export metrics | `export json` |
| `config [setting]` | Configure system | `config threshold 0.8` |
| `help` | Show help | `help` |
| `quit` | Exit | `quit` |

---

## Visualization

### Creating Performance Charts

```python
from src.interface.visualization import Visualizer

# Create visualizer
viz = Visualizer(memory)

# Individual plots
viz.plot_memory_growth(save_path='output/memory_growth.png')
viz.plot_solve_time_distribution(save_path='output/solve_times.png')
viz.plot_performance_metrics(save_path='output/metrics.png')

# Comprehensive dashboard
viz.create_dashboard(save_path='output/dashboard.png')
```

### Similarity Heatmap

```python
queries = [
    "What is 5 plus 3?",
    "Calculate 5 + 3",
    "Add 5 and 3",
    "What is 10 minus 2?"
]

viz.plot_similarity_heatmap(queries, save_path='output/heatmap.png')
```

---

## Troubleshooting

### Issue: Module Not Found

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the correct directory
cd ai-training-memory-system

# Install in editable mode
pip install -e .
```

### Issue: Slow Performance

**Problem**: Queries taking too long

**Solutions**:
1. Disable neural network:
   ```python
   memory = AITrainingMemory(enable_neural_network=False)
   ```

2. Reduce similarity threshold:
   ```python
   memory = AITrainingMemory(similarity_threshold=0.6)
   ```

3. Limit memory size:
   ```python
   memory = AITrainingMemory(max_memory_entries=1000)
   ```

### Issue: Out of Memory

**Problem**: System running out of RAM

**Solution**:
```python
# Reduce memory limit
memory = AITrainingMemory(max_memory_entries=500)

# Or save and clear periodically
memory.save('checkpoint.pkl')
memory = AITrainingMemory()  # Fresh start
```

### Issue: Model Download Fails

**Problem**: Can't download SentenceTransformer model

**Solution**:
```python
# Pre-download the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

---

## Best Practices

### 1. Regular Saving

```python
# Save after significant work
for i, problem in enumerate(problems):
    memory.solve_problem(problem)
    
    if i % 100 == 0:  # Every 100 problems
        memory.save(f'data/saved_memories/checkpoint_{i}.pkl')
```

### 2. Use Training Mode

```python
# Provide good training examples
memory.training_phase(high_quality_examples)

# Train periodically
if memory.total_queries % 500 == 0:
    memory.training_phase()
```

### 3. Monitor Performance

```python
# Check hit rate regularly
hit_rate = (memory.memory_hit_count / memory.total_queries) * 100
print(f"Current hit rate: {hit_rate:.1f}%")

if hit_rate < 20:
    print("Consider: More diverse training, adjust threshold")
```

### 4. Appropriate Thresholds

```python
# High precision (fewer false matches)
memory = AITrainingMemory(similarity_threshold=0.9)

# Balanced
memory = AITrainingMemory(similarity_threshold=0.7)

# High recall (more matches)
memory = AITrainingMemory(similarity_threshold=0.5)
```

### 5. Clean Data

```python
# Standardize input format
def clean_problem(text):
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Standardize punctuation
    text = text.replace('?', '').replace('.', '')
    return text

problem = clean_problem(user_input)
result = memory.solve_problem(problem)
```

---

## Next Steps

Now that you've completed the tutorial:

1. ✅ Try the demo scenarios: `python -m src.interface.demo`
2. ✅ Explore the API documentation: `docs/API.md`
3. ✅ Read the architecture guide: `docs/ARCHITECTURE.md`
4. ✅ Run the Jupyter notebooks: `notebooks/demonstration.ipynb`
5. ✅ Run the tests: `pytest tests/ -v`

---

## FAQ

**Q: How accurate is the system?**  
A: Rule-based solving is 95%+ accurate for standard math. Neural network improves with training.

**Q: Can I use it for other domains?**  
A: The architecture is generalizable. You'd need to adapt the solver components.

**Q: How much memory does it use?**  
A: Approximately 4KB per memory entry. 1000 entries ≈ 4MB.

**Q: Is it production-ready?**  
A: Yes, for appropriate scale. Consider database integration for >100K entries.

**Q: Can I contribute?**  
A: Absolutely! See the contributing guidelines in README.md.

---

**Tutorial Version**: 1.0.0  
**Last Updated**: 2026-02-06  
**Need Help?** Open an issue on GitHub
