# API Documentation

## AI Training Memory System - Complete API Reference

---

## Core Components

### SimpleDocument

Primary data container for memory storage.

#### Constructor

```python
SimpleDocument(page_content: str, metadata: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `page_content` (str): Original text content
- `metadata` (dict, optional): Structured metadata dictionary

**Metadata Schema:**
- `type` (str): Document type ('training', 'user_query', 'system')
- `problem` (str): Original problem statement
- `solution` (any): Computed solution
- `explanation` (str): Solution derivation method
- `timestamp` (datetime): Creation timestamp
- `operation` (str): Mathematical operation used
- `confidence` (float): Solution confidence (0.0-1.0)
- `similar_to` (list): Related problem references

#### Methods

##### `to_dict() -> Dict[str, Any]`
Convert document to dictionary format.

**Returns:** Dictionary with 'page_content' and 'metadata' keys

##### `from_dict(data: Dict[str, Any]) -> SimpleDocument`
Create document from dictionary format (class method).

**Parameters:**
- `data` (dict): Dictionary with document data

**Returns:** SimpleDocument instance

#### Example

```python
doc = SimpleDocument(
    page_content="What is 5 plus 3?",
    metadata={
        'type': 'user_query',
        'solution': 8,
        'operation': 'addition',
        'confidence': 1.0
    }
)

# Serialize
doc_dict = doc.to_dict()

# Deserialize
doc2 = SimpleDocument.from_dict(doc_dict)
```

---

### TextEmbedder

Text embedder using SentenceTransformer for vector generation.

#### Constructor

```python
TextEmbedder(model_name: str = 'all-MiniLM-L6-v2')
```

**Parameters:**
- `model_name` (str): Name of SentenceTransformer model (default: 'all-MiniLM-L6-v2')

**Attributes:**
- `vector_dimension` (int): 384 (fixed for default model)
- `model`: SentenceTransformer instance

#### Methods

##### `embed(text: Union[str, List[str]]) -> np.ndarray`
Convert text to vector embedding(s).

**Parameters:**
- `text` (str or list): Single text string or list of strings

**Returns:** numpy array of shape (384,) for single text or (n, 384) for list

##### `embed_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`
Convert batch of texts efficiently.

**Parameters:**
- `texts` (list): List of text strings
- `batch_size` (int): Batch size for encoding (default: 32)

**Returns:** numpy array of shape (n, 384)

#### Example

```python
embedder = TextEmbedder()

# Single text
embedding = embedder.embed("What is 5 plus 3?")
print(embedding.shape)  # (384,)

# Batch
embeddings = embedder.embed_batch(["Question 1", "Question 2"])
print(embeddings.shape)  # (2, 384)
```

---

### AITrainingMemory

Main memory system with persistent vector storage and semantic retrieval.

#### Constructor

```python
AITrainingMemory(
    similarity_threshold: float = 0.7,
    duplicate_check: bool = True,
    max_memory_entries: int = 10000,
    enable_neural_network: bool = True
)
```

**Parameters:**
- `similarity_threshold` (float): Minimum similarity for matches (0.0-1.0)
- `duplicate_check` (bool): Enable duplicate question detection
- `max_memory_entries` (int): Maximum number of memories to store
- `enable_neural_network` (bool): Enable neural network solver

**Attributes:**
- `memory_documents` (List[SimpleDocument]): Text documents with metadata
- `memory_embeddings` (List[np.ndarray]): Parallel 384D vector storage
- `vector_dimension` (int): 384
- `question_history` (defaultdict): Hash → history entries
- `user_questions` (List[dict]): Complete user interaction log
- `mistake_patterns_learned` (Set[str]): Accumulated error patterns
- `training_cycles` (int): Number of training cycles completed
- `total_queries` (int): Total queries processed
- `memory_hit_count` (int): Successful memory retrievals
- `performance_tracker` (PerformanceTracker): Metrics tracking

#### Methods

##### `add_to_memory(document: SimpleDocument)`
Add document to memory with vector embedding.

**Parameters:**
- `document` (SimpleDocument): Document to store

**Side Effects:**
- Generates 384D embedding
- Stores in memory_documents and memory_embeddings
- Updates question_history
- Enforces max_memory_entries limit

##### `similarity_search(query: str, k: int = 2, threshold: float = None) -> List[Tuple[SimpleDocument, float]]`
Search for similar documents using cosine similarity.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of results to return (default: 2)
- `threshold` (float): Similarity threshold (uses default if None)

**Returns:** List of (document, similarity_score) tuples, sorted by similarity

##### `check_duplicate_question(question: str) -> Tuple[bool, List[dict]]`
Check if question has been asked before.

**Parameters:**
- `question` (str): Question text

**Returns:** Tuple of (is_duplicate, history_entries)

##### `solve_problem(problem: str) -> Dict[str, Any]`
Solve mathematical problem using multi-phase approach.

**Parameters:**
- `problem` (str): Mathematical problem text

**Returns:** Dictionary with keys:
- `problem` (str): Original problem
- `answer` (float/None): Computed answer
- `explanation` (str): Solution explanation
- `method` (str): Method used ('memory_hit', 'similarity_match', 'hybrid', etc.)
- `time` (float): Solve time in seconds
- `confidence` (float): Confidence score (0.0-1.0)
- `numbers_extracted` (List[float]): Numbers found in problem
- `operation` (str): Identified operation

**Phases:**
1. Check for exact duplicates (MD5 hash)
2. Similarity search for similar problems
3. Rule-based solving
4. Neural network enhancement
5. Store solution and return

##### `memory_guided_solving(user_problem: str) -> Dict[str, Any]`
Enhanced problem solving using memory insights.

**Parameters:**
- `user_problem` (str): User's problem text

**Returns:** Dictionary with solution and similar_problems list

##### `training_phase(training_examples: List[Dict[str, Any]] = None)`
Load training examples and learn from mistakes.

**Parameters:**
- `training_examples` (list, optional): List of training example dictionaries
  - Each dict should have: 'problem', 'answer', 'operation'

**Side Effects:**
- Increments training_cycles
- Stores training examples in memory
- Trains neural network if enabled

##### `show_memory()`
Display memory contents and statistics.

**Output:** Prints formatted memory status to console

##### `interactive_demonstration()`
Interactive demonstration interface.

**Usage:** Provides command-line interface for interaction

##### `save(filepath: str)`
Save memory system to file.

**Parameters:**
- `filepath` (str): Path to save file (.pkl)

**Side Effects:**
- Saves memory data with pickle
- Saves neural network weights if trained

##### `load(filepath: str)`
Load memory system from file.

**Parameters:**
- `filepath` (str): Path to load file (.pkl)

**Side Effects:**
- Restores all memory data
- Loads neural network weights if available

#### Example

```python
# Initialize
memory = AITrainingMemory(similarity_threshold=0.8)

# Solve problems
result = memory.solve_problem("What is 5 plus 3?")
print(result['answer'])  # 8

# Check duplicate
is_dup, history = memory.check_duplicate_question("What is 5 plus 3?")
print(is_dup)  # True

# Search similar
similar = memory.similarity_search("Calculate 5 + 3", k=3)

# Train
memory.training_phase([
    {'problem': 'What is 7 * 8?', 'answer': 56, 'operation': 'multiplication'}
])

# Save/Load
memory.save('data/saved_memories/my_memory.pkl')
memory2 = AITrainingMemory()
memory2.load('data/saved_memories/my_memory.pkl')
```

---

## Solver Components

### Rule-Based Solver

#### Functions

##### `extract_numbers(text: str) -> List[float]`
Extract all numbers from text using regex.

**Parameters:**
- `text` (str): Input text

**Returns:** List of extracted numbers as floats

##### `identify_operation(text: str) -> str`
Identify mathematical operation from keywords.

**Parameters:**
- `text` (str): Input text

**Returns:** Operation name ('addition', 'subtraction', 'multiplication', 'division', 'unknown')

##### `solve_with_rules(problem: str) -> Dict[str, Any]`
Main rule-based solving function.

**Parameters:**
- `problem` (str): Mathematical problem text

**Returns:** Dictionary with:
- `numbers_extracted` (list): Numbers found
- `operation` (str): Identified operation
- `answer` (float/None): Computed answer
- `explanation` (str): Solution explanation
- `success` (bool): Whether solving succeeded

#### Example

```python
from src.solvers.rule_based import solve_with_rules

result = solve_with_rules("What is 5 plus 3?")
print(result['answer'])      # 8
print(result['operation'])   # 'addition'
print(result['success'])     # True
```

---

### Neural Network Solver

#### NeuralNetworkSolver

##### Constructor

```python
NeuralNetworkSolver(input_dim: int = 384, learning_rate: float = 0.001)
```

**Parameters:**
- `input_dim` (int): Dimension of input vectors (default: 384)
- `learning_rate` (float): Learning rate for optimizer (default: 0.001)

##### Methods

###### `train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) -> dict`
Train the neural network.

**Parameters:**
- `X_train` (np.ndarray): Training embeddings of shape (n_samples, 384)
- `y_train` (np.ndarray): Training labels of shape (n_samples,)
- `epochs` (int): Number of training epochs (default: 10)
- `batch_size` (int): Batch size (default: 32)
- `validation_split` (float): Fraction for validation (default: 0.2)

**Returns:** Dictionary with training history:
- `train_loss` (list): Training loss per epoch
- `val_loss` (list): Validation loss per epoch

###### `predict(X: np.ndarray) -> np.ndarray`
Make predictions.

**Parameters:**
- `X` (np.ndarray): Input embeddings of shape (n_samples, 384)

**Returns:** Predictions of shape (n_samples,)

###### `save_model(filepath: str)` / `load_model(filepath: str)`
Save/load model weights.

#### Example

```python
from src.solvers.neural_network import NeuralNetworkSolver
import numpy as np

solver = NeuralNetworkSolver()

# Train
X_train = np.random.randn(100, 384)
y_train = np.random.randn(100)
history = solver.train(X_train, y_train, epochs=5)

# Predict
X_test = np.random.randn(10, 384)
predictions = solver.predict(X_test)
```

---

## Utility Components

### Text Processing

#### Functions

- `normalize_text(text: str) -> str` - Lowercase and remove extra whitespace
- `clean_text_for_hash(text: str) -> str` - Clean text for consistent hashing
- `standardize_punctuation(text: str) -> str` - Standardize punctuation
- `remove_stop_words(text: str) -> str` - Remove common stop words

### Similarity Calculations

#### Functions

- `cosine_similarity_calc(vec1, vec2) -> float` - Calculate cosine similarity (0.0-1.0)
- `cosine_similarity_batch(query_vec, stored_vecs) -> np.ndarray` - Batch calculation
- `find_similar_vectors(query_vec, stored_vecs, threshold) -> List[Tuple]` - Find vectors above threshold
- `rank_by_similarity(similarities: List) -> List` - Sort by similarity score

### Performance Tracking

#### PerformanceTracker

##### Methods

- `record_query(method, solve_time, success, ...)` - Record query execution
- `calculate_hit_rate() -> float` - Calculate memory hit rate percentage
- `generate_report() -> dict` - Generate comprehensive performance report
- `export_metrics(filepath, format='json')` - Export metrics to file
- `reset()` - Reset all metrics

---

## Interface Components

### CLI

Command-line interface for the memory system.

#### Usage

```bash
python -m src.interface.cli
```

#### Commands

- `solve <problem>` - Solve mathematical problem
- `train` - Enter training mode
- `memory` - Show memory contents
- `stats` - Display performance statistics
- `clear` - Clear current session data
- `export <format> [filename]` - Export data (json/txt)
- `config [setting] [value]` - View or change configuration
- `save [filename]` - Save memory to file
- `load <filename>` - Load memory from file
- `help` - Show help message
- `quit/exit` - Exit the system

### Demo Scenarios

Run comprehensive demonstrations:

```python
from src.interface.demo import DemoScenarios

demo = DemoScenarios()
demo.run_all_demos()

# Or run individual demos
demo.demo_1_basic_memory()
demo.demo_2_semantic_similarity()
demo.demo_3_learning_from_mistakes()
demo.demo_4_complex_problem_solving()
```

### Visualization

Create performance visualizations:

```python
from src.interface.visualization import Visualizer

viz = Visualizer(memory_system)

# Individual plots
viz.plot_memory_growth(save_path='memory_growth.png')
viz.plot_solve_time_distribution(save_path='solve_times.png')
viz.plot_performance_metrics(save_path='metrics.png')
viz.plot_similarity_heatmap(queries, save_path='heatmap.png')

# Comprehensive dashboard
viz.create_dashboard(save_path='dashboard.png')
```

---

## Error Handling

All components include comprehensive error handling:

- Invalid inputs return graceful error messages
- Division by zero handled appropriately
- Missing data handled with defaults
- File I/O errors caught and reported

---

## Version Information

**API Version**: 1.0.0  
**Last Updated**: 2026-02-06  
**Compatibility**: Python 3.8+
