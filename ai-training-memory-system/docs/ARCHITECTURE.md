# Architecture Documentation

## AI Training Memory System - System Architecture

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Three-Layer Architecture](#three-layer-architecture)
3. [Component Interactions](#component-interactions)
4. [Data Flow](#data-flow)
5. [Memory Management](#memory-management)
6. [Vector Storage](#vector-storage)
7. [Solving Pipeline](#solving-pipeline)
8. [Design Decisions](#design-decisions)
9. [Performance Considerations](#performance-considerations)
10. [Scalability](#scalability)

---

## System Overview

The AI Training Memory System implements a memory-augmented architecture that enables continuous learning through persistent vector storage and semantic retrieval.

### Core Principles

1. **Stateful Learning**: Maintain persistent memory across sessions
2. **Vector-Based Retrieval**: Use embeddings for semantic matching
3. **Hybrid Solving**: Combine rule-based and neural approaches
4. **Performance Tracking**: Comprehensive metrics collection

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│          Interface Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   CLI    │  │   Demo   │  │  Visual  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          Core Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Document │  │ Embedder │  │  Memory  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          Solver Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   Rule   │  │  Neural  │  │  Hybrid  │ │
│  │  Based   │  │  Network │  │          │ │
│  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

### Layer 1: Interface Layer

**Purpose**: User interaction and visualization

**Components**:
- **CLI**: Command-line interface for interactive usage
- **Demo**: Pre-built demonstration scenarios
- **Visualization**: Performance charts and graphs

**Responsibilities**:
- Handle user input
- Display results
- Generate visualizations
- Manage demonstrations

### Layer 2: Core Layer

**Purpose**: Memory management and vector operations

**Components**:
- **SimpleDocument**: Data container class
- **TextEmbedder**: SentenceTransformer wrapper
- **AITrainingMemory**: Main memory system

**Responsibilities**:
- Store documents and embeddings
- Manage vector database
- Perform similarity searches
- Track history and metrics

### Layer 3: Solver Layer

**Purpose**: Problem solving and computation

**Components**:
- **RuleBasedSolver**: Mathematical rule engine
- **NeuralNetworkSolver**: Deep learning model
- **HybridSolver**: Combined approach

**Responsibilities**:
- Extract numbers and operations
- Apply mathematical rules
- Use neural network predictions
- Select optimal solving method

---

## Component Interactions

```
┌────────────────────────────────────────────────────┐
│                   User Query                       │
└────────────────┬───────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────┐
│            AITrainingMemory                        │
│  ┌──────────────────────────────────────────────┐ │
│  │  1. Hash Generation (MD5)                    │ │
│  │     ↓                                         │ │
│  │  2. Duplicate Check                          │ │
│  │     ↓ (if not duplicate)                     │ │
│  │  3. Text → Embedding (TextEmbedder)          │ │
│  │     ↓                                         │ │
│  │  4. Similarity Search (Cosine)               │ │
│  │     ↓ (if low similarity)                    │ │
│  │  5. Hybrid Solver                            │ │
│  │     ├─ Rule-Based Solver                     │ │
│  │     └─ Neural Network (optional)             │ │
│  │     ↓                                         │ │
│  │  6. Store Result                             │ │
│  │     ↓                                         │ │
│  │  7. Track Performance                        │ │
│  └──────────────────────────────────────────────┘ │
└────────────────┬───────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────┐
│                   Result                           │
└────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Query Processing Flow

```
User Query
    │
    ├─→ Text Normalization
    │       └─→ MD5 Hash Generation
    │
    ├─→ Duplicate Check
    │   ├─→ [FOUND] Return from Memory ⚡ (Fast Path)
    │   └─→ [NOT FOUND] Continue
    │
    ├─→ Text Embedding (384D)
    │       └─→ SentenceTransformer
    │
    ├─→ Similarity Search
    │   ├─→ Cosine Similarity Calculation
    │   ├─→ Threshold Filtering
    │   └─→ [HIGH MATCH] Return Similar 🔍 (Medium Path)
    │
    ├─→ Problem Solving
    │   ├─→ Rule-Based Solver
    │   │   ├─→ Number Extraction (Regex)
    │   │   ├─→ Operation Identification
    │   │   └─→ Mathematical Computation
    │   │
    │   └─→ Neural Network (if enabled)
    │       ├─→ Embedding Input
    │       ├─→ Forward Pass
    │       └─→ Prediction Output
    │
    ├─→ Result Storage
    │   ├─→ Create Document
    │   ├─→ Generate Embedding
    │   ├─→ Store in Memory
    │   └─→ Update History
    │
    └─→ Performance Tracking
        ├─→ Record Time
        ├─→ Update Metrics
        └─→ Log Query
```

### 2. Training Flow

```
Training Examples
    │
    ├─→ For Each Example:
    │   ├─→ Create Document
    │   ├─→ Generate Embedding
    │   └─→ Add to Memory
    │
    ├─→ Prepare Training Data
    │   ├─→ Collect Embeddings (X)
    │   ├─→ Collect Solutions (y)
    │   └─→ Filter Valid Data
    │
    ├─→ Neural Network Training
    │   ├─→ Split Train/Val
    │   ├─→ Mini-batch Training
    │   ├─→ Loss Calculation
    │   └─→ Weight Updates
    │
    └─→ Update System
        ├─→ Increment Training Cycles
        └─→ Update Hybrid Solver
```

---

## Memory Management

### Storage Structure

```python
AITrainingMemory:
    memory_documents: List[SimpleDocument]
        [0]: Document("What is 5+3?", {solution: 8, ...})
        [1]: Document("Calculate 10-4", {solution: 6, ...})
        [2]: Document("What is 7*6?", {solution: 42, ...})
        ...
    
    memory_embeddings: List[np.ndarray]
        [0]: array([0.12, -0.34, ..., 0.56])  # 384D
        [1]: array([0.23, 0.11, ..., -0.22])  # 384D
        [2]: array([-0.45, 0.67, ..., 0.89])  # 384D
        ...
    
    question_history: defaultdict(list)
        "a3f2b8c..." → [
            {timestamp: ..., solution: 8, index: 0},
            {timestamp: ..., solution: 8, index: 15}
        ]
```

### Memory Limit Enforcement

```python
if len(memory_documents) >= max_memory_entries:
    # FIFO eviction
    memory_documents.pop(0)
    memory_embeddings.pop(0)
```

### Parallel Storage Guarantee

**Critical Invariant**: 
```
len(memory_documents) == len(memory_embeddings)
memory_documents[i] ↔ memory_embeddings[i]
```

This 1:1 correspondence is maintained at all times.

---

## Vector Storage

### Embedding Generation

```
Text: "What is 5 plus 3?"
    ↓
[Tokenization]
    ↓
Tokens: ["what", "is", "5", "plus", "3", "?"]
    ↓
[SentenceTransformer Processing]
    ↓
Embedding: [0.123, -0.456, 0.789, ..., 0.321]  # 384 dimensions
```

### Similarity Calculation

```python
# Cosine Similarity Formula:
similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Batch Optimization:
# Instead of loop:
for vec in stored_vecs:
    sim = cosine_similarity(query, vec)

# Use vectorized operation:
similarities = dot(query, stored_vecs.T) / norms
```

### Similarity Threshold

```
Threshold = 0.7 (default)

0.9 - 1.0: Nearly identical (high confidence retrieval)
0.7 - 0.9: Similar (moderate confidence retrieval)
0.5 - 0.7: Somewhat related (low confidence)
< 0.5:     Different (compute new solution)
```

---

## Solving Pipeline

### Multi-Phase Approach

#### Phase 1: Instant Retrieval (Duplicate Check)
- **Method**: MD5 hash comparison
- **Speed**: ~0.001s
- **Accuracy**: 100% for exact matches

#### Phase 2: Semantic Retrieval (Similarity Search)
- **Method**: Cosine similarity
- **Speed**: ~0.01s per 1000 vectors
- **Accuracy**: High for similar wordings

#### Phase 3: Rule-Based Solving
- **Method**: Regex + operation rules
- **Speed**: ~0.1s
- **Accuracy**: 95%+ for standard math

#### Phase 4: Neural Network Enhancement
- **Method**: Feedforward NN
- **Speed**: ~0.05s
- **Accuracy**: Improves with training

### Decision Tree

```
Is exact duplicate?
├─ YES → Return from memory (Phase 1) ⚡
└─ NO → Check similarity
    ├─ High (≥0.9) → Return similar (Phase 2) 🔍
    └─ Low (<0.9) → Compute solution
        ├─ Rule-based solve (Phase 3) 📐
        ├─ Neural network enhance (Phase 4) 🧠
        └─ Store & return
```

---

## Design Decisions

### 1. Why 384 Dimensions?

**Rationale**: 
- Balance between expressiveness and efficiency
- Standard for `all-MiniLM-L6-v2` model
- Proven effective for semantic similarity
- Fits in memory for large datasets

### 2. Why MD5 Hashing?

**Rationale**:
- Fast computation (microseconds)
- Collision probability negligible for text
- Simple implementation
- Standard library support

### 3. Why Hybrid Approach?

**Rationale**:
- Rule-based: Reliable, explainable, fast
- Neural network: Learns patterns, handles edge cases
- Combination: Best of both worlds

### 4. Why In-Memory Storage?

**Rationale**:
- Fast access (no disk I/O)
- Simple implementation
- Sufficient for typical use cases
- Save/load for persistence

### 5. Why Cosine Similarity?

**Rationale**:
- Scale-invariant
- Normalized to [0, 1]
- Standard for embedding comparison
- Efficient batch computation

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Add to memory | O(d) | d=384, embedding generation |
| Duplicate check | O(1) | Hash table lookup |
| Similarity search | O(n·d) | n=documents, d=384 |
| Rule-based solve | O(m) | m=text length |
| NN forward pass | O(d·h) | h=hidden layer sizes |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Document | ~1KB | Text + metadata |
| Embedding | 3KB | 384 × 8 bytes |
| History entry | ~100B | Timestamp + refs |
| **Per memory** | **~4KB** | Total per entry |
| **1000 memories** | **~4MB** | Typical usage |

### Optimization Strategies

1. **Batch Operations**: Process multiple embeddings together
2. **Numpy Vectorization**: Use numpy for similarity calculations
3. **Lazy Loading**: Load neural network only if needed
4. **Memory Limits**: Cap total entries to prevent bloat
5. **FIFO Eviction**: Remove oldest entries when full

---

## Scalability

### Horizontal Scaling

```
┌─────────────┐
│   User 1    │ → Memory Instance 1
└─────────────┘
┌─────────────┐
│   User 2    │ → Memory Instance 2
└─────────────┘
┌─────────────┐
│   User 3    │ → Memory Instance 3
└─────────────┘
```

Each user gets isolated memory instance.

### Vertical Scaling

- **Small**: 1K memories, ~4MB, single-threaded
- **Medium**: 10K memories, ~40MB, batch processing
- **Large**: 100K memories, ~400MB, distributed search

### Database Integration (Future)

```
In-Memory Cache
    ↓ (miss)
Vector Database (e.g., Pinecone, Weaviate)
    ↓ (miss)
Traditional Database (PostgreSQL, MongoDB)
```

---

## Security Considerations

1. **Input Validation**: Sanitize all user inputs
2. **Resource Limits**: Enforce memory caps
3. **No Code Execution**: Pure mathematical operations
4. **Safe Serialization**: Use pickle with caution
5. **Error Handling**: Graceful degradation

---

## Future Enhancements

1. **Distributed Storage**: Support for remote vector databases
2. **Batch Inference**: GPU acceleration for neural network
3. **Advanced Retrieval**: HNSW or other approximate search
4. **Multi-Modal**: Support images, code, etc.
5. **Active Learning**: Identify and request labels for uncertain cases

---

**Architecture Version**: 1.0.0  
**Last Updated**: 2026-02-06  
**Maintained By**: AI Training Memory System Team
