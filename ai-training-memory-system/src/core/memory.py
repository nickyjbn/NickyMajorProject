"""
AITrainingMemory: Main memory system with vector storage and semantic retrieval.
Enables continuous learning through persistent memory.
"""
import hashlib
import time
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .document import SimpleDocument
from .embedder import TextEmbedder
from ..utils.text_processing import clean_text_for_hash
from ..utils.similarity import cosine_similarity_batch, rank_by_similarity
from ..utils.performance import PerformanceTracker
from ..solvers.rule_based import solve_with_rules
from ..solvers.neural_network import NeuralNetworkSolver
from ..solvers.hybrid import HybridSolver


class AITrainingMemory:
    """
    Main memory system with persistent vector storage and semantic retrieval.
    Implements continuous learning through experience accumulation.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        duplicate_check: bool = True,
        max_memory_entries: int = 10000,
        enable_neural_network: bool = True
    ):
        """
        Initialize the AI Training Memory system.
        
        Args:
            similarity_threshold: Minimum similarity for matches (0.0-1.0)
            duplicate_check: Enable duplicate question detection
            max_memory_entries: Maximum number of memories to store
            enable_neural_network: Enable neural network solver
        """
        # Core storage
        self.memory_documents: List[SimpleDocument] = []
        self.memory_embeddings: List[np.ndarray] = []
        self.vector_dimension = 384
        
        # History and tracking
        self.question_history: defaultdict = defaultdict(list)
        self.user_questions: List[Dict[str, Any]] = []
        self.mistake_patterns_learned: set = set()
        
        # Embedder
        self.embedder = TextEmbedder()
        
        # Training and solving
        self.training_cycles = 0
        self.nn_solver = NeuralNetworkSolver() if enable_neural_network else None
        self.hybrid_solver = HybridSolver(self.nn_solver)
        
        # Metrics
        self.query_timestamps: List[datetime] = []
        self.solve_times: List[float] = []
        self.memory_hit_count = 0
        self.total_queries = 0
        self.performance_tracker = PerformanceTracker()
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.duplicate_check = duplicate_check
        self.max_memory_entries = max_memory_entries
        self.enable_neural_network = enable_neural_network
    
    def _generate_hash(self, text: str) -> str:
        """
        Generate MD5 hash for duplicate detection.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash string
        """
        cleaned_text = clean_text_for_hash(text)
        return hashlib.md5(cleaned_text.encode()).hexdigest()
    
    def add_to_memory(self, document: SimpleDocument):
        """
        Add document to memory with vector embedding.
        
        Args:
            document: SimpleDocument to store
        """
        # Check memory limit
        if len(self.memory_documents) >= self.max_memory_entries:
            # Remove oldest entry
            self.memory_documents.pop(0)
            self.memory_embeddings.pop(0)
        
        # Generate embedding
        embedding = self.embedder.embed(document.page_content)
        
        # Store document and embedding
        self.memory_documents.append(document)
        self.memory_embeddings.append(embedding)
        
        # Update question history if duplicate checking enabled
        if self.duplicate_check and 'problem' in document.metadata:
            question_hash = self._generate_hash(document.metadata['problem'])
            self.question_history[question_hash].append({
                'timestamp': document.metadata.get('timestamp', datetime.now()),
                'solution': document.metadata.get('solution'),
                'index': len(self.memory_documents) - 1
            })
    
    def similarity_search(
        self,
        query: str,
        k: int = 2,
        threshold: float = None
    ) -> List[Tuple[SimpleDocument, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.memory_embeddings:
            return []
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Calculate similarities
        stored_vecs = np.array(self.memory_embeddings)
        similarities = cosine_similarity_batch(query_embedding, stored_vecs)
        
        # Filter by threshold and get top k
        results = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                doc = self.memory_documents[idx]
                # Add similarity score to metadata copy
                doc_copy = SimpleDocument(doc.page_content, doc.metadata.copy())
                doc_copy.metadata['similarity_score'] = float(sim)
                results.append((doc_copy, float(sim)))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def check_duplicate_question(self, question: str) -> Tuple[bool, List[dict]]:
        """
        Check if question has been asked before.
        
        Args:
            question: Question text
            
        Returns:
            Tuple of (is_duplicate, history_entries)
        """
        if not self.duplicate_check:
            return False, []
        
        question_hash = self._generate_hash(question)
        
        if question_hash in self.question_history:
            history = self.question_history[question_hash]
            return True, history
        
        return False, []
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve mathematical problem using multi-phase approach.
        
        Phases:
        1. Check for exact duplicates (instant return)
        2. Similarity search for similar problems
        3. Rule-based solving
        4. Neural network enhancement
        5. Store solution and return
        
        Args:
            problem: Mathematical problem text
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        self.total_queries += 1
        self.query_timestamps.append(datetime.now())
        
        result = {
            'problem': problem,
            'answer': None,
            'explanation': '',
            'method': 'unknown',
            'time': 0.0,
            'confidence': 0.0,
            'numbers_extracted': [],
            'operation': 'unknown'
        }
        
        # Phase 1: Check for exact duplicates
        is_duplicate, history = self.check_duplicate_question(problem)
        if is_duplicate and history:
            self.memory_hit_count += 1
            latest = history[-1]
            doc_idx = latest.get('index')
            
            if doc_idx is not None and doc_idx < len(self.memory_documents):
                doc = self.memory_documents[doc_idx]
                result['answer'] = doc.metadata.get('solution')
                result['explanation'] = f"Retrieved from memory (asked {len(history)} time(s) before)"
                result['method'] = 'memory_hit'
                result['confidence'] = 1.0
                result['time'] = time.time() - start_time
                
                self.solve_times.append(result['time'])
                self.performance_tracker.record_query('memory_hit', result['time'], True)
                return result
        
        # Phase 2: Similarity search
        similar_docs = self.similarity_search(problem, k=2, threshold=self.similarity_threshold)
        if similar_docs:
            doc, similarity = similar_docs[0]
            if similarity >= 0.9:  # High similarity threshold
                result['answer'] = doc.metadata.get('solution')
                result['explanation'] = f"Found similar problem (similarity: {similarity:.2f})"
                result['method'] = 'similarity_match'
                result['confidence'] = float(similarity)
                result['time'] = time.time() - start_time
                
                self.solve_times.append(result['time'])
                self.performance_tracker.record_query('similarity', result['time'], True)
                
                # Store this query
                self._store_solution(problem, result)
                return result
        
        # Phase 3: Rule-based + Neural network solving
        problem_embedding = self.embedder.embed(problem)
        solve_result = self.hybrid_solver.solve(problem, problem_embedding)
        
        result['answer'] = solve_result.get('final_answer')
        result['explanation'] = solve_result.get('explanation', '')
        result['method'] = solve_result.get('method', 'hybrid')
        result['confidence'] = solve_result.get('confidence', 0.5)
        result['numbers_extracted'] = solve_result.get('numbers_extracted', [])
        result['operation'] = solve_result.get('operation', 'unknown')
        result['time'] = time.time() - start_time
        
        self.solve_times.append(result['time'])
        
        if result['answer'] is not None:
            self.performance_tracker.record_query('computation', result['time'], True)
        else:
            self.performance_tracker.record_query('failed', result['time'], False)
        
        # Phase 5: Store solution
        self._store_solution(problem, result)
        
        return result
    
    def _store_solution(self, problem: str, result: Dict[str, Any]):
        """Store problem and solution in memory."""
        doc = SimpleDocument(
            page_content=problem,
            metadata={
                'type': 'user_query',
                'problem': problem,
                'solution': result['answer'],
                'explanation': result['explanation'],
                'operation': result.get('operation', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'method': result.get('method', 'unknown'),
                'timestamp': datetime.now()
            }
        )
        self.add_to_memory(doc)
    
    def memory_guided_solving(self, user_problem: str) -> Dict[str, Any]:
        """
        Enhanced problem solving using memory insights.
        
        Args:
            user_problem: User's problem text
            
        Returns:
            Dictionary with solution and insights
        """
        # Use solve_problem which already implements memory-guided approach
        result = self.solve_problem(user_problem)
        
        # Add memory insights
        similar_docs = self.similarity_search(user_problem, k=3)
        result['similar_problems'] = [
            {
                'problem': doc.page_content,
                'similarity': sim,
                'solution': doc.metadata.get('solution')
            }
            for doc, sim in similar_docs
        ]
        
        return result
    
    def training_phase(self, training_examples: List[Dict[str, Any]] = None):
        """
        Load training examples and learn from mistakes.
        
        Args:
            training_examples: List of training example dictionaries
        """
        self.training_cycles += 1
        
        if training_examples is None:
            # Use default training examples
            training_examples = [
                {'problem': 'What is 5 plus 3?', 'answer': 8, 'operation': 'addition'},
                {'problem': 'Calculate 10 minus 4', 'answer': 6, 'operation': 'subtraction'},
                {'problem': 'What is 7 times 6?', 'answer': 42, 'operation': 'multiplication'},
                {'problem': 'Divide 20 by 4', 'answer': 5, 'operation': 'division'},
                {'problem': 'Add 15 and 25', 'answer': 40, 'operation': 'addition'},
            ]
        
        # Store training examples in memory
        for example in training_examples:
            doc = SimpleDocument(
                page_content=example['problem'],
                metadata={
                    'type': 'training',
                    'problem': example['problem'],
                    'solution': example['answer'],
                    'operation': example.get('operation', 'unknown'),
                    'confidence': 1.0,
                    'timestamp': datetime.now()
                }
            )
            self.add_to_memory(doc)
        
        # Train neural network if enabled
        if self.enable_neural_network and len(self.memory_embeddings) > 10:
            try:
                # Prepare training data
                X_train = np.array(self.memory_embeddings)
                y_train = np.array([
                    doc.metadata.get('solution', 0) 
                    for doc in self.memory_documents
                ])
                
                # Filter out None values
                valid_indices = ~np.isnan(y_train)
                X_train = X_train[valid_indices]
                y_train = y_train[valid_indices]
                
                if len(X_train) > 0:
                    # Train the model
                    self.nn_solver.train(X_train, y_train, epochs=5, batch_size=16)
                    # Update hybrid solver
                    self.hybrid_solver = HybridSolver(self.nn_solver)
            except Exception as e:
                print(f"Warning: Neural network training failed: {e}")
    
    def show_memory(self):
        """Display memory contents and statistics."""
        print("\n" + "="*60)
        print("AI TRAINING MEMORY SYSTEM - MEMORY STATUS")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 MEMORY STATISTICS:")
        print(f"  Total questions stored: {len(self.memory_documents)}")
        print(f"  Unique questions: {len(self.question_history)}")
        print(f"  Repeated questions: {len(self.memory_documents) - len(self.question_history)}")
        print(f"  Vector dimension: {self.vector_dimension}D")
        
        # Memory efficiency
        memory_size_mb = (len(self.memory_embeddings) * self.vector_dimension * 8) / (1024 * 1024)
        print(f"  Estimated memory size: {memory_size_mb:.2f} MB")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Total queries: {self.total_queries}")
        print(f"  Memory hits: {self.memory_hit_count}")
        hit_rate = (self.memory_hit_count / self.total_queries * 100) if self.total_queries > 0 else 0
        print(f"  Hit rate: {hit_rate:.2f}%")
        
        if self.solve_times:
            avg_time = sum(self.solve_times) / len(self.solve_times)
            print(f"  Average solve time: {avg_time:.4f}s")
        
        # Learning progress
        print(f"\n🧠 LEARNING PROGRESS:")
        print(f"  Training cycles: {self.training_cycles}")
        print(f"  Neural network enabled: {self.enable_neural_network}")
        print(f"  Mistake patterns learned: {len(self.mistake_patterns_learned)}")
        
        # Sample problems
        if self.memory_documents:
            print(f"\n📝 SAMPLE PROBLEMS (last 5):")
            for doc in self.memory_documents[-5:]:
                problem = doc.metadata.get('problem', doc.page_content)
                solution = doc.metadata.get('solution', 'N/A')
                timestamp = doc.metadata.get('timestamp', 'Unknown')
                print(f"  • {problem[:50]}... = {solution} [{timestamp}]")
        
        print("\n" + "="*60 + "\n")
    
    def interactive_demonstration(self):
        """Interactive demonstration interface."""
        print("\n" + "="*60)
        print("AI TRAINING MEMORY SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("\nCommands:")
        print("  - Type a math problem to solve it")
        print("  - 'memory' - Show memory contents")
        print("  - 'stats' - Show performance statistics")
        print("  - 'train' - Enter training mode")
        print("  - 'quit' or 'exit' - Exit the system")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n💭 Enter problem or command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'memory':
                    self.show_memory()
                
                elif user_input.lower() == 'stats':
                    report = self.performance_tracker.generate_report()
                    print("\n" + "="*60)
                    print("PERFORMANCE REPORT")
                    print("="*60)
                    for section, data in report.items():
                        print(f"\n{section.upper()}:")
                        for key, value in data.items():
                            print(f"  {key}: {value}")
                
                elif user_input.lower() == 'train':
                    print("\n🎓 Entering training mode...")
                    self.training_phase()
                    print("✅ Training complete!")
                
                else:
                    # Solve the problem
                    print("\n🔍 Solving...")
                    result = self.solve_problem(user_input)
                    
                    print(f"\n✨ Answer: {result['answer']}")
                    print(f"📋 Explanation: {result['explanation']}")
                    print(f"⚙️  Method: {result['method']}")
                    print(f"🎯 Confidence: {result['confidence']:.2f}")
                    print(f"⏱️  Time: {result['time']:.4f}s")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def save(self, filepath: str):
        """Save memory system to file."""
        data = {
            'memory_documents': [doc.to_dict() for doc in self.memory_documents],
            'memory_embeddings': np.array(self.memory_embeddings),
            'question_history': dict(self.question_history),
            'user_questions': self.user_questions,
            'mistake_patterns_learned': list(self.mistake_patterns_learned),
            'training_cycles': self.training_cycles,
            'total_queries': self.total_queries,
            'memory_hit_count': self.memory_hit_count,
            'configuration': {
                'similarity_threshold': self.similarity_threshold,
                'duplicate_check': self.duplicate_check,
                'max_memory_entries': self.max_memory_entries,
                'enable_neural_network': self.enable_neural_network
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save neural network if trained
        if self.nn_solver and self.nn_solver.trained:
            nn_filepath = filepath.replace('.pkl', '_nn.pth')
            self.nn_solver.save_model(nn_filepath)
    
    def load(self, filepath: str):
        """Load memory system from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore data
        self.memory_documents = [SimpleDocument.from_dict(d) for d in data['memory_documents']]
        self.memory_embeddings = list(data['memory_embeddings'])
        self.question_history = defaultdict(list, data['question_history'])
        self.user_questions = data['user_questions']
        self.mistake_patterns_learned = set(data['mistake_patterns_learned'])
        self.training_cycles = data['training_cycles']
        self.total_queries = data['total_queries']
        self.memory_hit_count = data['memory_hit_count']
        
        # Restore configuration
        config = data.get('configuration', {})
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.duplicate_check = config.get('duplicate_check', True)
        self.max_memory_entries = config.get('max_memory_entries', 10000)
        self.enable_neural_network = config.get('enable_neural_network', True)
        
        # Load neural network if exists
        nn_filepath = filepath.replace('.pkl', '_nn.pth')
        if Path(nn_filepath).exists() and self.nn_solver:
            try:
                self.nn_solver.load_model(nn_filepath)
                self.hybrid_solver = HybridSolver(self.nn_solver)
            except Exception as e:
                print(f"Warning: Could not load neural network: {e}")
