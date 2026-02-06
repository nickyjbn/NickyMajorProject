"""
Unit tests for memory system components.
"""
import pytest
import numpy as np
from src.core.document import SimpleDocument
from src.core.embedder import TextEmbedder
from src.core.memory import AITrainingMemory


class TestSimpleDocument:
    """Test SimpleDocument class."""
    
    def test_document_creation(self):
        """Test document creation with content and metadata."""
        doc = SimpleDocument(
            page_content="What is 2 plus 2?",
            metadata={'type': 'user_query', 'solution': 4}
        )
        
        assert doc.page_content == "What is 2 plus 2?"
        assert doc.metadata['type'] == 'user_query'
        assert doc.metadata['solution'] == 4
        assert 'timestamp' in doc.metadata
    
    def test_document_to_dict(self):
        """Test document serialization."""
        doc = SimpleDocument("Test content", {'key': 'value'})
        doc_dict = doc.to_dict()
        
        assert 'page_content' in doc_dict
        assert 'metadata' in doc_dict
        assert doc_dict['page_content'] == "Test content"
    
    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            'page_content': "Test content",
            'metadata': {'key': 'value'}
        }
        doc = SimpleDocument.from_dict(data)
        
        assert doc.page_content == "Test content"
        assert doc.metadata['key'] == 'value'


class TestTextEmbedder:
    """Test TextEmbedder class."""
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        embedder = TextEmbedder()
        assert embedder.vector_dimension == 384
        assert embedder.model is not None
    
    def test_vector_generation(self):
        """Test 384D embedding generation."""
        embedder = TextEmbedder()
        text = "What is 5 plus 3?"
        embedding = embedder.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()
    
    def test_batch_embedding(self):
        """Test batch embedding generation."""
        embedder = TextEmbedder()
        texts = ["Question 1", "Question 2", "Question 3"]
        embeddings = embedder.embed_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)


class TestAITrainingMemory:
    """Test AITrainingMemory class."""
    
    def test_memory_initialization(self):
        """Test memory system initialization."""
        memory = AITrainingMemory()
        
        assert len(memory.memory_documents) == 0
        assert len(memory.memory_embeddings) == 0
        assert memory.vector_dimension == 384
        assert memory.total_queries == 0
        assert memory.memory_hit_count == 0
    
    def test_add_to_memory(self):
        """Test adding documents to memory."""
        memory = AITrainingMemory()
        doc = SimpleDocument(
            page_content="What is 2 plus 2?",
            metadata={'solution': 4}
        )
        
        memory.add_to_memory(doc)
        
        assert len(memory.memory_documents) == 1
        assert len(memory.memory_embeddings) == 1
        assert memory.memory_embeddings[0].shape == (384,)
    
    def test_duplicate_detection(self):
        """Test hash-based duplicate detection."""
        memory = AITrainingMemory()
        
        question = "What is 5 plus 3?"
        
        # First query
        result1 = memory.solve_problem(question)
        is_dup1, history1 = memory.check_duplicate_question(question)
        
        # Second query (should be duplicate)
        result2 = memory.solve_problem(question)
        is_dup2, history2 = memory.check_duplicate_question(question)
        
        assert is_dup2 == True
        assert len(history2) >= 1
        assert result2['method'] == 'memory_hit'
    
    def test_similarity_search(self):
        """Test cosine similarity search."""
        memory = AITrainingMemory()
        
        # Add some documents
        problems = [
            "What is 5 plus 3?",
            "Calculate 10 minus 4",
            "What is 7 times 6?"
        ]
        
        for problem in problems:
            memory.solve_problem(problem)
        
        # Search for similar
        results = memory.similarity_search("What is 5 + 3?", k=2)
        
        assert len(results) > 0
        assert all(isinstance(doc, SimpleDocument) for doc, _ in results)
        assert all(0 <= score <= 1 for _, score in results)
    
    def test_memory_storage(self):
        """Test data integrity in storage."""
        memory = AITrainingMemory()
        
        problem = "What is 10 plus 20?"
        result = memory.solve_problem(problem)
        
        # Verify storage
        assert len(memory.memory_documents) > 0
        doc = memory.memory_documents[-1]
        assert doc.metadata['problem'] == problem
        assert doc.metadata['solution'] == result['answer']
    
    def test_max_memory_limit(self):
        """Test memory limit enforcement."""
        memory = AITrainingMemory(max_memory_entries=10)
        
        # Add more than limit
        for i in range(15):
            problem = f"What is {i} plus 1?"
            memory.solve_problem(problem)
        
        # Should not exceed limit
        assert len(memory.memory_documents) <= 10
    
    def test_save_and_load(self, tmp_path):
        """Test save and load functionality."""
        memory = AITrainingMemory()
        
        # Add some data
        problems = ["What is 2 + 2?", "Calculate 5 * 3"]
        for problem in problems:
            memory.solve_problem(problem)
        
        # Save
        save_path = tmp_path / "test_memory.pkl"
        memory.save(str(save_path))
        
        # Load into new instance
        memory2 = AITrainingMemory()
        memory2.load(str(save_path))
        
        assert len(memory2.memory_documents) == len(memory.memory_documents)
        assert memory2.total_queries == memory.total_queries


class TestPerformanceMetrics:
    """Test performance tracking."""
    
    def test_query_counting(self):
        """Test query counting."""
        memory = AITrainingMemory()
        
        initial_count = memory.total_queries
        memory.solve_problem("What is 1 + 1?")
        
        assert memory.total_queries == initial_count + 1
    
    def test_hit_rate_calculation(self):
        """Test memory hit rate calculation."""
        memory = AITrainingMemory()
        
        problem = "What is 3 + 3?"
        
        # First time
        memory.solve_problem(problem)
        
        # Second time (should be hit)
        memory.solve_problem(problem)
        
        hit_rate = (memory.memory_hit_count / memory.total_queries) * 100
        assert hit_rate > 0
    
    def test_timing_tracking(self):
        """Test solve time tracking."""
        memory = AITrainingMemory()
        
        result = memory.solve_problem("What is 5 + 5?")
        
        assert 'time' in result
        assert result['time'] > 0
        assert len(memory.solve_times) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
