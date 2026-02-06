"""
Integration tests for the complete system.
"""
import pytest
import time
from pathlib import Path
from src.core.memory import AITrainingMemory


class TestEndToEndQuery:
    """Test complete query flow."""
    
    def test_full_query_lifecycle(self):
        """Test complete lifecycle from query to answer."""
        memory = AITrainingMemory()
        
        problem = "What is 5 plus 3?"
        
        # Execute query
        result = memory.solve_problem(problem)
        
        # Verify all components worked
        assert result['answer'] is not None
        assert 'explanation' in result
        assert 'method' in result
        assert 'time' in result
        assert 'confidence' in result
        
        # Verify memory stored
        assert len(memory.memory_documents) > 0
        assert len(memory.memory_embeddings) > 0
    
    def test_multiple_queries_sequence(self):
        """Test sequence of multiple queries."""
        memory = AITrainingMemory()
        
        problems = [
            "What is 5 plus 3?",
            "Calculate 10 minus 4",
            "What is 7 times 6?",
            "Divide 20 by 4"
        ]
        
        for problem in problems:
            result = memory.solve_problem(problem)
            assert result['answer'] is not None
        
        # Verify all queries tracked
        assert memory.total_queries == len(problems)
        assert len(memory.memory_documents) == len(problems)


class TestMemoryRetrievalCycle:
    """Test memory storage and retrieval."""
    
    def test_duplicate_retrieval(self):
        """Test retrieving duplicate questions."""
        memory = AITrainingMemory()
        
        problem = "What is 10 plus 20?"
        
        # First query - computation
        result1 = memory.solve_problem(problem)
        time1 = result1['time']
        method1 = result1['method']
        
        # Second query - should retrieve from memory
        result2 = memory.solve_problem(problem)
        time2 = result2['time']
        method2 = result2['method']
        
        # Verify memory hit
        assert result2['method'] == 'memory_hit'
        assert time2 < time1 * 2  # Should be faster (allowing some variance)
        assert memory.memory_hit_count > 0
    
    def test_similarity_based_retrieval(self):
        """Test similarity-based retrieval."""
        memory = AITrainingMemory()
        
        # Store original
        memory.solve_problem("What is 5 plus 3?")
        
        # Query with similar wording
        result = memory.solve_problem("Calculate 5 + 3")
        
        # Should find similar or compute correctly
        assert result['answer'] == 8
        assert result['method'] in ['similarity_match', 'memory_hit', 'hybrid', 'rule_based']


class TestTrainingPhase:
    """Test training functionality."""
    
    def test_training_increases_memory(self):
        """Test that training adds to memory."""
        memory = AITrainingMemory()
        
        initial_size = len(memory.memory_documents)
        
        # Run training
        memory.training_phase()
        
        # Verify memory grew
        assert len(memory.memory_documents) > initial_size
        assert memory.training_cycles == 1
    
    def test_multiple_training_cycles(self):
        """Test multiple training cycles."""
        memory = AITrainingMemory()
        
        # Multiple training rounds
        memory.training_phase()
        memory.training_phase()
        memory.training_phase()
        
        assert memory.training_cycles == 3


class TestPerformanceTracking:
    """Test performance tracking integration."""
    
    def test_performance_metrics_updated(self):
        """Test that metrics are tracked during queries."""
        memory = AITrainingMemory()
        
        # Execute queries
        memory.solve_problem("What is 5 + 5?")
        memory.solve_problem("What is 5 + 5?")  # Duplicate
        memory.solve_problem("What is 10 - 3?")
        
        # Check metrics
        report = memory.performance_tracker.generate_report()
        
        assert report['summary']['total_queries'] == 3
        assert report['summary']['memory_hits'] > 0
        assert 'timing' in report
        assert 'methods' in report
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation over multiple queries."""
        memory = AITrainingMemory()
        
        problem = "What is 7 + 7?"
        
        # First time
        memory.solve_problem(problem)
        
        # Repeat 3 times
        for _ in range(3):
            memory.solve_problem(problem)
        
        # Hit rate should be > 0
        hit_rate = (memory.memory_hit_count / memory.total_queries) * 100
        assert hit_rate > 0
        assert hit_rate <= 100


class TestSystemIntegration:
    """Test integration between components."""
    
    def test_embedder_memory_integration(self):
        """Test embedder works with memory system."""
        memory = AITrainingMemory()
        
        problem = "What is 2 plus 2?"
        result = memory.solve_problem(problem)
        
        # Verify embedding was created and stored
        assert len(memory.memory_embeddings) > 0
        embedding = memory.memory_embeddings[0]
        assert embedding.shape == (384,)
    
    def test_solver_memory_integration(self):
        """Test solver integration with memory."""
        memory = AITrainingMemory()
        
        # Test different operations
        problems = {
            "What is 5 plus 3?": 8,
            "Calculate 10 minus 4": 6,
            "What is 6 times 7?": 42,
            "Divide 20 by 4": 5.0
        }
        
        for problem, expected in problems.items():
            result = memory.solve_problem(problem)
            assert result['answer'] == expected
    
    def test_save_load_integration(self, tmp_path):
        """Test complete save/load cycle."""
        # Create and populate memory
        memory1 = AITrainingMemory()
        problems = [
            "What is 5 + 5?",
            "Calculate 10 * 2",
            "What is 100 / 4?"
        ]
        
        for problem in problems:
            memory1.solve_problem(problem)
        
        # Save
        save_path = tmp_path / "integration_test.pkl"
        memory1.save(str(save_path))
        
        # Load into new instance
        memory2 = AITrainingMemory()
        memory2.load(str(save_path))
        
        # Verify everything restored
        assert len(memory2.memory_documents) == len(memory1.memory_documents)
        assert len(memory2.memory_embeddings) == len(memory1.memory_embeddings)
        assert memory2.total_queries == memory1.total_queries
        
        # Test that loaded memory works
        result = memory2.solve_problem("What is 5 + 5?")
        assert result['method'] == 'memory_hit'  # Should find in loaded memory


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_problem(self):
        """Test handling empty problem."""
        memory = AITrainingMemory()
        result = memory.solve_problem("")
        
        # Should handle gracefully
        assert 'answer' in result
    
    def test_no_numbers_in_problem(self):
        """Test problem with no numbers."""
        memory = AITrainingMemory()
        result = memory.solve_problem("Hello world")
        
        assert 'answer' in result
        assert 'explanation' in result
    
    def test_invalid_operation(self):
        """Test problem with unclear operation."""
        memory = AITrainingMemory()
        result = memory.solve_problem("5 and 3 something")
        
        # Should still extract numbers
        assert 'numbers_extracted' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
