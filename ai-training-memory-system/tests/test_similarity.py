"""
Unit tests for similarity calculations.
"""
import pytest
import numpy as np
from src.utils.similarity import (
    cosine_similarity_calc,
    cosine_similarity_batch,
    find_similar_vectors,
    rank_by_similarity
)


class TestCosineSimilarity:
    """Test cosine similarity calculations."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec1 = np.array([1, 2, 3, 4])
        vec2 = np.array([1, 2, 3, 4])
        
        similarity = cosine_similarity_calc(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        
        similarity = cosine_similarity_calc(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        
        similarity = cosine_similarity_calc(vec1, vec2)
        # Due to clipping, should be 0.0
        assert similarity >= 0.0
    
    def test_zero_vector(self):
        """Test handling of zero vectors."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])
        
        similarity = cosine_similarity_calc(vec1, vec2)
        assert similarity == 0.0
    
    def test_batch_similarity(self):
        """Test batch similarity calculation."""
        query_vec = np.array([1, 2, 3, 4])
        stored_vecs = np.array([
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        
        similarities = cosine_similarity_batch(query_vec, stored_vecs)
        
        assert len(similarities) == 4
        assert abs(similarities[0] - 1.0) < 1e-6  # Identical
        assert abs(similarities[1] - 1.0) < 1e-6  # Parallel
        assert similarities[2] == 0.0  # Zero vector
        assert 0 <= similarities[3] <= 1.0


class TestVectorSearch:
    """Test vector search functions."""
    
    def test_find_similar_vectors(self):
        """Test finding similar vectors above threshold."""
        query_vec = np.array([1, 2, 3, 4])
        stored_vecs = np.array([
            [1, 2, 3, 4],      # Should match (similarity = 1.0)
            [2, 4, 6, 8],      # Should match (parallel)
            [10, 0, 0, 0],     # Below threshold
            [1, 2, 3, 5]       # Should match (very similar)
        ])
        
        results = find_similar_vectors(query_vec, stored_vecs, threshold=0.9)
        
        assert len(results) >= 2
        assert all(idx < len(stored_vecs) for idx, _ in results)
        assert all(sim >= 0.9 for _, sim in results)
    
    def test_rank_by_similarity(self):
        """Test ranking results by similarity."""
        similarities = [
            (0, 0.95),
            (1, 0.85),
            (2, 0.99),
            (3, 0.80)
        ]
        
        ranked = rank_by_similarity(similarities)
        
        # Should be in descending order
        assert ranked[0][1] == 0.99
        assert ranked[1][1] == 0.95
        assert ranked[2][1] == 0.85
        assert ranked[3][1] == 0.80
    
    def test_empty_similarity_list(self):
        """Test ranking empty list."""
        ranked = rank_by_similarity([])
        assert ranked == []


class TestSimilarityEdgeCases:
    """Test edge cases in similarity calculations."""
    
    def test_single_dimension_vectors(self):
        """Test single dimension vectors."""
        vec1 = np.array([5.0])
        vec2 = np.array([5.0])
        
        similarity = cosine_similarity_calc(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_high_dimensional_vectors(self):
        """Test high dimensional vectors (like 384D)."""
        vec1 = np.random.randn(384)
        vec2 = vec1.copy()
        
        similarity = cosine_similarity_calc(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_batch_with_empty_array(self):
        """Test batch calculation with empty array."""
        query_vec = np.array([1, 2, 3])
        stored_vecs = np.array([]).reshape(0, 3)
        
        similarities = cosine_similarity_batch(query_vec, stored_vecs)
        assert len(similarities) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
