"""
Similarity calculation utilities.
Implements cosine similarity and vector ranking.
"""
import numpy as np
from typing import List, Tuple


def cosine_similarity_calc(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    # Clip to valid range [0, 1]
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return float(similarity)


def cosine_similarity_batch(query_vec: np.ndarray, stored_vecs: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query and multiple vectors efficiently.
    
    Args:
        query_vec: Query vector of shape (384,)
        stored_vecs: Stored vectors of shape (n, 384)
        
    Returns:
        Array of similarity scores of shape (n,)
    """
    if len(stored_vecs) == 0:
        return np.array([])
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(stored_vecs))
    
    query_normalized = query_vec / query_norm
    
    # Normalize stored vectors
    stored_norms = np.linalg.norm(stored_vecs, axis=1, keepdims=True)
    stored_norms[stored_norms == 0] = 1  # Avoid division by zero
    stored_normalized = stored_vecs / stored_norms
    
    # Calculate similarities
    similarities = np.dot(stored_normalized, query_normalized)
    
    # Clip to valid range [0, 1]
    similarities = np.clip(similarities, 0.0, 1.0)
    
    return similarities


def find_similar_vectors(
    query_vec: np.ndarray,
    stored_vecs: np.ndarray,
    threshold: float = 0.7
) -> List[Tuple[int, float]]:
    """
    Find vectors above similarity threshold.
    
    Args:
        query_vec: Query vector
        stored_vecs: Array of stored vectors
        threshold: Minimum similarity threshold
        
    Returns:
        List of (index, similarity) tuples above threshold
    """
    similarities = cosine_similarity_batch(query_vec, stored_vecs)
    
    # Find indices above threshold
    above_threshold = np.where(similarities >= threshold)[0]
    
    # Create list of (index, similarity) tuples
    results = [(int(idx), float(similarities[idx])) for idx in above_threshold]
    
    return results


def rank_by_similarity(similarities: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Sort results by similarity score in descending order.
    
    Args:
        similarities: List of (index, similarity) tuples
        
    Returns:
        Sorted list of (index, similarity) tuples
    """
    return sorted(similarities, key=lambda x: x[1], reverse=True)
