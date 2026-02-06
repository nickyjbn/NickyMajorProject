"""
Text embedder for converting text to vector representations.
Uses SentenceTransformer model to generate 384-dimensional embeddings.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union


class TextEmbedder:
    """
    Text embedder using SentenceTransformer.
    Generates 384-dimensional vectors for text.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedder with specified model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_dimension = 384  # Standard dimension for all-MiniLM-L6-v2
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to vector embedding(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            numpy array of shape (384,) for single text or (n, 384) for list
        """
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert batch of texts to embeddings efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (n, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
