"""
SimpleDocument: Primary data container for memory storage.
Combines text content with structured metadata.
"""
from datetime import datetime
from typing import Dict, Any, Optional


class SimpleDocument:
    """
    Primary data container for memory storage.
    Combines text content with structured metadata.
    
    Metadata schema includes:
    - type: str ('training', 'user_query', 'system')
    - problem: str (Original problem statement)
    - solution: any (Computed solution)
    - explanation: str (Solution derivation method)
    - timestamp: datetime (Creation timestamp)
    - operation: str (Mathematical operation used)
    - confidence: float (Solution confidence 0.0-1.0)
    - similar_to: list (Related problem references)
    """
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a SimpleDocument.
        
        Args:
            page_content: Original text content
            metadata: Structured metadata dictionary
        """
        self.page_content = page_content
        self.metadata = metadata or {}
        
        # Set default timestamp if not provided
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now()
    
    def __repr__(self) -> str:
        """String representation of the document."""
        return f"SimpleDocument(content='{self.page_content[:50]}...', metadata={list(self.metadata.keys())})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            'page_content': self.page_content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleDocument':
        """Create document from dictionary format."""
        return cls(
            page_content=data['page_content'],
            metadata=data.get('metadata', {})
        )
