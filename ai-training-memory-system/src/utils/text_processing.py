"""
Text processing utilities for normalization and standardization.
"""
import re
from typing import Set


def normalize_text(text: str) -> str:
    """
    Convert text to lowercase and remove extra whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def preserve_numbers(text: str) -> str:
    """
    Keep numerical tokens in text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with numbers preserved
    """
    # This function primarily ensures numbers are kept during processing
    # Already handled by normalize_text, but can be extended
    return text


def remove_stop_words(text: str, stop_words: Set[str] = None) -> str:
    """
    Remove common stop words from text.
    
    Args:
        text: Input text string
        stop_words: Set of stop words to remove (optional)
        
    Returns:
        Text with stop words removed
    """
    if stop_words is None:
        # Basic stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'should', 'could', 'may', 
            'might', 'must', 'can'
        }
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def standardize_punctuation(text: str) -> str:
    """
    Standardize punctuation in text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with standardized punctuation
    """
    # Replace multiple punctuation with single
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    # Remove special characters except those needed for numbers
    text = re.sub(r'[^\w\s.+\-*/=?!,]', '', text)
    return text


def clean_text_for_hash(text: str) -> str:
    """
    Clean text for consistent hash generation.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text suitable for hashing
    """
    text = normalize_text(text)
    text = standardize_punctuation(text)
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([?.!,])', r'\1', text)
    return text.strip()
