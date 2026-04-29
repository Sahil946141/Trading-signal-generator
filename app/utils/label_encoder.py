import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class PatternLabelEncoder:
    """
    Encodes pattern labels to one-hot vectors for LSTM input.
    """
    
    def __init__(self, pattern_labels: List[str]):
        """
        Initialize encoder.
        
        Args:
            pattern_labels: List of pattern class names
        
        Example:
            labels = ["Double Bottom", "Double Top", "Head and Shoulders", ...]
            encoder = PatternLabelEncoder(labels)
        """
        self.pattern_labels = pattern_labels
        self.num_patterns = len(pattern_labels)
        
        # Create bidirectional mappings
        self.label_to_idx = {
            label: idx for idx, label in enumerate(pattern_labels)
        }
        self.idx_to_label = {
            idx: label for idx, label in enumerate(pattern_labels)
        }
        
        logger.info(f"Initialized PatternLabelEncoder with {self.num_patterns} patterns")
    
    def encode(
        self,
        pattern_label: str,
        confidence: float
    ) -> np.ndarray:
        """
        Encode pattern label to vector.
        
        Args:
            pattern_label: Pattern class name
            confidence: Confidence score (0-1)
        
        Returns:
            np.ndarray of shape (num_patterns + 1,)
            Format: [one_hot_pattern..., confidence]
        
        Example:
            >>> encoder = PatternLabelEncoder([...])
            >>> encoded = encoder.encode("Double Bottom", 0.87)
            >>> encoded.shape
            (11,)  # 10 patterns + 1 confidence
            >>> encoded
            array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.87])
        """
        # Create one-hot vector
        one_hot = np.zeros(self.num_patterns)
        
        if pattern_label not in self.label_to_idx:
            logger.warning(f"Unknown pattern label: {pattern_label}")
            # Default to first class if unknown
            pattern_idx = 0
        else:
            pattern_idx = self.label_to_idx[pattern_label]
        
        one_hot[pattern_idx] = 1.0
        
        # Append confidence
        encoded = np.append(one_hot, confidence)
        
        logger.debug(f"Encoded {pattern_label} with confidence {confidence:.4f}")
        return encoded
    
    def decode(self, one_hot_vector: np.ndarray) -> str:
        """
        Decode one-hot vector back to pattern label.
        
        Args:
            one_hot_vector: Array of shape (num_patterns + 1,)
        
        Returns:
            Pattern label string
        """
        # Get pattern index (ignore confidence at end)
        pattern_one_hot = one_hot_vector[:-1]
        pattern_idx = np.argmax(pattern_one_hot)
        
        return self.idx_to_label[pattern_idx]
    
    def get_pattern_names(self) -> List[str]:
        """Get all pattern class names."""
        return self.pattern_labels
    
    def get_num_patterns(self) -> int:
        """Get number of pattern classes."""
        return self.num_patterns