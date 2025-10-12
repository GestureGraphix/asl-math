"""Vocabulary and codebook definitions for product VQ.

This module defines the vocabulary structure and codebook sizes as specified:
- Σ_H: 64 codes for hands (finger flexions + thumb angle)
- Σ_L: 128 codes for location (palm centers)
- Σ_O: 32 codes for orientation (palm normals)
- Σ_M: 64 codes for motion (deltas)
- Σ_N: 32 codes for non-manual features
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class Vocabulary:
    """Vocabulary management for ASL translation."""
    
    def __init__(self):
        """Initialize vocabulary with codebook sizes."""
        self.codebook_sizes = {
            'hands': 64,      # Σ_H
            'location': 128,  # Σ_L
            'orientation': 32,  # Σ_O
            'motion': 64,     # Σ_M
            'non_manual': 32,  # Σ_N
        }
        
        # Total vocabulary size
        self.total_size = sum(self.codebook_sizes.values())
        
        # Special tokens
        self.blank_token = 0
        self.unk_token = 1
        
        # Create index mappings
        self._create_index_mappings()
        
    def _create_index_mappings(self) -> None:
        """Create index mappings for each codebook."""
        self.index_ranges = {}
        start_idx = 2  # Reserve 0 for blank, 1 for unk
        
        for modality, size in self.codebook_sizes.items():
            end_idx = start_idx + size
            self.index_ranges[modality] = (start_idx, end_idx)
            start_idx = end_idx
            
    def get_codebook_indices(self, modality: str) -> Tuple[int, int]:
        """Get start and end indices for a modality.
        
        Args:
            modality: One of ['hands', 'location', 'orientation', 'motion', 'non_manual']
            
        Returns:
            Tuple of (start_idx, end_idx)
        """
        if modality not in self.index_ranges:
            raise ValueError(f"Unknown modality: {modality}")
        return self.index_ranges[modality]
    
    def get_codebook_size(self, modality: str) -> int:
        """Get codebook size for a modality.
        
        Args:
            modality: One of ['hands', 'location', 'orientation', 'motion', 'non_manual']
            
        Returns:
            Codebook size
        """
        if modality not in self.codebook_sizes:
            raise ValueError(f"Unknown modality: {modality}")
        return self.codebook_sizes[modality]
    
    def encode_product_code(self, codes: Dict[str, int]) -> int:
        """Encode individual codes into a product code.
        
        Args:
            codes: Dictionary mapping modality to code index
            
        Returns:
            Combined product code
        """
        product_code = 1  # Start with unk token
        
        for modality, code in codes.items():
            start_idx, _ = self.get_codebook_indices(modality)
            product_code = product_code * self.total_size + (start_idx + code)
            
        return product_code
    
    def decode_product_code(self, product_code: int) -> Dict[str, int]:
        """Decode product code into individual modality codes.
        
        Args:
            product_code: Combined product code
            
        Returns:
            Dictionary mapping modality to code index
        """
        codes = {}
        remaining = product_code
        
        # Reverse order for decoding
        modalities = list(self.codebook_sizes.keys())[::-1]
        
        for modality in modalities:
            start_idx, _ = self.get_codebook_indices(modality)
            code = remaining % self.total_size - start_idx
            codes[modality] = max(0, code)  # Ensure non-negative
            remaining = remaining // self.total_size
            
        return codes
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for each modality.
        
        Returns:
            Dictionary mapping modality to feature dimension
        """
        return {
            'hands': 10,      # 5 finger flexions + thumb angle
            'location': 6,    # palm centres c^L_t, c^R_t
            'orientation': 6,  # unit normals n^L_t, n^R_t
            'motion': 9,      # Δc, Δ²c, Δa_t, Δg_t
            'non_manual': 5,  # gaze proxy g_t, mouth a_t, eyebrow height
        }