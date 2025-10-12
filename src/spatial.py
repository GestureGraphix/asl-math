"""Spatial Discourse Module with Bayesian Fusion.

This module implements:
1. Voxelized locus set C_t with window W_t of past 2 seconds
2. Pointing vector g(t) from dominant hand index base to neck
3. Bayesian fusion: p(r|C) ∝ ∏_c ℓ_c(r) · p(r|r_{t-1})
4. Likelihoods ℓ_c(r) = exp(-0.5(∠(g(t),r̂)/σ_pt)²) with σ_pt=2°
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import math


class SpatialDiscourse(nn.Module):
    """Spatial discourse module for referent resolution.
    
    Maintains a spatial memory of signed locations and uses Bayesian
    fusion to resolve referents based on pointing gestures.
    """
    
    def __init__(self, 
                 window_size: float = 2.0,  # 2 seconds
                 voxel_size: float = 0.08,   # 8 cm
                 pointing_sigma: float = 2.0,  # degrees
                 fps: int = 30):
        """Initialize spatial discourse module.
        
        Args:
            window_size: Temporal window size in seconds
            voxel_size: Voxel size in meters
            pointing_sigma: Pointing likelihood standard deviation in degrees
            fps: Frames per second
        """
        super().__init__()
        
        self.window_size = window_size
        self.voxel_size = voxel_size
        self.pointing_sigma = math.radians(pointing_sigma)  # Convert to radians
        self.fps = fps
        self.window_frames = int(window_size * fps)
        
        # Spatial memory - deque of (timestamp, position, referent_id)
        self.spatial_memory = deque(maxlen=self.window_frames)
        
        # Referent tracking
        self.referent_positions = {}  # referent_id -> last_position
        self.referent_priors = {}     # referent_id -> prior probability
        
        # Landmark indices
        self.INDEX_BASE = 5  # Index finger base
        self.NECK = 0       # Neck position
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        
    def forward(self, 
                landmarks: torch.Tensor,
                timestamp: float,
                referent_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Process frame and update spatial discourse.
        
        Args:
            landmarks: Normalized landmarks of shape (N, 3)
            timestamp: Current timestamp in seconds
            referent_id: Optional referent ID if known
            
        Returns:
            Dictionary containing:
                - pointing_vector: Pointing direction g(t)
                - locus_set: Voxelized locus set C_t
                - referent_probs: Posterior probabilities for each referent
                - spatial_features: Spatial feature vector
        """
        # Extract pointing vector g(t)
        pointing_vector = self._compute_pointing_vector(landmarks)
        
        # Update spatial memory
        if referent_id is not None:
            # If we know the referent, add to spatial memory
            position = self._extract_hand_position(landmarks)
            self._update_spatial_memory(timestamp, position, referent_id)
            
            # Update referent position
            self.referent_positions[referent_id] = position
        
        # Voxelize spatial memory to get locus set C_t
        locus_set = self._voxelize_locus_set()
        
        # Compute referent probabilities using Bayesian fusion
        referent_probs = self._bayesian_fusion(pointing_vector, locus_set)
        
        # Extract spatial features
        spatial_features = self._extract_spatial_features(
            pointing_vector, locus_set, referent_probs
        )
        
        return {
            'pointing_vector': pointing_vector,
            'locus_set': locus_set,
            'referent_probs': referent_probs,
            'spatial_features': spatial_features
        }
    
    def _compute_pointing_vector(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute pointing vector g(t) from dominant hand to neck.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Pointing vector of shape (3,)
        """
        # Get index finger base (dominant hand assumption)
        index_base = landmarks[self.INDEX_BASE]
        
        # Get neck position
        neck = landmarks[self.NECK]
        
        # Compute pointing vector
        pointing_vector = neck - index_base
        pointing_vector = pointing_vector / (torch.norm(pointing_vector) + 1e-8)
        
        return pointing_vector
    
    def _extract_hand_position(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Extract hand position for spatial memory.
        
        Args:
            landmarks: Normalized landmarks
            
        Returns:
            Hand position of shape (3,)
        """
        # Use palm center as hand position
        hand_landmarks = landmarks[:21]  # First 21 are hand landmarks
        palm_indices = [0, 1, 5, 9, 13, 17]  # Palm landmarks
        palm_center = hand_landmarks[palm_indices].mean(dim=0)
        
        return palm_center
    
    def _update_spatial_memory(self, 
                              timestamp: float, 
                              position: torch.Tensor,
                              referent_id: str):
        """Update spatial memory with new observation.
        
        Args:
            timestamp: Current timestamp
            position: Signed position
            referent_id: Referent identifier
        """
        self.spatial_memory.append({
            'timestamp': timestamp,
            'position': position,
            'referent_id': referent_id
        })
        
        # Update referent prior (frequency-based)
        if referent_id not in self.referent_priors:
            self.referent_priors[referent_id] = 0.0
        self.referent_priors[referent_id] += 1.0
    
    def _voxelize_locus_set(self) -> Dict[str, torch.Tensor]:
        """Voxelize spatial memory to get locus set C_t.
        
        Returns:
            Dictionary mapping voxel coordinates to referent information
        """
        locus_set = {}
        
        for memory in self.spatial_memory:
            position = memory['position']
            referent_id = memory['referent_id']
            
            # Convert to voxel coordinates
            voxel_coord = (position / self.voxel_size).floor().int()
            voxel_key = f"{voxel_coord[0]}_{voxel_coord[1]}_{voxel_coord[2]}"
            
            if voxel_key not in locus_set:
                locus_set[voxel_key] = {
                    'position': position,
                    'referents': set(),
                    'count': 0
                }
            
            locus_set[voxel_key]['referents'].add(referent_id)
            locus_set[voxel_key]['count'] += 1
        
        return locus_set
    
    def _bayesian_fusion(self, 
                        pointing_vector: torch.Tensor,
                        locus_set: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Apply Bayesian fusion for referent resolution.
        
        Args:
            pointing_vector: Current pointing direction g(t)
            locus_set: Voxelized locus set C_t
            
        Returns:
            Dictionary mapping referent_id to posterior probability
        """
        if not locus_set:
            return {}
        
        # Compute total prior count for normalization
        total_prior = sum(self.referent_priors.values()) + 1e-8
        
        # Compute likelihoods for each referent
        likelihoods = {}
        
        for referent_id in self.referent_priors:
            # Get referent position (or use a default)
            if referent_id in self.referent_positions:
                referent_pos = self.referent_positions[referent_id]
            else:
                # Use center of relevant voxels
                relevant_voxels = [
                    voxel for voxel in locus_set.values()
                    if referent_id in voxel['referents']
                ]
                if relevant_voxels:
                    referent_pos = torch.stack([v['position'] for v in relevant_voxels]).mean(dim=0)
                else:
                    continue
            
            # Compute direction to referent
            direction = referent_pos / (torch.norm(referent_pos) + 1e-8)
            
            # Compute angle between pointing vector and referent direction
            cos_angle = torch.dot(pointing_vector, direction)
            cos_angle = torch.clamp(cos_angle, -1, 1)
            angle = torch.acos(cos_angle)
            
            # Likelihood: ℓ_c(r) = exp(-0.5(∠(g(t),r̂)/σ_pt)²)
            likelihood = torch.exp(-0.5 * (angle / self.pointing_sigma)**2)
            likelihoods[referent_id] = likelihood.item()
        
        # Normalize likelihoods
        total_likelihood = sum(likelihoods.values()) + 1e-8
        for referent_id in likelihoods:
            likelihoods[referent_id] = likelihoods[referent_id] / total_likelihood
        
        # Bayesian fusion: p(r|C) ∝ ℓ_c(r) · p(r)
        posteriors = {}
        for referent_id in self.referent_priors:
            prior = self.referent_priors[referent_id] / total_prior
            likelihood = likelihoods.get(referent_id, 1e-8)
            posteriors[referent_id] = likelihood * prior
        
        # Normalize posteriors
        total_posterior = sum(posteriors.values()) + 1e-8
        for referent_id in posteriors:
            posteriors[referent_id] = posteriors[referent_id] / total_posterior
        
        return posteriors
    
    def _extract_spatial_features(self,
                                 pointing_vector: torch.Tensor,
                                 locus_set: Dict[str, torch.Tensor],
                                 referent_probs: Dict[str, float]) -> torch.Tensor:
        """Extract spatial features for the model.
        
        Args:
            pointing_vector: Pointing direction g(t)
            locus_set: Voxelized locus set C_t
            referent_probs: Posterior probabilities
            
        Returns:
            Spatial feature vector of shape (D,)
        """
        features = []
        
        # Pointing vector components
        features.extend(pointing_vector.tolist())
        
        # Locus set statistics
        num_voxels = len(locus_set)
        avg_density = np.mean([v['count'] for v in locus_set.values()]) if locus_set else 0
        
        features.extend([num_voxels, avg_density])
        
        # Referent probability distribution features
        if referent_probs:
            max_prob = max(referent_probs.values())
            entropy = -sum(p * math.log(p + 1e-8) for p in referent_probs.values())
            num_referents = len(referent_probs)
        else:
            max_prob = 0.0
            entropy = 0.0
            num_referents = 0
            
        features.extend([max_prob, entropy, num_referents])
        
        # Pad to fixed size if needed
        target_size = 16
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        
        return torch.tensor(features[:target_size])
    
    def reset(self):
        """Reset spatial memory."""
        self.spatial_memory.clear()
        self.referent_positions.clear()
        self.referent_priors.clear()
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about spatial memory.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            'memory_size': len(self.spatial_memory),
            'num_referents': len(self.referent_positions),
            'window_capacity': self.window_frames
        }