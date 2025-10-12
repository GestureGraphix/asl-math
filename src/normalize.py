"""Sim(3) Normalization Module.

This module implements the Sim(3) normalization described in the paper:
X̃_t = (X_t - T_t)R_t^⊤ / s_t

Where:
- s_t = ‖B_t[RS] - B_t[LS]‖₂ (shoulder distance)
- T_t = B_t[NECK] (neck position)  
- R_t = yaw_align(B_t[RS] - B_t[LS]) (rotation alignment)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class Sim3Normalizer(nn.Module):
    """Sim(3) normalization for ASL landmark sequences.
    
    Normalizes landmark coordinates using shoulder distance for scale,
    neck position for translation, and shoulder vector for rotation alignment.
    
    Attributes:
        epsilon: Small value for numerical stability
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """Initialize Sim(3) normalizer.
        
        Args:
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
        
        # Landmark indices (from MediaPipe)
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.NECK = 0  # Approximate neck position
        
    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply Sim(3) normalization to landmark sequence.
        
        Args:
            landmarks: Landmark tensor of shape (T, N, 3) where
                      T = time steps, N = number of landmarks, 3 = (x,y,z)
                      
        Returns:
            Normalized landmarks of shape (T, N, 3)
        """
        if landmarks.dim() == 2:
            landmarks = landmarks.unsqueeze(0)  # Add time dimension if missing
            
        T, N, _ = landmarks.shape
        normalized = torch.zeros_like(landmarks)
        
        for t in range(T):
            normalized[t] = self._normalize_frame(landmarks[t])
            
        return normalized
    
    def _normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Normalize a single frame of landmarks.
        
        Args:
            frame: Landmarks for one frame, shape (N, 3)
            
        Returns:
            Normalized landmarks, shape (N, 3)
        """
        # Extract key points
        left_shoulder = frame[self.LEFT_SHOULDER]
        right_shoulder = frame[self.RIGHT_SHOULDER]
        neck = frame[self.NECK]
        
        # Compute scale factor s_t = ‖B_t[RS] - B_t[LS]‖₂
        shoulder_vector = right_shoulder - left_shoulder
        s_t = torch.norm(shoulder_vector, p=2) + self.epsilon
        
        # Compute translation T_t = B_t[NECK]
        T_t = neck
        
        # Compute rotation R_t = yaw_align(B_t[RS] - B_t[LS])
        R_t = self._compute_yaw_alignment(shoulder_vector)
        
        # Apply normalization: X̃_t = (X_t - T_t)R_t^⊤ / s_t
        centered = frame - T_t.unsqueeze(0)
        rotated = torch.mm(centered, R_t.T)
        normalized = rotated / s_t
        
        return normalized
    
    def _compute_yaw_alignment(self, shoulder_vector: torch.Tensor) -> torch.Tensor:
        """Compute yaw alignment rotation matrix.
        
        Args:
            shoulder_vector: Vector from left to right shoulder, shape (3,)
            
        Returns:
            Rotation matrix R_t, shape (3, 3)
        """
        # Project shoulder vector to xz-plane for yaw alignment
        shoulder_xz = torch.tensor([shoulder_vector[0], 0.0, shoulder_vector[2]])
        
        # Normalize
        shoulder_xz = shoulder_xz / (torch.norm(shoulder_xz) + self.epsilon)
        
        # Compute rotation angle (yaw)
        # Align shoulder vector with x-axis
        target = torch.tensor([1.0, 0.0, 0.0])
        
        # Rotation axis (y-axis)
        axis = torch.tensor([0.0, 1.0, 0.0])
        
        # Compute rotation matrix using Rodrigues' formula
        cos_theta = torch.dot(shoulder_xz, target)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Numerical stability
        
        # Handle parallel vectors
        if abs(cos_theta) > 0.9999:
            return torch.eye(3)
        
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        
        # Rodrigues' rotation formula
        K = torch.tensor([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        
        R = (torch.eye(3) + sin_theta * K + 
             (1.0 - cos_theta) * torch.mm(K, K))
        
        return R
    
    def normalize_numpy(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply Sim(3) normalization using NumPy backend.
        
        Args:
            landmarks: Landmark array of shape (T, N, 3)
            
        Returns:
            Normalized landmarks, shape (T, N, 3)
        """
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis, ...]
            
        T, N, _ = landmarks.shape
        normalized = np.zeros_like(landmarks)
        
        for t in range(T):
            normalized[t] = self._normalize_frame_numpy(landmarks[t])
            
        return normalized
    
    def _normalize_frame_numpy(self, frame: np.ndarray) -> np.ndarray:
        """Normalize a single frame using NumPy.
        
        Args:
            frame: Landmarks for one frame, shape (N, 3)
            
        Returns:
            Normalized landmarks, shape (N, 3)
        """
        # Extract key points
        left_shoulder = frame[self.LEFT_SHOULDER]
        right_shoulder = frame[self.RIGHT_SHOULDER]
        neck = frame[self.NECK]
        
        # Compute scale factor
        shoulder_vector = right_shoulder - left_shoulder
        s_t = np.linalg.norm(shoulder_vector) + self.epsilon
        
        # Translation
        T_t = neck
        
        # Rotation
        R_t = self._compute_yaw_alignment_numpy(shoulder_vector)
        
        # Apply normalization
        centered = frame - T_t[np.newaxis, :]
        rotated = centered @ R_t.T
        normalized = rotated / s_t
        
        return normalized
    
    def _compute_yaw_alignment_numpy(self, shoulder_vector: np.ndarray) -> np.ndarray:
        """Compute yaw alignment rotation matrix using NumPy.
        
        Args:
            shoulder_vector: Vector from left to right shoulder, shape (3,)
            
        Returns:
            Rotation matrix R_t, shape (3, 3)
        """
        # Project to xz-plane
        shoulder_xz = np.array([shoulder_vector[0], 0.0, shoulder_vector[2]])
        shoulder_xz = shoulder_xz / (np.linalg.norm(shoulder_xz) + self.epsilon)
        
        # Target alignment
        target = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 1.0, 0.0])
        
        cos_theta = np.clip(np.dot(shoulder_xz, target), -1.0, 1.0)
        
        if abs(cos_theta) > 0.9999:
            return np.eye(3)
        
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        # Rodrigues' formula
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        
        R = (np.eye(3) + sin_theta * K + 
             (1.0 - cos_theta) * (K @ K))
        
        return R