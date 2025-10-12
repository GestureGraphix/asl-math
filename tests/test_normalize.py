"""Tests for Sim(3) normalization module."""

import pytest
import torch
import numpy as np
from src.normalize import Sim3Normalizer


class TestSim3Normalizer:
    """Test cases for Sim3Normalizer."""
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = Sim3Normalizer(epsilon=1e-8)
        assert normalizer.epsilon == 1e-8
    
    def test_forward_pass_torch(self):
        """Test forward pass with PyTorch tensors."""
        normalizer = Sim3Normalizer()
        
        # Create dummy landmarks (T=10, N=1623, 3)
        landmarks = torch.randn(10, 1623, 3)
        
        # Set shoulder positions for meaningful normalization
        landmarks[:, 11, :] = torch.tensor([1.0, 0.0, 0.0])  # Left shoulder
        landmarks[:, 12, :] = torch.tensor([-1.0, 0.0, 0.0])  # Right shoulder
        landmarks[:, 0, :] = torch.tensor([0.0, 1.0, 0.0])   # Neck
        
        # Apply normalization
        normalized = normalizer(landmarks)
        
        # Check output shape
        assert normalized.shape == landmarks.shape
        
        # Check that values are reasonable (should be normalized)
        assert torch.isfinite(normalized).all()
        assert normalized.abs().max() < 10.0  # Should be bounded
    
    def test_forward_pass_numpy(self):
        """Test forward pass with NumPy arrays."""
        normalizer = Sim3Normalizer()
        
        # Create dummy landmarks
        landmarks = np.random.randn(5, 1623, 3).astype(np.float32)
        
        # Set shoulder positions
        landmarks[:, 11, :] = [1.0, 0.0, 0.0]
        landmarks[:, 12, :] = [-1.0, 0.0, 0.0]
        landmarks[:, 0, :] = [0.0, 1.0, 0.0]
        
        # Apply normalization
        normalized = normalizer.normalize_numpy(landmarks)
        
        # Check output shape
        assert normalized.shape == landmarks.shape
        
        # Check that values are reasonable
        assert np.isfinite(normalized).all()
        assert np.abs(normalized).max() < 10.0
    
    def test_single_frame(self):
        """Test normalization with single frame."""
        normalizer = Sim3Normalizer()
        
        # Single frame (N=1623, 3)
        landmarks = torch.randn(1623, 3)
        landmarks[11] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[12] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[0] = torch.tensor([0.0, 1.0, 0.0])
        
        normalized = normalizer(landmarks)
        
        # Should add time dimension
        assert normalized.shape == (1, 1623, 3)
    
    def test_yaw_alignment(self):
        """Test yaw alignment computation."""
        normalizer = Sim3Normalizer()
        
        # Test shoulder vector alignment
        shoulder_vector = torch.tensor([2.0, 0.0, 1.0])
        rotation = normalizer._compute_yaw_alignment(shoulder_vector)
        
        # Check rotation matrix properties
        assert rotation.shape == (3, 3)
        
        # Should be orthogonal
        identity_approx = rotation @ rotation.T
        identity = torch.eye(3)
        assert torch.allclose(identity_approx, identity, atol=1e-5)
        
        # Should have determinant 1
        det = torch.det(rotation)
        assert abs(det - 1.0) < 1e-5
    
    def test_yaw_alignment_numpy(self):
        """Test yaw alignment with NumPy."""
        normalizer = Sim3Normalizer()
        
        shoulder_vector = np.array([2.0, 0.0, 1.0], dtype=np.float32)
        rotation = normalizer._compute_yaw_alignment_numpy(shoulder_vector)
        
        # Check rotation matrix properties
        assert rotation.shape == (3, 3)
        
        # Should be orthogonal
        identity_approx = rotation @ rotation.T
        identity = np.eye(3, dtype=np.float32)
        assert np.allclose(identity_approx, identity, atol=1e-5)
        
        # Should have determinant 1
        det = np.linalg.det(rotation)
        assert abs(det - 1.0) < 1e-5
    
    def test_edge_cases(self):
        """Test edge cases."""
        normalizer = Sim3Normalizer()
        
        # Test with zero shoulder distance (should use epsilon)
        landmarks = torch.randn(1, 1623, 3)
        landmarks[0, 11] = torch.tensor([0.0, 0.0, 0.0])
        landmarks[0, 12] = torch.tensor([0.0, 0.0, 0.0])  # Zero distance
        landmarks[0, 0] = torch.tensor([0.0, 1.0, 0.0])
        
        normalized = normalizer(landmarks)
        
        # Should not produce NaN or inf
        assert torch.isfinite(normalized).all()
    
    def test_determinism(self):
        """Test that normalization is deterministic."""
        normalizer = Sim3Normalizer()
        
        landmarks = torch.randn(5, 1623, 3)
        landmarks[:, 11] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[:, 12] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[:, 0] = torch.tensor([0.0, 1.0, 0.0])
        
        # Run normalization twice
        normalized1 = normalizer(landmarks)
        normalized2 = normalizer(landmarks)
        
        # Should be identical
        assert torch.allclose(normalized1, normalized2, atol=1e-6)
    
    def test_batch_processing(self):
        """Test batch processing."""
        normalizer = Sim3Normalizer()
        
        # Different batch sizes
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            landmarks = torch.randn(batch_size, 1623, 3)
            landmarks[:, 11] = torch.tensor([1.0, 0.0, 0.0])
            landmarks[:, 12] = torch.tensor([-1.0, 0.0, 0.0])
            landmarks[:, 0] = torch.tensor([0.0, 1.0, 0.0])
            
            normalized = normalizer(landmarks)
            
            # Should preserve batch dimension
            assert normalized.shape[0] == batch_size
            assert normalized.shape[1:] == landmarks.shape[1:]
    
    def test_normalization_invariance(self):
        """Test that normalization provides scale and rotation invariance."""
        normalizer = Sim3Normalizer()
        
        # Create base landmarks
        base_landmarks = torch.randn(1, 1623, 3)
        base_landmarks[0, 11] = torch.tensor([1.0, 0.0, 0.0])
        base_landmarks[0, 12] = torch.tensor([-1.0, 0.0, 0.0])
        base_landmarks[0, 0] = torch.tensor([0.0, 1.0, 0.0])
        
        base_normalized = normalizer(base_landmarks)
        
        # Apply scaling
        scaled_landmarks = base_landmarks * 2.0
        scaled_normalized = normalizer(scaled_landmarks)
        
        # Apply translation
        translated_landmarks = base_landmarks + torch.tensor([5.0, 3.0, -2.0])
        translated_normalized = normalizer(translated_landmarks)
        
        # Normalized results should be similar (up to numerical precision)
        assert torch.allclose(base_normalized, scaled_normalized, atol=1e-4)
        assert torch.allclose(base_normalized, translated_normalized, atol=1e-4)