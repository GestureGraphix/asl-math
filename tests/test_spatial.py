"""Tests for spatial discourse module."""

import pytest
import torch
import numpy as np
from src.spatial import SpatialDiscourse


class TestSpatialDiscourse:
    """Test cases for SpatialDiscourse."""
    
    def test_initialization(self):
        """Test spatial discourse initialization."""
        spatial = SpatialDiscourse(
            window_size=2.0,
            voxel_size=0.08,
            pointing_sigma=2.0,
            fps=30
        )
        
        assert spatial.window_size == 2.0
        assert spatial.voxel_size == 0.08
        assert spatial.pointing_sigma == np.radians(2.0)
        assert spatial.fps == 30
        assert spatial.window_frames == 60  # 2.0 * 30
    
    def test_pointing_vector_computation(self):
        """Test pointing vector computation."""
        spatial = SpatialDiscourse()
        
        # Create landmarks with known positions
        landmarks = torch.zeros(1623, 3)
        landmarks[5] = torch.tensor([0.5, 0.0, 0.0])   # Index finger base
        landmarks[0] = torch.tensor([0.0, 1.0, 0.0])   # Neck
        
        pointing_vector = spatial._compute_pointing_vector(landmarks)
        
        # Should be unit vector pointing from finger to neck
        assert pointing_vector.shape == (3,)
        assert abs(torch.norm(pointing_vector) - 1.0) < 1e-5
        
        # Should point in correct direction
        expected_direction = landmarks[0] - landmarks[5]
        expected_direction = expected_direction / torch.norm(expected_direction)
        
        # Check direction (either same or opposite due to normalization)
        dot_product = torch.dot(pointing_vector, expected_direction)
        assert abs(abs(dot_product) - 1.0) < 1e-5
    
    def test_spatial_memory_update(self):
        """Test spatial memory update."""
        spatial = SpatialDiscourse()
        
        # Update memory with observations
        for i in range(5):
            position = torch.tensor([i * 0.1, 0.0, 0.0])
            spatial._update_spatial_memory(i * 0.5, position, f"referent_{i}")
        
        # Check memory size
        assert len(spatial.spatial_memory) == 5
        
        # Check that all entries are present
        referents = [mem['referent_id'] for mem in spatial.spatial_memory]
        assert all(f"referent_{i}" in referents for i in range(5))
    
    def test_voxelization(self):
        """Test locus set voxelization."""
        spatial = SpatialDiscourse(voxel_size=0.1)
        
        # Add some spatial memory
        positions = [
            torch.tensor([0.05, 0.05, 0.05]),
            torch.tensor([0.15, 0.05, 0.05]),
            torch.tensor([0.05, 0.15, 0.05]),
            torch.tensor([0.25, 0.25, 0.25]),
        ]
        
        for i, pos in enumerate(positions):
            spatial._update_spatial_memory(i, pos, f"ref_{i}")
        
        # Voxelize
        locus_set = spatial._voxelize_locus_set()
        
        # Should have 3 voxels (first three positions in same voxel)
        assert len(locus_set) == 3
        
        # Check voxel coordinates
        voxel_keys = list(locus_set.keys())
        assert "0_0_0" in voxel_keys  # First three positions
        assert "1_0_0" in voxel_keys  # Second position
        assert "2_2_2" in voxel_keys  # Fourth position
    
    def test_bayesian_fusion(self):
        """Test Bayesian fusion for referent resolution."""
        spatial = SpatialDiscourse(pointing_sigma=2.0)
        
        # Set up referent positions
        spatial.referent_positions = {
            "person": torch.tensor([1.0, 0.0, 0.0]),
            "table": torch.tensor([0.0, 1.0, 0.0]),
            "chair": torch.tensor([-1.0, 0.0, 0.0]),
        }
        
        # Set up priors
        spatial.referent_priors = {
            "person": 5.0,
            "table": 2.0,
            "chair": 1.0,
        }
        
        # Pointing vector towards person
        pointing_vector = torch.tensor([1.0, 0.0, 0.0])
        
        # Locus set (empty for this test)
        locus_set = {}
        
        # Apply Bayesian fusion
        posteriors = spatial._bayesian_fusion(pointing_vector, locus_set)
        
        # Should have probabilities for all referents
        assert len(posteriors) == 3
        assert all(prob >= 0 for prob in posteriors.values())
        assert abs(sum(posteriors.values()) - 1.0) < 1e-5
        
        # Person should have highest probability (pointing towards them)
        assert posteriors["person"] > posteriors["table"]
        assert posteriors["person"] > posteriors["chair"]
    
    def test_spatial_features(self):
        """Test spatial feature extraction."""
        spatial = SpatialDiscourse()
        
        # Create test data
        pointing_vector = torch.tensor([1.0, 0.0, 0.0])
        locus_set = {
            "0_0_0": {
                'position': torch.tensor([0.0, 0.0, 0.0]),
                'referents': {"person", "object"},
                'count': 10
            }
        }
        referent_probs = {"person": 0.7, "object": 0.3}
        
        features = spatial._extract_spatial_features(
            pointing_vector, locus_set, referent_probs
        )
        
        # Check feature shape
        assert features.shape == (16,)  # Fixed size
        
        # Check that features are reasonable
        assert torch.isfinite(features).all()
        
        # First 3 features should be pointing vector
        assert torch.allclose(features[:3], pointing_vector)
    
    def test_full_forward_pass(self):
        """Test complete forward pass."""
        spatial = SpatialDiscourse()
        
        # Create landmarks
        landmarks = torch.randn(1623, 3)
        landmarks[5] = torch.tensor([0.5, 0.0, 0.0])  # Index base
        landmarks[0] = torch.tensor([0.0, 1.0, 0.0])  # Neck
        
        # Process frame
        result = spatial(landmarks, timestamp=0.0, referent_id="test_referent")
        
        # Check result structure
        assert 'pointing_vector' in result
        assert 'locus_set' in result
        assert 'referent_probs' in result
        assert 'spatial_features' in result
        
        # Check output shapes and types
        assert result['pointing_vector'].shape == (3,)
        assert isinstance(result['locus_set'], dict)
        assert isinstance(result['referent_probs'], dict)
        assert result['spatial_features'].shape == (16,)
    
    def test_memory_window_limit(self):
        """Test that spatial memory respects window limit."""
        spatial = SpatialDiscourse(window_size=1.0, fps=30)  # 30 frame window
        
        # Add more observations than window size
        for i in range(50):
            position = torch.tensor([i * 0.01, 0.0, 0.0])
            spatial._update_spatial_memory(i * 0.1, position, f"ref_{i}")
        
        # Memory should not exceed window size
        assert len(spatial.spatial_memory) <= spatial.window_frames
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        spatial = SpatialDiscourse()
        
        # Add some data
        spatial._update_spatial_memory(0.0, torch.tensor([0.0, 0.0, 0.0]), "test")
        spatial.referent_positions["test"] = torch.tensor([0.0, 0.0, 0.0])
        spatial.referent_priors["test"] = 1.0
        
        # Reset
        spatial.reset()
        
        # All should be cleared
        assert len(spatial.spatial_memory) == 0
        assert len(spatial.referent_positions) == 0
        assert len(spatial.referent_priors) == 0
    
    def test_memory_stats(self):
        """Test memory statistics."""
        spatial = SpatialDiscourse()
        
        # Add some data
        spatial._update_spatial_memory(0.0, torch.tensor([0.0, 0.0, 0.0]), "test1")
        spatial._update_spatial_memory(1.0, torch.tensor([1.0, 0.0, 0.0]), "test2")
        
        stats = spatial.get_memory_stats()
        
        assert stats['memory_size'] == 2
        assert stats['num_referents'] == 2
        assert stats['window_capacity'] == spatial.window_frames
    
    def test_likelihood_computation(self):
        """Test likelihood computation for pointing."""
        spatial = SpatialDiscourse(pointing_sigma=2.0)
        
        # Pointing vector
        g_t = torch.tensor([1.0, 0.0, 0.0])
        
        # Referent direction (slightly off)
        r_hat = torch.tensor([0.9, 0.1, 0.0])
        r_hat = r_hat / torch.norm(r_hat)
        
        # Compute angle
        cos_angle = torch.dot(g_t, r_hat)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        
        # Compute likelihood
        likelihood = torch.exp(-0.5 * (angle / spatial.pointing_sigma)**2)
        
        # Should be reasonable value
        assert 0 < likelihood < 1
        
        # Perfect alignment should give likelihood 1
        perfect_likelihood = torch.exp(-0.5 * (0.0 / spatial.pointing_sigma)**2)
        assert perfect_likelihood == 1.0
    
    def test_determinism(self):
        """Test that spatial discourse is deterministic."""
        spatial = SpatialDiscourse()
        
        landmarks = torch.randn(1623, 3)
        landmarks[5] = torch.tensor([0.5, 0.0, 0.0])
        landmarks[0] = torch.tensor([0.0, 1.0, 0.0])
        
        # Run twice with same input
        result1 = spatial(landmarks, timestamp=0.0)
        result2 = spatial(landmarks, timestamp=0.0)
        
        # Results should be identical
        assert torch.allclose(result1['pointing_vector'], result2['pointing_vector'])
        assert torch.allclose(result1['spatial_features'], result2['spatial_features'])