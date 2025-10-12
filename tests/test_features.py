"""Tests for feature extraction and product VQ modules."""

import pytest
import torch
import numpy as np
from src.features import FeatureExtractor, ProductVQ, VectorQuantizer
from src.vocab import Vocabulary


class TestVectorQuantizer:
    """Test cases for VectorQuantizer."""
    
    def test_initialization(self):
        """Test VQ initialization."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=10, beta=0.25)
        
        assert vq.num_embeddings == 64
        assert vq.embedding_dim == 10
        assert vq.beta == 0.25
        assert vq.embedding.weight.shape == (64, 10)
    
    def test_forward_pass(self):
        """Test VQ forward pass."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=10, beta=0.25)
        
        # Test with 2D input
        x = torch.randn(32, 10)
        quantized, indices, loss = vq(x)
        
        assert quantized.shape == x.shape
        assert indices.shape == (32,)
        assert loss >= 0
        assert indices.min() >= 0
        assert indices.max() < 64
    
    def test_forward_pass_3d(self):
        """Test VQ forward pass with 3D input."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=10, beta=0.25)
        
        # Test with 3D input (B, T, D)
        x = torch.randn(8, 20, 10)
        quantized, indices, loss = vq(x)
        
        assert quantized.shape == x.shape
        assert indices.shape == (8, 20)
        assert loss >= 0
    
    def test_kmeans_initialization(self):
        """Test K-means initialization."""
        vq = VectorQuantizer(num_embeddings=8, embedding_dim=5, beta=0.25)
        
        # Create training data
        data = torch.randn(1000, 5)
        
        # Initialize with K-means
        vq.initialize_kmeans(data, num_samples=100)
        
        # Check that embeddings are updated
        assert not torch.allclose(vq.embedding.weight, torch.zeros_like(vq.embedding.weight))
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=10, beta=0.25)
        
        x = torch.randn(16, 10, requires_grad=True)
        quantized, indices, loss = vq(x)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert vq.embedding.weight.grad is not None


class TestProductVQ:
    """Test cases for ProductVQ."""
    
    def test_initialization(self):
        """Test ProductVQ initialization."""
        vocab = Vocabulary()
        product_vq = ProductVQ(vocab, beta=0.25)
        
        assert len(product_vq.quantizers) == 5  # 5 modalities
        assert 'hands' in product_vq.quantizers
        assert 'location' in product_vq.quantizers
        assert 'orientation' in product_vq.quantizers
        assert 'motion' in product_vq.quantizers
        assert 'non_manual' in product_vq.quantizers
    
    def test_forward_pass(self):
        """Test ProductVQ forward pass."""
        vocab = Vocabulary()
        product_vq = ProductVQ(vocab, beta=0.25)
        
        # Create feature dictionary
        features = {
            'hands': torch.randn(8, 10),      # (B, D_hands)
            'location': torch.randn(8, 6),    # (B, D_location)
            'orientation': torch.randn(8, 6), # (B, D_orientation)
            'motion': torch.randn(8, 9),      # (B, D_motion)
            'non_manual': torch.randn(8, 5),  # (B, D_non_manual)
        }
        
        quantized, indices, loss = product_vq(features)
        
        # Check outputs
        assert len(quantized) == 5
        assert len(indices) == 5
        assert loss >= 0
        
        # Check shapes are preserved
        for modality in features:
            assert quantized[modality].shape == features[modality].shape
            assert indices[modality].shape == (8,)
    
    def test_forward_pass_temporal(self):
        """Test ProductVQ with temporal data."""
        vocab = Vocabulary()
        product_vq = ProductVQ(vocab, beta=0.25)
        
        # Create temporal features (B, T, D)
        features = {
            'hands': torch.randn(4, 20, 10),
            'location': torch.randn(4, 20, 6),
            'orientation': torch.randn(4, 20, 6),
            'motion': torch.randn(4, 20, 9),
            'non_manual': torch.randn(4, 20, 5),
        }
        
        quantized, indices, loss = product_vq(features)
        
        # Check temporal shapes
        for modality in features:
            assert quantized[modality].shape == features[modality].shape
            assert indices[modality].shape == (4, 20)
    
    def test_codebook_initialization(self):
        """Test codebook initialization."""
        vocab = Vocabulary()
        product_vq = ProductVQ(vocab, beta=0.25)
        
        # Create training features
        features = {
            'hands': torch.randn(1000, 10),
            'location': torch.randn(1000, 6),
            'orientation': torch.randn(1000, 6),
            'motion': torch.randn(1000, 9),
            'non_manual': torch.randn(1000, 5),
        }
        
        # Initialize codebooks
        product_vq.initialize_codebooks(features)
        
        # Check that embeddings are updated
        for modality in product_vq.quantizers:
            weight = product_vq.quantizers[modality].embedding.weight
            assert not torch.allclose(weight, torch.zeros_like(weight))
    
    def test_total_loss_computation(self):
        """Test total loss computation."""
        vocab = Vocabulary()
        product_vq = ProductVQ(vocab, beta=0.25)
        
        features = {
            'hands': torch.randn(16, 10),
            'location': torch.randn(16, 6),
            'orientation': torch.randn(16, 6),
            'motion': torch.randn(16, 9),
            'non_manual': torch.randn(16, 5),
        }
        
        quantized, indices, total_loss = product_vq(features)
        
        # Loss should be sum of individual losses
        individual_losses = []
        for modality in product_vq.quantizers:
            _, _, loss = product_vq.quantizers[modality](features[modality])
            individual_losses.append(loss.item())
        
        expected_loss = sum(individual_losses)
        assert abs(total_loss.item() - expected_loss) < 1e-5


class TestFeatureExtractor:
    """Test cases for FeatureExtractor."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        
        # Check that landmark indices are defined
        assert hasattr(extractor, 'hand_landmarks')
        assert hasattr(extractor, 'face_landmarks')
        assert 'thumb' in extractor.hand_landmarks
        assert 'palm' in extractor.hand_landmarks
    
    def test_forward_pass(self):
        """Test feature extraction forward pass."""
        extractor = FeatureExtractor()
        
        # Create dummy landmarks (T=10, N=1623, 3)
        landmarks = torch.randn(10, 1623, 3)
        
        # Set hand landmarks (first 21 points)
        landmarks[:, :21, :] = torch.randn(10, 21, 3)
        
        # Extract features
        features = extractor(landmarks)
        
        # Check output structure
        assert isinstance(features, dict)
        assert len(features) == 5  # 5 modalities
        
        # Check expected modalities
        expected_modalities = ['hands', 'location', 'orientation', 'motion', 'non_manual']
        for modality in expected_modalities:
            assert modality in features
        
        # Check output shapes
        assert features['hands'].shape == (10, 10)      # 5 flexions + thumb angle
        assert features['location'].shape == (10, 6)    # palm centers
        assert features['orientation'].shape == (10, 6)  # palm normals
        assert features['motion'].shape == (10, 9)      # deltas
        assert features['non_manual'].shape == (10, 5)  # gaze, mouth, etc.
    
    def test_hand_feature_extraction(self):
        """Test hand feature extraction."""
        extractor = FeatureExtractor()
        
        # Create hand landmarks
        hand_landmarks = torch.randn(1, 21, 3)
        
        # Set reasonable finger positions
        # Thumb
        hand_landmarks[0, [1, 2, 3, 4]] = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0], 
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0]
        ])
        
        # Index finger
        hand_landmarks[0, [5, 6, 7, 8]] = torch.tensor([
            [0.1, 0.1, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.1, 0.0],
            [0.4, 0.1, 0.0]
        ])
        
        features = extractor._extract_hand_features(hand_landmarks)
        
        # Check output shape
        assert features.shape == (1, 10)
        
        # Check that features are reasonable
        assert torch.isfinite(features).all()
        assert features.min() >= 0  # Angles should be non-negative
    
    def test_finger_flexion_computation(self):
        """Test finger flexion angle computation."""
        extractor = FeatureExtractor()
        
        # Create hand with bent finger
        hand = torch.zeros(21, 3)
        
        # Bent index finger
        hand[extractor.hand_landmarks['index']] = torch.tensor([
            [0.0, 0.0, 0.0],  # Base
            [0.1, 0.0, 0.0],  # First joint
            [0.15, 0.05, 0.0],  # Bent second joint
            [0.18, 0.08, 0.0],  # Tip
        ])
        
        flexion = extractor._compute_finger_flexion(hand, 'index')
        
        # Should compute positive flexion angle
        assert flexion > 0
        assert np.isfinite(flexion)
    
    def test_location_feature_extraction(self):
        """Test location feature extraction."""
        extractor = FeatureExtractor()
        
        hand_landmarks = torch.randn(5, 21, 3)
        locations = extractor._extract_location_features(hand_landmarks)
        
        # Check output shape
        assert locations.shape == (5, 6)  # 2 palm centers * 3 coordinates
        
        # Check that values are reasonable
        assert torch.isfinite(locations).all()
    
    def test_orientation_feature_extraction(self):
        """Test orientation feature extraction."""
        extractor = FeatureExtractor()
        
        hand_landmarks = torch.randn(5, 21, 3)
        orientations = extractor._extract_orientation_features(hand_landmarks)
        
        # Check output shape
        assert orientations.shape == (5, 6)  # 2 palm normals * 3 coordinates
        
        # Check that normals are unit vectors
        for t in range(5):
            normal_left = orientations[t, :3]
            normal_right = orientations[t, 3:]
            
            norm_left = torch.norm(normal_left)
            norm_right = torch.norm(normal_right)
            
            assert abs(norm_left - 1.0) < 1e-5 or norm_left < 1e-5
            assert abs(norm_right - 1.0) < 1e-5 or norm_right < 1e-5
    
    def test_determinism(self):
        """Test that feature extraction is deterministic."""
        extractor = FeatureExtractor()
        
        landmarks = torch.randn(10, 1623, 3)
        landmarks[:, :21, :] = torch.randn(10, 21, 3)
        
        # Run extraction twice
        features1 = extractor(landmarks)
        features2 = extractor(landmarks)
        
        # Should be identical
        for modality in features1:
            assert torch.allclose(features1[modality], features2[modality], atol=1e-6)