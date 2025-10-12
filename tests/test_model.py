"""Tests for ASL translation model."""

import pytest
import torch
import numpy as np
from src.model import (
    CausalConv1d, CausalTCNBlock, CausalTCNEncoder, 
    CTCHead, ASLTranslationModel
)


class TestCausalConv1d:
    """Test cases for CausalConv1d."""
    
    def test_initialization(self):
        """Test causal convolution initialization."""
        conv = CausalConv1d(in_channels=10, out_channels=20, kernel_size=5)
        
        assert conv.kernel_size == 5
        assert conv.padding == 4  # (5-1)*1
        assert conv.dilation == 1
    
    def test_forward_pass(self):
        """Test causal convolution forward pass."""
        conv = CausalConv1d(in_channels=10, out_channels=20, kernel_size=5)
        
        # Input: (B, C, T)
        x = torch.randn(2, 10, 100)
        output = conv(x)
        
        # Check output shape
        assert output.shape == (2, 20, 100)  # Same temporal length
        
        # Check causality: output at time t should only depend on inputs up to t
        # Create input with zeros after time 50
        x_causal = torch.randn(1, 10, 100)
        x_causal[:, :, 50:] = 0
        
        output_causal = conv(x_causal)
        
        # Output after time 50 should be zero
        assert torch.allclose(output_causal[:, :, 50:], torch.zeros_like(output_causal[:, :, 50:]))
    
    def test_dilated_convolution(self):
        """Test dilated causal convolution."""
        conv = CausalConv1d(in_channels=10, out_channels=20, kernel_size=5, dilation=2)
        
        x = torch.randn(1, 10, 50)
        output = conv(x)
        
        assert output.shape == (1, 20, 50)
        assert conv.padding == 8  # (5-1)*2


class TestCausalTCNBlock:
    """Test cases for CausalTCNBlock."""
    
    def test_initialization(self):
        """Test TCN block initialization."""
        block = CausalTCNBlock(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            dilation=2,
            dropout=0.2
        )
        
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'relu')
        assert hasattr(block, 'dropout')
    
    def test_forward_pass(self):
        """Test TCN block forward pass."""
        block = CausalTCNBlock(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            dilation=1,
            dropout=0.0  # No dropout for testing
        )
        
        # Input: (B, C, T)
        x = torch.randn(2, 10, 50)
        output = block(x)
        
        # Check output shape
        assert output.shape == (2, 20, 50)
        
        # Check that residual connection works
        # For this test, we need matching dimensions
        block_same = CausalTCNBlock(
            in_channels=10,
            out_channels=10,  # Same as input
            kernel_size=5,
            dilation=1,
            dropout=0.0
        )
        
        x_same = torch.randn(1, 10, 30)
        output_same = block_same(x_same)
        
        assert output_same.shape == x_same.shape
    
    def test_residual_connection(self):
        """Test residual connection."""
        block = CausalTCNBlock(
            in_channels=10,
            out_channels=10,  # Same channels for residual
            kernel_size=5,
            dilation=1,
            dropout=0.0
        )
        
        x = torch.randn(1, 10, 20)
        output = block(x)
        
        # Output should be different from input (due to convolutions)
        assert not torch.allclose(output, x, atol=1e-6)
        
        # But should have same shape
        assert output.shape == x.shape


class TestCausalTCNEncoder:
    """Test cases for CausalTCNEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CausalTCNEncoder(
            input_dim=50,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.2
        )
        
        assert encoder.input_dim == 50
        assert encoder.hidden_dim == 256
        assert encoder.num_layers == 3
        assert len(encoder.tcn_layers) == 3
    
    def test_forward_pass(self):
        """Test encoder forward pass."""
        encoder = CausalTCNEncoder(
            input_dim=50,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Input: (B, T, D)
        x = torch.randn(2, 100, 50)
        output = encoder(x)
        
        # Check output shape
        assert output.shape == (2, 100, 256)
        
        # Check that temporal dimension is preserved
        assert output.shape[1] == x.shape[1]
    
    def test_causality(self):
        """Test that encoder maintains causality."""
        encoder = CausalTCNEncoder(
            input_dim=10,
            hidden_dim=20,
            num_layers=2,
            kernel_size=3,
            dropout=0.0
        )
        
        # Create causal input: zeros after time 50
        x = torch.randn(1, 100, 10)
        x[:, 50:, :] = 0
        
        output = encoder(x)
        
        # Output after time 50 should be close to zero
        # (allowing for small numerical errors)
        assert torch.allclose(output[:, 50:, :], torch.zeros_like(output[:, 50:, :]), atol=1e-4)
    
    def test_dilation_growth(self):
        """Test that dilation grows exponentially."""
        encoder = CausalTCNEncoder(
            input_dim=10,
            hidden_dim=20,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Check dilations: should be 1, 2, 4
        expected_dilations = [1, 2, 4]
        for i, layer in enumerate(encoder.tcn_layers):
            assert layer.conv1.dilation == expected_dilations[i]
            assert layer.conv2.dilation == expected_dilations[i]
    
    def test_determinism(self):
        """Test that encoder is deterministic."""
        encoder = CausalTCNEncoder(
            input_dim=10,
            hidden_dim=20,
            num_layers=2,
            kernel_size=5,
            dropout=0.0
        )
        
        x = torch.randn(1, 50, 10)
        
        # Run twice
        output1 = encoder(x)
        output2 = encoder(x)
        
        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


class TestCTCHead:
    """Test cases for CTCHead."""
    
    def test_initialization(self):
        """Test CTC head initialization."""
        ctc_head = CTCHead(input_dim=256, vocab_size=1000, blank_idx=0)
        
        assert ctc_head.input_dim == 256
        assert ctc_head.vocab_size == 1000
        assert ctc_head.blank_idx == 0
    
    def test_forward_pass(self):
        """Test CTC head forward pass."""
        ctc_head = CTCHead(input_dim=256, vocab_size=1000, blank_idx=0)
        
        # Input: (B, T, D)
        x = torch.randn(2, 100, 256)
        log_probs = ctc_head(x)
        
        # Check output shape
        assert log_probs.shape == (2, 100, 1000)
        
        # Check that outputs are log probabilities
        # Sum over vocabulary should be close to 1 for each time step
        probs = torch.exp(log_probs)
        sum_probs = probs.sum(dim=-1)
        assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5)
    
    def test_log_softmax(self):
        """Test that outputs are proper log probabilities."""
        ctc_head = CTCHead(input_dim=10, vocab_size=5, blank_idx=0)
        
        x = torch.randn(1, 10, 10)
        log_probs = ctc_head(x)
        
        # Check log probabilities
        assert log_probs.shape == (1, 10, 5)
        
        # Convert to probabilities and check
        probs = torch.exp(log_probs)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        
        # Check normalization
        sum_probs = probs.sum(dim=-1)
        assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5)


class TestASLTranslationModel:
    """Test cases for ASLTranslationModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.2,
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        assert model.input_dim == 50
        assert model.vocab_size == 1000
        assert model.blank_idx == 0
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'ctc_head')
    
    def test_forward_pass(self):
        """Test model forward pass."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Input: (B, T, D)
        batch_size = 2
        seq_len = 100
        features = torch.randn(batch_size, seq_len, 50)
        feature_lengths = torch.tensor([100, 80])
        
        outputs = model(features, feature_lengths)
        
        # Check outputs
        assert 'log_probs' in outputs
        assert outputs['log_probs'].shape == (batch_size, seq_len, 1000)
    
    def test_forward_pass_with_targets(self):
        """Test model forward pass with targets."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Input data
        batch_size = 2
        features = torch.randn(batch_size, 100, 50)
        feature_lengths = torch.tensor([100, 80])
        targets = torch.randint(1, 1000, (batch_size, 50))  # Skip blank
        target_lengths = torch.tensor([50, 40])
        vq_loss = torch.tensor(0.5)
        
        outputs = model(features, feature_lengths, targets, target_lengths, vq_loss)
        
        # Check all outputs
        assert 'loss' in outputs
        assert 'ctc_loss' in outputs
        assert 'cal_loss' in outputs
        assert 'log_probs' in outputs
        
        # Check loss values
        assert outputs['loss'] > 0
        assert outputs['ctc_loss'] > 0
        assert outputs['cal_loss'] >= 0
    
    def test_greedy_decoding(self):
        """Test greedy decoding."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=10,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Create log probabilities
        seq_len = 20
        log_probs = torch.randn(1, seq_len, 10)
        
        # Make some tokens more likely
        log_probs[0, 5, 3] = 10.0  # Strong peak at time 5, token 3
        log_probs[0, 10, 7] = 10.0  # Strong peak at time 10, token 7
        
        decoded = model.decode_greedy(log_probs)
        
        # Check output shape
        assert decoded.shape == (1, seq_len)
        
        # Check that decoding makes sense
        decoded_list = decoded[0].tolist()
        
        # Remove padding
        decoded_clean = [token for token in decoded_list if token != 0]
        
        # Should contain our high-probability tokens
        assert 3 in decoded_clean or 7 in decoded_clean
    
    def test_determinism(self):
        """Test that model is deterministic."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0  # No dropout for determinism
        )
        
        features = torch.randn(1, 50, 50)
        feature_lengths = torch.tensor([50])
        
        # Run twice
        output1 = model(features, feature_lengths)
        output2 = model(features, feature_lengths)
        
        # Should be identical
        assert torch.allclose(output1['log_probs'], output2['log_probs'], atol=1e-6)
    
    def test_batch_processing(self):
        """Test batch processing."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, 50, 50)
            feature_lengths = torch.full((batch_size,), 50)
            
            outputs = model(features, feature_lengths)
            
            # Check batch dimension is preserved
            assert outputs['log_probs'].shape[0] == batch_size
            assert outputs['log_probs'].shape[1] == 50
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        model = ASLTranslationModel(
            input_dim=50,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.0
        )
        
        # Different sequence lengths
        seq_lengths = [20, 50, 100, 200]
        
        for seq_len in seq_lengths:
            features = torch.randn(1, seq_len, 50)
            feature_lengths = torch.tensor([seq_len])
            
            outputs = model(features, feature_lengths)
            
            # Output length should match input length
            assert outputs['log_probs'].shape[1] == seq_len