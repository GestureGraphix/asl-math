"""Integration tests for complete ASL translation pipeline."""

import pytest
import torch
import numpy as np
import tempfile
import os
import json

from src.normalize import Sim3Normalizer
from src.features import FeatureExtractor, ProductVQ
from src.spatial import SpatialDiscourse
from src.model import ASLTranslationModel
from src.vocab import Vocabulary
from src.train import ASLTrainer
from src.evaluate import ASLEvaluator
from src.export import CompletePipelineExporter


class TestCompletePipeline:
    """Integration tests for complete ASL translation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test that all pipeline components can be initialized."""
        # Initialize vocabulary
        vocab = Vocabulary()
        
        # Initialize components
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Initialize model
        feature_dims = vocab.get_feature_dimensions()
        total_feature_dim = sum(feature_dims.values())
        
        model = ASLTranslationModel(
            input_dim=total_feature_dim,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=3,
            kernel_size=5,
            dropout=0.2,
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        # Check that all components exist
        assert normalizer is not None
        assert feature_extractor is not None
        assert product_vq is not None
        assert spatial_discourse is not None
        assert model is not None
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing of landmarks."""
        # Initialize components
        vocab = Vocabulary()
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Create dummy landmarks
        landmarks = torch.randn(1, 30, 1623, 3)  # (B, T, N, 3)
        
        # Set shoulder positions for meaningful normalization
        landmarks[0, :, 11, :] = torch.tensor([1.0, 0.0, 0.0])   # Left shoulder
        landmarks[0, :, 12, :] = torch.tensor([-1.0, 0.0, 0.0])  # Right shoulder
        landmarks[0, :, 0, :] = torch.tensor([0.0, 1.0, 0.0])    # Neck
        
        # Process through pipeline
        batch_size, seq_len, _, _ = landmarks.shape
        processed_features = []
        
        for b in range(batch_size):
            # Normalize landmarks
            normalized = normalizer(landmarks[b])  # (T, N, 3)
            
            # Extract features
            features = feature_extractor(normalized)
            
            # Apply product VQ
            quantized_features, vq_indices, vq_loss = product_vq(features)
            
            # Combine features
            combined_features = []
            for modality, feature in quantized_features.items():
                combined_features.append(feature)
            
            combined = torch.cat(combined_features, dim=-1)
            processed_features.append(combined)
        
        # Check that we got features
        assert len(processed_features) == batch_size
        assert all(feat.shape[0] == seq_len for feat in processed_features)
        
        # Check feature dimensions
        feature_dims = vocab.get_feature_dimensions()
        expected_dim = sum(feature_dims.values())
        assert processed_features[0].shape[1] == expected_dim
    
    def test_model_training_step(self):
        """Test a single training step."""
        # Initialize components
        vocab = Vocabulary()
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Initialize model
        feature_dims = vocab.get_feature_dimensions()
        total_feature_dim = sum(feature_dims.values())
        
        model = ASLTranslationModel(
            input_dim=total_feature_dim,
            vocab_size=1000,
            hidden_dim=128,  # Smaller for testing
            num_layers=2,    # Fewer layers for testing
            kernel_size=5,
            dropout=0.0,     # No dropout for determinism
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        # Create dummy batch
        batch_size = 2
        seq_len = 20
        landmarks = torch.randn(batch_size, seq_len, 1623, 3)
        
        # Set shoulder positions
        landmarks[:, :, 11, :] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[:, :, 12, :] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[:, :, 0, :] = torch.tensor([0.0, 1.0, 0.0])
        
        # Process batch
        processed_features = []
        feature_lengths = []
        
        for b in range(batch_size):
            normalized = normalizer(landmarks[b])
            features = feature_extractor(normalized)
            quantized_features, vq_indices, vq_loss = product_vq(features)
            
            combined_features = []
            for modality, feature in quantized_features.items():
                combined_features.append(feature)
            
            combined = torch.cat(combined_features, dim=-1)
            processed_features.append(combined)
            feature_lengths.append(combined.shape[0])
        
        # Pad features
        max_feat_len = max(f.shape[0] for f in processed_features)
        padded_features = torch.zeros(batch_size, max_feat_len, total_feature_dim)
        
        for b, feat in enumerate(processed_features):
            feat_len = feat.shape[0]
            padded_features[b, :feat_len, :] = feat
        
        feature_lengths = torch.tensor(feature_lengths)
        
        # Create dummy targets
        targets = torch.randint(1, 100, (batch_size, 10))
        target_lengths = torch.tensor([10, 8])
        
        # Forward pass
        outputs = model(padded_features, feature_lengths, targets, target_lengths, vq_loss)
        
        # Check outputs
        assert 'loss' in outputs
        assert 'ctc_loss' in outputs
        assert 'log_probs' in outputs
        
        # Check that loss is computed
        assert outputs['loss'].item() > 0
        assert outputs['ctc_loss'].item() > 0
        assert torch.isfinite(outputs['loss'])
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Initialize components
        vocab = Vocabulary()
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Initialize model
        feature_dims = vocab.get_feature_dimensions()
        total_feature_dim = sum(feature_dims.values())
        
        model = ASLTranslationModel(
            input_dim=total_feature_dim,
            vocab_size=100,
            hidden_dim=64,   # Small for testing
            num_layers=1,    # Single layer for testing
            kernel_size=5,
            dropout=0.0,
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        # Initialize evaluator
        evaluator = ASLEvaluator(
            model=model,
            feature_extractor=feature_extractor,
            product_vq=product_vq,
            normalizer=normalizer,
            spatial_discourse=spatial_discourse,
            device='cpu'
        )
        
        # Create dummy evaluation data
        landmarks = torch.randn(2, 30, 1623, 3)
        
        # Set shoulder positions
        landmarks[:, :, 11, :] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[:, :, 12, :] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[:, :, 0, :] = torch.tensor([0.0, 1.0, 0.0])
        
        # Create dummy dataloader-like object
        class DummyDataloader:
            def __init__(self, landmarks):
                self.landmarks = landmarks
            
            def __iter__(self):
                yield {
                    'landmarks': self.landmarks,
                    'targets': torch.randint(1, 50, (2, 10)),
                    'target_lengths': torch.tensor([10, 8]),
                    'referent_targets': ['person', 'table']
                }
        
        dataloader = DummyDataloader(landmarks)
        
        # Evaluate
        metrics = evaluator.evaluate(dataloader)
        
        # Check that we got metrics
        assert isinstance(metrics, dict)
        assert 'top1_accuracy' in metrics
        assert 'top5_accuracy' in metrics
        assert 'wer' in metrics
        assert 'referent_accuracy' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['top1_accuracy'] <= 1
        assert 0 <= metrics['top5_accuracy'] <= 1
        assert 0 <= metrics['wer'] <= 1
        assert 0 <= metrics['referent_accuracy'] <= 1
    
    def test_model_export(self):
        """Test model export functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            vocab = Vocabulary()
            normalizer = Sim3Normalizer()
            feature_extractor = FeatureExtractor()
            product_vq = ProductVQ(vocab, beta=0.25)
            spatial_discourse = SpatialDiscourse()
            
            # Initialize model
            feature_dims = vocab.get_feature_dimensions()
            total_feature_dim = sum(feature_dims.values())
            
            model = ASLTranslationModel(
                input_dim=total_feature_dim,
                vocab_size=100,
                hidden_dim=64,
                num_layers=1,
                kernel_size=5,
                dropout=0.0,
                blank_idx=0,
                lambda_vq=0.1,
                lambda_cal=0.05
            )
            
            # Create configuration
            config = {
                'model': {
                    'encoder': {'hidden_dim': 64, 'num_layers': 1},
                    'ctc': {'blank_idx': 0}
                },
                'training': {
                    'lambda_vq': 0.1,
                    'lambda_cal': 0.05
                },
                'export': {
                    'optimize_for_mobile': True
                }
            }
            
            # Export model
            exporter = CompletePipelineExporter(
                model=model,
                feature_extractor=feature_extractor,
                product_vq=product_vq,
                normalizer=normalizer,
                spatial_discourse=spatial_discourse,
                vocab=vocab,
                config=config
            )
            
            exported = exporter.export_complete_pipeline(tmpdir)
            
            # Check that files were created
            assert os.path.exists(exported['model'])
            assert os.path.exists(exported['feature_extractor'])
            assert os.path.exists(exported['normalizer'])
            assert os.path.exists(exported['config'])
            assert os.path.exists(exported['vocab'])
            assert os.path.exists(exported['deployment_package'])
            
            # Check deployment package contents
            deploy_dir = exported['deployment_package']
            assert os.path.exists(os.path.join(deploy_dir, "asl_model.pt"))
            assert os.path.exists(os.path.join(deploy_dir, "inference.py"))
            assert os.path.exists(os.path.join(deploy_dir, "README.md"))
    
    def test_information_theoretic_analysis(self):
        """Test information-theoretic analysis."""
        # Initialize components
        vocab = Vocabulary()
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Initialize model
        feature_dims = vocab.get_feature_dimensions()
        total_feature_dim = sum(feature_dims.values())
        
        model = ASLTranslationModel(
            input_dim=total_feature_dim,
            vocab_size=50,   # Small vocab for testing
            hidden_dim=32,   # Small hidden dim
            num_layers=1,
            kernel_size=5,
            dropout=0.0,
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        # Initialize evaluator
        evaluator = ASLEvaluator(
            model=model,
            feature_extractor=feature_extractor,
            product_vq=product_vq,
            normalizer=normalizer,
            spatial_discourse=spatial_discourse,
            device='cpu'
        )
        
        # Create dummy data for analysis
        landmarks = torch.randn(10, 30, 1623, 3)
        landmarks[:, :, 11, :] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[:, :, 12, :] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[:, :, 0, :] = torch.tensor([0.0, 1.0, 0.0])
        
        class DummyDataloader:
            def __init__(self, landmarks):
                self.landmarks = landmarks
            
            def __iter__(self):
                for i in range(len(self.landmarks)):
                    yield {
                        'landmarks': self.landmarks[i:i+1],
                        'targets': torch.randint(1, 50, (1, 10)),
                        'target_lengths': torch.tensor([10]),
                    }
        
        dataloader = DummyDataloader(landmarks)
        
        # Perform information-theoretic analysis
        info_metrics = evaluator.information_theoretic_analysis(dataloader)
        
        # Check metrics
        assert 'mutual_information' in info_metrics
        assert 'target_entropy' in info_metrics
        assert 'fano_bound' in info_metrics
        
        # Check reasonable ranges
        assert info_metrics['mutual_information'] >= 0
        assert info_metrics['target_entropy'] >= 0
        assert 0 <= info_metrics['fano_bound'] <= 1
    
    def test_latency_profiling(self):
        """Test latency profiling."""
        # Initialize components
        vocab = Vocabulary()
        normalizer = Sim3Normalizer()
        feature_extractor = FeatureExtractor()
        product_vq = ProductVQ(vocab, beta=0.25)
        spatial_discourse = SpatialDiscourse()
        
        # Initialize model
        feature_dims = vocab.get_feature_dimensions()
        total_feature_dim = sum(feature_dims.values())
        
        model = ASLTranslationModel(
            input_dim=total_feature_dim,
            vocab_size=50,
            hidden_dim=32,
            num_layers=1,
            kernel_size=5,
            dropout=0.0,
            blank_idx=0,
            lambda_vq=0.1,
            lambda_cal=0.05
        )
        
        # Initialize evaluator
        evaluator = ASLEvaluator(
            model=model,
            feature_extractor=feature_extractor,
            product_vq=product_vq,
            normalizer=normalizer,
            spatial_discourse=spatial_discourse,
            device='cpu'
        )
        
        # Create test landmarks
        landmarks = torch.randn(1, 30, 1623, 3)
        landmarks[0, :, 11, :] = torch.tensor([1.0, 0.0, 0.0])
        landmarks[0, :, 12, :] = torch.tensor([-1.0, 0.0, 0.0])
        landmarks[0, :, 0, :] = torch.tensor([0.0, 1.0, 0.0])
        
        # Profile inference
        profile_results = evaluator.profile_inference(landmarks, num_runs=10)
        
        # Check profiling results
        assert 'mean_latency' in profile_results
        assert 'std_latency' in profile_results
        assert 'p95_latency' in profile_results
        assert 'fps' in profile_results
        
        # Check reasonable values
        assert profile_results['mean_latency'] > 0
        assert profile_results['fps'] > 0