"""Training module for ASL translation model.

Implements:
- AdamW optimizer with lr=3e-4
- Cosine learning rate decay
- 100 epochs training
- Batch size 32 videos
- Early stopping with patience=10
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

from .model import ASLTranslationModel
from .features import FeatureExtractor, ProductVQ
from .normalize import Sim3Normalizer
from .spatial import SpatialDiscourse
from .vocab import Vocabulary


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement required
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def state_dict(self) -> Dict:
        """Get state dictionary."""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dictionary."""
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']
        self.best_epoch = state_dict['best_epoch']


class ASLTrainer:
    """Trainer for ASL translation model."""
    
    def __init__(self,
                 model: ASLTranslationModel,
                 feature_extractor: FeatureExtractor,
                 product_vq: ProductVQ,
                 normalizer: Sim3Normalizer,
                 spatial_discourse: SpatialDiscourse,
                 vocab: Vocabulary,
                 config: Dict):
        """Initialize trainer.
        
        Args:
            model: ASL translation model
            feature_extractor: Feature extractor
            product_vq: Product VQ module
            normalizer: Sim3 normalizer
            spatial_discourse: Spatial discourse module
            vocab: Vocabulary
            config: Training configuration
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.product_vq = product_vq
        self.normalizer = normalizer
        self.spatial_discourse = spatial_discourse
        self.vocab = vocab
        self.config = config
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_vq_loss = 0.0
        total_cal_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            landmarks = batch['landmarks']  # (B, T, N, 3)
            targets = batch['targets']      # (B, max_target_len)
            target_lengths = batch['target_lengths']
            video_ids = batch['video_ids']
            
            batch_size, seq_len, num_landmarks, _ = landmarks.shape
            
            # Process landmarks
            processed_features = []
            
            for b in range(batch_size):
                # Normalize landmarks
                normalized = self.normalizer(landmarks[b])  # (T, N, 3)
                
                # Extract features
                features = self.feature_extractor(normalized)
                
                # Apply product VQ
                quantized_features, vq_indices, vq_loss = self.product_vq(features)
                
                # Combine features
                combined_features = []
                for modality, feature in quantized_features.items():
                    combined_features.append(feature)
                
                # Concatenate features
                combined = torch.cat(combined_features, dim=-1)  # (T, D)
                processed_features.append(combined)
            
            # Pad features to same length
            max_feat_len = max(f.shape[0] for f in processed_features)
            feat_dim = processed_features[0].shape[1]
            
            padded_features = torch.zeros(batch_size, max_feat_len, feat_dim)
            feature_lengths = []
            
            for b, feat in enumerate(processed_features):
                feat_len = feat.shape[0]
                padded_features[b, :feat_len, :] = feat
                feature_lengths.append(feat_len)
            
            feature_lengths = torch.tensor(feature_lengths)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                padded_features,
                feature_lengths,
                targets,
                target_lengths,
                vq_loss
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ctc_loss += outputs['ctc_loss'].item()
            total_vq_loss += outputs.get('vq_loss', torch.tensor(0.0)).item()
            total_cal_loss += outputs.get('cal_loss', torch.tensor(0.0)).item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ctc': f"{outputs['ctc_loss'].item():.4f}",
                'vq': f"{outputs.get('vq_loss', torch.tensor(0.0)).item():.4f}"
            })
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_ctc_loss = total_ctc_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_cal_loss = total_cal_loss / num_batches
        
        metrics = {
            'train_loss': avg_loss,
            'train_ctc_loss': avg_ctc_loss,
            'train_vq_loss': avg_vq_loss,
            'train_cal_loss': avg_cal_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return metrics
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_ctc_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Process batch similar to training
                landmarks = batch['landmarks']
                targets = batch['targets']
                target_lengths = batch['target_lengths']
                
                batch_size = landmarks.shape[0]
                
                # Process landmarks
                processed_features = []
                feature_lengths = []
                
                for b in range(batch_size):
                    normalized = self.normalizer(landmarks[b])
                    features = self.feature_extractor(normalized)
                    quantized_features, _, vq_loss = self.product_vq(features)
                    
                    combined_features = []
                    for modality, feature in quantized_features.items():
                        combined_features.append(feature)
                    
                    combined = torch.cat(combined_features, dim=-1)
                    processed_features.append(combined)
                    feature_lengths.append(combined.shape[0])
                
                # Pad features
                max_feat_len = max(f.shape[0] for f in processed_features)
                feat_dim = processed_features[0].shape[1]
                
                padded_features = torch.zeros(batch_size, max_feat_len, feat_dim)
                for b, feat in enumerate(processed_features):
                    feat_len = feat.shape[0]
                    padded_features[b, :feat_len, :] = feat
                
                feature_lengths = torch.tensor(feature_lengths)
                
                # Forward pass
                outputs = self.model(
                    padded_features,
                    feature_lengths,
                    targets,
                    target_lengths,
                    vq_loss
                )
                
                total_loss += outputs['loss'].item()
                total_ctc_loss += outputs['ctc_loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ctc_loss = total_ctc_loss / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_ctc_loss': avg_ctc_loss
        }
    
    def train(self, 
              train_dataloader,
              val_dataloader,
              num_epochs: int,
              save_dir: str,
              save_every: int = 10):
        """Train model.
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        training_history = []
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            training_history.append(epoch_metrics)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"train_loss={epoch_metrics['train_loss']:.4f}, "
                f"val_loss={epoch_metrics['val_loss']:.4f}, "
                f"lr={epoch_metrics['learning_rate']:.6f}"
            )
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, epoch, epoch_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_path = os.path.join(save_dir, "best_model.pt")
                self.save_checkpoint(best_model_path, epoch, epoch_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss'], epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        self.logger.info("Training completed!")
        return training_history
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint.
        
        Args:
            path: Checkpoint path
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stopping_state_dict': self.early_stopping.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
        
        return checkpoint['epoch'], checkpoint['metrics']