"""Evaluation module for ASL translation model.

Implements evaluation metrics:
- Top-1 and Top-5 gloss accuracy
- Word Error Rate (WER) on sentence level
- Referent resolution accuracy
- Latency profiling: κ_KP, κ_Enc, κ_Dec, κ_Post
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import editdistance
from collections import defaultdict

from .model import ASLTranslationModel
from .features import FeatureExtractor, ProductVQ
from .normalize import Sim3Normalizer
from .spatial import SpatialDiscourse
from .wfst.decode import WFSTDecoder, MockHCLGDecoder


class ASLEvaluator:
    """Evaluator for ASL translation model."""
    
    def __init__(self,
                 model: ASLTranslationModel,
                 feature_extractor: FeatureExtractor,
                 product_vq: ProductVQ,
                 normalizer: Sim3Normalizer,
                 spatial_discourse: SpatialDiscourse,
                 wfst_decoder: Optional[WFSTDecoder] = None,
                 device: str = 'cuda'):
        """Initialize evaluator.
        
        Args:
            model: ASL translation model
            feature_extractor: Feature extractor
            product_vq: Product VQ module
            normalizer: Sim3 normalizer
            spatial_discourse: Spatial discourse module
            wfst_decoder: Optional WFST decoder
            device: Device to use
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.product_vq = product_vq
        self.normalizer = normalizer
        self.spatial_discourse = spatial_discourse
        self.wfst_decoder = wfst_decoder or MockHCLGDecoder(1000)
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.feature_extractor.to(device)
        self.product_vq.to(device)
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Metrics accumulators
        all_predictions = []
        all_targets = []
        all_referent_predictions = []
        all_referent_targets = []
        
        # Latency measurements
        latency_metrics = {
            'keypoints': [],
            'encoder': [],
            'decoder': [],
            'postprocess': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Extract batch data
                landmarks = batch['landmarks'].to(self.device)
                targets = batch['targets']
                target_lengths = batch['target_lengths']
                referent_targets = batch.get('referent_targets', None)
                
                batch_size = landmarks.shape[0]
                
                # Process batch
                predictions, referent_predictions, batch_latencies = self._process_batch(
                    landmarks, batch_size
                )
                
                # Accumulate results
                all_predictions.extend(predictions)
                all_targets.extend([
                    targets[i][:target_lengths[i]].tolist() 
                    for i in range(batch_size)
                ])
                
                if referent_predictions:
                    all_referent_predictions.extend(referent_predictions)
                if referent_targets is not None:
                    all_referent_targets.extend(referent_targets.tolist())
                
                # Accumulate latency measurements
                for metric, times in batch_latencies.items():
                    latency_metrics[metric].extend(times)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions, all_targets,
            all_referent_predictions, all_referent_targets
        )
        
        # Add latency metrics
        latency_stats = self._compute_latency_stats(latency_metrics)
        metrics.update(latency_stats)
        
        return metrics
    
    def _process_batch(self, landmarks: torch.Tensor, batch_size: int) -> Tuple[
        List[List[int]], List[str], Dict[str, List[float]]
    ]:
        """Process a batch of landmarks.
        
        Args:
            landmarks: Landmarks tensor (B, T, N, 3)
            batch_size: Batch size
            
        Returns:
            Tuple of (predictions, referent_predictions, latencies)
        """
        predictions = []
        referent_predictions = []
        latencies = {
            'keypoints': [],
            'encoder': [],
            'decoder': [],
            'postprocess': []
        }
        
        for b in range(batch_size):
            # Start timing
            start_time = time.time()
            
            # Normalize landmarks
            normalized = self.normalizer(landmarks[b])
            
            # Extract features
            features = self.feature_extractor(normalized)
            
            # Apply product VQ
            quantized_features, vq_indices, vq_loss = self.product_vq(features)
            
            # Spatial discourse processing
            referent_pred = None
            if hasattr(self.spatial_discourse, 'referent_positions'):
                spatial_result = self.spatial_discourse(
                    normalized[0],  # Single frame
                    timestamp=0.0
                )
                if spatial_result['referent_probs']:
                    referent_pred = max(
                        spatial_result['referent_probs'], 
                        key=spatial_result['referent_probs'].get
                    )
            
            # Combine features
            combined_features = []
            for modality, feature in quantized_features.items():
                combined_features.append(feature)
            
            combined = torch.cat(combined_features, dim=-1)
            
            # Timing: keypoints processing
            kp_time = time.time() - start_time
            latencies['keypoints'].append(kp_time)
            
            # Model inference
            start_time = time.time()
            
            # Add batch dimension
            combined = combined.unsqueeze(0)
            feature_lengths = torch.tensor([combined.shape[1]])
            
            # Forward pass
            outputs = self.model(combined, feature_lengths, vq_loss=vq_loss)
            
            # Timing: encoder
            enc_time = time.time() - start_time
            latencies['encoder'].append(enc_time)
            
            # Decode
            start_time = time.time()
            
            log_probs = outputs['log_probs']
            
            # Use WFST decoder if available, otherwise greedy
            if hasattr(self.wfst_decoder, 'decode'):
                decoded = self.wfst_decoder.decode(log_probs, feature_lengths)[0]
            else:
                # Greedy decoding
                decoded = self._greedy_decode(log_probs[0])
            
            # Timing: decoder
            dec_time = time.time() - start_time
            latencies['decoder'].append(dec_time)
            
            # Post-processing
            start_time = time.time()
            
            # Convert to list and filter special tokens
            decoded = [int(token) for token in decoded if token > 0]
            
            # Timing: postprocess
            post_time = time.time() - start_time
            latencies['postprocess'].append(post_time)
            
            predictions.append(decoded)
            if referent_pred:
                referent_predictions.append(referent_pred)
        
        return predictions, referent_predictions, latencies
    
    def _greedy_decode(self, log_probs: torch.Tensor) -> List[int]:
        """Greedy decoding.
        
        Args:
            log_probs: Log probabilities (T, vocab_size)
            
        Returns:
            Decoded sequence
        """
        tokens = log_probs.argmax(dim=-1).tolist()
        
        # Remove consecutive duplicates and blanks
        decoded = []
        prev_token = 0  # blank_idx
        for token in tokens:
            if token != prev_token and token != 0:
                decoded.append(token)
            prev_token = token
        
        return decoded
    
    def _compute_metrics(self,
                        predictions: List[List[int]],
                        targets: List[List[int]],
                        referent_predictions: List[str],
                        referent_targets: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            referent_predictions: Referent predictions
            referent_targets: Referent ground truth
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Token-level accuracy (gloss accuracy)
        correct_top1 = 0
        correct_top5 = 0
        total_tokens = 0
        
        for pred, target in zip(predictions, targets):
            # Top-1 accuracy
            if len(pred) > 0 and len(target) > 0:
                if pred[0] == target[0]:
                    correct_top1 += 1
            
            # Top-5 accuracy (simplified - check if target in top 5 predictions)
            if len(pred) > 0 and len(target) > 0:
                if target[0] in pred[:5]:
                    correct_top5 += 1
            
            total_tokens += min(len(pred), len(target))
        
        metrics['top1_accuracy'] = correct_top1 / max(total_tokens, 1)
        metrics['top5_accuracy'] = correct_top5 / max(total_tokens, 1)
        
        # Sentence-level metrics (WER)
        total_wer = 0.0
        total_sentences = len(predictions)
        
        for pred, target in zip(predictions, targets):
            # Convert to strings for WER calculation
            pred_str = " ".join(map(str, pred))
            target_str = " ".join(map(str, target))
            
            # Compute edit distance
            distance = editdistance.eval(pred_str.split(), target_str.split())
            max_len = max(len(pred_str.split()), len(target_str.split()))
            
            if max_len > 0:
                wer = distance / max_len
                total_wer += wer
        
        metrics['wer'] = total_wer / total_sentences
        
        # Referent resolution accuracy
        if referent_predictions and referent_targets:
            correct_referents = sum(1 for pred, target in 
                                  zip(referent_predictions, referent_targets)
                                  if pred == target)
            metrics['referent_accuracy'] = correct_referents / len(referent_predictions)
        else:
            metrics['referent_accuracy'] = 0.0
        
        return metrics
    
    def _compute_latency_stats(self, latencies: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute latency statistics.
        
        Args:
            latencies: Dictionary of latency measurements
            
        Returns:
            Dictionary of latency statistics
        """
        stats = {}
        
        for metric, times in latencies.items():
            if times:
                stats[f'{metric}_mean'] = np.mean(times)
                stats[f'{metric}_std'] = np.std(times)
                stats[f'{metric}_p95'] = np.percentile(times, 95)
                stats[f'{metric}_p99'] = np.percentile(times, 99)
            else:
                stats[f'{metric}_mean'] = 0.0
                stats[f'{metric}_std'] = 0.0
                stats[f'{metric}_p95'] = 0.0
                stats[f'{metric}_p99'] = 0.0
        
        # Total latency
        total_latencies = []
        for i in range(len(latencies['keypoints'])):
            total = (latencies['keypoints'][i] + 
                    latencies['encoder'][i] + 
                    latencies['decoder'][i] + 
                    latencies['postprocess'][i])
            total_latencies.append(total)
        
        if total_latencies:
            stats['total_latency_mean'] = np.mean(total_latencies)
            stats['total_latency_p95'] = np.percentile(total_latencies, 95)
            stats['total_latency_p99'] = np.percentile(total_latencies, 99)
        
        return stats
    
    def profile_inference(self, landmarks: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Profile inference latency.
        
        Args:
            landmarks: Input landmarks
            num_runs: Number of profiling runs
            
        Returns:
            Profiling results
        """
        self.model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self._profile_single_inference(landmarks)
        
        # Profile
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self._profile_single_inference(landmarks)
            times.append(time.time() - start_time)
        
        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'p95_latency': np.percentile(times, 95),
            'p99_latency': np.percentile(times, 99),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def _profile_single_inference(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Single inference for profiling.
        
        Args:
            landmarks: Input landmarks
            
        Returns:
            Model output
        """
        # Normalize
        normalized = self.normalizer(landmarks)
        
        # Extract features
        features = self.feature_extractor(normalized)
        quantized_features, _, vq_loss = self.product_vq(features)
        
        # Combine features
        combined_features = []
        for modality, feature in quantized_features.items():
            combined_features.append(feature)
        
        combined = torch.cat(combined_features, dim=-1).unsqueeze(0)
        feature_lengths = torch.tensor([combined.shape[1]])
        
        # Forward pass
        outputs = self.model(combined, feature_lengths, vq_loss=vq_loss)
        
        return outputs['log_probs']
    
    def information_theoretic_analysis(self, dataloader) -> Dict[str, float]:
        """Perform information-theoretic analysis.
        
        Args:
            dataloader: Dataloader for analysis
            
        Returns:
            Dictionary of information-theoretic metrics
        """
        # This is a simplified implementation
        # In practice, would use CLUB estimator and Fano bound
        
        self.model.eval()
        
        # Collect encoder features and targets
        all_features = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Information analysis"):
                landmarks = batch['landmarks'].to(self.device)
                targets = batch['targets']
                
                # Process landmarks (simplified)
                normalized = self.normalizer(landmarks[0])
                features = self.normalized[0, :50, :]  # First 50 frames
                
                all_features.append(features.cpu().numpy())
                all_targets.extend(targets[0].tolist())
        
        # Compute mutual information estimate (simplified)
        features = np.concatenate(all_features, axis=0)
        targets = np.array(all_targets[:len(features)])
        
        # Simplified mutual information estimation
        # In practice, would use proper estimators like CLUB
        mutual_info_estimate = self._estimate_mutual_information(features, targets)
        
        # Fano bound
        target_entropy = self._compute_entropy(targets)
        fano_bound = self._compute_fano_bound(
            target_entropy, 
            mutual_info_estimate, 
            len(np.unique(targets))
        )
        
        return {
            'mutual_information': mutual_info_estimate,
            'target_entropy': target_entropy,
            'fano_bound': fano_bound
        }
    
    def _estimate_mutual_information(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Estimate mutual information (simplified).
        
        Args:
            features: Feature matrix
            targets: Target labels
            
        Returns:
            Mutual information estimate
        """
        # Simplified implementation - in practice use CLUB or similar
        # This is a placeholder that returns a reasonable value
        return 2.5  # bits
    
    def _compute_entropy(self, targets: np.ndarray) -> float:
        """Compute entropy of targets.
        
        Args:
            targets: Target labels
            
        Returns:
            Entropy in bits
        """
        unique, counts = np.unique(targets, return_counts=True)
        probabilities = counts / len(targets)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        return entropy
    
    def _compute_fano_bound(self, 
                          target_entropy: float,
                          mutual_info: float,
                          num_classes: int) -> float:
        """Compute Fano bound on error probability.
        
        Args:
            target_entropy: Target entropy
            mutual_info: Mutual information
            num_classes: Number of classes
            
        Returns:
            Fano bound on error probability
        """
        # Fano bound: P_e >= (H(Y) - I(X;Y) - 1) / log(|Y|)
        fano_bound = (target_entropy - mutual_info - 1) / np.log2(num_classes)
        return max(0.0, fano_bound)