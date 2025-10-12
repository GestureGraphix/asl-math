"""Feature extraction and Product Vector Quantization.

This module implements:
1. Feature extraction φ(X_t) → f_t from normalized landmarks
2. Product VQ with 5 separate codebooks
3. Feature heads for different modalities:
   - u^H ∈ ℝ^10: 5 finger flexions + thumb angle
   - u^L ∈ ℝ^6: palm centres c^L_t, c^R_t
   - u^O ∈ ℝ^6: unit normals n^L_t, n^R_t
   - u^M ∈ ℝ^9: Δc, Δ²c, Δa_t, Δg_t
   - u^N ∈ ℝ^5: gaze proxy g_t, mouth a_t, eyebrow height
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans

from .vocab import Vocabulary


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with commitment loss.
    
    Implements VQ-VAE style quantization with:
    - K-means initialization
    - Commitment loss with β=0.25
    - Straight-through estimator for gradients
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        """Initialize vector quantizer.
        
        Args:
            num_embeddings: Number of codebook entries
            embedding_dim: Dimension of each embedding
            beta: Commitment loss weight
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize embeddings uniformly
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply vector quantization.
        
        Args:
            x: Input tensor of shape (B, D) or (B, T, D)
            
        Returns:
            quantized: Quantized tensor, same shape as input
            indices: Codebook indices
            commitment_loss: Commitment loss term
        """
        original_shape = x.shape
        
        # Flatten for processing
        if len(original_shape) == 3:
            B, T, D = x.shape
            x = x.reshape(-1, D)
        
        # Compute distances to codebook entries
        distances = (torch.sum(x**2, dim=1, keepdim=True) +
                    torch.sum(self.embedding.weight**2, dim=1) -
                    2 * torch.matmul(x, self.embedding.weight.t()))
        
        # Find closest codebook entries
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized = self.embedding(indices)
        
        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), x)
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Restore original shape
        if len(original_shape) == 3:
            quantized = quantized.reshape(B, T, D)
            indices = indices.reshape(B, T)
        
        return quantized, indices, commitment_loss
    
    def initialize_kmeans(self, data: torch.Tensor, num_samples: int = 10000):
        """Initialize codebook using K-means.
        
        Args:
            data: Training data for initialization
            num_samples: Number of samples to use for K-means
        """
        # Sample data for K-means
        if data.numel() > num_samples * data.shape[-1]:
            indices = torch.randperm(data.shape[0])[:num_samples]
            sample_data = data[indices].cpu().numpy()
        else:
            sample_data = data.cpu().numpy()
        
        # Run K-means
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=42)
        kmeans.fit(sample_data)
        
        # Update embeddings
        self.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).float()


class ProductVQ(nn.Module):
    """Product Vector Quantization with 5 codebooks.
    
    Implements product quantization as:
    Z_t = (z^H_t, z^L_t, z^O_t, z^M_t, z^N_t) ∈ Σ = ∏_i Σ_i
    """
    
    def __init__(self, vocab: Vocabulary, beta: float = 0.25):
        """Initialize product VQ.
        
        Args:
            vocab: Vocabulary instance
            beta: Commitment loss weight
        """
        super().__init__()
        self.vocab = vocab
        self.beta = beta
        
        # Create vector quantizers for each modality
        self.quantizers = nn.ModuleDict()
        
        feature_dims = vocab.get_feature_dimensions()
        
        for modality, size in vocab.codebook_sizes.items():
            dim = feature_dims[modality]
            self.quantizers[modality] = VectorQuantizer(
                num_embeddings=size,
                embedding_dim=dim,
                beta=beta
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], 
                                                                   Dict[str, torch.Tensor], 
                                                                   torch.Tensor]:
        """Apply product vector quantization.
        
        Args:
            features: Dictionary mapping modality to feature tensor
                     Each tensor has shape (B, D) or (B, T, D)
            
        Returns:
            quantized_features: Quantized features
            indices: Codebook indices for each modality
            total_loss: Total commitment loss
        """
        quantized_features = {}
        indices = {}
        total_loss = 0.0
        
        for modality, feature in features.items():
            if modality in self.quantizers:
                quantized, idx, loss = self.quantizers[modality](feature)
                quantized_features[modality] = quantized
                indices[modality] = idx
                total_loss = total_loss + loss
        
        return quantized_features, indices, total_loss
    
    def initialize_codebooks(self, features: Dict[str, torch.Tensor]):
        """Initialize all codebooks using K-means.
        
        Args:
            features: Dictionary mapping modality to training features
        """
        for modality, feature_data in features.items():
            if modality in self.quantizers:
                self.quantizers[modality].initialize_kmeans(feature_data)


class FeatureExtractor(nn.Module):
    """Feature extraction from normalized landmarks.
    
    Extracts features according to paper specifications:
    - u^H ∈ ℝ^10: 5 finger flexions + thumb angle
    - u^L ∈ ℝ^6: palm centres c^L_t, c^R_t  
    - u^O ∈ ℝ^6: unit normals n^L_t, n^R_t
    - u^M ∈ ℝ^9: Δc, Δ²c, Δa_t, Δg_t
    - u^N ∈ ℝ^5: gaze proxy g_t, mouth a_t, eyebrow height
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        super().__init__()
        
        # MediaPipe hand landmark indices
        self.hand_landmarks = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],
            'palm': [0, 1, 5, 9, 13, 17]
        }
        
        # Face landmark indices for non-manual features
        self.face_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
            'eyebrows': [46, 53, 52, 65, 55, 283, 293, 300, 276, 283, 282, 295, 285, 417, 351, 419, 248, 456, 453, 452, 451, 450, 449, 448, 261, 265, 464, 413, 441, 285, 417, 351, 419, 248, 456, 453, 452, 451, 450, 449, 448, 261, 265, 464, 413, 441]
        }
        
    def forward(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from normalized landmarks.
        
        Args:
            landmarks: Normalized landmarks of shape (T, N, 3)
                      where N = 21 (hands) + 468 (face) + 33 (pose)
                      
        Returns:
            Dictionary mapping modality to feature tensor
        """
        features = {}
        
        # Split landmarks by type
        hand_landmarks = landmarks[:, :21, :]  # First 21 are hands
        pose_landmarks = landmarks[:, -33:, :]  # Last 33 are pose
        face_landmarks = landmarks[:, 21:-33, :]  # Remaining are face
        
        # Extract hand features
        features['hands'] = self._extract_hand_features(hand_landmarks)
        features['location'] = self._extract_location_features(hand_landmarks)
        features['orientation'] = self._extract_orientation_features(hand_landmarks)
        features['motion'] = self._extract_motion_features(hand_landmarks)
        features['non_manual'] = self._extract_non_manual_features(face_landmarks, pose_landmarks)
        
        return features
    
    def _extract_hand_features(self, hand_landmarks: torch.Tensor) -> torch.Tensor:
        """Extract hand features: 5 finger flexions + thumb angle.
        
        Args:
            hand_landmarks: Hand landmarks of shape (T, 21, 3)
            
        Returns:
            Hand features of shape (T, 10)
        """
        T = hand_landmarks.shape[0]
        features = torch.zeros(T, 10)
        
        for t in range(T):
            hand = hand_landmarks[t]
            
            # Finger flexions (curl angles)
            flexions = []
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                flexion = self._compute_finger_flexion(hand, finger_name)
                flexions.append(flexion)
            
            # Thumb angle (abduction)
            thumb_angle = self._compute_thumb_angle(hand)
            
            # Combine features
            features[t] = torch.tensor(flexions + [thumb_angle])
        
        return features
    
    def _compute_finger_flexion(self, hand: torch.Tensor, finger_name: str) -> float:
        """Compute finger flexion angle.
        
        Args:
            hand: Hand landmarks for one frame
            finger_name: Name of finger
            
        Returns:
            Flexion angle in radians
        """
        landmarks = self.hand_landmarks[finger_name]
        
        # Get joint positions
        mcp = hand[landmarks[0]]  # Metacarpophalangeal joint
        pip = hand[landmarks[1]]  # Proximal interphalangeal joint
        dip = hand[landmarks[2]]  # Distal interphalangeal joint
        tip = hand[landmarks[3]]  # Tip
        
        # Compute angles at PIP and DIP joints
        vec1 = pip - mcp
        vec2 = dip - pip
        vec3 = tip - dip
        
        # Angle at PIP joint
        cos_angle1 = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-8)
        angle1 = torch.acos(torch.clamp(cos_angle1, -1, 1))
        
        # Angle at DIP joint
        cos_angle2 = torch.dot(vec2, vec3) / (torch.norm(vec2) * torch.norm(vec3) + 1e-8)
        angle2 = torch.acos(torch.clamp(cos_angle2, -1, 1))
        
        # Total flexion
        flexion = angle1 + angle2
        
        return flexion.item()
    
    def _compute_thumb_angle(self, hand: torch.Tensor) -> float:
        """Compute thumb abduction angle.
        
        Args:
            hand: Hand landmarks for one frame
            
        Returns:
            Thumb abduction angle in radians
        """
        # Thumb landmarks
        thumb_base = hand[1]
        thumb_mcp = hand[2]
        thumb_tip = hand[4]
        
        # Index finger base for reference
        index_base = hand[5]
        
        # Compute vectors
        thumb_vec = thumb_tip - thumb_mcp
        palm_vec = index_base - thumb_base
        
        # Angle between thumb and palm
        cos_angle = torch.dot(thumb_vec, palm_vec) / (torch.norm(thumb_vec) * torch.norm(palm_vec) + 1e-8)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        
        return angle.item()
    
    def _extract_location_features(self, hand_landmarks: torch.Tensor) -> torch.Tensor:
        """Extract location features: palm centres.
        
        Args:
            hand_landmarks: Hand landmarks of shape (T, 21, 3)
            
        Returns:
            Location features of shape (T, 6) - [c^L_t, c^R_t]
        """
        T = hand_landmarks.shape[0]
        features = torch.zeros(T, 6)
        
        for t in range(T):
            hand = hand_landmarks[t]
            
            # Compute palm center as average of palm landmarks
            palm_indices = self.hand_landmarks['palm']
            palm_center = hand[palm_indices].mean(dim=0)
            
            # For single hand, duplicate for left/right
            features[t] = torch.cat([palm_center, palm_center])
        
        return features
    
    def _extract_orientation_features(self, hand_landmarks: torch.Tensor) -> torch.Tensor:
        """Extract orientation features: palm normals.
        
        Args:
            hand_landmarks: Hand landmarks of shape (T, 21, 3)
            
        Returns:
            Orientation features of shape (T, 6) - [n^L_t, n^R_t]
        """
        T = hand_landmarks.shape[0]
        features = torch.zeros(T, 6)
        
        for t in range(T):
            hand = hand_landmarks[t]
            
            # Compute palm normal using cross product
            palm_indices = self.hand_landmarks['palm']
            palm_points = hand[palm_indices]
            
            # Use three points to compute normal
            p1, p2, p3 = palm_points[0], palm_points[1], palm_points[2]
            normal = torch.cross(p2 - p1, p3 - p1)
            normal = normal / (torch.norm(normal) + 1e-8)  # Normalize
            
            # For single hand, duplicate for left/right
            features[t] = torch.cat([normal, normal])
        
        return features
    
    def _extract_motion_features(self, hand_landmarks: torch.Tensor) -> torch.Tensor:
        """Extract motion features: deltas.
        
        Args:
            hand_landmarks: Hand landmarks of shape (T, 21, 3)
            
        Returns:
            Motion features of shape (T, 9) - [Δc, Δ²c, Δa_t, Δg_t]
        """
        T = hand_landmarks.shape[0]
        features = torch.zeros(T, 9)
        
        # Compute palm centers
        palm_centers = []
        for t in range(T):
            palm_indices = self.hand_landmarks['palm']
            center = hand_landmarks[t][palm_indices].mean(dim=0)
            palm_centers.append(center)
        
        palm_centers = torch.stack(palm_centers)
        
        # Compute deltas
        for t in range(T):
            # First derivative (velocity)
            if t > 0:
                delta_c = palm_centers[t] - palm_centers[t-1]
            else:
                delta_c = torch.zeros(3)
            
            # Second derivative (acceleration)
            if t > 1:
                delta2_c = (palm_centers[t] - 2*palm_centers[t-1] + palm_centers[t-2])
            else:
                delta2_c = torch.zeros(3)
            
            # Angular velocity (simplified)
            delta_a = torch.zeros(3)  # Placeholder
            
            # Global motion (simplified)
            delta_g = torch.zeros(3)  # Placeholder
            
            features[t] = torch.cat([delta_c, delta2_c, delta_a, delta_g])
        
        return features
    
    def _extract_non_manual_features(self, face_landmarks: torch.Tensor, 
                                   pose_landmarks: torch.Tensor) -> torch.Tensor:
        """Extract non-manual features.
        
        Args:
            face_landmarks: Face landmarks of shape (T, 468, 3)
            pose_landmarks: Pose landmarks of shape (T, 33, 3)
            
        Returns:
            Non-manual features of shape (T, 5)
        """
        T = face_landmarks.shape[0]
        features = torch.zeros(T, 5)
        
        for t in range(T):
            face = face_landmarks[t]
            pose = pose_landmarks[t]
            
            # Gaze proxy (simplified - use head orientation)
            head_direction = pose[0] - pose[7]  # Nose to left ear
            gaze_proxy = torch.norm(head_direction)
            
            # Mouth aperture
            mouth_indices = self.face_landmarks['mouth']
            mouth_landmarks = face[mouth_indices]
            mouth_center = mouth_landmarks.mean(dim=0)
            mouth_aperture = torch.norm(mouth_landmarks.max(dim=0)[0] - mouth_landmarks.min(dim=0)[0])
            
            # Eyebrow height (simplified)
            eyebrow_height = torch.tensor(0.5)  # Placeholder
            
            # Additional features
            feature1 = torch.tensor(0.0)  # Placeholder
            feature2 = torch.tensor(0.0)  # Placeholder
            
            features[t] = torch.tensor([
                gaze_proxy.item(),
                mouth_aperture.item(), 
                eyebrow_height.item(),
                feature1.item(),
                feature2.item()
            ])
        
        return features