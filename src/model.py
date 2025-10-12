"""Model implementation: Causal TCN Encoder + CTC Head.

This module implements:
1. 3-layer causal TCN encoder with hidden=256, kernel=5, dropout=0.2
2. CTC head with blank_idx=0
3. Complete model combining encoder and CTC loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer.
    
    Ensures that output at time t only depends on inputs up to time t.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 **kwargs):
        """Initialize causal convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation factor
            **kwargs: Additional arguments for Conv1d
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Output tensor of shape (B, C_out, T)
        """
        # Apply convolution
        out = self.conv(x)
        
        # Remove padding from the right to ensure causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return out


class CausalTCNBlock(nn.Module):
    """Causal TCN residual block.
    
    Implements a residual block with:
    - Causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.2):
        """Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First causal convolution
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # Second causal convolution
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 convolution for residual connection if needed
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN block.
        
        Args:
            x: Input tensor of shape (B, C_in, T)
            
        Returns:
            Output tensor of shape (B, C_out, T)
        """
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x
        
        return self.relu(out + residual)


class CausalTCNEncoder(nn.Module):
    """Causal TCN Encoder.
    
    Implements a 3-layer causal TCN encoder with:
    - Hidden dimension: 256
    - Kernel size: 5
    - Dropout: 0.2
    - Exponential dilation growth
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 kernel_size: int = 5,
                 dropout: float = 0.2):
        """Initialize causal TCN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # TCN layers with exponential dilation
        self.tcn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4
            
            layer = CausalTCNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            )
            
            self.tcn_layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            Encoded features of shape (B, T, hidden_dim)
        """
        # Transpose for Conv1d: (B, D, T)
        x = x.transpose(1, 2)
        
        # Input projection
        out = self.input_proj(x)
        
        # Apply TCN layers
        for layer in self.tcn_layers:
            out = layer(out)
        
        # Transpose back: (B, T, hidden_dim)
        out = out.transpose(1, 2)
        
        return out


class CTCHead(nn.Module):
    """CTC head for sequence modeling.
    
    Implements a CTC-compatible output layer with:
    - Linear projection to vocabulary size
    - Log softmax for CTC loss
    """
    
    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 blank_idx: int = 0):
        """Initialize CTC head.
        
        Args:
            input_dim: Input feature dimension
            vocab_size: Vocabulary size
            blank_idx: Blank token index for CTC
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        
        # Linear projection
        self.linear = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CTC head.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            Log probabilities of shape (B, T, vocab_size)
        """
        # Linear projection
        logits = self.linear(x)
        
        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs


class ASLTranslationModel(nn.Module):
    """Complete ASL Translation Model.
    
    Combines:
    1. Feature extraction and normalization
    2. Causal TCN encoder
    3. CTC head
    4. Loss computation including CTC, VQ, and calibration
    """
    
    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 kernel_size: int = 5,
                 dropout: float = 0.2,
                 blank_idx: int = 0,
                 lambda_vq: float = 0.1,
                 lambda_cal: float = 0.05):
        """Initialize ASL translation model.
        
        Args:
            input_dim: Input feature dimension
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            blank_idx: Blank token index
            lambda_vq: VQ loss weight
            lambda_cal: Calibration loss weight
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.lambda_vq = lambda_vq
        self.lambda_cal = lambda_cal
        
        # Encoder
        self.encoder = CausalTCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # CTC head
        self.ctc_head = CTCHead(
            input_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_idx=blank_idx
        )
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx,
            reduction='mean',
            zero_infinity=True
        )
    
    def forward(self, 
                features: torch.Tensor,
                feature_lengths: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                target_lengths: Optional[torch.Tensor] = None,
                vq_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            features: Input features of shape (B, T, D)
            feature_lengths: Feature sequence lengths of shape (B,)
            targets: Target sequences of shape (B, max_target_len)
            target_lengths: Target sequence lengths of shape (B,)
            vq_loss: Optional VQ loss term
            
        Returns:
            Dictionary containing:
                - log_probs: Log probabilities (B, T, vocab_size)
                - loss: Total loss (if targets provided)
                - ctc_loss: CTC loss component
                - vq_loss: VQ loss component
        """
        # Encode features
        encoded = self.encoder(features)
        
        # Apply CTC head
        log_probs = self.ctc_head(encoded)
        
        result = {'log_probs': log_probs}
        
        # Compute loss if targets provided
        if targets is not None and target_lengths is not None:
            # CTC loss
            ctc_loss = self.ctc_loss(
                log_probs.transpose(0, 1),  # (T, B, vocab_size)
                targets,
                feature_lengths,
                target_lengths
            )
            
            # Total loss
            total_loss = ctc_loss
            
            # Add VQ loss if provided
            if vq_loss is not None:
                total_loss = total_loss + self.lambda_vq * vq_loss
                result['vq_loss'] = vq_loss
            
            # Add calibration loss (simplified)
            cal_loss = self._compute_calibration_loss(log_probs, targets)
            total_loss = total_loss + self.lambda_cal * cal_loss
            
            result.update({
                'loss': total_loss,
                'ctc_loss': ctc_loss,
                'cal_loss': cal_loss
            })
        
        return result
    
    def _compute_calibration_loss(self, 
                                 log_probs: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """Compute calibration loss (Expected Calibration Error).
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            targets: Target sequences of shape (B, max_target_len)
            
        Returns:
            Calibration loss
        """
        # Simplified calibration loss
        # In practice, this would use binning and ECE computation
        probs = torch.exp(log_probs)
        confidence = probs.max(dim=-1)[0].mean()
        
        # Encourage confidence to match accuracy
        target_confidence = 0.8  # Desired confidence level
        cal_loss = F.mse_loss(confidence, torch.tensor(target_confidence))
        
        return cal_loss
    
    def decode_greedy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Greedy decoding of CTC outputs.
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            
        Returns:
            Decoded sequences of shape (B, max_len)
        """
        # Get most likely tokens
        tokens = log_probs.argmax(dim=-1)
        
        # Remove consecutive duplicates and blanks
        decoded = []
        for seq in tokens:
            decoded_seq = []
            prev_token = self.blank_idx
            for token in seq:
                if token != prev_token and token != self.blank_idx:
                    decoded_seq.append(token)
                prev_token = token
            decoded.append(decoded_seq)
        
        # Pad to same length
        max_len = max(len(seq) for seq in decoded)
        padded = []
        for seq in decoded:
            padded.append(seq + [self.blank_idx] * (max_len - len(seq)))
        
        return torch.tensor(padded)
    
    def get_encoder_features(self, features: torch.Tensor) -> torch.Tensor:
        """Get encoder features for analysis.
        
        Args:
            features: Input features of shape (B, T, D)
            
        Returns:
            Encoder features of shape (B, T, hidden_dim)
        """
        return self.encoder(features)