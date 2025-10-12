"""WFST Decoder with beam search.

This module implements beam search decoding on H∘C∘L∘G using:
- Beam width: 12
- Lattice beam: 8  
- Max active states: 10,000
- Log-semiring for probabilities
"""

import torch
import pynini
from pynini import Fst, ShortestPath
from pynini.lib import pynutil
import openfst_python as fst
from typing import List, Tuple, Dict, Optional
import math


class WFSTDecoder:
    """WFST decoder for ASL translation.
    
    Implements beam search decoding on composed HCLG FST.
    """
    
    def __init__(self,
                 hcl_fst_path: str,
                 g_fst_path: str,
                 beam: int = 12,
                 lattice_beam: int = 8,
                 max_active: int = 10000):
        """Initialize WFST decoder.
        
        Args:
            hcl_fst_path: Path to HCL FST
            g_fst_path: Path to G FST (language model)
            beam: Beam width for search
            lattice_beam: Lattice generation beam width
            max_active: Maximum number of active states
        """
        self.beam = beam
        self.lattice_beam = lattice_beam
        self.max_active = max_active
        
        # Load FSTs
        self.hcl_fst = Fst.read(hcl_fst_path)
        self.g_fst = Fst.read(g_fst_path)
        
        # Compose HCL and G
        print("Composing HCL and G FSTs...")
        self.hclg_fst = fst.compose(self.hcl_fst, self.g_fst)
        
        # Optimize composed FST
        print("Optimizing HCLG FST...")
        self.hclg_fst = fst.determinize(self.hclg_fst)
        self.hclg_fst.minimize()
        
        print(f"HCLG FST has {self.hclg_fst.num_states()} states")
        
        # Get symbol tables
        self.input_symbols = self.hclg_fst.input_symbols()
        self.output_symbols = self.hclg_fst.output_symbols()
    
    def decode(self, 
               log_probs: torch.Tensor,
               input_lengths: torch.Tensor) -> List[List[int]]:
        """Decode log probabilities using WFST beam search.
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            input_lengths: Input sequence lengths of shape (B,)
            
        Returns:
            List of decoded sequences (as token IDs)
        """
        batch_size, max_len, vocab_size = log_probs.shape
        decoded_sequences = []
        
        for b in range(batch_size):
            # Get sequence for this batch element
            seq_len = input_lengths[b].item()
            seq_log_probs = log_probs[b, :seq_len, :]  # (T, vocab_size)
            
            # Convert to FST format
            recognition_fst = self._build_recognition_fst(seq_log_probs)
            
            # Compose with HCLG and decode
            decoded = self._decode_sequence(recognition_fst)
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def _build_recognition_fst(self, log_probs: torch.Tensor) -> Fst:
        """Build recognition FST from log probabilities.
        
        Args:
            log_probs: Log probabilities of shape (T, vocab_size)
            
        Returns:
            Recognition FST
        """
        T, vocab_size = log_probs.shape
        rec_fst = Fst()
        
        # Create symbol table for input/output
        symbols = SymbolTable()
        symbols.add_symbol("<eps>")
        for i in range(vocab_size):
            symbols.add_symbol(f"token_{i}")
        
        rec_fst.set_input_symbols(symbols)
        rec_fst.set_output_symbols(symbols)
        
        # Add states and arcs
        states = []
        for t in range(T + 1):
            state = rec_fst.add_state()
            states.append(state)
            if t == 0:
                rec_fst.set_start(state)
            if t == T:
                rec_fst.set_final(state)
        
        # Add arcs between time steps
        for t in range(T):
            current_state = states[t]
            next_state = states[t + 1]
            
            # Add arcs for each token
            for token_id in range(vocab_size):
                log_prob = log_probs[t, token_id].item()
                weight = fst.Weight(rec_fst.weight_type(), -log_prob)
                
                rec_fst.add_arc(current_state, fst.Arc(
                    token_id, token_id, weight, next_state
                ))
        
        return rec_fst
    
    def _decode_sequence(self, recognition_fst: Fst) -> List[int]:
        """Decode single sequence using WFST composition.
        
        Args:
            recognition_fst: Recognition FST
            
        Returns:
            Decoded token sequence
        """
        # Compose recognition FST with HCLG
        composed = fst.compose(recognition_fst, self.hclg_fst)
        
        # Find shortest path (Viterbi)
        shortest_path = fst.shortestpath(composed, nshortest=1)
        
        # Extract best path
        if shortest_path.num_states() == 0:
            return []
        
        # Trace path and extract output labels
        decoded_tokens = []
        visited = set()
        
        # Start from initial state
        state = shortest_path.start()
        
        while state != -1 and state not in visited:
            visited.add(state)
            
            # Find outgoing arcs
            for arc in shortest_path.arcs(state):
                # Get output label
                output_label = arc.olabel
                
                # Skip epsilon and special symbols
                if output_label != 0:  # 0 is typically epsilon
                    symbol = self.output_symbols.find(output_label)
                    if symbol and symbol not in ["<eps>", "<s>", "</s>"]:
                        decoded_tokens.append(output_label)
                
                # Move to next state
                state = arc.nextstate
                break  # Follow best path
        
        return decoded_tokens
    
    def decode_with_lattice(self, 
                           log_probs: torch.Tensor,
                           input_lengths: torch.Tensor) -> Tuple[List[List[int]], List[float]]:
        """Decode with lattice generation and scoring.
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            input_lengths: Input sequence lengths of shape (B,)
            
        Returns:
            Tuple of (decoded_sequences, sequence_scores)
        """
        batch_size = log_probs.shape[0]
        decoded_sequences = []
        sequence_scores = []
        
        for b in range(batch_size):
            seq_len = input_lengths[b].item()
            seq_log_probs = log_probs[b, :seq_len, :]
            
            # Build recognition FST
            rec_fst = self._build_recognition_fst(seq_log_probs)
            
            # Compose and generate lattice
            composed = fst.compose(rec_fst, self.hclg_fst)
            
            # Generate n-best list
            nbest = fst.shortestpath(composed, nshortest=self.beam)
            
            # Score and rank paths
            best_path, best_score = self._score_lattice(nbest)
            
            decoded_sequences.append(best_path)
            sequence_scores.append(best_score)
        
        return decoded_sequences, sequence_scores
    
    def _score_lattice(self, lattice: Fst) -> Tuple[List[int], float]:
        """Score lattice and return best path.
        
        Args:
            lattice: WFST lattice
            
        Returns:
            Tuple of (best_path, best_score)
        """
        # Use shortest path algorithm to find best path
        shortest = fst.shortestpath(lattice, nshortest=1)
        
        # Extract path and score
        best_path = []
        best_score = float('inf')
        
        if shortest.num_states() > 0:
            # Get distance to final state
            for state in range(shortest.num_states()):
                if shortest.final(state) != fst.Weight.zero(shortest.weight_type()):
                    # This is a final state, get its weight
                    weight = shortest.final(state)
                    best_score = float(weight)
                    break
            
            # Extract output labels from best path
            state = shortest.start()
            visited = set()
            
            while state != -1 and state not in visited:
                visited.add(state)
                
                for arc in shortest.arcs(state):
                    output_label = arc.olabel
                    
                    # Skip epsilon and special symbols
                    if output_label != 0:
                        symbol = self.output_symbols.find(output_label)
                        if symbol and symbol not in ["<eps>", "<s>", "</s>"]:
                            best_path.append(output_label)
                    
                    state = arc.nextstate
                    break
        
        return best_path, best_score
    
    def decode_with_confusion_network(self, 
                                    log_probs: torch.Tensor,
                                    input_lengths: torch.Tensor) -> List[Dict[str, float]]:
        """Decode with confusion network generation.
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            input_lengths: Input sequence lengths of shape (B,)
            
        Returns:
            List of confusion networks (token -> probability)
        """
        batch_size = log_probs.shape[0]
        confusion_networks = []
        
        for b in range(batch_size):
            seq_len = input_lengths[b].item()
            seq_log_probs = log_probs[b, :seq_len, :]
            
            # Build recognition FST
            rec_fst = self._build_recognition_fst(seq_log_probs)
            
            # Compose with HCLG
            composed = fst.compose(rec_fst, self.hclg_fst)
            
            # Generate confusion network
            cn = self._generate_confusion_network(composed)
            confusion_networks.append(cn)
        
        return confusion_networks
    
    def _generate_confusion_network(self, lattice: Fst) -> Dict[str, float]:
        """Generate confusion network from lattice.
        
        Args:
            lattice: WFST lattice
            
        Returns:
            Confusion network as token -> probability mapping
        """
        confusion_network = {}
        
        # Collect all possible outputs and their weights
        for state in range(lattice.num_states()):
            for arc in lattice.arcs(state):
                output_label = arc.olabel
                
                if output_label != 0:  # Non-epsilon
                    symbol = self.output_symbols.find(output_label)
                    if symbol and symbol not in ["<eps>", "<s>", "</s>"]:
                        weight = float(arc.weight)
                        prob = math.exp(-weight)  # Convert from log space
                        
                        if symbol in confusion_network:
                            confusion_network[symbol] += prob
                        else:
                            confusion_network[symbol] = prob
        
        # Normalize probabilities
        total_prob = sum(confusion_network.values())
        if total_prob > 0:
            for symbol in confusion_network:
                confusion_network[symbol] /= total_prob
        
        return confusion_network
    
    def get_hclg_stats(self) -> Dict[str, int]:
        """Get statistics about HCLG FST.
        
        Returns:
            Dictionary with FST statistics
        """
        return {
            'num_states': self.hclg_fst.num_states(),
            'num_arcs': sum(len(list(self.hclg_fst.arcs(s))) 
                          for s in range(self.hclg_fst.num_states())),
            'input_symbols': self.hclg_fst.input_symbols().num_symbols(),
            'output_symbols': self.hclg_fst.output_symbols().num_symbols()
        }


class MockHCLGDecoder:
    """Mock HCLG decoder for testing without actual WFST files."""
    
    def __init__(self, vocab_size: int, blank_idx: int = 0):
        """Initialize mock decoder.
        
        Args:
            vocab_size: Vocabulary size
            blank_idx: Blank token index
        """
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
    
    def decode(self, 
               log_probs: torch.Tensor,
               input_lengths: torch.Tensor) -> List[List[int]]:
        """Mock decoding - returns greedy decoding.
        
        Args:
            log_probs: Log probabilities of shape (B, T, vocab_size)
            input_lengths: Input sequence lengths of shape (B,)
            
        Returns:
            Greedy decoded sequences
        """
        batch_size = log_probs.shape[0]
        decoded_sequences = []
        
        for b in range(batch_size):
            seq_len = input_lengths[b].item()
            seq_log_probs = log_probs[b, :seq_len, :]
            
            # Greedy decoding
            tokens = seq_log_probs.argmax(dim=-1).tolist()
            
            # Remove consecutive duplicates and blanks
            decoded = []
            prev_token = self.blank_idx
            for token in tokens:
                if token != prev_token and token != self.blank_idx:
                    decoded.append(token)
                prev_token = token
            
            decoded_sequences.append(decoded)
        
        return decoded_sequences


if __name__ == "__main__":
    # Example usage
    print("WFST Decoder example usage")
    
    # Create mock decoder for testing
    vocab_size = 1000
    mock_decoder = MockHCLGDecoder(vocab_size)
    
    # Create dummy log probabilities
    batch_size = 2
    seq_len = 10
    log_probs = torch.randn(batch_size, seq_len, vocab_size)
    input_lengths = torch.tensor([10, 8])
    
    # Decode
    decoded = mock_decoder.decode(log_probs, input_lengths)
    print(f"Decoded sequences: {decoded}")