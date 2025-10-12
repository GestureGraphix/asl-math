"""Build Language Model FST (G).

This module implements:
1. N-gram language model using KenLM
2. FST conversion for WFST decoding
3. Pruning and optimization
"""

import pynini
from pynini import Fst, SymbolTable
import kenlm
import os
import tempfile
from typing import List, Dict, Tuple
import math


def train_kenlm_model(text_file: str,
                     output_dir: str,
                     order: int = 4,
                     prune_threshold: float = 1e-8) -> str:
    """Train KenLM language model.
    
    Args:
        text_file: Path to training text file
        output_dir: Output directory for LM files
        order: N-gram order
        prune_threshold: Pruning threshold
        
    Returns:
        Path to trained ARPA file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    arpa_file = os.path.join(output_dir, f"lm_{order}gram.arpa")
    binary_file = os.path.join(output_dir, f"lm_{order}gram.bin")
    
    # Train KenLM model
    print(f"Training {order}-gram KenLM model...")
    
    # Build vocabulary
    vocab_file = os.path.join(output_dir, "vocab.txt")
    cmd = f"kenlm/build/bin/lmplz -o {order} --text {text_file} --arpa {arpa_file}"
    
    if prune_threshold > 0:
        cmd += f" --prune {prune_threshold}"
    
    # Run command (in practice, use subprocess)
    print(f"Running: {cmd}")
    # os.system(cmd)  # Uncomment in actual implementation
    
    # Convert to binary for faster loading
    if os.path.exists(arpa_file):
        cmd = f"kenlm/build/bin/build_binary {arpa_file} {binary_file}"
        print(f"Running: {cmd}")
        # os.system(cmd)  # Uncomment in actual implementation
    
    return arpa_file if not os.path.exists(binary_file) else binary_file


def load_kenlm_model(model_path: str) -> kenlm.Model:
    """Load KenLM model.
    
    Args:
        model_path: Path to KenLM model
        
    Returns:
        Loaded KenLM model
    """
    return kenlm.Model(model_path)


def arpa_to_fst(arpa_file: str,
               word_symbols: SymbolTable,
               output_fst: str,
               backoff_weight: float = 0.4) -> Fst:
    """Convert ARPA format language model to FST.
    
    Args:
        arpa_file: Path to ARPA file
        word_symbols: Word symbol table
        output_fst: Output FST path
        backoff_weight: Backoff weight for smoothing
        
    Returns:
        Language model FST
    """
    lm_fst = Fst()
    lm_fst.set_input_symbols(word_symbols)
    lm_fst.set_output_symbols(word_symbols)
    
    # Start state
    start_state = lm_fst.add_state()
    lm_fst.set_start(start_state)
    
    # Read ARPA file and convert to FST
    with open(arpa_file, 'r') as f:
        lines = f.readlines()
    
    in_ngrams = False
    current_order = 0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("\\data\\"):
            in_ngrams = False
            continue
        elif line.startswith("\\1-grams:"):
            in_ngrams = True
            current_order = 1
            continue
        elif line.startswith("\\2-grams:"):
            current_order = 2
            continue
        elif line.startswith("\\3-grams:"):
            current_order = 3
            continue
        elif line.startswith("\\4-grams:"):
            current_order = 4
            continue
        elif line.startswith("\\end\\"):
            break
        
        if in_ngrams and line and not line.startswith("ngram"):
            parts = line.split()
            
            if current_order == 1:
                # Unigram format: log_prob word [backoff_weight]
                if len(parts) >= 2:
                    log_prob = float(parts[0])
                    word = parts[1]
                    
                    # Convert to FST arc
                    word_id = word_symbols.find(word)
                    if word_id != -1:
                        state = lm_fst.add_state()
                        weight = fst.Weight(lm_fst.weight_type(), -log_prob)
                        lm_fst.add_arc(start_state, fst.Arc(
                            word_id, word_id, weight, state
                        ))
                        lm_fst.set_final(state)
                        
                        # Add backoff arc if present
                        if len(parts) >= 3:
                            backoff_log_prob = float(parts[2])
                            backoff_weight = fst.Weight(
                                lm_fst.weight_type(), -backoff_log_prob
                            )
                            lm_fst.add_arc(state, fst.Arc(
                                word_symbols.find("<eps>"),
                                word_symbols.find("<eps>"),
                                backoff_weight,
                                start_state
                            ))
    
    # Add sentence beginning and end markers
    bos_state = lm_fst.add_state()
    eos_state = lm_fst.add_state()
    
    # Beginning of sentence
    lm_fst.add_arc(start_state, fst.Arc(
        word_symbols.find("<s>"),
        word_symbols.find("<s>"),
        fst.Weight.one(lm_fst.weight_type()),
        bos_state
    ))
    
    # End of sentence
    for state in range(lm_fst.num_states()):
        if lm_fst.final(state) != fst.Weight.zero(lm_fst.weight_type()):
            lm_fst.add_arc(state, fst.Arc(
                word_symbols.find("</s>"),
                word_symbols.find("</s>"),
                fst.Weight.one(lm_fst.weight_type()),
                eos_state
            ))
    
    lm_fst.set_final(eos_state)
    
    # Optimize the FST
    lm_fst = pynini.determinize(lm_fst)
    lm_fst.minimize()
    
    if output_fst:
        lm_fst.write(output_fst)
    
    return lm_fst


def build_language_model(text_files: List[str],
                        output_dir: str,
                        word_symbols: SymbolTable,
                        order: int = 4,
                        prune_threshold: float = 1e-8,
                        vocab_size: int = 50000) -> str:
    """Build complete language model pipeline.
    
    Args:
        text_files: List of training text files
        output_dir: Output directory
        word_symbols: Word symbol table
        order: N-gram order
        prune_threshold: Pruning threshold
        vocab_size: Maximum vocabulary size
        
    Returns:
        Path to final G.fst file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine text files
    combined_text = os.path.join(output_dir, "combined_text.txt")
    with open(combined_text, 'w') as outf:
        for text_file in text_files:
            with open(text_file, 'r') as inf:
                for line in inf:
                    outf.write(line.strip() + "\n")
    
    # Train KenLM model
    lm_model_path = train_kenlm_model(
        combined_text,
        output_dir,
        order=order,
        prune_threshold=prune_threshold
    )
    
    # Convert to FST
    if lm_model_path.endswith('.arpa'):
        arpa_file = lm_model_path
    else:
        # Convert binary to ARPA
        arpa_file = os.path.join(output_dir, f"lm_{order}gram.arpa")
        # kenlm.Model(lm_model_path).write_arpa(arpa_file)  # Uncomment in actual implementation
    
    # Build G.fst
    g_fst_path = os.path.join(output_dir, "G.fst")
    g_fst = arpa_to_fst(arpa_file, word_symbols, g_fst_path)
    
    # Further optimize G.fst
    g_fst = pynini.determinize(g_fst)
    g_fst.minimize()
    g_fst.write(g_fst_path)
    
    print(f"Language model FST saved to: {g_fst_path}")
    return g_fst_path


def create_dummy_language_model(word_symbols: SymbolTable,
                               output_path: str) -> str:
    """Create a dummy language model for testing.
    
    Args:
        word_symbols: Word symbol table
        output_path: Output path for G.fst
        
    Returns:
        Path to created G.fst
    """
    g_fst = Fst()
    g_fst.set_input_symbols(word_symbols)
    g_fst.set_output_symbols(word_symbols)
    
    # Simple unigram model
    states = {}
    
    # Start state
    start_state = g_fst.add_state()
    g_fst.set_start(start_state)
    
    # Create states for each word
    for i in range(word_symbols.num_symbols()):
        word = word_symbols.find(i)
        if word and word not in ["<eps>", "<s>", "</s>"]:
            state = g_fst.add_state()
            states[word] = state
            
            # Arc from start to word state
            g_fst.add_arc(start_state, fst.Arc(
                i, i, fst.Weight.one(g_fst.weight_type()), state
            ))
            
            # Self-loop
            g_fst.add_arc(state, fst.Arc(
                i, i, fst.Weight.one(g_fst.weight_type()), state
            ))
            
            g_fst.set_final(state)
    
    # Add sentence markers
    bos_state = g_fst.add_state()
    eos_state = g_fst.add_state()
    
    g_fst.add_arc(start_state, fst.Arc(
        word_symbols.find("<s>"),
        word_symbols.find("<s>"),
        fst.Weight.one(g_fst.weight_type()),
        bos_state
    ))
    
    g_fst.add_arc(bos_state, fst.Arc(
        word_symbols.find("</s>"),
        word_symbols.find("</s>"),
        fst.Weight.one(g_fst.weight_type()),
        eos_state
    ))
    
    g_fst.set_final(eos_state)
    
    # Optimize
    g_fst = pynini.determinize(g_fst)
    g_fst.minimize()
    g_fst.write(output_path)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    import tempfile
    import os
    
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create example text file
        text_file = os.path.join(tmpdir, "train.txt")
        with open(text_file, 'w') as f:
            f.write("hello world\n")
            f.write("asl translation\n")
            f.write("sign language\n")
        
        # Create word symbols
        word_symbols = SymbolTable()
        word_symbols.add_symbol("<eps>")
        word_symbols.add_symbol("<unk>")
        word_symbols.add_symbol("<s>")
        word_symbols.add_symbol("</s>")
        
        words = ["hello", "world", "asl", "translation", "sign", "language"]
        for word in words:
            word_symbols.add_symbol(word)
        
        # Build language model
        output_dir = os.path.join(tmpdir, "lm")
        g_fst_path = build_language_model(
            [text_file],
            output_dir,
            word_symbols,
            order=2,
            prune_threshold=0.0
        )
        
        print(f"Created language model at: {g_fst_path}")