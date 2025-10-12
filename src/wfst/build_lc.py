"""Build lexicon (L) and context (C) WFSTs.

This module implements:
1. Lexicon FST (L): Maps phonological tokens to words
2. Context FST (C): Bi-phone context (left=1, right=1)
3. HMM FST (H): 3-state left-to-right HMM per phonological token
"""

import pynini
from pynini import Fst, SymbolTable
from pynini.lib import pynutil
import openfst_python as fst
from typing import Dict, List, Tuple, Set


def build_hmm(phonemes: List[str], 
              num_states: int = 3,
              self_loop_prob: float = 0.7) -> Fst:
    """Build HMM FST for phonological tokens.
    
    Creates a 3-state left-to-right HMM for each phoneme with:
    - Self-loop probability: 0.7
    - Transition to next state: 0.3
    
    Args:
        phonemes: List of phonological tokens
        num_states: Number of HMM states per phoneme
        self_loop_prob: Self-loop probability
        
    Returns:
        HMM FST
    """
    hmm_fst = Fst()
    
    # Create symbol tables
    input_symbols = SymbolTable()
    output_symbols = SymbolTable()
    
    # Add epsilon
    epsilon = "<eps>"
    input_symbols.add_symbol(epsilon)
    output_symbols.add_symbol(epsilon)
    
    # Add phonemes to symbol tables
    for phoneme in phonemes:
        input_symbols.add_symbol(phoneme)
        output_symbols.add_symbol(phoneme)
    
    hmm_fst.set_input_symbols(input_symbols)
    hmm_fst.set_output_symbols(output_symbols)
    
    # Start state
    start_state = hmm_fst.add_state()
    hmm_fst.set_start(start_state)
    
    # Build HMM for each phoneme
    state_offset = 1
    
    for phoneme in phonemes:
        phoneme_states = []
        
        # Create states for this phoneme
        for i in range(num_states):
            state = hmm_fst.add_state()
            phoneme_states.append(state)
            
            # Add self-loop
            if i > 0 or num_states == 1:
                weight = -math.log(self_loop_prob)  # Convert to log space
                hmm_fst.add_arc(state, fst.Arc(
                    input_symbols.find(phoneme),
                    output_symbols.find(phoneme),
                    fst.Weight(hmm_fst.weight_type(), weight),
                    state
                ))
        
        # Connect to start state for first phoneme
        if state_offset == 1:
            hmm_fst.add_arc(start_state, fst.Arc(
                input_symbols.find(epsilon),
                output_symbols.find(epsilon),
                fst.Weight.one(hmm_fst.weight_type()),
                phoneme_states[0]
            ))
        
        # Add transitions between states
        for i in range(num_states - 1):
            next_state_prob = 1.0 - self_loop_prob
            weight = -math.log(next_state_prob)
            hmm_fst.add_arc(phoneme_states[i], fst.Arc(
                input_symbols.find(epsilon),
                output_symbols.find(epsilon),
                fst.Weight(hmm_fst.weight_type(), weight),
                phoneme_states[i + 1]
            ))
        
        # Last state is final
        hmm_fst.set_final(phoneme_states[-1])
        
        state_offset += num_states
    
    return hmm_fst


def build_lexicon(pronunciation_dict: Dict[str, List[str]],
                  phoneme_symbols: SymbolTable,
                  word_symbols: SymbolTable) -> Fst:
    """Build lexicon FST (L).
    
    Creates a left-deterministic FST that maps phonological tokens to words.
    
    Args:
        pronunciation_dict: Dictionary mapping words to phoneme sequences
        phoneme_symbols: Phoneme symbol table
        word_symbols: Word symbol table
        
    Returns:
        Lexicon FST
    """
    lex_fst = Fst()
    
    # Set symbol tables
    lex_fst.set_input_symbols(phoneme_symbols)
    lex_fst.set_output_symbols(word_symbols)
    
    # Start state
    start_state = lex_fst.add_state()
    lex_fst.set_start(start_state)
    
    # Build lexicon from pronunciation dictionary
    for word, phonemes in pronunciation_dict.items():
        current_state = start_state
        
        # Add arcs for each phoneme
        for i, phoneme in enumerate(phonemes):
            next_state = lex_fst.add_state()
            
            # Input is phoneme
            phoneme_id = phoneme_symbols.find(phoneme)
            
            # Output is word only at the end
            if i == len(phonemes) - 1:
                word_id = word_symbols.find(word)
            else:
                word_id = word_symbols.find("<eps>")
            
            lex_fst.add_arc(current_state, fst.Arc(
                phoneme_id,
                word_id,
                fst.Weight.one(lex_fst.weight_type()),
                next_state
            ))
            
            current_state = next_state
        
        # Mark final state
        lex_fst.set_final(current_state)
    
    # Add unknown word handling
    unk_state = lex_fst.add_state()
    unk_phoneme = "<unk>"
    if phoneme_symbols.find(unk_phoneme) != -1:
        lex_fst.add_arc(start_state, fst.Arc(
            phoneme_symbols.find(unk_phoneme),
            word_symbols.find("<unk>"),
            fst.Weight.one(lex_fst.weight_type()),
            unk_state
        ))
        lex_fst.set_final(unk_state)
    
    return lex_fst


def build_context(phonemes: List[str],
                  left_context: int = 1,
                  right_context: int = 1) -> Fst:
    """Build context FST (C) for bi-phone modeling.
    
    Creates a context FST that models left and right phoneme context.
    
    Args:
        phonemes: List of phonological tokens
        left_context: Left context size
        right_context: Right context size
        
    Returns:
        Context FST
    """
    context_fst = Fst()
    
    # Create symbol tables
    input_symbols = SymbolTable()
    output_symbols = SymbolTable()
    
    # Add epsilon and phonemes
    epsilon = "<eps>"
    input_symbols.add_symbol(epsilon)
    output_symbols.add_symbol(epsilon)
    
    for phoneme in phonemes:
        input_symbols.add_symbol(phoneme)
        output_symbols.add_symbol(phoneme)
    
    context_fst.set_input_symbols(input_symbols)
    context_fst.set_output_symbols(output_symbols)
    
    # Context states represent (left_context, current_phoneme, right_context)
    # For bi-phone: (prev_phoneme, current_phoneme)
    
    # Start state (no context)
    start_state = context_fst.add_state()
    context_fst.set_start(start_state)
    
    # Create context states
    context_states = {}
    state_id = 1
    
    # Add states for all context combinations
    for prev_phoneme in [epsilon] + phonemes:
        for curr_phoneme in phonemes:
            state_key = (prev_phoneme, curr_phoneme)
            state = context_fst.add_state()
            context_states[state_key] = state
            
            # Add transition from start state for initial phonemes
            if prev_phoneme == epsilon:
                context_fst.add_arc(start_state, fst.Arc(
                    input_symbols.find(curr_phoneme),
                    input_symbols.find(curr_phoneme),
                    fst.Weight.one(context_fst.weight_type()),
                    state
                ))
    
    # Add transitions between context states
    for (prev_phoneme, curr_phoneme), state in context_states.items():
        for next_phoneme in phonemes:
            next_state_key = (curr_phoneme, next_phoneme)
            if next_state_key in context_states:
                next_state = context_states[next_state_key]
                
                # Input: next_phoneme, Output: curr_phoneme with context
                context_output = f"{prev_phoneme}_{curr_phoneme}_{next_phoneme}"
                if not output_symbols.find(context_output):
                    output_symbols.add_symbol(context_output)
                
                context_fst.add_arc(state, fst.Arc(
                    input_symbols.find(next_phoneme),
                    output_symbols.find(context_output),
                    fst.Weight.one(context_fst.weight_type()),
                    next_state
                ))
    
    # Mark all context states as final
    for state in context_states.values():
        context_fst.set_final(state)
    
    return context_fst


def compose_hcl(hmm_fst: Fst, context_fst: Fst, lex_fst: Fst) -> Fst:
    """Compose H, C, and L FSTs to create HCL FST.
    
    Args:
        hmm_fst: HMM FST
        context_fst: Context FST
        lex_fst: Lexicon FST
        
    Returns:
        Composed HCL FST
    """
    # Compose H and C
    hc_fst = fst.compose(hmm_fst, context_fst)
    
    # Compose HC and L
    hcl_fst = fst.compose(hc_fst, lex_fst)
    
    # Optimize the composed FST
    hcl_fst = fst.determinize(hcl_fst)
    hcl_fst.minimize()
    
    return hcl_fst


def load_pronunciation_dict(file_path: str) -> Dict[str, List[str]]:
    """Load pronunciation dictionary from file.
    
    File format: word phoneme1 phoneme2 ...
    
    Args:
        file_path: Path to pronunciation dictionary file
        
    Returns:
        Dictionary mapping words to phoneme sequences
    """
    pronunciation_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                phonemes = parts[1:]
                pronunciation_dict[word] = phonemes
    
    return pronunciation_dict


def extract_phonemes(pronunciation_dict: Dict[str, List[str]]) -> Set[str]:
    """Extract unique phonemes from pronunciation dictionary.
    
    Args:
        pronunciation_dict: Dictionary mapping words to phoneme sequences
        
    Returns:
        Set of unique phonemes
    """
    phonemes = set()
    
    for phoneme_seq in pronunciation_dict.values():
        phonemes.update(phoneme_seq)
    
    return phonemes


def create_symbol_tables(phonemes: Set[str], 
                        pronunciation_dict: Dict[str, List[str]]) -> Tuple[SymbolTable, SymbolTable]:
    """Create symbol tables for phonemes and words.
    
    Args:
        phonemes: Set of phonemes
        pronunciation_dict: Dictionary mapping words to phoneme sequences
        
    Returns:
        Tuple of (phoneme_symbol_table, word_symbol_table)
    """
    # Phoneme symbol table
    phoneme_symbols = SymbolTable()
    phoneme_symbols.add_symbol("<eps>")
    phoneme_symbols.add_symbol("<unk>")
    
    for phoneme in sorted(phonemes):
        phoneme_symbols.add_symbol(phoneme)
    
    # Word symbol table
    word_symbols = SymbolTable()
    word_symbols.add_symbol("<eps>")
    word_symbols.add_symbol("<unk>")
    word_symbols.add_symbol("<s>")
    word_symbols.add_symbol("</s>")
    
    for word in sorted(pronunciation_dict.keys()):
        word_symbols.add_symbol(word)
    
    return phoneme_symbols, word_symbols


def build_complete_hcl(pronunciation_file: str,
                      output_dir: str,
                      left_context: int = 1,
                      right_context: int = 1,
                      hmm_states: int = 3,
                      self_loop_prob: float = 0.7) -> str:
    """Build complete HCL FST from pronunciation dictionary.
    
    Args:
        pronunciation_file: Path to pronunciation dictionary
        output_dir: Output directory for FST files
        left_context: Left context size
        right_context: Right context size
        hmm_states: Number of HMM states
        self_loop_prob: Self-loop probability
        
    Returns:
        Path to composed HCL FST file
    """
    # Load data
    pronunciation_dict = load_pronunciation_dict(pronunciation_file)
    phonemes = extract_phonemes(pronunciation_dict)
    phoneme_symbols, word_symbols = create_symbol_tables(phonemes, pronunciation_dict)
    
    # Build individual FSTs
    print("Building HMM FST...")
    hmm_fst = build_hmm(list(phonemes), hmm_states, self_loop_prob)
    
    print("Building Context FST...")
    context_fst = build_context(list(phonemes), left_context, right_context)
    
    print("Building Lexicon FST...")
    lex_fst = build_lexicon(pronunciation_dict, phoneme_symbols, word_symbols)
    
    # Compose FSTs
    print("Composing HCL FST...")
    hcl_fst = compose_hcl(hmm_fst, context_fst, lex_fst)
    
    # Save FSTs
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    hmm_path = os.path.join(output_dir, "hmm.fst")
    context_path = os.path.join(output_dir, "context.fst")
    lex_path = os.path.join(output_dir, "lexicon.fst")
    hcl_path = os.path.join(output_dir, "hcl.fst")
    
    hmm_fst.write(hmm_path)
    context_fst.write(context_path)
    lex_fst.write(lex_path)
    hcl_fst.write(hcl_path)
    
    # Save symbol tables
    phoneme_symbols.write_text(os.path.join(output_dir, "phonemes.syms"))
    word_symbols.write_text(os.path.join(output_dir, "words.syms"))
    
    print(f"HCL FST saved to: {hcl_path}")
    return hcl_path


if __name__ == "__main__":
    # Example usage
    pronunciation_dict_path = "data/pronunciation.dict"
    output_directory = "models/wfst"
    
    # Create example pronunciation dictionary if it doesn't exist
    import os
    os.makedirs(os.path.dirname(pronunciation_dict_path), exist_ok=True)
    
    if not os.path.exists(pronunciation_dict_path):
        with open(pronunciation_dict_path, 'w') as f:
            # Example entries
            f.write("HELLO HH AH L OW\n")
            f.write("WORLD W ER L D\n")
            f.write("ASL AE S EH L\n")
            f.write("SIGN S AY N\n")
    
    hcl_path = build_complete_hcl(
        pronunciation_dict_path,
        output_directory,
        left_context=1,
        right_context=1,
        hmm_states=3,
        self_loop_prob=0.7
    )