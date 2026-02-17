#!/usr/bin/env python3
"""
Generate colors for all 4096 6-mers using DNA-BERT and UMAP.
"""
import itertools
import json
import numpy as np
import torch
import umap
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def generate_kmers(k=6):
    """Generate all k-mers of length k"""
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in itertools.product(bases, repeat=k)]

def get_embeddings(model, tokenizer, kmers, batch_size=32):
    """Get embeddings for kmers using the model"""
    model.eval()
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(kmers), batch_size), desc="Generating embeddings"):
        batch_kmers = kmers[i:i+batch_size]
        # DNA-BERT expects space-separated k-mers as input.
        # But here we are embedding the 6-mer itself. 
        # The model is trained on sequences of k-mers. 
        # To get an embedding for a specific 6-mer, we can just feed it as a single token sequence
        # or a sequence constructed of that k-mer.
        # However, checking the tokenizer behavior is important.
        # The tokenizer usually splits sequence into k-mers.
        # If we pass a 6-mer "AAAAAA", it is effectively one token in the vocab (if k=6).
        
        # DNA-BERT tokenizer expects a string of k-mers separated by spaces?
        # Actually, for DNA-BERT-6, the vocab is 6-mers.
        # Let's try to just pass the raw 6-mer string.
        # Since the model is trained on sentences, we should probably just treat the 6-mer as a "sentence" of length 1.
        
        inputs = tokenizer(batch_kmers, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
             # Use the pooler_output (CLS token embedding) which represents the sequence
            # batch_embeddings = outputs.pooler_output
            # OR use the average of the last hidden state
            # outputs.last_hidden_state is [batch, seq_len, hidden_size]
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
        embeddings.append(batch_embeddings.numpy())
        
    return np.vstack(embeddings)

def main():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "deepsv" / "visualization" / "kmer_colors.json"
    
    print("Loading model...")
    model_name = "zhihan1996/DNA_bert_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    print("Generating 6-mers...")
    kmers = generate_kmers(6)
    print(f"Total k-mers: {len(kmers)}")
    
    print("Computing embeddings...")
    embeddings = get_embeddings(model, tokenizer, kmers)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    print("Normalizing to [0, 255]...")
    # Normalize each dimension to 0-1 then scale to 0-255
    min_vals = umap_embeddings.min(axis=0)
    max_vals = umap_embeddings.max(axis=0)
    
    normalized = (umap_embeddings - min_vals) / (max_vals - min_vals)
    rgb_values = (normalized * 255).astype(int)
    
    # Create dictionary
    kmer_colors = {}
    for kmer, rgb in zip(kmers, rgb_values):
        kmer_colors[kmer] = rgb.tolist()
        
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(kmer_colors, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
