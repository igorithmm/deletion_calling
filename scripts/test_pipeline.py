#!/usr/bin/env python3
"""End-to-end test of the DeepSV3 pipeline components.

Demonstrates:
  1. Reference genome reading (chr1, first 5000 bases)
  2. BAM file reading (first 5000 bases from a read)
  3. Full alignment tensor features for a single read
  4. DNABERT-2 embedding for a 50bp window
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler, NUM_ALIGNMENT_CHANNELS
from deepsv.data.genomic_context import ReferenceGenome, DNABERT2Embedder

# Paths — adjust if needed
BAM_PATH = "raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
FASTA_PATH = "raw/hs37d5.fa"


def test_1_reference_genome():
    """Read reference genome (chr1) and print first 5000 bases."""
    print("=" * 70)
    print("TEST 1: Reference Genome — chr1, first 5000 bases")
    print("=" * 70)

    with ReferenceGenome(FASTA_PATH) as ref:
        seq = ref.get_sequence("1", 0, 5000)
        print(f"Sequence length: {len(seq)}")
        print(f"Composition: A={seq.count('A')} C={seq.count('C')} "
              f"G={seq.count('G')} T={seq.count('T')} N={seq.count('N')}")
        print()

        # Print in rows of 80
        for i in range(0, len(seq), 80):
            print(f"  {i:5d}  {seq[i:i+80]}")

    print()


def test_2_bam_read():
    """Read .bam file and print first 5000 bases from a read."""
    print("=" * 70)
    print("TEST 2: BAM File — first read with ≥100 bases, showing up to 5000 bases")
    print("=" * 70)

    with BAMHandler(BAM_PATH) as bam:
        # Find a read with a decent sequence in a well-covered region
        reads = bam.get_reads("1", 100000, 200000)
        print(f"Fetched {len(reads)} reads from chr1:100000-200000")

        # Pick the first read with a long enough sequence
        chosen = None
        for r in reads:
            if r.query_sequence and len(r.query_sequence) >= 50:
                chosen = r
                break

        if chosen is None:
            print("No suitable read found!")
            return

        seq = chosen.query_sequence[:5000]
        print(f"\nRead name:      {chosen.query_name}")
        print(f"Ref position:   chr1:{chosen.reference_start}-{chosen.reference_end}")
        print(f"Sequence length: {len(chosen.query_sequence)}")
        print(f"CIGAR:          {chosen.cigarstring}")
        print(f"MAPQ:           {chosen.mapping_quality}")
        print(f"Is paired:      {chosen.is_paired}")
        print(f"Is proper pair: {chosen.is_proper_pair}")
        print(f"Is reverse:     {chosen.is_reverse}")
        print(f"Is supplementary: {chosen.is_supplementary}")
        print(f"Insert size:    {chosen.template_length}")
        print(f"\nFirst {len(seq)} bases:")
        for i in range(0, len(seq), 80):
            print(f"  {i:5d}  {seq[i:i+80]}")

    print()


def test_3_alignment_features():
    """Show all alignment features for a single read in a 50bp window."""
    print("=" * 70)
    print("TEST 3: Alignment Tensor — all 13 channels at chr1:100000-100050")
    print("=" * 70)

    region_start, region_end = 100000, 100050
    width = region_end - region_start

    with BAMHandler(BAM_PATH) as bam:
        tensor = bam.get_alignment_tensor("1", region_start, region_end, max_reads=100)
        print(f"Tensor shape: {tensor.shape}  (channels={tensor.shape[0]}, "
              f"reads={tensor.shape[1]}, positions={tensor.shape[2]})")

        # Count reads that actually have data
        active_reads = (tensor.sum(axis=(0, 2)) != 0).sum()
        print(f"Active reads (non-zero rows): {active_reads}")
        print()

        channel_names = [
            "A (one-hot)", "C (one-hot)", "G (one-hot)", "T (one-hot)",
            "is_paired", "is_proper_pair", "base_quality",
            "strand", "mapping_quality", "insert_size_zscore",
            "is_supplementary", "cigar_deletion", "cigar_soft_clip",
        ]

        # Show per-channel statistics
        print("  Ch  Channel Name            Min       Max      Mean    NonZero")
        print("  " + "-" * 65)
        for ch in range(NUM_ALIGNMENT_CHANNELS):
            data = tensor[ch]
            nz = (data != 0).sum()
            print(f"  {ch:2d}  {channel_names[ch]:<22s}  {data.min():8.4f}  "
                  f"{data.max():8.4f}  {data.mean():8.4f}  {nz:7d}")

        # Show detailed view for the first active read
        print(f"\n── Detailed view: Read 0, positions {region_start}-{region_end} ──")
        print(f"  {'Pos':>7s}", end="")
        for name in ["Nuc", "Pair", "Prop", "BQ", "Str", "MQ", "ISZ", "Sup", "Del", "Clip"]:
            print(f"  {name:>5s}", end="")
        print()

        for w in range(min(width, 50)):
            pos = region_start + w
            # Reconstruct nucleotide
            nuc_vals = [tensor[ch, 0, w] for ch in range(4)]
            nuc = "ACGT"[np.argmax(nuc_vals)] if max(nuc_vals) > 0 else "."
            print(f"  {pos:7d}  {nuc:>3s}", end="")
            for ch in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
                v = tensor[ch, 0, w]
                print(f"  {v:5.2f}", end="")
            print()

    print()


def test_4_dnabert_embedding():
    """Show DNABERT-2 embedding for a 50bp window."""
    print("=" * 70)
    print("TEST 4: DNABERT-2 Embedding — 50bp window at chr1:100000-100050")
    print("=" * 70)

    center = 100025  # midpoint of 100000-100050

    with ReferenceGenome(FASTA_PATH) as ref:
        # Extract the extended context window (2500bp)
        extended_seq = ref.get_extended_window("1", center, 2500)
        print(f"Extended context: {len(extended_seq)}bp centered at chr1:{center}")
        print(f"  First 100bp: {extended_seq[:100]}")
        print(f"  Non-N bases: {sum(1 for c in extended_seq if c != 'N')}")
        print()

        # Also show the actual 50bp target window
        target_seq = ref.get_sequence("1", 100000, 100050)
        print(f"Target 50bp:   {target_seq}")
        print()

    # Run DNABERT-2
    print("Loading DNABERT-2 model …")
    embedder = DNABERT2Embedder(device="cpu")

    print("Running inference …")
    embedding = embedder.embed_sequence(extended_seq)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"L2 norm:         {np.linalg.norm(embedding):.4f}")
    print(f"Min / Max:       {embedding.min():.4f} / {embedding.max():.4f}")
    print(f"Mean / Std:      {embedding.mean():.4f} / {embedding.std():.4f}")
    print()

    # Show first 32 dimensions
    print("First 32 dimensions:")
    for i in range(0, 32, 8):
        vals = "  ".join(f"{v:+.4f}" for v in embedding[i:i+8])
        print(f"  [{i:3d}-{i+7:3d}]  {vals}")

    # Show a histogram-like distribution
    print(f"\nEmbedding distribution (768 dims):")
    hist, edges = np.histogram(embedding, bins=10)
    for i in range(len(hist)):
        bar = "█" * (hist[i] * 50 // max(hist))
        print(f"  [{edges[i]:+6.3f}, {edges[i+1]:+6.3f})  {bar}  ({hist[i]})")

    print()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧬 DeepSV3 Pipeline Test Script\n")

    test_1_reference_genome()
    test_2_bam_read()
    test_3_alignment_features()

    # Test 4 requires downloading DNABERT-2 (~450 MB)
    if "--skip-dnabert" in sys.argv:
        print("Skipping DNABERT-2 test (--skip-dnabert)")
    else:
        test_4_dnabert_embedding()

    print("✅ All tests complete!")
