#!/usr/bin/env python3
"""Alignment visualization script for DeepSV3.

Provides a clear ASCII "pileup" view of genomic alignments compared to the reference.
Useful for verifying tensor generation logic and inspecting SV regions.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.genomic_context import ReferenceGenome

# ANSI Color codes
COLORS = {
    'A': '\033[92mA\033[0m',  # Green
    'T': '\033[91mT\033[0m',  # Red
    'C': '\033[94mC\033[0m',  # Blue
    'G': '\033[93mG\033[0m',  # Yellow
    'N': '\033[90mN\033[0m',  # Grey
    'match': '\033[90m.\033[0m',  # Grey dot
    'deletion': '\033[31m-\033[0m',  # Red hyphen
    'soft_clip': '\033[35mS\033[0m',  # Magenta S
    'ref_header': '\033[1m',
    'reset': '\033[0m'
}

def color_base(base, ref_base=None, use_colors=True):
    if not use_colors:
        if ref_base and base == ref_base:
            return "."
        return base
        
    if ref_base and base == ref_base:
        return COLORS['match']
    
    return COLORS.get(base, base)

def visualize_region(bam_path: str, fasta_path: str, chrom: str, start: int, end: int, max_reads: int = 50, use_colors: bool = True):
    print(f"\n🧬 Visualization of {chrom}:{start}-{end} ({end-start}bp)")
    print(f"BAM:   {bam_path}")
    print(f"Ref:   {fasta_path}")
    print("-" * 80)

    with ReferenceGenome(fasta_path) as ref, BAMHandler(bam_path) as bam:
        ref_seq = ref.get_sequence(chrom, start, end)
        reads = bam.get_reads(chrom, start, end)
        
        # Header with positions
        # Take every 10th position for labeling
        pos_labels = []
        for i in range(start, end):
            if i % 10 == 0:
                pos_labels.append(str(i)[-1]) # Last digit
            else:
                pos_labels.append(" ")
        
        print(f"       {''.join(pos_labels)}")
        
        # Reference row
        formatted_ref = []
        for b in ref_seq:
            if use_colors:
                formatted_ref.append(f"{COLORS['ref_header']}{b}{COLORS['reset']}")
            else:
                formatted_ref.append(b)
        
        print(f"Ref:   {''.join(formatted_ref)}")
        print("-" * 80)
        
        # Read rows
        # Limit to max_reads
        reads = sorted(reads, key=lambda r: r.reference_start)[:max_reads]
        
        for i, read in enumerate(reads):
            # Alignment map: ref_pos -> sequence_char or special status
            # get_aligned_pairs returns (query_pos, ref_pos)
            pairs = read.get_aligned_pairs(with_seq=False)
            
            read_line = [" "] * (end - start)
            
            for qpos, rpos in pairs:
                if rpos is None or rpos < start or rpos >= end:
                    continue
                
                col_idx = rpos - start
                ref_base = ref_seq[col_idx]
                
                if qpos is None:
                    # Deletion in read (gap compared to reference)
                    if use_colors:
                        read_line[col_idx] = COLORS['deletion']
                    else:
                        read_line[col_idx] = "-"
                else:
                    # Match or mismatch
                    qbase = read.query_sequence[qpos].upper()
                    base_char = color_base(qbase, ref_base, use_colors)
                    read_line[col_idx] = base_char
            
            # Map quality and strand
            strand = "+" if not read.is_reverse else "-"
            metadata = f" MQ:{read.mapping_quality:2d} {strand}"
            
            read_name = f"R{i:02d}:"
            print(f"{read_name}  {''.join(read_line)}{metadata}")

    print("-" * 80)
    print("Legend: '.'=match ref, 'A,C,G,T'=mismatch, '-'=deletion, 'MQ'=mapq, '+/-'=strand")
    print()

def main():
    parser = argparse.ArgumentParser(description="Visualize genomic alignment pileup.")
    parser.add_argument("--bam", default="raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam", help="Path to BAM file")
    parser.add_argument("--fasta", default="raw/hs37d5.fa", help="Path to FASTA file")
    parser.add_argument("--chrom", required=True, help="Chromosome name (e.g., 1)")
    parser.add_argument("--start", type=int, required=True, help="Start position (0-based)")
    parser.add_argument("--end", type=int, required=True, help="End position (0-based)")
    parser.add_argument("--max_reads", type=int, default=30, help="Maximum reads to show")
    parser.add_argument("--no_color", action="store_true", help="Disable ANSI colors")
    
    args = parser.parse_args()
    
    visualize_region(args.bam, args.fasta, args.chrom, args.start, args.end, args.max_reads, not args.no_color)

if __name__ == "__main__":
    main()
