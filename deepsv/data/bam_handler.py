"""BAM file handling and read extraction"""
from typing import List, Tuple, Optional, Dict
import pysam
import numpy as np
import pandas as pd


# Nucleotide to one-hot channel index mapping
_NUC_TO_IDX: Dict[str, int] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Channel index constants for the alignment tensor
CH_NUC_A, CH_NUC_C, CH_NUC_G, CH_NUC_T = 0, 1, 2, 3
CH_IS_PAIRED = 4
CH_IS_PROPER_PAIR = 5
CH_BASE_QUALITY = 6
CH_STRAND = 7
CH_MAPPING_QUALITY = 8
CH_INSERT_SIZE_ZSCORE = 9
CH_IS_SUPPLEMENTARY = 10
CH_CIGAR_DELETION = 11
CH_CIGAR_SOFT_CLIP = 12
NUM_ALIGNMENT_CHANNELS = 13


class BAMHandler:
    """Handles BAM file operations and read extraction"""
    
    def __init__(self, bam_path: str):
        """
        Initialize BAM handler
        
        Args:
            bam_path: Path to BAM file
        """
        self.bam_path = bam_path
        self._bam_file: Optional[pysam.AlignmentFile] = None
    
    def __enter__(self):
        """Context manager entry"""
        self._bam_file = pysam.AlignmentFile(self.bam_path, "rb")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._bam_file:
            self._bam_file.close()
    
    def get_reads(self, chrom: str, start: int, end: int) -> List[pysam.AlignedSegment]:
        """
        Extract reads from a genomic region
        
        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            List of aligned segments
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")
        
        reads = []
        for read in self._bam_file.fetch(chrom, start, end):
            if read.cigarstring is not None:
                reads.append(read)
        return reads
    
    def get_coverage_depth(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Calculate coverage depth for a region
        
        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            Array of depth values
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")
        
        coverage = self._bam_file.count_coverage(chrom, start, end)
        depth = np.array(list(coverage)).sum(axis=0)
        return depth
    
    def get_clipping_info(self, chrom: str, start: int, end: int) -> dict:
        """
        Extract soft-clipping information for a region
        
        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            Dictionary mapping position to clipping count
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")
        
        from collections import defaultdict
        clip_counts = defaultdict(int)
        for read in self._bam_file.fetch(chrom, start, end):
            if not read.cigartuples:
                continue
            
            # Check for soft/hard clips at the alignment boundaries
            first_op, _ = read.cigartuples[0]
            last_op, _ = read.cigartuples[-1]
            
            if first_op in (4, 5):
                pos = read.reference_start
                if start <= pos <= end:
                    clip_counts[pos] += 1
                    
            if last_op in (4, 5) and read.reference_end is not None:
                pos = read.reference_end - 1
                if start <= pos <= end:
                    clip_counts[pos] += 1
                    
        return dict(clip_counts)
    
    def _get_cigar_at_position(self, read: pysam.AlignedSegment, query_pos: int) -> int:
        """Get CIGAR operation type at a query position"""
        index = 0
        for op, length in read.cigartuples:
            consumes_query = op in (0, 1, 4, 7, 8)  # M, I, S, =, X
            if consumes_query:
                if query_pos < index + length:
                    return op
                index += length
        return -1
    
    def get_pileup_data(self, chrom: str, start: int, end: int) -> List[Tuple]:
        """
        Extract pileup data for a region
        
        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            List of pileup records (pos, is_paired, is_proper_pair, mapq, cigar_type, base, kmer)
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")
        
        pileup_records = []
        for pileup_column in self._bam_file.pileup(chrom, start, end):
            if start <= pileup_column.pos <= end:
                for pileup_read in pileup_column.pileups:
                    alignment = pileup_read.alignment
                    if alignment.cigarstring is None or pileup_read.query_position is None:
                        continue
                    
                    # Get CIGAR operation at query position
                    cigar_type = self._get_cigar_at_position(
                        alignment, pileup_read.query_position
                    )
                    
                    base = alignment.query_sequence[pileup_read.query_position]
                    record = (
                        pileup_column.pos,
                        alignment.is_paired,
                        alignment.is_proper_pair,
                        alignment.mapping_quality,
                        cigar_type,
                        base,
                        self._get_kmer(alignment.query_sequence, pileup_read.query_position)
                    )
                    pileup_records.append(record)
        
        return pileup_records

    def _get_kmer(self, sequence: str, pos: int, k: int = 6) -> str:
        """
        Extract k-mer centered around the position.
        For k=6, we take positions [pos-3, pos+3) to center the context.
        Boundary checks are handled by padding with 'N'.
        """
        start = pos - (k // 2)
        end = start + k
        
        # Determine actual slice indices
        actual_start = max(0, start)
        actual_end = min(len(sequence), end)
        
        # Extract available sequence
        if actual_start < actual_end:
            subseq = sequence[actual_start:actual_end]
        else:
            subseq = ""
            
        # Pad if necessary
        left_pad = "N" * (actual_start - start)
        right_pad = "N" * (end - actual_end)
        
        kmer = left_pad + subseq + right_pad
        # Ensure it's exactly k length (in case of logic errors, though math above should match)
        return kmer[:k]

    # ------------------------------------------------------------------
    # Multichannel alignment tensor extraction (DeepSV3 / tensor mode)
    # ------------------------------------------------------------------


    def get_alignment_tensor(
        self,
        chrom: str,
        start: int,
        end: int,
        max_reads: int = 100,
    ) -> np.ndarray:
        """Build a (NUM_ALIGNMENT_CHANNELS, max_reads, W) float32 tensor.

        Channels (13 total):
            0-3   nucleotide_identity  one-hot (A, C, G, T)
            4     is_paired            binary
            5     is_proper_pair       binary
            6     base_quality         Phred / 40, clipped [0, 1]
            7     strand               0 = forward, 1 = reverse
            8     mapping_quality      MAPQ / 60, clipped [0, 1]
            9     insert_size_zscore   Z-score (computed on the reads in this window)
            10    is_supplementary     binary
            11    cigar_deletion       binary (CIGAR == D at this position)
            12    cigar_soft_clip      binary (CIGAR == S at this position)

        Rows where a read does not cover a position are left as all zeros.

        Args:
            chrom: Chromosome name.
            start: 0-based inclusive start of the genomic window.
            end:   0-based exclusive end of the genomic window (width = end - start).
            max_reads: Maximum number of reads (H dimension).  Excess reads
                       are discarded; missing rows are zero-padded.

        Returns:
            np.ndarray of shape (NUM_ALIGNMENT_CHANNELS, max_reads, W),
            dtype float32.
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")

        width = end - start
        tensor = np.zeros(
            (NUM_ALIGNMENT_CHANNELS, max_reads, width), dtype=np.float32
        )

        # Fetch reads that overlap the window
        reads = []
        insert_sizes = []
        for read in self._bam_file.fetch(chrom, start, end):
            if read.cigarstring is None or read.query_sequence is None:
                continue
            if read.is_duplicate or read.is_qcfail or read.is_secondary:
                continue
            reads.append(read)
            if read.is_paired and read.template_length != 0:
                insert_sizes.append(abs(read.template_length))

        # Compute insert-size statistics for Z-score normalisation
        if len(insert_sizes) >= 2:
            is_mean = float(np.mean(insert_sizes))
            is_std = float(np.std(insert_sizes))
            if is_std == 0:
                is_std = 1.0
        else:
            is_mean, is_std = 0.0, 1.0

        # Truncate to max_reads
        reads = reads[:max_reads]

        for row_idx, read in enumerate(reads):
            # Build a map: ref_pos -> (query_pos, cigar_op) using
            # get_aligned_pairs which handles all CIGAR complexities.
            aligned_pairs = read.get_aligned_pairs(with_seq=False)
            # aligned_pairs: list of (query_pos | None, ref_pos | None)

            # Per-read features (constant across positions)
            is_paired = float(read.is_paired)
            is_proper = float(read.is_proper_pair)
            strand = 1.0 if read.is_reverse else 0.0
            mapq = min(read.mapping_quality / 60.0, 1.0)
            is_supp = float(read.is_supplementary)

            # Insert size z-score
            if read.is_paired and read.template_length != 0:
                isz = (abs(read.template_length) - is_mean) / is_std
            else:
                isz = 0.0

            # Pre-calculate soft-clipped positions in reference coordinates
            if read.cigartuples:
                first_op, _ = read.cigartuples[0]
                last_op, _ = read.cigartuples[-1]
                
                if first_op in (4, 5):
                    clip_pos = read.reference_start
                    if start <= clip_pos < end:
                        tensor[CH_CIGAR_SOFT_CLIP, row_idx, clip_pos - start] = 1.0
                
                if last_op in (4, 5):
                    clip_pos = read.reference_end - 1
                    if start <= clip_pos < end:
                        tensor[CH_CIGAR_SOFT_CLIP, row_idx, clip_pos - start] = 1.0

            # Determine which columns are active for this read
            read_start_col = max(0, read.reference_start - start)
            read_end_col = min(width, read.reference_end - start)
            
            if read_start_col < read_end_col:
                tensor[CH_IS_PAIRED, row_idx, read_start_col:read_end_col] = is_paired
                tensor[CH_IS_PROPER_PAIR, row_idx, read_start_col:read_end_col] = is_proper
                tensor[CH_STRAND, row_idx, read_start_col:read_end_col] = strand
                tensor[CH_MAPPING_QUALITY, row_idx, read_start_col:read_end_col] = mapq
                tensor[CH_INSERT_SIZE_ZSCORE, row_idx, read_start_col:read_end_col] = isz
                tensor[CH_IS_SUPPLEMENTARY, row_idx, read_start_col:read_end_col] = is_supp

            # Build ref_pos -> (query_pos, cigar_op) index for the window
            for qpos, rpos in aligned_pairs:
                if rpos is None or rpos < start or rpos >= end:
                    continue
                col_idx = rpos - start

                if qpos is not None:
                    # Per-position channels
                    seq = read.query_sequence
                    quals = read.query_qualities

                    # Nucleotide identity (one-hot)
                    nuc = seq[qpos].upper()
                    nuc_idx = _NUC_TO_IDX.get(nuc)
                    if nuc_idx is not None:
                        tensor[nuc_idx, row_idx, col_idx] = 1.0

                    # Base quality
                    if quals is not None:
                        bq = min(quals[qpos] / 40.0, 1.0)
                        tensor[CH_BASE_QUALITY, row_idx, col_idx] = bq
                else:
                    # qpos is None → deletion or skipped region in the read at this ref pos
                    tensor[CH_CIGAR_DELETION, row_idx, col_idx] = 1.0
                    # Still fill per-read channels (already done above)

        return tensor
