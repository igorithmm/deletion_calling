"""BAM file handling and read extraction"""
from typing import List, Tuple, Optional
import pysam
import numpy as np


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
                if start <= pos < end:
                    clip_counts[pos] += 1
                    
            if last_op in (4, 5) and read.reference_end is not None:
                pos = read.reference_end - 1
                if start <= pos < end:
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
            List of pileup records (pos, is_paired, is_proper_pair, mapq, cigar_type, base)
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")

        pileup_records = []
        for pileup_column in self._bam_file.pileup(chrom, start, end):
            if start <= pileup_column.pos < end:
                for pileup_read in pileup_column.pileups:
                    alignment = pileup_read.alignment
                    if alignment.cigarstring is None or pileup_read.query_position is None:
                        continue

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
                    )
                    pileup_records.append(record)

        return pileup_records
