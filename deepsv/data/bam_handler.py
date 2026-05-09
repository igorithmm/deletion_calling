"""BAM file handling and read extraction"""
from typing import List, Tuple, Optional
import pysam
import numpy as np
import pandas as pd


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
        Extract clipping/mapping-anomaly signal for a region.

        Replicates the legacy ``get_clip_num()`` algorithm exactly:

        1. For every read overlapping [start, end], determine the CIGAR
           operation type at each genomic position using the legacy walk
           (cumulative length vs genomic position — always yields the
           *last* CIGAR operation whose cumulative length is still less
           than the genomic position).
        2. Accumulate ``-map_type`` per position across all reads.
        3. Integer-divide the per-position sum by 4.

        The resulting values are typically negative (dominated by soft-clip
        op=4 contributing −4 each).  The image generator negates them
        before applying to pixel channels.

        Args:
            chrom: Chromosome name
            start: Start position (inclusive)
            end: End position (exclusive in pysam, but legacy used inclusive
                 — we add 1 internally to match)

        Returns:
            Dictionary mapping position → clipping signal value
        """
        if not self._bam_file:
            raise RuntimeError("BAM file not opened. Use context manager.")

        clip_temp: List[Tuple[int, int]] = []

        for read in self._bam_file.fetch(chrom, start, end):
            if read.cigarstring is None:
                continue

            # Determine effective read span the same way legacy did:
            #   base_pos = read.get_reference_positions(full_length=True)
            #   count leading Nones → soft-clipped query bases at the start
            #   read_start = reference_start - leading_nones
            #   read_end   = read_start + read_len - 1
            base_pos = read.get_reference_positions(full_length=True)
            read_len = len(base_pos)

            leading_nones = 0
            for p in base_pos:
                if p is not None:
                    break
                leading_nones += 1

            read_start = read.reference_start - leading_nones
            read_end = read_start + read_len - 1

            # For every position in the query window that this read covers,
            # walk the CIGAR to find map_type using legacy logic.
            for i in range(end - start):
                pos = start + i
                if pos < read_start or pos > read_end:
                    continue

                # Legacy CIGAR walk: compare *genomic position* against a
                # cumulative CIGAR-length counter starting at 0.  Because
                # genomic positions are much larger than read lengths, this
                # effectively always selects the last CIGAR operation.
                index_ptr = 0
                map_type = -1
                for cigar_op, cigar_len in read.cigartuples:
                    if pos > index_ptr:
                        index_ptr = cigar_len + index_ptr
                        map_type = cigar_op

                clip_temp.append((pos, -map_type))

        if not clip_temp:
            return {}

        clip_record_np = np.array(clip_temp)
        df = pd.DataFrame(clip_record_np, columns=[0, 1])
        clip_record_df = df.groupby(0)[1].sum()
        clip_record_df = clip_record_df // 4
        temp = clip_record_df.reset_index()
        clip_record = np.array(temp).tolist()
        return {int(row[0]): int(row[1]) for row in clip_record}

    def _get_cigar_at_position(self, read: pysam.AlignedSegment, query_pos: int) -> int:
        """Get CIGAR operation type at a query position.

        Replicates the legacy ``pipeup_column`` CIGAR walk: advance a
        cumulative index by **every** CIGAR operation length (not only
        query-consuming ones) and compare against ``query_position``.
        """
        index = 0
        map_type = -1
        for op, length in read.cigartuples:
            if query_pos > index:
                index = length + index
                map_type = op
        return map_type
    
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
