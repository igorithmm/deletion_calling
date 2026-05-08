"""VCF file handling and variant processing"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Variant:
    """Represents a structural variant"""
    chrom: str
    start: int
    end: int
    sv_type: str = "DEL"
    
    @property
    def length(self) -> int:
        """Calculate variant length"""
        return abs(self.end - self.start)


class VCFHandler:
    """Handles VCF file parsing using pysam"""
    
    def __init__(self, vcf_path: str):
        """
        Initialize VCF handler
        
        Args:
            vcf_path: Path to VCF file (gzip compressed and indexed)
        """
        self.vcf_path = vcf_path
    
    def load_variants(self, variant_type: str = "deletion", sample_id: str = None) -> List[Variant]:
        """
        Load variants from VCF file using pysam
        
        Args:
            variant_type: Type of variants to load ('deletion' or 'non_deletion')
            sample_id: Optional sample ID to filter variants for.
            
        Returns:
            List of Variant objects
        """
        import pysam
        variants = []
        
        # Open VCF file with pysam
        vcf = pysam.VariantFile(self.vcf_path)
        
        # Check if sample exists if requested
        if sample_id:
            if sample_id not in vcf.header.samples:
                raise ValueError(f"Sample {sample_id} not found in VCF header")
            vcf.subset_samples([sample_id])
        
        for record in vcf:
            # Check SV type from INFO
            sv_type = None
            if 'SVTYPE' in record.info:
                sv_type = record.info['SVTYPE']
                if isinstance(sv_type, tuple) or isinstance(sv_type, list):
                    sv_type = sv_type[0]
            
            # Check ALT for <DEL>
            is_del_alt = False
            if record.alts:
                for alt in record.alts:
                    if alt == '<DEL>':
                        is_del_alt = True
                        break
            
            # Strict filtering for deletions
            if variant_type == "deletion":
                is_deletion = (sv_type == "DEL") or is_del_alt
                if not is_deletion:
                    continue
            
            # Determine final type
            final_type = sv_type if sv_type else ("DEL" if is_del_alt else "UNKNOWN")

            # Filter by sample genotype if requested
            if sample_id:
                # Get genotype for the sample
                sample_record = record.samples[sample_id]
                gt = sample_record['GT']
                # Check if variant is present (at least one allele is non-reference and not missing)
                has_variant = False
                for allele in gt:
                    if allele is not None and allele > 0:
                        has_variant = True
                        break
                
                if not has_variant:
                    continue
            
            chrom = record.chrom
            start = record.pos
            
            end = None
            if 'END' in record.info:
                end = record.info['END']
                if isinstance(end, tuple) or isinstance(end, list):
                    end = end[0]
            else:
                 end = record.stop
            
            variant = Variant(chrom=chrom, start=start, end=end, sv_type=final_type)
            variants.append(variant)
        
        vcf.close()
        
        return variants
    
    def get_non_deletion_regions(self, variants: List[Variant],
                                 anchor_type: str = "up") -> List[Variant]:
        """
        Generate non-deletion anchor regions for training (Level 1 – hard negatives).

        Args:
            variants: List of deletion variants
            anchor_type: 'up' or 'down' anchor

        Returns:
            List of non-deletion regions
        """
        anchor_regions = []

        for variant in variants:
            del_length = variant.length
            if del_length > 700:
                del_length = 4 * del_length // 5  # Cap at 80% of original

            if anchor_type == "up":
                start = max(0, variant.start - del_length - 150)
                end = start + del_length
            else:  # down anchor
                start = variant.end + 150
                end = start + del_length

            anchor = Variant(
                chrom=variant.chrom,
                start=start,
                end=end,
                sv_type="NON_DEL",
            )
            anchor_regions.append(anchor)

        return anchor_regions

    # ------------------------------------------------------------------
    # Level-2 & Level-3 negative samplers (E3 mixed strategy)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sv_index(variants: List[Variant]) -> Dict[str, List[Tuple[int, int]]]:
        """Build a per-chromosome index of known-SV intervals for fast overlap checks."""
        index: Dict[str, List[Tuple[int, int]]] = {}
        for v in variants:
            index.setdefault(v.chrom, []).append((v.start, v.end))
        return index

    @staticmethod
    def _overlaps_sv(chrom: str, start: int, end: int,
                     sv_index: Dict[str, List[Tuple[int, int]]]) -> bool:
        """Return True if [start, end) overlaps any known SV on *chrom*."""
        for sv_start, sv_end in sv_index.get(chrom, []):
            if start < sv_end and end > sv_start:
                return True
        return False

    def sample_regional_negatives(
        self,
        variant: Variant,
        all_variants: List[Variant],
        chrom_lengths: Dict[str, int],
        distance_kb: int = 200,
        max_tries: int = 10,
        rng: Optional[random.Random] = None,
    ) -> List[Variant]:
        """Level-2 negatives: same chromosome, ~distance_kb away from the variant.

        Samples two candidate positions (upstream / downstream of the SV) and
        returns those that:
          * stay within chromosome bounds
          * do not overlap any known SV

        Args:
            variant:        The positive deletion being paired.
            all_variants:   Full list of known variants (for overlap filtering).
            chrom_lengths:  Dict mapping chrom name → length in bp.
            distance_kb:    Minimum distance from the deletion in kilobases.
            max_tries:      How many jittered offsets to attempt per side.
            rng:            Optional seeded random.Random for reproducibility.

        Returns:
            List of 0–2 Variant objects (NON_DEL) from the same chromosome.
        """
        if rng is None:
            rng = random.Random()

        sv_index = self._build_sv_index(all_variants)
        win_len = variant.length
        if win_len > 700:
            win_len = 4 * win_len // 5

        offset_bp = distance_kb * 1_000
        chrom_len = chrom_lengths.get(variant.chrom, 0)
        results: List[Variant] = []

        for direction in (-1, +1):          # upstream, downstream
            for attempt in range(max_tries):
                # Add a small random jitter (±20 % of offset) to avoid always
                # landing at the exact same genomic locus.
                jitter = rng.randint(-offset_bp // 5, offset_bp // 5)
                raw_offset = offset_bp + jitter

                if direction == -1:
                    start = variant.start - raw_offset - win_len
                else:
                    start = variant.end + raw_offset

                end = start + win_len

                if start < 0 or (chrom_len and end > chrom_len):
                    continue
                if self._overlaps_sv(variant.chrom, start, end, sv_index):
                    continue

                results.append(Variant(
                    chrom=variant.chrom,
                    start=start,
                    end=end,
                    sv_type="NON_DEL",
                ))
                break  # found a valid candidate for this direction

        return results

    def sample_random_negatives(
        self,
        all_variants: List[Variant],
        chrom_lengths: Dict[str, int],
        win_len: int,
        n_windows: int,
        centromere_mask: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        max_tries_per_window: int = 50,
        rng: Optional[random.Random] = None,
    ) -> List[Variant]:
        """Level-3 negatives: random positions across the entire genome.

        Filters out windows that overlap:
          * any known SV (from *all_variants*)
          * centromere / telomere regions (optional *centromere_mask*)

        Args:
            all_variants:          Full list of known variants.
            chrom_lengths:         Dict mapping chrom name → length in bp.
            win_len:               Window length in bp (should match variant length).
            n_windows:             Number of random windows to return.
            centromere_mask:       Optional per-chrom list of (start, end) masked regions.
            max_tries_per_window:  Rejection-sampling attempts before giving up.
            rng:                   Optional seeded random.Random.

        Returns:
            List of up to *n_windows* NON_DEL Variant objects.
        """
        if rng is None:
            rng = random.Random()

        sv_index = self._build_sv_index(all_variants)
        mask_index: Dict[str, List[Tuple[int, int]]] = centromere_mask or {}

        chroms = [c for c, length in chrom_lengths.items() if length > win_len]
        if not chroms:
            return []

        results: List[Variant] = []
        global_tries = 0
        max_global = n_windows * max_tries_per_window

        while len(results) < n_windows and global_tries < max_global:
            global_tries += 1
            chrom = rng.choice(chroms)
            chrom_len = chrom_lengths[chrom]
            start = rng.randint(0, chrom_len - win_len)
            end = start + win_len

            # Filter: overlaps known SV?
            if self._overlaps_sv(chrom, start, end, sv_index):
                continue
            # Filter: overlaps centromere/telomere mask?
            if self._overlaps_sv(chrom, start, end, mask_index):
                continue

            results.append(Variant(
                chrom=chrom,
                start=start,
                end=end,
                sv_type="NON_DEL",
            ))

        return results
