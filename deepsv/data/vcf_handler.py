"""VCF file handling and variant processing"""
from typing import List, Tuple
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
        Generate non-deletion anchor regions for training
        
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
                sv_type="NON_DEL"
            )
            anchor_regions.append(anchor)
        
        return anchor_regions
