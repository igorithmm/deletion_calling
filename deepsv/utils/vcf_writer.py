"""VCF file generation utilities"""
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DeletionCall:
    """Represents a deletion call"""
    chrom: str
    start: int
    end: int
    quality: float
    filter_status: str = "PASS"
    
    @property
    def length(self) -> int:
        """Calculate deletion length"""
        return abs(self.end - self.start)


class VCFWriter:
    """Writes deletion calls to VCF format"""
    
    def __init__(self, reference: Optional[str] = None, sample: str = "SAMPLE"):
        """
        Initialize VCF writer
        
        Args:
            reference: Path to reference genome (optional)
            sample: Sample name
        """
        self.reference = reference
        self.sample = sample
        self.calls: List[DeletionCall] = []
    
    def add_call(self, call: DeletionCall):
        """Add a deletion call"""
        self.calls.append(call)
    
    def write(self, output_path: Path, min_quality: float = 0.5):
        """
        Write calls to VCF file
        
        Args:
            output_path: Path to output VCF file
            min_quality: Minimum quality threshold
        """
        with open(output_path, 'w') as f:
            # Write header
            self._write_header(f)
            
            # Write calls
            for call in sorted(self.calls, key=lambda x: (x.chrom, x.start)):
                if call.quality >= min_quality:
                    self._write_record(f, call)
    
    def _write_header(self, f):
        """Write VCF header"""
        f.write("##fileformat=VCFv4.3\n")
        f.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
        f.write("##source=DeepSV\n")
        f.write("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Length of variant\">\n")
        f.write("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of structural variant\">\n")
        f.write("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of variant\">\n")
        f.write("##ALT=<ID=DEL,Description=\"Deletion\">\n")
        f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{self.sample}\n")
    
    def _write_record(self, f, call: DeletionCall):
        """Write a single VCF record"""
        chrom = call.chrom
        pos = call.start
        end = call.end
        svlen = -call.length  # Negative for deletions
        
        ref = "N"  # Placeholder
        alt = "<DEL>"
        qual = int(call.quality * 100)
        filter_status = call.filter_status
        
        info = f"SVTYPE=DEL;END={end};SVLEN={svlen}"
        format_fields = "GT"
        sample_data = "0/1"  # Heterozygous
        
        f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t{filter_status}\t"
               f"{info}\t{format_fields}\t{sample_data}\n")

