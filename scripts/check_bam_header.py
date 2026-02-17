
import pysam
import sys

def check_bam(bam_path):
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            print(f"BAM file: {bam_path}")
            print("Chromosomes:")
            for i, chrom in enumerate(bam.references):
                if i < 25:
                    print(f"  {chrom}: length {bam.lengths[i]}")
            if len(bam.references) > 25:
                print(f"  ... and {len(bam.references) - 25} more")

            # Check read names/chroms
            print("\nFirst 5 reads:")
            count = 0
            for read in bam.fetch():
                print(f"  {read.reference_name}:{read.reference_start}-{read.reference_end}")
                count += 1
                if count >= 5:
                    break
    except Exception as e:
        print(f"Error reading BAM: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_bam(sys.argv[1])
    else:
        print("Usage: python check_bam.py <bam_file>")
