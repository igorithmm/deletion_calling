import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler, NUM_ALIGNMENT_CHANNELS

BAM_PATH = "raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"

def test_deletion():
    print("--- Test 1: Deletion Region (chr1:4204667-4204717) ---")
    # Using a 50bp window starting at the deletion start
    start, end = 4204667, 4204717
    with BAMHandler(BAM_PATH) as bam:
        tensor = bam.get_alignment_tensor("1", start, end, max_reads=100)
        
        # Channel 11 is CH_CIGAR_DELETION
        del_channel = tensor[11]
        active_reads = (tensor.sum(axis=(0, 2)) != 0).sum()
        del_signals = (del_channel > 0).sum()
        
        print(f"Active reads: {active_reads}")
        print(f"Deletion signals (total cells): {del_signals}")
        
        # In a deletion region, we expect many reads to have 'D' signals
        if del_signals > 0:
            print("SUCCESS: Deletion signals detected.")
        else:
            print("FAILURE: No deletion signals in known deletion region.")
    print()

def test_soft_clipping():
    print("--- Test 2: Soft Clipping Region (around chr1:4200237) ---")
    # Read SRR622461.90926 has 47M54S starting at 4200190. Clip at 4200237.
    start, end = 4200230, 4200250
    with BAMHandler(BAM_PATH) as bam:
        tensor = bam.get_alignment_tensor("1", start, end, max_reads=100)
        
        # Channel 12 is CH_CIGAR_SOFT_CLIP
        clip_channel = tensor[12]
        
        # Find all positions with signals
        rows, cols = np.where(clip_channel > 0)
        if len(cols) > 0:
            for c in np.unique(cols):
                count = (clip_channel[:, c] > 0).sum()
                print(f"Soft clip signals at pos {start + c} (idx {c}): {count}")
        
        if len(cols) > 0:
            print("SUCCESS: Soft clip signals detected.")
        else:
            print("FAILURE: No soft clip signals in known clipping region.")
    print()

def test_insert_size():
    print("--- Test 3: Insert Size Z-score ---")
    start, end = 4200000, 4200100
    with BAMHandler(BAM_PATH) as bam:
        tensor = bam.get_alignment_tensor("1", start, end, max_reads=100)
        isz_channel = tensor[9]
        non_zero = isz_channel[isz_channel != 0]
        if len(non_zero) > 0:
            print(f"Z-score stats: min={non_zero.min():.4f}, max={non_zero.max():.4f}, mean={non_zero.mean():.4f}")
            print("SUCCESS: Insert size Z-scores calculated.")
        else:
            print("FAILURE: No insert size signals.")
    print()

def test_nucleotides():
    print("--- Test 3: Nucleotide One-Hot Encoding ---")
    start, end = 4200000, 4200010
    with BAMHandler(BAM_PATH) as bam:
        tensor = bam.get_alignment_tensor("1", start, end, max_reads=10)
        
        # Channels 0-3 are A, C, G, T
        for r in range(tensor.shape[1]):
            if tensor[:, r, :].sum() == 0: continue
            print(f"Read {r}:")
            for w in range(tensor.shape[2]):
                nuc_vec = tensor[0:4, r, w]
                nuc = "ACGT"[np.argmax(nuc_vec)] if nuc_vec.sum() > 0 else "."
                print(f"  Pos {start+w}: {nuc} (vec: {nuc_vec})")
    print()

if __name__ == "__main__":
    test_deletion()
    test_soft_clipping()
    test_insert_size()
    test_nucleotides()
