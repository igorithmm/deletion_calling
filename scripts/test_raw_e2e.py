#!/usr/bin/env python3
"""
DeepSV2 Raw End-to-End Test (Multi-sample version).
This script verifies the pipeline starting from RAW files for multiple regions:
1. Extraction of read signals from BAM (BAMHandler).
2. Generation of pileup images (ImageGenerator) and saving them.
3. Extraction of sequence from FASTA (pyfaidx).
4. Computation of sequence embeddings (HyenaDNA).
5. Comparison of RGB-only vs Fused (FiLM) model predictions.

Usage:
    python3 scripts/test_raw_e2e.py
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import VCFHandler
from deepsv.models import FusedDeepSV
from deepsv.visualization.image_generator import ImageGenerator


def run_test():
    print("=" * 70)
    print("DeepSV2: RAW End-to-End Pipeline Check (Multi-sample)")
    print("=" * 70)

    # 1. Config
    bam_path = "raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
    vcf_path = "raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"
    fasta_path = "raw/hs37d5.fa"
    model_id = "LongSafari/hyenadna-small-32k-seqlen-hf"
    output_img_dir = Path("scripts/test_images")
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # 2. Load Models
    print(f"[*] Loading HyenaDNA model ({model_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    hyena = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    hyena.eval()

    print(f"[*] Initializing FusedDeepSV model ...")
    model = FusedDeepSV(embed_dim=256, num_classes=2).to(device)
    model.eval()

    # 3. Load Variants
    print(f"[*] Loading variants from {vcf_path} (Sample: NA12878) ...")
    vcf = VCFHandler(vcf_path)
    # Take first 10 deletions
    all_dels = vcf.load_variants(variant_type="deletion", sample_id="NA12878")
    deletions = all_dels[:10]
    # Get 10 non-deletions (anchors)
    non_deletions = vcf.get_non_deletion_regions(deletions, anchor_type="up")
    
    targets = [("DEL", v) for v in deletions] + [("NON_DEL", v) for v in non_deletions]
    
    # 4. Pipeline Components
    ref = Fasta(fasta_path)
    gen = ImageGenerator()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n" + "-" * 70)
    print(f"{'Type':<8} | {'Region':<25} | {'RGB-only':<10} | {'Fused':<10}")
    print("-" * 70)

    with BAMHandler(bam_path) as bam:
        for label_str, var in targets:
            chrom = var.chrom
            # For testing, we just take the first 50bp window of the variant
            start = var.start
            window_bp = 50
            end = start + window_bp
            
            # Step A: Image
            try:
                pileup = bam.get_pileup_data(chrom, start, end)
                clipping = bam.get_clipping_info(chrom, start, end)
                if not pileup:
                    print(f"{label_str:<8} | {chrom}:{start:<18} | SKIPPED (no reads)")
                    continue
                
                img = gen.generate_image(pileup, clipping, start, window_bp)
                img_path = output_img_dir / f"{label_str}_{chrom}_{start}.png"
                gen.save_image(img, str(img_path))
                
                img_tensor = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"{label_str:<8} | {chrom}:{start:<18} | ERROR (Image): {e}")
                continue

            # Step B: Embedding
            try:
                flank = 100
                full_start = max(1, start - flank)
                full_end = end + flank
                seq = str(ref[chrom][full_start - 1 : full_end])
                
                inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    outputs = hyena(**inputs)
                    hidden = outputs.last_hidden_state
                    embedding = hidden[:, flank : flank + window_bp, :].mean(dim=1)
                
                emb_sample = embedding[0, :20].cpu().numpy()
            except Exception as e:
                print(f"{label_str:<8} | {chrom}:{start:<18} | ERROR (Emb): {e}")
                continue

            # Step C: Inference
            with torch.no_grad():
                # RGB-only (backbone only)
                logits_rgb = model.cnn(img_tensor)
                prob_rgb = torch.softmax(logits_rgb, dim=1)[0, 1].item()
                
                # Fused (with FiLM)
                logits_fused = model(img_tensor, embedding)
                prob_fused = torch.softmax(logits_fused, dim=1)[0, 1].item()

            # Output results
            region_str = f"{chrom}:{start}-{end}"
            print(f"{label_str:<8} | {region_str:<25} | {prob_rgb:.4f}     | {prob_fused:.4f}")
            print(f"         └─ Embedding (first 20): {emb_sample}")
            print(f"         └─ Image saved to: {img_path.name}")

    print("-" * 70)
    print(f"\n[SUCCESS] Processed {len(targets)} regions. Images in {output_img_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
