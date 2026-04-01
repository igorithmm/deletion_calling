#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Change to repo root
REPO_ROOT = Path(__file__).parent.parent
os.chdir(REPO_ROOT)

MODEL_ID = "zhihan1996/DNABERT-2-117M"
OUTPUT_DIR = REPO_ROOT / "models" / "dnabert2"

def main():
    print(f"=== Downloading {MODEL_ID} to {OUTPUT_DIR} ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading all model files (including custom code)...")
    # snapshot_download ensures all files like bert_layers.py are copied
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False
    )
    
    print(f"✅ Success! All files saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
