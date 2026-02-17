"""Generate images from sequence read data"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass


@dataclass
class BaseColor:
    """Base color configuration for nucleotides"""
    A: Tuple[int, int, int] = (255, 0, 0)      # Red
    T: Tuple[int, int, int] = (0, 255, 0)       # Green
    C: Tuple[int, int, int] = (0, 0, 255)       # Blue
    G: Tuple[int, int, int] = (0, 0, 0)          # Black


class ImageGenerator:
    """Generates images from pileup and clipping data"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), coloring_mode: str = "standard"):
        """
        Initialize image generator
        
        Args:
            image_size: Size of output images (width, height)
            coloring_mode: 'standard' or 'kmer'
        """
        self.image_size = image_size
        self.base_colors = BaseColor()
        self.pixel_size = 5
        self.coloring_mode = coloring_mode
        self.kmer_colors = {}
        
        if self.coloring_mode == "kmer":
            import json
            import os
            # Attempt to load kmer colors
            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
                color_path = os.path.join(base_path, "kmer_colors.json")
                with open(color_path, 'r') as f:
                    self.kmer_colors = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load kmer colors: {e}")

    def generate_image(self, 
                      pileup_data: List[Tuple],
                      clipping_data: Dict[int, int],
                      region_start: int,
                      region_length: int) -> Image.Image:
        """
        Generate an image from pileup and clipping data
        
        Args:
            pileup_data: List of (pos, is_paired, is_proper_pair, mapq, cigar_type, base, kmer)
            clipping_data: Dictionary mapping position to clipping count
            region_start: Start position of the region
            region_length: Length of the region
            
        Returns:
            PIL Image object
        """
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)
        
        y_index = 0
        last_x = None
        
        for pileup_record in pileup_data:
            # Handle both 6-element (legacy) and 7-element (new) tuples temporarily or assume 7
            if len(pileup_record) == 7:
                pos, is_paired, is_proper_pair, mapq, cigar_type, base, kmer = pileup_record
            else:
                pos, is_paired, is_proper_pair, mapq, cigar_type, base = pileup_record
                kmer = None
            
            # Calculate x position
            x_start = (pos - region_start) * self.pixel_size + self.pixel_size
            
            # Reset y index if we've moved to a new x position
            if last_x != x_start:
                y_index = 0
                last_x = x_start
            
            # Calculate y position
            y_start = self.pixel_size + y_index * self.pixel_size
            
            # Calculate rectangle bounds
            x_end = x_start + self.pixel_size
            y_end = y_start + self.pixel_size
            
            # Get clipping value for this position
            clip_value = clipping_data.get(pos, 0)
            
            # Get RGB color based on base and read properties
            rgb = self._get_base_color(
                base, is_paired, is_proper_pair, mapq, cigar_type, clip_value, kmer
            )
            
            # Draw rectangle
            draw.rectangle((x_start, y_start, x_end, y_end), fill=rgb)
            
            y_index += 1
        
        return image
    
    def _get_base_color(self,
                       base: str,
                       is_paired: bool,
                       is_proper_pair: bool,
                       mapq: int,
                       cigar_type: int,
                       clip_value: int,
                       kmer: Optional[str] = None) -> Tuple[int, int, int]:
        """
        Calculate RGB color for a base based on its properties
        
        Args:
            base: Nucleotide base (A, T, C, G)
            is_paired: Whether read is paired
            is_proper_pair: Whether pair is proper
            mapq: Mapping quality
            cigar_type: CIGAR operation type
            clip_value: Clipping value
            kmer: 6-mer context (required for kmer coloring)
            
        Returns:
            RGB tuple
        """
        # K-mer coloring mode
        if self.coloring_mode == "kmer":
            if kmer and kmer in self.kmer_colors:
                return tuple(self.kmer_colors[kmer])
            # Fallback to black if kmer not found or None
            return (0, 0, 0)

        # Standard coloring mode
        # Get base color
        base_color = list(self.base_colors.__dict__.get(base, (0,0,0)))
        
        # Adjust color based on read properties
        # High quality: no adjustment
        # Low quality or clipped: add offset
        if not (is_paired and is_proper_pair and mapq >= 20 and cigar_type != 4):
            # Calculate offset based on properties
            offset = self._calculate_color_offset(
                is_paired, is_proper_pair, mapq, cigar_type
            )
            base_color = [min(255, c + offset) for c in base_color]
        
        # Add clipping value
        base_color = [min(255, max(0, c + clip_value)) for c in base_color]
        
        return tuple(base_color)
    
    def _calculate_color_offset(self,
                            is_paired: bool,
                            is_proper_pair: bool,
                            mapq: int,
                            cigar_type: int) -> int:
        """
        Calculate color offset based on the specific 4-bit index logic:
        Bit 3 (8): is_paired is False
        Bit 2 (4): is_proper_pair is False
        Bit 1 (2): mapping_quality < 20
        Bit 0 (1): map_type == 4
        """

        # 1. Calculate the Index (0 to 15)
        # We use boolean arithmetic where True=1 and False=0
        idx = (
            ((not is_paired)      * 8) +  # Bit 3
            ((not is_proper_pair) * 4) +  # Bit 2
            ((mapq < 20)          * 2) +  # Bit 1
            ((cigar_type == 4)    * 1)    # Bit 0
        )

        # 2. Apply the Formula
        # If index is 0 (all good), return 0. 
        # Otherwise, return 50 + (Index * 10).
        return 0 if idx == 0 else 50 + (idx * 10)
    
    def save_image(self, image: Image.Image, output_path: str):
        """Save image to file"""
        image.save(output_path, "PNG")

