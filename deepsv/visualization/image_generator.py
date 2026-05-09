"""Generate images from sequence read data"""
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from dataclasses import dataclass


# Which RGB channels are "non-primary" for each base.
# Legacy only applies the clipping adjustment to these channels,
# leaving the primary (255) channel untouched.
#   A (Red):   primary = R(0), non-primary = G(1), B(2)
#   T (Green): primary = G(1), non-primary = R(0), B(2)
#   C (Blue):  primary = B(2), non-primary = R(0), G(1)
#   G (Black): ALL channels are non-primary (all start at 0)
_NON_PRIMARY_CHANNELS: Dict[str, Tuple[int, ...]] = {
    'A': (1, 2),
    'T': (0, 2),
    'C': (0, 1),
    'G': (0, 1, 2),
}


@dataclass
class BaseColor:
    """Base color configuration for nucleotides"""
    A: Tuple[int, int, int] = (255, 0, 0)      # Red
    T: Tuple[int, int, int] = (0, 255, 0)       # Green
    C: Tuple[int, int, int] = (0, 0, 255)       # Blue
    G: Tuple[int, int, int] = (0, 0, 0)          # Black


class ImageGenerator:
    """Generates images from pileup and clipping data"""

    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        """
        Initialize image generator

        Args:
            image_size: Size of output images (width, height)
        """
        self.image_size = image_size
        self.base_colors = BaseColor()
        self.pixel_size = 5

    def generate_image(self,
                      pileup_data: List[Tuple],
                      clipping_data: Dict[int, int],
                      region_start: int,
                      region_length: int) -> Image.Image:
        """
        Generate an image from pileup and clipping data

        Args:
            pileup_data: List of (pos, is_paired, is_proper_pair, mapq, cigar_type, base)
            clipping_data: Dictionary mapping position to clipping signal
                (from ``BAMHandler.get_clipping_info``; values are typically
                negative — the legacy pipeline negates them before applying).
            region_start: Start position of the region
            region_length: Length of the region

        Returns:
            PIL Image object
        """
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)

        y_index = 0
        last_x = None

        for pos, is_paired, is_proper_pair, mapq, cigar_type, base in pileup_data:
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

            # Get clipping value for this position.
            # Legacy passes ``-clip_dict_record[pos]`` to get_rgb, i.e. the
            # stored (negative) value is negated so the effective clip_value
            # applied to channels is positive.  We replicate that here.
            clip_value = -clipping_data.get(pos, 0)

            # Get RGB color based on base and read properties
            rgb = self._get_base_color(
                base, is_paired, is_proper_pair, mapq, cigar_type, clip_value
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
                       clip_value: int) -> Tuple[int, int, int]:
        """
        Calculate RGB color for a base based on its properties.

        Matches the legacy ``get_rgb()`` colour formation:
        1. Start from the base colour (A=Red, T=Green, C=Blue, G=Black).
        2. Compute a quality offset from the 4 read-property flags and add
           it to **all** channels (for A/T/C the primary channel is already
           255 so ``min(255, 255+offset) == 255`` — no visible change on
           the primary channel, matching legacy which only wrote the offset
           into non-primary channels explicitly).
        3. Apply the clipping value to **non-primary channels only**, as
           the legacy code did (e.g. for base A it touched G and B but
           *not* R).

        Args:
            base: Nucleotide base (A, T, C, G)
            is_paired: Whether read is paired
            is_proper_pair: Whether pair is proper
            mapq: Mapping quality
            cigar_type: CIGAR operation type
            clip_value: Clipping value (already negated by caller so that
                positive = lighten non-primary channels)

        Returns:
            RGB tuple
        """
        # Get base color
        base_color = list(self.base_colors.__dict__.get(base, (0, 0, 0)))

        # Adjust color based on read properties
        # High quality: no adjustment
        # Low quality or clipped: add offset
        if not (is_paired and is_proper_pair and mapq >= 20 and cigar_type != 4):
            offset = self._calculate_color_offset(
                is_paired, is_proper_pair, mapq, cigar_type
            )
            base_color = [min(255, c + offset) for c in base_color]

        # Apply clipping value to non-primary channels only (legacy behaviour).
        # For G (Black) all three channels are non-primary.
        non_primary = _NON_PRIMARY_CHANNELS.get(base, (0, 1, 2))
        for ch in non_primary:
            base_color[ch] = min(255, max(0, base_color[ch] + clip_value))

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
        idx = (
            ((not is_paired)      * 8) +  # Bit 3
            ((not is_proper_pair) * 4) +  # Bit 2
            ((mapq < 20)          * 2) +  # Bit 1
            ((cigar_type == 4)    * 1)    # Bit 0
        )

        # If index is 0 (all good), return 0.
        # Otherwise, return 50 + (Index * 10).
        return 0 if idx == 0 else 50 + (idx * 10)

    def save_image(self, image: Image.Image, output_path: str):
        """Save image to file"""
        image.save(output_path, "PNG")
