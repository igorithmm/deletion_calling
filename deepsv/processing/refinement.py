import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import Variant, DeletionSize
from deepsv.utils.kmeans import kmeans

logger = logging.getLogger(__name__)

class BoundaryRefiner:
    """Refines deletion boundaries using K-Means clustering on depth and clipping data"""
    
    def __init__(self, k: int = 3, max_iterations: int = 100000):
        self.k = k
        self.max_iterations = max_iterations

    def refine_boundaries(self, bam_handler: BAMHandler, variant: Variant) -> Variant:
        """
        Refine the start and end coordinates of a deletion variant
        
        Args:
            bam_handler: Open BAMHandler instance
            variant: The variant to refine
            
        Returns:
            New Variant with potentially adjusted coordinates
        """
        # Original logic uses a 200bp padding window for analysis
        # "int(vcf_del[i][1]-200), int(vcf_del[i][2]+200)"
        padding = 200
        analyze_start = variant.start - padding
        analyze_end = variant.end + padding
        chrom = variant.chrom
        
        # 1. Get raw depth
        # Note: count_coverage returns tuple of arrays (A, C, G, T)
        # We need total depth
        try:
            depth_data = bam_handler.get_coverage_depth(chrom, analyze_start, analyze_end)
        except Exception as e:
            logger.warning(f"Could not get coverage for refinement: {e}")
            return variant

        # 2. Get clipping info
        try:
            clip_dict = bam_handler.get_clipping_info(chrom, analyze_start, analyze_end)
        except Exception as e:
            logger.warning(f"Could not get clipping info for refinement: {e}")
            return variant

        # Prepare features for initial 1D K-means (Depth only)
        # Old code: seq_depth array [(pos, depth), ...]
        positions = np.arange(analyze_start, analyze_end) # count_coverage usually returns length = end - start
        
        # Ensure dimensions match
        # pysam count_coverage length might differ slightly if near chromosome end?
        # Assuming safe for now based on BAMHandler implementation
        if len(depth_data) != len(positions):
            # Fallback
            min_len = min(len(depth_data), len(positions))
            depth_data = depth_data[:min_len]
            positions = positions[:min_len]

        seq_depth_array = np.column_stack((positions, depth_data))
        
        # Prepare clipping array in matching format [(pos, clip_val*2)]
        # Old code multiplies clip value by 2 for plotting, but also uses it as feature
        # "ax.plot(seq_clip_array[:,0],seq_clip_array[:,1]*2,color='g')" -> visual only?
        # later: "df_merge_np_roll = np.array(df_merge_fill)" -> merges depth and clip
        
        clip_values = np.array([clip_dict.get(pos, 0) for pos in positions])
        
        # 3. First K-Means (on Depth only)
        # Old code: result = kmeans(seq_depth_array,3,100000)
        # Actually it clusters on (pos, depth), so position is a feature!
        # This seems odd for a "level" detection, but we strictly follow the logic.
        result_1d = kmeans(seq_depth_array, self.k, self.max_iterations)
        result_1d = np.nan_to_num(result_1d)
        
        # 4. Prepare for 3D K-Means (Smoothed Depth + Position + Clips)
        # Old code logic:
        # - Merge depth and clip data
        # - Rolling median on depth (window 61)
        # - Features: [Position, Smoothed_Depth*5, Clip_Count]
        # - 3D Kmeans
        
        df_depth = pd.DataFrame(seq_depth_array, columns=[0, 1]) # 0=Pos, 1=Depth
        df_clip = pd.DataFrame({'pos': positions, 'clip': clip_values})
        
        # Merge not strictly necessary if we aligned arrays, but robust
        # Since we generated both from 'positions', we can just stack
        
        # Rolling median
        # "pd.rolling_median(df_merge_np_roll[:,1],61,center=True)" -> modern pandas: .rolling(61, center=True).median()
        depth_series = pd.Series(depth_data)
        smoothed_depth = depth_series.rolling(61, center=True).median().fillna(0).values
        
        # Trim edges as per old code "df_merge_fill_re = df_merge_fill_re[30:-30]"
        # This removes the NaN/boundary effects of the rolling window
        trim = 30
        if len(positions) <= 2 * trim:
             return variant # Too short
             
        valid_indices = slice(trim, -trim)
        
        feat_pos = positions[valid_indices]
        feat_depth_smooth = smoothed_depth[valid_indices] * 5 # Multiplied by 5 as per old code
        feat_clip = clip_values[valid_indices] # Old code seems to use raw clip here for clustering
        
        # "smooth_kmeans = np.c_[df_merge_np_roll[30:-30,0],df_merge_fill_re]"
        # "smooth_kmeans = np.c_[smooth_kmeans, df_merge_np_roll[30:-30,2]]" -> (Pos, SmoothDepth, Clip)
        
        features_3d = np.column_stack((feat_pos, feat_depth_smooth, feat_clip))
        
        # 5. Run 3D K-Means
        result_3d = kmeans(features_3d, self.k, self.max_iterations)
        result_3d = np.nan_to_num(result_3d)
        
        # 6. Analyze Clusters to Logic
        # Group by cluster label (last column)
        unique_labels = np.unique(result_3d[:, -1])
        clusters = []
        for label in unique_labels:
            cluster_data = result_3d[result_3d[:, -1] == label, :-1]
            clusters.append(cluster_data)
            
        # Need exactly 3 clusters for logic to work implicitly
        if len(clusters) < 3:
            return variant
            
        # Calculate stats for logic
        # "class_one_3D_mean = np.mean(class_one_3D[:,1])//5"
        # Remember col 1 is SmoothDepth*5, so this gets back to raw-ish depth
        cluster_means = [(np.mean(c[:, 1]) // 5) for c in clusters]
        cluster_mins = [(np.min(c[:, 1]) // 5) for c in clusters]
        
        # "class_sort = np.sort(class_array)"
        sorted_means = sorted(cluster_means)
        
        # "if class_sort[0] < 4*class_sort[1]//5 and class_sort[0] < 4*class_sort[2]//5:"
        # Logic: Is the lowest depth cluster significantly lower ( < 80%) than the others?
        if sorted_means[0] < (4 * sorted_means[1] // 5) and sorted_means[0] < (4 * sorted_means[2] // 5):
            
            # Identify which cluster is the "deletion" (min depth)
            min_idx = np.argmin(cluster_means)
            max_idx = np.argmax(cluster_means)
            # Find the remaining ("middle") cluster index safely
            all_indices = {0, 1, 2}
            all_indices.discard(min_idx)
            all_indices.discard(max_idx)
            if not all_indices:
                # All clusters have same mean — can't distinguish deletion
                return variant
            mid_idx = all_indices.pop()
            
            deletion_cluster = clusters[min_idx]
            
            # "class_one_3D = class_one_3D[np.lexsort(class_one_3D[:,::-1].T)]" -> Sort by position
            # Our data is already sorted by position from construction
            
            del_cluster_start_pos = int(deletion_cluster[0, 0])
            del_cluster_end_pos = int(deletion_cluster[-1, 0])
             
            # "int(class_one_3D[0,0]) != smooth_kmeans[0,0]"
            # Logic: verify the deletion cluster doesn't start at the very beginning of the analyzed window
            # (which would imply the window didn't capture the start of the deletion)
            window_start_pos = features_3d[0, 0]
            window_end_pos = features_3d[-1, 0]
            
            if del_cluster_start_pos != window_start_pos and del_cluster_end_pos != window_end_pos:
                
                # Refine boundaries
                # "while del_left_pos: ... seq_depth_dict[del_left_pos] > class_array_min[class_mid_index] ..."
                # Walk INWARDS from the cluster edges (or outwards? Code says:
                # del_left_pos = ...; while del_left_pos: if depth > thresh break; del_left_pos -= 1)
                # It walks OUTWARDS (left goes left, right goes right) looking for a depth spike
                # Threshold is the min depth of the "middle" cluster
                
                threshold = cluster_mins[mid_idx]
                
                # Create dict for fast lookup of raw depth using original depth array
                # Note: old code used "seq_depth" which was the raw depth
                seq_depth_dict = dict(zip(seq_depth_array[:, 0], seq_depth_array[:, 1]))
                
                # Walk Left
                final_left = del_cluster_start_pos
                # Safety bounds
                min_valid = int(seq_depth_array[0, 0])
                while final_left > min_valid:
                    depth = seq_depth_dict.get(final_left, 0)
                    if depth > threshold:
                        break
                    final_left -= 1
                    
                # Walk Right
                final_right = del_cluster_end_pos
                max_valid = int(seq_depth_array[-1, 0])
                while final_right < max_valid:
                    depth = seq_depth_dict.get(final_right, 0)
                    if depth > threshold:
                        break
                    final_right += 1
                
                # Construct new variant
                # We need to maintain the original ID/info probably, but Variant class is simple data class
                new_start = int(final_left)
                new_end = int(final_right)
                
                logger.info(f"Refined deletion: {chrom}:{variant.start}-{variant.end} -> {chrom}:{new_start}-{new_end}")
                    
                return Variant(
                    chrom=chrom, 
                    start=new_start, 
                    end=new_end, 
                    sv_type=variant.sv_type
                )
        
        # Fallback if conditions not met
        return variant
