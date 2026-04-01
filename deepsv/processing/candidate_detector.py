"""Detect deletion candidates using depth and clipping analysis"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter


class CandidateDetector:
    """Detects deletion candidates from depth and clipping data"""
    
    def __init__(self, n_clusters: int = 3, window_size: int = 61):
        """
        Initialize candidate detector
        
        Args:
            n_clusters: Number of clusters for K-means
            window_size: Window size for median filtering
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
    
    def detect_deletion(self,
                       depth_data: np.ndarray,
                       clipping_data: np.ndarray,
                       positions: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect deletion boundaries from depth and clipping data
        
        Args:
            depth_data: Array of depth values
            clipping_data: Array of clipping values
            positions: Array of genomic positions
            
        Returns:
            Tuple of (left_pos, right_pos) or None if no deletion detected
        """
        if len(depth_data) < self.window_size:
            return None
        
        # Merge depth and clipping data
        df = pd.DataFrame({
            'pos': positions,
            'depth': depth_data,
            'clip': clipping_data
        })
        
        # Apply median filter to smooth depth
        smoothed_depth = median_filter(depth_data, size=self.window_size)
        smoothed_depth = smoothed_depth[self.window_size//2:-self.window_size//2]
        smoothed_positions = positions[self.window_size//2:-self.window_size//2]
        smoothed_clip = clipping_data[self.window_size//2:-self.window_size//2]
        
        # Prepare features for clustering (position, depth, clipping)
        features = np.column_stack([
            smoothed_positions,
            smoothed_depth * 5,  # Scale depth
            smoothed_clip
        ])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Find cluster with lowest mean depth (likely deletion)
        cluster_means = [
            np.mean(features[labels == i][:, 1]) for i in range(self.n_clusters)
        ]
        cluster_mins = [
            np.min(features[labels == i][:, 1]) for i in range(self.n_clusters)
        ]
        
        min_cluster_idx = np.argmin(cluster_means)
        min_mean = cluster_means[min_cluster_idx]
        
        # Check if minimum cluster is significantly lower
        sorted_means = sorted(cluster_means)
        if min_mean < 4 * sorted_means[1] // 5 and min_mean < 4 * sorted_means[2] // 5:
            # Extract deletion region
            deletion_points = features[labels == min_cluster_idx]
            
            if len(deletion_points) == 0:
                return None
            
            # Get boundaries (don't assume sorted order from K-means)
            del_left = int(deletion_points[:, 0].min())
            del_right = int(deletion_points[:, 0].max())
            
            # Refine boundaries using original depth data
            depth_dict = dict(zip(positions, depth_data))
            mid_cluster_idx = sorted(range(self.n_clusters), 
                                   key=lambda i: cluster_means[i])[1]
            threshold = cluster_mins[mid_cluster_idx] // 5
            
            # Extend left boundary
            refined_left = del_left
            while refined_left > positions[0]:
                if depth_dict.get(refined_left, 0) > threshold:
                    break
                refined_left -= 1
            
            # Extend right boundary
            refined_right = del_right
            while refined_right < positions[-1]:
                if depth_dict.get(refined_right, 0) > threshold:
                    break
                refined_right += 1
            
            return (refined_left, refined_right)
        
        return None

