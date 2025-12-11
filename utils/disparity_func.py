import numpy as np
import cv2

# Sum of Absolute Differences (SAD)
def _calculate_sad(roi_l, roi_r_shifted, window_size):
        diff = np.abs(roi_l - roi_r_shifted)
        cost = cv2.boxFilter(diff, -1, (window_size, window_size), normalize=False)
        return cost

# Sum of Squared Differences (SSD)
def _calculate_ssd(roi_l, roi_r_shifted, window_size):
        diff = (roi_l - roi_r_shifted)**2
        cost = cv2.boxFilter(diff, -1, (window_size, window_size), normalize=False)
        return cost

# precomputation of left image squared sum (it's a one-time operation, useless to do it for every disparity in the range)
def precompute_ncc(roi_l, window_size):
        sum_sq_l = cv2.boxFilter(roi_l**2, -1, (window_size, window_size), normalize=False)
        return sum_sq_l
        
# same as ncc + zero mean
def precompute_zncc(roi_l, window_size):
        mean_l = cv2.boxFilter(roi_l, -1, (window_size, window_size), normalize=True)
        roi_l_zm = roi_l - mean_l
        sum_sq_l = cv2.boxFilter(roi_l_zm**2, -1, (window_size, window_size), normalize=False)
        return roi_l_zm, sum_sq_l

# Normalized Cross-Correlation (NCC)
def _calculate_ncc(roi_l, sum_sq_l, roi_r_shifted, window_size):
        num = cv2.boxFilter(roi_l * roi_r_shifted, -1, (window_size, window_size), normalize=False)
        sum_sq_r = cv2.boxFilter(roi_r_shifted**2, -1, (window_size, window_size), normalize=False)
        den = np.sqrt(sum_sq_l * sum_sq_r)
        cost = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        return cost

# Zero-Mean Normalized Cross-Correlation (ZNCC)
def _calculate_zncc(roi_l_zm, sum_sq_l, roi_r_shifted, window_size):
        mean_r = cv2.boxFilter(roi_r_shifted, -1, (window_size, window_size), normalize=True)
        roi_r_zm = roi_r_shifted - mean_r
         
        num = cv2.boxFilter(roi_l_zm * roi_r_zm, -1, (window_size, window_size), normalize=False)
        sum_sq_r = cv2.boxFilter(roi_r_zm**2, -1, (window_size, window_size), normalize=False)
        den = np.sqrt(sum_sq_l * sum_sq_r)
        cost = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        return cost

# differeneciate between different (dis)similarity functions
def disparity_function(metric, roi_l, roi_r_shifted, window_size, sum_sq_l=None, roi_l_zm=None):
        if metric == 'SAD':
                return _calculate_sad(roi_l, roi_r_shifted, window_size)
        elif metric == 'SSD':
                return _calculate_ssd(roi_l, roi_r_shifted, window_size)
        elif metric == 'NCC':
                return _calculate_ncc(roi_l_zm, sum_sq_l, roi_r_shifted, window_size)
        elif metric == 'ZNCC':
                return _calculate_zncc(roi_l_zm, sum_sq_l, roi_r_shifted, window_size)