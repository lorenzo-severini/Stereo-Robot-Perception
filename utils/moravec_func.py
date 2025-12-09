import cv2
import numpy as np

def compute_moravec(roi, window_size=3, threshold=2000):
    # converting to float32 to avoid overflow when squaring numbers
    roi = roi.astype(np.float32)
    h, w = roi.shape
    
    # initializing with infinity because we want to find the minimum difference
    min_ssd = np.full((h, w), np.inf, dtype=np.float32)
    
    # we check all 8 directions around the pixel
    shifts = [
        (0, 1),     # right
        (0, -1),    # left
        (1, 0),     # down
        (-1, 0),    # up
        (-1, -1),   # up-left
        (-1, 1),    # up-right
        (1, 1),     # down-right
        (1, -1)     # down-left
    ]
    
    # adding a 1-pixel padding all around so we can shift in any direction
    padded_roi = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    
    # original image coordinates inside the padded version
    base_y_start, base_y_end = 1, h + 1
    base_x_start, base_x_end = 1, w + 1
    
    for dy, dx in shifts:
        # figuring out the slice coordinates for the shifted image
        shift_y_start = base_y_start + dy
        shift_y_end   = base_y_end   + dy
        shift_x_start = base_x_start + dx
        shift_x_end   = base_x_end   + dx
        
        # grabbing the shifted view
        shifted_view = padded_roi[shift_y_start:shift_y_end, shift_x_start:shift_x_end]
        
        # ssd using boxfilter
        # no normalization for the sum
        diff_sq = (roi - shifted_view) ** 2
        ssd = cv2.boxFilter(diff_sq, -1, (window_size, window_size), normalize=False)
        
        # keeping the minimum differences
        min_ssd = np.minimum(min_ssd, ssd)

    # mask to avoid pixels with ssd value below the threshold
    mask = min_ssd > threshold
    
    return mask