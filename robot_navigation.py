import cv2
import numpy as np
import time
import argparse
import sys
from utils.disparity_func import disparity_function, precompute_ncc, precompute_zncc
from utils.draw_func import draw_planar_view
from utils.moravec_func import compute_moravec

class RobotNavigation:
    def __init__(self, metric='SAD', aggregation='mode', roi_size=80, window_size=9, vertical_stripes=4, moravec_threshold=2000):
        
        # camera parameters provided in the pdf
        self.focal = 567.2        # pixel
        self.baseline = 92.226     # mm
        
        # stereo algorithm parameters
        self.max_disp_value = 128
        self.disp_range = 64
        self.window_size = window_size
        self.half_window = window_size // 2
        
        # dynamic stereo parameters
        self.d_prev = 0.0
        self.dynamic_offset = 0
        
        self.vertical_stripes = vertical_stripes
        
        self.moravec_threshold = moravec_threshold
        
        # region
        self.roi_size = roi_size
        self.stripe_width = 0
        self.center_x = 0
        self.center_y = 0
        
        # real chessboard dimensions for verification
        self.real_board_w_target = 125.0 
        self.real_board_h_target = 178.0 
        
        # metric configuration
        self.metric = metric.upper()
        self.aggregation = aggregation.lower()
        self.maximize = False
        
        if self.metric in ['NCC', 'ZNCC']:
            self.maximize = True
        
        # kernel for image preprocessing
        self.kernel_sharpening = np.array([
            [0, -1,  0], 
            [-1, 5, -1], 
            [0, -1,  0]
        ], dtype=np.float32)

    def preprocess_img(self, img):
        return cv2.filter2D(img, -1, self.kernel_sharpening)
    
    def compute_disparity(self, img_left, img_right, roi_x, roi_y):
        # range goes from d_prev - 32 to d_prev + 32    ->  offset is where the dynamic range begins, so d_prev - 32
        half_disp_range = self.disp_range // 2                          # 32
        offset = max(0, int(self.d_prev - half_disp_range))             # max(0, d_prev - 32) ->  need to check if d_prev - 32 is below 0 
        offset = min(offset, self.max_disp_value - self.disp_range)     # min(offset, 64)   ->  need to check if the offset is above 64   ->  the range is 64, so offset + range would go over 128 if offset > 64
        self.dynamic_offset = offset                                    # 0 + offset is where the disparity range begins
        
        # extract left roi
        roi_l = img_left[roi_y : roi_y + self.roi_size, roi_x : roi_x + self.roi_size].astype(np.float32)
        
        # extract wider right strip
        start_x_r = max(0, roi_x - self.max_disp_value)
        end_x_r = roi_x + self.roi_size
        strip_r = img_right[roi_y : roi_y + self.roi_size, start_x_r : end_x_r].astype(np.float32)
        
        # initialize cost volume
        init_val = -np.inf if self.maximize else np.inf
        cost_volume = np.full((self.roi_size, self.roi_size, self.disp_range), init_val, dtype=np.float32)

        texture_mask = compute_moravec(roi_l, self.window_size, self.moravec_threshold)
        
        roi_l_zm, sum_sq_l = None, None
        
        # computing left frame calculations for zncc and ncc speedup
        if self.metric == 'NCC':
            sum_sq_l = precompute_ncc(roi_l, self.window_size)
        elif self.metric == 'ZNCC':
            roi_l_zm, sum_sq_l = precompute_zncc(roi_l, self.window_size)

        # loop over disparity range
        for d_local in range(0, self.disp_range):
            
            d_global = d_local + offset
            
            # calculate indices for right patch corresponding to disparity d
            local_x_start = (roi_x - d_global) - start_x_r
            local_x_end = local_x_start + self.roi_size
            
            # boundary check
            if local_x_start < 0 or local_x_end > strip_r.shape[1]:
                continue
            
            roi_r_shifted = strip_r[:, local_x_start : local_x_end]
            
            if roi_r_shifted.shape != roi_l.shape: 
                continue

            cost = disparity_function(self.metric, roi_l, roi_r_shifted, self.window_size, sum_sq_l, roi_l_zm)

            # cost_volume is, as initialized before, [roi_size, roi_size, disp_range], so for example [80, 80, 64] ->   we need to remove again the offset to "re-normalize" it between 0-63
            cost_volume[:, :, d_local] = cost

        # we take the best indices for every y,x along depth (axis=2)
        if self.maximize:
            best_indices = np.argmax(cost_volume, axis=2)
        else:
            best_indices = np.argmin(cost_volume, axis=2)
            
        # add the offset (best_indices are from 0 to 64)
        disparity_map = best_indices.astype(np.float32) + offset

        # to avoid border miscalculations
        border = self.window_size // 2
        
        # cropping the disparity map to remove bad borders
        if disparity_map.shape[0] > 2*border:
            true_disparity_map = disparity_map[border:-border, border:-border]
        else:
            true_disparity_map = disparity_map
        
        # cropping the texture mask exactly like the disparity map
        if texture_mask.shape[0] > 2*border:
            true_texture_mask = texture_mask[border:-border, border:-border]
        else:
            true_texture_mask = texture_mask
        
        # selecting only the valid pixels using the mask
        values_for_calc = true_disparity_map[true_texture_mask]

        d_main = 0.0
        
        # calculating the final value only on good pixels
        if values_for_calc.size > 0:
            if self.aggregation == 'mean':
                d_main = np.mean(values_for_calc)
            elif self.aggregation == 'median':
                d_main = np.median(values_for_calc)
            else: 
                # mode
                vals, counts = np.unique(values_for_calc.astype(int), return_counts=True)
                d_main = float(vals[np.argmax(counts)])
        else:
            # if the mask filtered everything out (e.g. valid inside borders but flat), keep d_prev
            d_main = self.d_prev if self.d_prev is not None else 0.0

        # updating d_prev for the next frame
        self.d_prev = d_main
        
        return d_main, true_disparity_map, texture_mask
    
    def calculate_distance(self, d_main):
        if d_main <= 0: return float('inf') 
        z_mm = (self.baseline * self.focal) / d_main
        return z_mm

    def process_chessboard(self, img, z_distance):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 6 cols x 8 rows of internal corners
        pattern_size = (6, 8) 
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if found:
            # calculate pixel dimensions w and h
            w_pix = np.linalg.norm(corners[0] - corners[pattern_size[0] - 1])
            h_pix = np.linalg.norm(corners[0] - corners[pattern_size[0] * (pattern_size[1] - 1)])

            # calculate real dimensions using estimated z
            w_mm = (z_distance * w_pix) / self.focal
            h_mm = (z_distance * h_pix) / self.focal

            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            
            # verification output
            print(f"Disparity range: [{self.dynamic_offset}, {self.disp_range + self.dynamic_offset}]\t\tDisparity main: {self.d_prev:.2f}\t\tCalculated dims.: w={w_mm:.1f}, h={h_mm:.1f}\t(target: {self.real_board_w_target}x{self.real_board_h_target})")
            
        return img

    def calculate_obstacle_angle(self, planar_points):
        # at least 2 points
        if planar_points.shape[0] < 2:
            return 0.0

        # first row and last row
        p_start = planar_points[0]
        p_end = planar_points[-1]

        # calculate deltas (x and z differences)
        delta_x = p_end[0] - p_start[0]
        delta_z = p_end[1] - p_start[1]

        # compute angle
        angle_rad = np.arctan2(delta_z, delta_x)

        return np.degrees(angle_rad)
    
    def compute_planar_view(self, d_stripes, roi_start_x):
        # d_stripes is not an np.array, it's a simple []
        d_mains = np.array(d_stripes)
        
        indices = np.arange(len(d_mains))

        # calculating the centers of the stripes (all in once, thanks to indices array)
        u_stripes_px = roi_start_x + (indices * self.stripe_width) + (self.stripe_width // 2)

        # filter unvalid d values
        valid_mask = d_mains > 0
        
        # keep only the good values
        disp_valid = d_mains[valid_mask]
        u_valid = u_stripes_px[valid_mask]

        # z = (focal * baseline) / disparity
        # calculating diparities for pixels at the center of every stripe (in mm)
        z_world = (self.focal * self.baseline) / disp_valid

        # calculation of the real x coordinate in the frame
        x_world = ((u_valid - self.center_x) * z_world) / self.focal

        planar_points = np.column_stack((x_world, z_world))
        
        # calculating the "tau"
        angle = self.calculate_obstacle_angle(planar_points)

        return planar_points, angle
    
    def process_stripes(self, disparity_map):
        _, w = disparity_map.shape
        
        self.stripe_width = w // self.vertical_stripes
        d_stripes = []
        map_stripes = np.zeros_like(disparity_map)
        
        for i in range(self.vertical_stripes):
            col_start = i * self.stripe_width
            col_end = (i + 1) * self.stripe_width if i < self.vertical_stripes - 1 else w
            
            stripe_data = disparity_map[:, col_start:col_end]
            
            val = 0.0
            if stripe_data.size > 0:
                # decided to use the classic mean to have smoother visualization of the stripes
                val = np.mean(stripe_data)
            
            map_stripes[:, col_start:col_end] = val
            d_stripes.append(val)
        
        return d_stripes, map_stripes

    def run(self, video_left_path, video_right_path):
        cap_l = cv2.VideoCapture(video_left_path)
        cap_r = cv2.VideoCapture(video_right_path)

        if not cap_l.isOpened() or not cap_r.isOpened():
            print("Error: Unable to open video files")
            return

        prev_time = 0 
        
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r: break
            
            # fps calculation
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            prev_time = curr_time

            gray_l = self.preprocess_img(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY))
            gray_r = self.preprocess_img(cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY))
            #gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            #gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            h, w = gray_l.shape
            self.center_x, self.center_y = w // 2, h // 2
            
            # shift roi down to center the obstacle
            roi_x = self.center_x - (self.roi_size // 2)
            roi_y = self.center_y - (self.roi_size // 2)

            # task 1 & 2 (+ improv 1): compute map and d_main
            d_main, disparity_map, texture_map = self.compute_disparity(gray_l, gray_r, roi_x, roi_y)
            
            # task 3: compute distance z
            distance_mm = self.calculate_distance(d_main)
            distance_m = distance_mm / 1000.0

            # task 4: alarm output
            color = (0, 255, 0) 
            alarm_text = "SAFE"
            if distance_m < 0.8:
                color = (0, 0, 255) 
                alarm_text = "ALARM! < 0.8m"

            # task 5: verification
            frame_l = self.process_chessboard(frame_l, distance_mm)            

            # disparity map
            d_map_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # stripes map + angle (improv 2)
            d_stripes, stripes_map = self.process_stripes(disparity_map)
            planar_points, angle = self.compute_planar_view(d_stripes, roi_x)
            planar_view = draw_planar_view(planar_points)
            angle_text = f"Angle: {angle:.1f} deg"
            stripes_map_norm = cv2.normalize(stripes_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # display data
        # frame l
            cv2.rectangle(frame_l, (roi_x, roi_y),  (roi_x+self.roi_size, roi_y+self.roi_size), color, 2)
            cv2.putText(frame_l, f"z: {distance_m:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_l, alarm_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_l, f"fps: {int(fps)}", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame_l, f"{self.metric}-{self.aggregation}", (w - 160, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Robot Navigation left', frame_l)
            
        # disparity map
            cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
            # for better visualization (bigger pixels)
            cv2.resizeWindow('Disparity Map', 200, 200)
            cv2.imshow('Disparity Map', d_map_norm) 

        # stripes map
            cv2.namedWindow('Disparity Map Stripes', cv2.WINDOW_NORMAL)
            # for better visualization (bigger pixels)
            stripes_map_norm = cv2.resize(stripes_map_norm, (stripes_map_norm.shape[1]*3, stripes_map_norm.shape[0 ]*3), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Disparity Map Stripes', stripes_map_norm)
            
        # texture map
            cv2.namedWindow('Texture Map', cv2.WINDOW_NORMAL)
            texture_map_coloured = np.full((texture_map.shape[1], texture_map.shape[0], 3), 255, dtype=np.uint8)
            texture_map_coloured[texture_map] = [0, 0, 255]
            cv2.resizeWindow('Texture Map', 200, 200)
            cv2.imshow('Texture Map', texture_map_coloured)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
        # planar view
            cv2.putText(planar_view, angle_text, (370, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('ROI Planar Map', planar_view)
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Robot Navigation Project")
    
    parser.add_argument("--metric", "-m", type=str, default="SSD",
                        choices=['SAD', 'SSD', 'NCC', 'ZNCC'],
                        help="Similarity metric")
    
    parser.add_argument("--aggregation", "-a", type=str, default="mean",
                        choices=['mean', 'median', 'mode'],
                        help="Disparity aggregation method (default: mode)")
    
    parser.add_argument("--roi-size", "-r", type=int, default=104,                  
                        help="ROI size in pixels")
    
    parser.add_argument("--window-size", "-w", type=int, default=9,                 
                        choices=[5, 7, 9, 11, 13],
                        help="Window size in pixels")
    
    parser.add_argument("--vertical-stripes", "-v", type=int, default=10,           
                        help="Number of vertical stripes")
    
    parser.add_argument("--moravec-threshold", "-t", type=int, default=15000,           
                        help="Number of vertical stripes")
    
    parser.add_argument("video_left", nargs='?', default="video/robotL.avi",
                        help="path to left video")
    parser.add_argument("video_right", nargs='?', default="video/robotR.avi",
                        help="Path to right video")

    args = parser.parse_args()
    
    
    if (args.roi_size - (args.window_size // 2)) % args.vertical_stripes != 0:
        print("Error:\t (roi_size - window_size // 2) divided by vertical_stripes must have a discard of 0.")
        print(f"Here:\t ({args.roi_size} - ({args.window_size} // 2)) % {args.vertical_stripes} == 0\t{args.roi_size - (args.window_size // 2)} % {args.vertical_stripes} = {(args.roi_size - (args.window_size // 2)) % args.vertical_stripes}")
        print("Example: (80 - (9 // 2)) % 4 == 0\t76 % 4 == 0")
        sys.exit()


    print(f"Starting...\n")
    print(f"-------------------------------------")
    print(f"Metric: {args.metric}")
    print(f"Aggregation method: {args.aggregation}")
    print(f"ROI size: {args.roi_size}x{args.roi_size}")
    print(f"Window size: {args.window_size}x{args.window_size}")
    print(f"-------------------------------------\n")
    
    
    bot = RobotNavigation(metric=args.metric, aggregation=args.aggregation, roi_size=args.roi_size, window_size=args.window_size, vertical_stripes=args.vertical_stripes, moravec_threshold=args.moravec_threshold)
    bot.run(args.video_left, args.video_right)