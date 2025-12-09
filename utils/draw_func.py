import cv2
import numpy as np

def draw_planar_view(planar_points, map_width=600, map_height=600, max_distance_m=3.0, block_h=15):
    
    # offset to do not overlap with y axis
    start_offset_x = 60
    
    # white map wallpaper
    planar_map = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255
    
    if len(planar_points) == 0:
        _draw_camera(planar_map, map_width, map_height, start_offset_x)
        return planar_map
    
    planar_points /= 1000.0  # conversion of x and z in meters

    scale = (map_height - 20) / max_distance_m
    
    # draw y axis  (meter measuring)
    axis_x = 40 
    cv2.line(planar_map, (axis_x, 20), (axis_x, map_height - 20), (0, 0, 0), 2)
    
    for d in np.arange(0, max_distance_m + 0.5, 0.5):
        y_pos = int(map_height - (d * scale))
        if y_pos <= map_height - 20:
            cv2.line(planar_map, (axis_x - 5, y_pos), (axis_x + 5, y_pos), (0, 0, 0), 2)
            cv2.putText(planar_map, f"{d:.1f}m", (axis_x + 10, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(planar_map, (axis_x - 5, map_height - 20), (axis_x + 5, map_height - 20), (0, 0, 0), 2)
    cv2.putText(planar_map, f"{0:.1f}m", (axis_x + 10, map_height - 20 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # draw obstacles
    num_stripes = len(planar_points)
    
    
    blocks_total_width = map_width - 80
    block_width = blocks_total_width / num_stripes

    z_meters = planar_points[:, 1]
    
    # calculating y (planar map)
    pixel_y = map_height - (z_meters * scale)
    pixel_y = pixel_y.astype(int)
    
    half_h = block_h // 2

    for i in range(num_stripes):
        x_start = int(start_offset_x + (i * block_width))
        x_end   = int(start_offset_x + ((i + 1) * block_width))
        
        y_center = pixel_y[i]
        y_top = y_center - half_h
        y_bottom = y_center + half_h
        
        # Clip Visibilità
        draw_y_top = max(0, y_top)
        draw_y_bottom = min(map_height, y_bottom)
        
        if draw_y_bottom > draw_y_top:
                cv2.rectangle(planar_map, (x_start, draw_y_top), (x_end, draw_y_bottom), (40, 40, 40), -1)

    # 4. Camera
    _draw_camera(planar_map, map_width, map_height, start_offset_x)
    
    return planar_map

def _draw_camera(image, w, h, offset=0):
    camera_width, camera_height = 40, 20
    
    # camera's starting positions (for drawing)
    top_left = ( (w // 2) - (camera_width // 2) + offset//2, h - camera_height - 10 )
    bottom_right = ( (w // 2) + (camera_width // 2) + offset//2, h - 10 )
    
    # camera's red body
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)
    
    # camera's black rectangles
    cv2.rectangle(image, (top_left[0], top_left[1]-5), (top_left[0]+10, top_left[1]), (0,0,0), -1)
    cv2.rectangle(image, (bottom_right[0]-10, top_left[1]-5), (bottom_right[0], top_left[1]), (0,0,0), -1)
    