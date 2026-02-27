import pyrealsense2 as rs
import numpy as np
import cv2

# Configuration parameters
MAX_DISTANCE = 1.5  # meters - objects within this distance will be analyzed
HEIGHT_THRESHOLD = 0.3  # meters - max height the robot can climb

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Align depth frames to color frames
align = rs.align(rs.stream.color)

# Get camera intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
ppx, ppy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy

# Get depth scale (converts depth units to meters)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    while True:
        # Get aligned frames
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth to meters
        depth_meters = depth_image.astype(np.float32) * depth_scale

        # Create mask for objects within MAX_DISTANCE
        # Filter out invalid depth readings (0) and distances beyond threshold
        close_objects_mask = (depth_meters > 0) & (depth_meters < MAX_DISTANCE)

        # Calculate height for each pixel
        # Height calculation: for a pixel at (x, y) with depth Z,
        # the real-world Y coordinate is: (y - ppy) * Z / fy
        # We assume camera is horizontal, so vertical pixel offset = height
        h, w = depth_image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Calculate real-world heights (vertical position relative to camera center)
        # Negative means above center, positive means below
        heights = np.abs((y_coords - ppy) * depth_meters / fy)

        # Find heights of objects within the distance threshold
        close_heights = heights[close_objects_mask]

        # Determine max height of close objects
        max_height = np.max(close_heights) if close_heights.size > 0 else 0

        # Activate climbing mode if obstacles are climbable (at or below threshold)
        # Turn OFF climbing mode if any obstacle exceeds the max climbable height
        # If no close objects detected, also activate climbing mode
        climbing_mode = (close_heights.size == 0) or (max_height <= HEIGHT_THRESHOLD)

        # Visualization
        display_image = color_image.copy()

        # Overlay mask showing close objects
        overlay = display_image.copy()
        overlay[close_objects_mask] = [0, 255, 255]  # Yellow tint for close objects
        display_image = cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0)

        # Draw status information
        status_color = (0, 255, 0) if climbing_mode else (255, 255, 255)
        mode_text = "CLIMBING MODE: ON" if climbing_mode else "CLIMBING MODE: OFF"

        cv2.putText(display_image, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        cv2.putText(display_image, f"Max Height: {max_height:.3f} m", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display_image, f"Distance Threshold: {MAX_DISTANCE:.1f} m", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(display_image, f"Height Threshold: {HEIGHT_THRESHOLD:.1f} m", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Create depth colormap for visualization (optional second window)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Show combined view
        cv2.imshow("Climbing Mode Detection", display_image)
        cv2.imshow("Depth View", depth_colormap)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
