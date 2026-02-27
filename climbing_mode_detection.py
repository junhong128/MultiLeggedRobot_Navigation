import pyrealsense2 as rs
import numpy as np
import cv2

# Configuration parameters
MAX_DISTANCE = 1.5  # meters - objects within this distance will be analyzed
HEIGHT_THRESHOLD = 0.3  # meters - max height the robot can climb
GROUND_RANSAC_ITERATIONS = 100  # RANSAC iterations for ground plane fitting
GROUND_RANSAC_THRESHOLD = 0.02  # meters - inlier threshold for ground plane
CAMERA_HEIGHT = 0.15  # meters - approximate camera height from ground (used as fallback)


def depth_to_pointcloud(depth_meters, fx, fy, ppx, ppy):
    """Convert depth image to 3D point cloud in camera coordinates.

    Camera coordinate system:
    - X: right
    - Y: down
    - Z: forward (depth)
    """
    h, w = depth_meters.shape

    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Convert to 3D coordinates
    z = depth_meters
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy

    return x, y, z


def fit_ground_plane_ransac(points_3d, n_iterations=100, threshold=0.02):
    """Fit ground plane using RANSAC.

    Returns plane coefficients (a, b, c, d) where ax + by + cz + d = 0
    The plane normal (a, b, c) points upward (negative y in camera coords).
    """
    if len(points_3d) < 3:
        return None

    best_plane = None
    best_inliers = 0

    for _ in range(n_iterations):
        # Randomly sample 3 points
        idx = np.random.choice(len(points_3d), 3, replace=False)
        p1, p2, p3 = points_3d[idx]

        # Compute plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            continue
        normal = normal / norm_length

        # Ensure normal points "up" (negative y direction in camera coords)
        if normal[1] > 0:
            normal = -normal

        # Plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(np.dot(points_3d, normal) + d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)

    return best_plane


def height_above_plane(x, y, z, plane):
    """Calculate signed height of points above a plane.

    Positive = above plane, Negative = below plane.
    """
    a, b, c, d = plane
    # Signed distance from point to plane
    # Normal points up (negative Y in camera coords), so positive distance = above ground
    distances = (a * x + b * y + c * z + d) / np.sqrt(a*a + b*b + c*c)
    return distances


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

# Store recent ground planes for temporal smoothing
ground_plane_history = []
PLANE_HISTORY_SIZE = 5

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
        h, w = depth_meters.shape

        # Convert to 3D point cloud
        x_3d, y_3d, z_3d = depth_to_pointcloud(depth_meters, fx, fy, ppx, ppy)

        # Create mask for valid depth readings
        valid_mask = depth_meters > 0

        # --- Ground Plane Estimation ---
        # Use bottom portion of image (more likely to be ground)
        ground_region_mask = np.zeros_like(valid_mask)
        ground_region_mask[int(h * 0.6):, :] = True  # Bottom 40% of image
        ground_sample_mask = valid_mask & ground_region_mask

        # Extract points for ground plane fitting
        ground_points = np.column_stack([
            x_3d[ground_sample_mask],
            y_3d[ground_sample_mask],
            z_3d[ground_sample_mask]
        ])

        # Subsample for speed if too many points
        if len(ground_points) > 5000:
            idx = np.random.choice(len(ground_points), 5000, replace=False)
            ground_points = ground_points[idx]

        # Fit ground plane
        ground_plane = None
        if len(ground_points) >= 100:
            ground_plane = fit_ground_plane_ransac(
                ground_points,
                n_iterations=GROUND_RANSAC_ITERATIONS,
                threshold=GROUND_RANSAC_THRESHOLD
            )

        # Temporal smoothing of ground plane
        if ground_plane is not None:
            ground_plane_history.append(ground_plane)
            if len(ground_plane_history) > PLANE_HISTORY_SIZE:
                ground_plane_history.pop(0)

        # Use averaged plane for stability
        if ground_plane_history:
            avg_plane = np.mean(ground_plane_history, axis=0)
            # Re-normalize the normal vector (but NOT the d value)
            norm = np.sqrt(avg_plane[0]**2 + avg_plane[1]**2 + avg_plane[2]**2)
            ground_plane = (avg_plane[0]/norm, avg_plane[1]/norm, avg_plane[2]/norm, avg_plane[3])

        # --- Height Calculation ---
        if ground_plane is not None:
            # Calculate height above ground for all points
            heights = height_above_plane(x_3d, y_3d, z_3d, ground_plane)
        else:
            # Fallback: assume camera is at CAMERA_HEIGHT, looking horizontally
            # Height = camera_height - y_3d (since y is down in camera coords)
            heights = CAMERA_HEIGHT - y_3d

        # --- Obstacle Detection ---
        # Mask for close objects (within MAX_DISTANCE)
        close_objects_mask = valid_mask & (z_3d < MAX_DISTANCE)

        # Filter out ground points (heights close to 0)
        # Only consider points above 2cm as potential obstacles
        obstacle_mask = close_objects_mask & (heights > 0.02)

        # Get obstacle heights
        obstacle_heights = heights[obstacle_mask]

        # Determine max obstacle height
        max_obstacle_height = np.max(obstacle_heights) if obstacle_heights.size > 0 else 0

        # Climbing mode logic
        climbing_mode = (obstacle_heights.size == 0) or (max_obstacle_height <= HEIGHT_THRESHOLD)

        # --- Visualization ---
        display_image = color_image.copy()

        # Create height colormap for visualization
        height_viz = np.zeros_like(color_image)
        if ground_plane is not None:
            # Normalize heights for visualization (0 to HEIGHT_THRESHOLD*2)
            height_normalized = np.clip(heights / (HEIGHT_THRESHOLD * 2), 0, 1)
            height_colored = cv2.applyColorMap(
                (height_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            height_viz[valid_mask] = height_colored[valid_mask]

        # Highlight obstacles above threshold in red
        too_high_mask = obstacle_mask & (heights > HEIGHT_THRESHOLD)
        display_image[too_high_mask] = [0, 0, 255]  # Red for too high

        # Highlight climbable obstacles in green
        climbable_mask = obstacle_mask & (heights <= HEIGHT_THRESHOLD) & (heights > 0.02)
        display_image[climbable_mask] = [0, 255, 0]  # Green for climbable

        # Draw status information
        status_color = (0, 255, 0) if climbing_mode else (0, 0, 255)
        mode_text = "CLIMBING MODE: ON" if climbing_mode else "CLIMBING MODE: OFF"

        cv2.putText(display_image, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        cv2.putText(display_image, f"Max Obstacle Height: {max_obstacle_height:.3f} m", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display_image, f"Height Threshold: {HEIGHT_THRESHOLD:.2f} m", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        plane_status = "Ground plane: DETECTED" if ground_plane else "Ground plane: FALLBACK"
        plane_color = (0, 255, 0) if ground_plane else (0, 165, 255)
        cv2.putText(display_image, plane_status, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, plane_color, 1, cv2.LINE_AA)

        # Show views
        cv2.imshow("Climbing Mode Detection", display_image)
        cv2.imshow("Height Map", height_viz)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
