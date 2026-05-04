import cv2
import pyrealsense2 as rs
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# INTEGRATION GUIDE
# ---------------------------------------------------------------------------
# How to record RGB video from Intel RealSense depth camera? 
# ---------------------------------------------------------------------------


# --- [STEP 1] OUTPUT SETUP -------------------------------------------------
# create the output folder and build a timestamped filename 
# change OUTPUT_DIR to the folder path
OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"rgb_{timestamp}.avi")
# ---------------------------------------------------------------------------


# --- [STEP 2] CAMERA SETUP -------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
# ---------------------------------------------------------------------------


# --- [STEP 3] VIDEO WRITER SETUP -------------------------------------------
# initialise video writer after the pipeline starts
# if need .mp4, swap codec to MP4V and change the file extension above.
# The resolution tuple (640, 480) and FPS 30 must match the stream config
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
# ---------------------------------------------------------------------------

print(f"Recording to {output_path} — press 'q' to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        colorFrame = frames.get_color_frame()
        if not colorFrame:
            continue

        color = np.asanyarray(colorFrame.get_data())

        # --- [STEP 4] PER-FRAME WRITE --------------------------------------
        # call writer.write(color) once per iteration
        writer.write(color)
        # -------------------------------------------------------------------

        cv2.imshow("Recording (q to quit)", color)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # --- CLEANUP -----------------------------------------------------------
    writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved: {output_path}")
