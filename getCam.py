#https://docs.google.com/document/d/1e9i8UenG0LrFPWIwc598e_yiBIcuYSt5axyimfpwHvM/edit?usp=sharing
import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()
        if not depthFrame or not colorFrame:
            continue
        
        color = np.asanyarray(colorFrame.get_data())
        cv2.imshow("cam", color)
        
        # depth = np.asanyarray(depthFrame.get_data())
        
        # depthVis = cv2.convertScaleAbs(depth, alpha=0.03)
        # depthVis = cv2.applyColorMap(depthVis, cv2.COLORMAP_JET)
        # both = np.hstack((color, depthVis))
        # cv2.imshow("cam", both)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()