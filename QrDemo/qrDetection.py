import cv2
import time
import pyrealsense2 as rs
import numpy as np

# -----------------------------
# FUNCTIONS TO RUN
# -----------------------------
def climb():
    print("CLIMBING MODE ON")



def detectTunnel():
    print("TUNNEL DETECTION ON")



def detectTurn():
    print("TURN TYPE DETECTION ON")



# -----------------------------
# QR CODE -> FUNCTION MAPPING
# -----------------------------
QR_ACTIONS = {
    "climb": climb,
    "tunnel": detectTunnel,
    "turn": detectTurn,
}



# -----------------------------
# CAMERA + QR DETECTOR
# -----------------------------
COOLDOWN_SECONDS = 2.0
last_trigger_time = {}

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    pipeline.start(config)
except Exception as e:
    raise RuntimeError(f"Could not start RealSense camera: {e}")

detector = cv2.QRCodeDetector()

print("Camera started. Show one of the 3 QR codes to the camera.")
print("Press 'q' to quit.")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Failed to read frame.")
        continue

    frame = np.asanyarray(color_frame.get_data())

    success, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

    if success and points is not None:
        for i, qr_data in enumerate(decoded_info):
            if not qr_data:
                continue

            qr_data = qr_data.strip()

            #find qr
            pts = points[i].astype(int)
            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            #decode qr
            x, y = pts[0]
            cv2.putText(
                frame,
                qr_data,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            #trigger function
            if qr_data in QR_ACTIONS:
                now = time.time()
                last_time = last_trigger_time.get(qr_data, 0)

                if now - last_time >= COOLDOWN_SECONDS:
                    print(f"Detected: {qr_data}")
                    QR_ACTIONS[qr_data]()
                    last_trigger_time[qr_data] = now

    cv2.imshow("QR Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()