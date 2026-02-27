import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Normalize

# -----------------------------
# Load model checkpoint (.pt)
# -----------------------------
CKPT_PATH = "turn_resnet18.pt"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
idx_to_class = ckpt["idx_to_class"]  # e.g. {0:"curved", 1:"90-degree"}

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(ckpt["model"])
model.eval()

# Normalization used by ResNet
normalize = Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))

# -----------------------------
# RealSense setup
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Warm-up
for _ in range(10):
    pipeline.wait_for_frames(15000)

print("Live inference running. Press Q or ESC to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames(15000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame_bgr = np.asanyarray(color_frame.get_data())

        # -----------------------------
        # Preprocess (cv2 -> tensor)
        # -----------------------------
        # Resize to 224x224 for ResNet
        small = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        # BGR -> RGB, uint8 -> float32 [0,1]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # HWC -> CHW tensor
        x = torch.from_numpy(rgb).permute(2, 0, 1)  # (3,224,224)
        x = normalize(x).unsqueeze(0)               # (1,3,224,224)

        # -----------------------------
        # Inference
        # -----------------------------
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = int(torch.argmax(probs).item())
            conf = float(probs[pred_id].item())

        pred_name = idx_to_class[pred_id]

        # -----------------------------
        # Overlay result on live frame
        # -----------------------------
        text = f"{pred_name} ({conf*100:.1f}%)"
        cv2.putText(frame_bgr, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Turn Classifier (Live)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()