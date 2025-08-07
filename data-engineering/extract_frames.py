import cv2

video_path = "./data/video/splits/Clip_3_35s.mp4"
output_dir = "./data/video/frames"
fps = None

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Cannot open video")

# Get metadata
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = frame_count / fps

# Compute start frame for last 60 seconds
start_time_sec = max(0, duration_sec - 60)
start_frame = int(start_time_sec * fps)

# Set video to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Extract and save frames
i = 0
j = 2320
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_dir}/frame_{j:04d}.jpg", frame)
    i += 1
    j += 1

cap.release()