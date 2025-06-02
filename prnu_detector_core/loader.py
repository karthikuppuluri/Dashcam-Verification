import cv2
import os

video_path = "C:/Users/karth/Documents/DashCam Proj/Dashcam-Verification/test_dashcam_clip.mp4"
output_dir = "C:/Users/karth/Documents/DashCam Proj/Dashcam-Verification/frames"

def extract_frames(video_path: str, output_dir: str, step: int = 5):
    """
    Extract frames from video every `step` frames.
    :param video_path: Path to video file
    :param output_dir: Directory to save extracted frames
    :param step: Frame interval
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()

extract_frames(video_path, output_dir)