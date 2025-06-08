from prnu_detector_core.detector import detect_forgery

if __name__ == "__main__":
    video_path = "C:/Users/karth/Documents/DashCam Proj/Dashcam-Verification/test_dashcam_clip.mp4"  # Replace with your actual path
    results = detect_forgery(video_path)

    for frame, score in results.items():
        status = "TAMPERED" if score < 0.01 else "AUTHENTIC"
        print(f"{frame}: {score:.4f} -> {status}")