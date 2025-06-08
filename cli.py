import argparse
from prnu_detector_core.detector import detect_forgery

def main():
    parser = argparse.ArgumentParser(description="Detect tampered frames in a video using PRNU")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--step", type=int, default=5, help="Frame sampling step")
    parser.add_argument("--threshold", type=float, default=0.01, help="Tampering threshold correlation")
    args = parser.parse_args()

    results = detect_forgery(args.video, args.step, args.threshold)

    print("\nForgery Detection Results:")
    for frame, score in results.items():
        status = "TAMPERED" if score < args.threshold else "AUTHENTIC"
        print(f"{frame}: {score:.4f} -> {status}")

if __name__ == "__main__":
    main()