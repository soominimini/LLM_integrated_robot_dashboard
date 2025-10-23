import argparse
import os
import time
import cv2


def open_capture(device: str) -> cv2.VideoCapture:
    """Open a V4L2 camera; fall back to default backend if needed."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(device)
    if cap.isOpened():
        print(f"[INFO] Opened {device}")
    return cap


def run_gui(cap: cv2.VideoCapture, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    window = "Camera (press c=save, q=quit)"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] Failed to read frame")
            break
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(save_dir, f"capture-{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[OK] Saved {path}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Simple V4L2 camera capture")
    p.add_argument("--device", default="/dev/video4", help="/dev/videoX (e.g., /dev/video4)")
    p.add_argument("--save-dir", default="captures")
    args = p.parse_args()

    cap = open_capture(args.device)
    if not cap.isOpened():
        print(f"[ERR] Could not open camera {args.device}. Try a different node (e.g., /dev/video2), or add user to 'video' group: sudo usermod -a -G video $USER; re-login.")
        return

    try:
        run_gui(cap, args.save_dir)
    finally:
        cap.release()


if __name__ == "__main__":
    main()


