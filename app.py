import argparse
from pathlib import Path
import sys

import cv2


WINDOW_TITLE = "Face Mask Detector"


def run_camera(camera_index):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's face detector.")

    camera = cv2.VideoCapture(camera_index)
    preview_opened = False
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}.")

        while True:
            captured, frame = camera.read()
            if not captured:
                raise RuntimeError(f"Could not read a frame from camera {camera_index}.")

            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(grayscale, minSize=(40, 40))
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)

            cv2.imshow(WINDOW_TITLE, frame)
            preview_opened = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        if preview_opened:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect faces in a live webcam feed.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be zero or greater")

    try:
        run_camera(args.camera)
    except (RuntimeError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
