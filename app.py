import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


WINDOW_TITLE = "Face Mask Detector"


def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's face detector.")
    return face_detector


def mark_faces(frame, face_detector):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayscale, minSize=(40, 40))
    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
    return len(faces)


def run_image(image_path, output_path):
    if not cv2.haveImageWriter(output_path.suffix):
        raise RuntimeError("Choose a supported output extension such as .png or .jpg.")
    image_data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR) if image_data.size else None
    if frame is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    face_count = mark_faces(frame, load_face_detector())
    success, encoded_image = cv2.imencode(output_path.suffix, frame)
    if not success:
        raise RuntimeError(f"Could not encode image: {output_path}")
    with output_path.open("xb") as output_file:
        output_file.write(encoded_image.tobytes())
    print(f"Faces detected: {face_count}. Saved image to {output_path}")


def run_camera(camera_index):
    face_detector = load_face_detector()

    camera = cv2.VideoCapture(camera_index)
    preview_opened = False
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}.")

        while True:
            captured, frame = camera.read()
            if not captured:
                raise RuntimeError(f"Could not read a frame from camera {camera_index}.")

            mark_faces(frame, face_detector)
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
    parser = argparse.ArgumentParser(description="Detect faces in a webcam feed or image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Webcam index (default: 0).")
    source.add_argument("--image", type=Path, help="Path to an input image.")
    parser.add_argument("--output", type=Path, help="Path for the annotated image.")
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.image is not None and args.output is None:
        parser.error("--image requires --output")
    if args.output is not None and args.image is None:
        parser.error("--output requires --image")

    try:
        if args.image is not None:
            run_image(args.image, args.output)
        else:
            run_camera(args.camera or 0)
    except (OSError, RuntimeError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
