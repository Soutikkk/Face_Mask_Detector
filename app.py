import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from mask_classifier import MaskClassifier


WINDOW_TITLE = "Face Mask Detector"


def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's face detector.")
    return face_detector


def mark_faces(frame, face_detector, classifier=None, *, min_mask_score=None):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayscale, minSize=(40, 40))
    predictions = [None] * len(faces)
    if classifier is not None:
        predictions = [
            classifier.predict(frame[y:y + height, x:x + width])
            for x, y, width, height in faces
        ]

    for (x, y, width, height), prediction in zip(faces, predictions):
        color = (0, 200, 0)
        if prediction is not None:
            label, confidence = prediction
            if min_mask_score is not None and confidence < min_mask_score:
                label = "Uncertain"
                color = (0, 165, 255)
            elif label == "No mask":
                color = (0, 0, 255)
            cv2.putText(
                frame, f"{label} {confidence:.0%}", (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    return len(faces)


def run_image(image_path, output_path, classifier=None, *, min_mask_score=None):
    if not cv2.haveImageWriter(output_path.suffix):
        raise RuntimeError("Choose a supported output extension such as .png or .jpg.")
    image_data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR) if image_data.size else None
    if frame is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    face_count = mark_faces(frame, load_face_detector(), classifier, min_mask_score=min_mask_score)
    success, encoded_image = cv2.imencode(output_path.suffix, frame)
    if not success:
        raise RuntimeError(f"Could not encode image: {output_path}")
    with output_path.open("xb") as output_file:
        output_file.write(encoded_image.tobytes())
    print(f"Faces detected: {face_count}. Saved image to {output_path}")


def run_camera(camera_index, classifier=None, *, min_mask_score=None):
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

            mark_faces(frame, face_detector, classifier, min_mask_score=min_mask_score)
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
    parser.add_argument("--mask-model", type=Path, help="Path to the ONNX mask classifier.")
    parser.add_argument(
        "--min-mask-score", type=float,
        help="Mark predictions below this score as Uncertain (0.5 to 1, requires --mask-model).",
    )
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.image is not None and args.output is None:
        parser.error("--image requires --output")
    if args.output is not None and args.image is None:
        parser.error("--output requires --image")
    if args.min_mask_score is not None:
        if not 0.5 <= args.min_mask_score <= 1:
            parser.error("--min-mask-score must be between 0.5 and 1")
        if args.mask_model is None:
            parser.error("--min-mask-score requires --mask-model")

    try:
        classifier = MaskClassifier(args.mask_model) if args.mask_model is not None else None
        if args.image is not None:
            run_image(args.image, args.output, classifier, min_mask_score=args.min_mask_score)
        else:
            run_camera(args.camera or 0, classifier, min_mask_score=args.min_mask_score)
    except (OSError, RuntimeError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
