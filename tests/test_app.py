from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app import mark_faces, run_image
from mask_classifier import MaskClassifier


class PhotoExportTests(unittest.TestCase):
    def test_exports_photo_without_detected_faces(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "annotated.png"
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            success, png = cv2.imencode(".png", frame)
            self.assertTrue(success)
            source.write_bytes(png.tobytes())

            with patch("builtins.print") as output_message:
                run_image(source, output)

            saved = cv2.imdecode(np.frombuffer(output.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(saved, frame)
            output_message.assert_called_once_with(f"Faces detected: 0. Saved image to {output}")

    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "annotated.png"
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            success, png = cv2.imencode(".png", frame)
            self.assertTrue(success)
            source.write_bytes(png.tobytes())
            output.write_bytes(b"existing file")

            with self.assertRaises(FileExistsError):
                run_image(source, output)

            self.assertEqual(output.read_bytes(), b"existing file")

    def test_rejects_corrupt_input_without_creating_output(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "corrupt.png"
            output = Path(directory) / "annotated.png"
            source.write_bytes(b"invalid image")

            with self.assertRaisesRegex(RuntimeError, "Could not decode image"):
                run_image(source, output)

            self.assertFalse(output.exists())


class MaskAnnotationTests(unittest.TestCase):
    def test_draws_labels_and_colors_for_both_predictions(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        detector = Mock()
        detector.detectMultiScale.return_value = [(10, 40, 45, 50), (100, 40, 45, 50)]
        classifier = Mock()
        classifier.predict.side_effect = [("Mask", 0.94), ("No mask", 0.91)]

        with patch.object(cv2, "putText", wraps=cv2.putText) as draw_label:
            self.assertEqual(mark_faces(frame, detector, classifier), 2)

        self.assertEqual([call.args[1] for call in draw_label.call_args_list], ["Mask 94%", "No mask 91%"])
        self.assertEqual(tuple(frame[40, 10]), (0, 200, 0))
        self.assertEqual(tuple(frame[40, 100]), (0, 0, 255))

    def test_preprocesses_face_as_normalized_rgb_and_maps_model_scores(self):
        classifier = MaskClassifier.__new__(MaskClassifier)
        classifier.input_name = "face"
        classifier.session = Mock()
        classifier.session.run.return_value = [np.array([[0.25, 0.75]], dtype=np.float32)]
        face = np.full((30, 40, 3), (0, 128, 255), dtype=np.uint8)

        self.assertEqual(classifier.predict(face), ("No mask", 0.75))

        batch = classifier.session.run.call_args.args[1]["face"]
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(batch[0, 0, 0], [1, 128 / 127.5 - 1, -1], atol=1e-6)
