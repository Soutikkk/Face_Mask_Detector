import cv2
import numpy as np


class MaskClassifier:
    """Run the MobileNetV2 mask classifier documented in the README."""

    def __init__(self, model_path):
        try:
            import onnxruntime
        except ImportError as error:
            raise RuntimeError(
                "Mask classification requires ONNX Runtime. "
                "Install it with python -m pip install -r requirements-mask.txt."
            ) from error

        model_data = model_path.read_bytes()
        try:
            self.session = onnxruntime.InferenceSession(
                model_data, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise RuntimeError(f"Could not load mask model {model_path}: {error}") from error

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape[1:] != [224, 224, 3]
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 1
            or outputs[0].shape[1:] != [2]
            or outputs[0].type != "tensor(float)"
        ):
            raise RuntimeError("Use the MobileNetV2 ONNX mask model linked in the README.")
        self.input_name = inputs[0].name

    def predict(self, face):
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(rgb_face, (224, 224)).astype(np.float32)
        batch = (resized_face / 127.5 - 1.0)[None, ...]
        try:
            probabilities = self.session.run(None, {self.input_name: batch})[0]
        except Exception as error:
            raise RuntimeError(f"Mask prediction failed: {error}") from error

        mask, no_mask = probabilities[0]
        if mask >= no_mask:
            return "Mask", float(mask)
        return "No mask", float(no_mask)
