# Face Mask Detector

A Python project for detecting faces in webcam feeds and photos and classifying whether they are wearing face masks.

The app draws a box around each detected face and can optionally label it as wearing a mask or not wearing one.

## Run locally

Create a virtual environment with `python -m venv .venv` and activate it. In PowerShell, run `.\.venv\Scripts\Activate.ps1`.

```powershell
python -m pip install -r requirements.txt
python app.py
```

Use `python app.py --camera 1` to select a different camera. Press `Q` or `Esc`, or close the preview window, to quit.

The app processes frames locally and displays them in the preview window.

## Annotate a photo

```powershell
python app.py --image "photo.jpg" --output "annotated.png"
```

The app prints the number of detected faces and saves a new image with their bounding boxes. Choose a new output file in an existing folder.

## Classify face masks

Install the optional dependency and download [mask_detector.onnx](https://huggingface.co/Dao-Dung/face-mask-detector-onnx/blob/7f17625a18e809737e4ac44d39fc949994c4c1d1/mask_detector.onnx) to the project folder. The [MobileNetV2 model card](https://huggingface.co/Dao-Dung/face-mask-detector-onnx) credits the original Face-Mask-Detection project and includes its MIT license.

```powershell
python -m pip install -r requirements-mask.txt
python app.py --mask-model "mask_detector.onnx"
python app.py --image "photo.jpg" --output "classified.png" --mask-model "mask_detector.onnx"
```

The classifier runs locally and labels each detected face with a prediction and model score. Faces missed by the OpenCV detector cannot be classified.

Add `--min-mask-score 0.80` to either command to mark predictions below 80% as **Uncertain** in orange. The option accepts scores from `0.5` to `1`. By default the app displays the model's highest-scoring label for each detected face.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

The tests use the base dependencies and run without a camera or a model download.
