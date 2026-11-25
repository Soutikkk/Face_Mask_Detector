# Face Mask Detector

A Python project for detecting faces in webcam feeds and photos and classifying whether they are wearing face masks.

The current version draws a box around each detected face. Mask classification is planned for a later update.

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
