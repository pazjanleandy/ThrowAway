# ThrowAway
ThrowAway is a smart waste segregation system that combines YOLOv8 object detection with a PyTorch classifier, a Tkinter GUI, and optional Arduino-controlled servos for physical sorting.

This repository contains a GUI app for training and live classification, dataset utilities, and scripts for training YOLO models.

**Quick Start**
1. Install dependencies for your environment (PyTorch, OpenCV, Ultralytics, Albumentations, scikit-learn, matplotlib, Pillow, pyserial, tkinter).
2. Run the GUI:

```powershell
python app.py
```

3. Use the app to train or run live classification.

**Project Layout**
`app.py`  
Main Tkinter application. Provides the modern GUI, navigation, logging, and orchestration for training, retraining, live prediction, and database browsing. It calls into `testing.py` for model training, dataset prep, and real-time prediction. It also uses `webcam_utils.py` utilities to correct fisheye distortion and adjust brightness/contrast/saturation when enabled.

Key responsibilities:
1. User interface: Home, Training, Live Dashboard, Database, Retrain, and Maintenance panels.
2. Training workflow: dataset selection, model type selection (CNN vs MobileNetV2 transfer), epoch config, progress logging.
3. Live prediction: runs the YOLO + classifier pipeline from `testing.py`, shows overlays, and manages logging.
4. Database and history: reads stored images/records, displays summaries, and exports where applicable.

`testing.py`  
Core ML and inference logic. Implements dataset loading, augmentation, training loops, evaluation, plotting, and real-time prediction with YOLOv8 + PyTorch classifier integration. Also handles serial control for Arduino servos during live sorting.

Key responsibilities:
1. Dataset management: folder-based datasets, optional manual train/validation split, stratified split fallback.
2. Augmentation: Albumentations pipelines for training and validation, plus heavy augmentation options.
3. Model definitions: custom CNN and MobileNetV2 transfer learning model builder.
4. Training and retraining: progress callbacks, model checkpointing, training plots, and dataset preparation.
5. Real-time prediction: YOLOv8 detection, classifier inference, object tracking, overlay rendering, and serial commands.

`crop.py`  
Utility for cropping YOLO-annotated objects into class folders, mirroring YOLO train/val splits. Includes optional re-splitting and randomization of cropped images.

Key responsibilities:
1. Parse YOLO label files and crop bounding boxes from images.
2. Save crops into class folders, optionally preserving train/val split.
3. Optional reshuffle to a fresh 80/20 per-class train/val split.

Example usage:

```powershell
python crop.py --base yolodataset --out cropped
python crop.py --out cropped -randomize
```

`ir.py`  
Simple serial controller for testing Arduino servo commands from the console. Opens a COM port and sends raw one-character commands used by the Arduino firmware.

Key responsibilities:
1. Establish serial connection to Arduino.
2. Provide a CLI loop for sending servo commands.
3. Read and print Arduino responses.

`webcam_utils.py`  
Small image processing helpers used by the GUI pipeline.

Key responsibilities:
1. Fisheye distortion correction via OpenCV.
2. Brightness, contrast, and saturation adjustment.

`yolov8training.py`  
Minimal Ultralytics YOLOv8 training launcher. It loads a YOLO model checkpoint and trains on a dataset defined by a YOLO `data.yaml`.

Key responsibilities:
1. Configure a YOLO model checkpoint path and dataset YAML.
2. Launch YOLO training with basic parameters.

**Notes**
1. Some paths in the scripts are hard-coded (for example, dataset roots and checkpoint paths). Update them for your environment.
2. Serial control defaults to `COM4`. Change the port if your Arduino is on a different COM device.
3. The GUI expects certain dataset folder structures; review `testing.py` if you plan to reorganize datasets.
