# River hydromorphology detection from drone imagery (RF-DETR, Roboflow)

Object detection of hydromorphological features (woody debris jams, fallen trees, riffles, islands, erosion, longitudinal bars, roots and others) in drone stills and video of the Erft river, Germany. The goal is a feature **inventory**, so the main check is whether the model counts each class correctly.

The work was done in 2026 in the module *Project 2: AI-Supported River Hydromorphology Mapping* (M.Sc. Integrated Water Resources Management, TH Köln).

## Model

| Item | Value |
|---|---|
| Architecture | RF-DETR Medium, 1280 px input, trained on Roboflow |
| Data | 436 labelled images (361 train, 50 valid, 25 test), forked from the CC BY 4.0 dataset [River Feature Object Detection](https://universe.roboflow.com/avizit-biswas/river-feature-object-detection) |
| Augmentation | 3× (flip, crop, brightness, exposure, hue, saturation, blur, noise) plus online motion blur and rotation |
| Validation metrics (Roboflow) | mAP@50 57.5 %, precision 66.4 %, recall 51.3 % |

All IDs, settings and the training recipe are in [`project_info.json`](project_info.json). The two deployed Roboflow Workflows (image detection, and video detection with ByteTrack so each object is counted once) are in [`workflows/`](workflows).

## Choosing the confidence threshold

I ran the model on 74 of the 75 held-out images (49 valid, 25 test; one valid image is excluded) and compared predicted with labelled counts per class and image.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.65 | 0.82 | 0.73 |
| **0.35** | **0.74** | **0.77** | **0.75** |
| 0.40 | 0.80 | 0.70 | 0.75 |
| 0.50 | 0.87 | 0.60 | 0.71 |

At 0.35 precision and recall are about equal, so the total count is close to the labels (871 predicted, 838 labelled). Single classes can be further off: building is over-counted (226 against 177). Per-class thresholds fitted on the validation split did worse on the test split (F1 0.712 against 0.735), so one global threshold is used. The script and raw counts are in [`evaluation/`](evaluation).

**Limitations.** Several classes have fewer than 10 held-out labels and are not reliable yet. Train, validation and test were split by frame, not by video, so neighbouring frames of one clip can sit in different splits and the scores are likely optimistic. More labels for rare classes and a split by video would help most.

## Usage

```bash
pip install "inference-sdk[webrtc]>=1.7" opencv-python
export ROBOFLOW_API_KEY=<your key>          # Windows: set ROBOFLOW_API_KEY=<your key>
python hydromorph_detect.py photo.jpg
python hydromorph_detect.py clip.mp4 --min-frames 5     # unique-object inventory from video
```

Running the model needs a Roboflow API key with access to the workspace. The code shows the full pipeline, from typed API client and error handling to video tracking, inventory output and threshold evaluation. Details are in [`docs/USAGE.md`](docs/USAGE.md).

Tests run offline against a recorded response:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Author

Rifah Muhaimina Rahman, M.Sc. student in Integrated Water Resources Management, TH Köln. Code released under the MIT License. The training images belong to the original dataset authors (CC BY 4.0) and are not included here.
