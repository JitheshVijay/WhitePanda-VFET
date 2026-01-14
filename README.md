# Video Feature Extraction Tool

A Python-based video analysis tool that extracts visual and temporal features from any video file. Built with OpenCV and designed for quick insights into video content.

## What Does It Do?

This tool analyzes your video and extracts four key features:

### 1. Shot Cut Detection
Finds "hard cuts" — those abrupt transitions between scenes. Great for understanding the editing pace of trailers, music videos, or any fast-paced content. Uses histogram comparison to detect when the visual content changes dramatically.

### 2. Motion Analysis
Measures how much movement is happening in your video using Optical Flow. Returns an intensity rating (low/medium/high) so you can quickly tell if it's a calm interview or an action-packed sequence.

### 3. Text Detection (OCR)
Scans sampled frames for on-screen text using Tesseract OCR. Useful for detecting subtitles, title cards, or any text overlays. Returns the percentage of frames containing text and extracts sample keywords.

### 4. Object vs. Person Detection (YOLO)
Uses YOLOv8 to detect objects and people, then calculates which dominates the video. Tells you if your video is person-focused (interviews, vlogs) or object-focused (product demos, nature footage).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the tool
python video_feature_extractor.py --video path/to/your/video.mp4
```

Results are printed as JSON.

---

## Installation

### Step 1: Python Dependencies

```bash
cd D:\WhitePanda-VFET
pip install -r requirements.txt
```

### Step 2: Tesseract OCR (for text detection) I did it using chocolately and it worked fine. You can use winget or brew or apt-get as well. But winget was not working for me.

**Windows (using Chocolatey):**
```bash
choco install tesseract
```

**Windows (using winget):**
```bash
winget install UB-Mannheim.TesseractOCR
```

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

> **Note:** The script auto-detects Tesseract's location on Windows, so you don't need to mess with PATH variables.

### Step 3: YOLO Object Detection (optional) P.S. I was not able to test this feature. It's probably an issue with my system. The rest of the features work fine. But if you want to test it, follow the instructions below:

If you want object detection, you'll need PyTorch:

```bash
# CPU-only version (smaller, faster to install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

Or skip it entirely with `--skip-yolo` — the other 3 features work fine without it.

---

## Usage

### Basic Usage

```bash
python video_feature_extractor.py --video my_video.mp4
```

### Save Results to a File

```bash
python video_feature_extractor.py --video my_video.mp4 --output results.json
```

### Skip YOLO (Faster, No PyTorch Needed)

```bash
python video_feature_extractor.py --video my_video.mp4 --skip-yolo
```

### All Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video`, `-v` | Path to your video file | Required |
| `--output`, `-o` | Save JSON to this file | Print to screen |
| `--skip-yolo` | Skip object detection | False |
| `--shot-threshold` | Sensitivity for cut detection (0-1) | 0.5 |
| `--motion-sample-rate` | Analyze every Nth frame for motion | 10 |
| `--ocr-interval` | Check every Nth frame for text | 30 |
| `--pretty` | Pretty-print the JSON output | False |

---

## Sample Output

Here's what you get for a typical movie trailer (I used Avengers Infinity War Trailer.mp4): 

```json
{
  "video_name": "Avengers_Infinity_War_Trailer.mp4",
  "duration_seconds": 144.14,
  "fps": 23.98,
  "resolution": [1280, 532],
  "features": {
    "shot_cuts": {
      "total_cuts": 81,
      "avg_shot_duration_seconds": 1.76
    },
    "motion_analysis": {
      "average_motion_magnitude": 3.96,
      "motion_intensity": "medium"
    },
    "text_detection": {
      "text_present_ratio": 0.06,
      "frames_with_text": 6,
      "sample_keywords": ["MARVEL", "STUDIOS"]
    },
    "object_detection": {
      "person_ratio": 0.65,
      "dominant_category": "person",
      "top_objects": [{"type": "car", "count": 12}]
    }
  }
}
```

---

## Using It in Your Own Code

```python
from video_feature_extractor import VideoFeatureExtractor

# Create an extractor
extractor = VideoFeatureExtractor("my_video.mp4")

# Get all features at once
results = extractor.extract_all_features()

# Or pick and choose
cuts = extractor.detect_shot_cuts()
motion = extractor.analyze_motion()
text = extractor.detect_text()
objects = extractor.detect_objects()  # Requires YOLO
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video processing, optical flow |
| `numpy` | Number crunching |
| `pytesseract` | OCR text detection |
| `Pillow` | Image handling for OCR |
| `ultralytics` | YOLOv8 object detection (optional) |

---

## Troubleshooting

**"Tesseract not found"**  
Make sure Tesseract is installed. The script auto-detects common install paths, but if it still fails, you can set the path manually in the script.

**YOLO is slow to start**  
First run downloads the model (~6MB). After that it's cached.(Didnt work for me) Use `--skip-yolo` if you don't need object detection. 

**Text detection showing 0%**  
Stylized text (3D logos, fancy fonts) is hard for OCR. Works best with plain text, subtitles, or simple title cards. So 3D title cards will be hard to see.

**Motion analysis takes forever**  
Try increasing `--motion-sample-rate` to 15 or 20 for faster (but less accurate) results.

---

## Reference

https://stackoverflow.com/questions/75097042/why-is-the-module-ultralytics-not-found-even-after-pip-installing-it-in-the-py

https://forums.raspberrypi.com/viewtopic.php?t=387759

https://github.com/ultralytics/ultralytics/issues/3097

https://docs.ultralytics.com/guides/yolo-common-issues/

https://pypi.org/project/pillow/

https://numpy.org/

https://tesseract-ocr.github.io/

https://github.com/tesseract-ocr/tesseract

https://download.pytorch.org/whl/cpu

https://discuss.pytorch.org/t/pytorch-taking-forever-to-install/172326

---
## License

              DO WTFPL

 Copyright (C) 2026 Jithesh Vijay <jitheshvijay67@gmail.com>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

              DO WTFPL
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
    0. You just DO WTFPL.
