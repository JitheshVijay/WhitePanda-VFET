#!/usr/bin/env python3
"""
Video Feature Extraction Tool
==============================

A tool for analyzing video files and pulling out useful visual features.
Whether you're analyzing movie trailers, vlogs, or any video content, this
tool gives you quick insights into what's happening on screen.

Features:
    - Shot Cut Detection: Finds hard cuts between scenes
    - Motion Analysis: Measures how much action is happening (uses Optical Flow)
    - Text Detection: Spots on-screen text using OCR
    - Object Detection: Figures out if people or objects dominate the video (YOLO) P.S. I was not able to test this feature. It's probably an issue with my system. The rest of the 3 features work fine.

Quick Start:
    python video_feature_extractor.py --video my_video.mp4
    python video_feature_extractor.py --video my_video.mp4 --output results.json
    python video_feature_extractor.py --video my_video.mp4 --skip-yolo  # faster!

Author: Jithesh Vijay <jitheshvijay67@gmail.com> (check out Refringence at https://refringence.com)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

# pytesseract for OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    
    # Auto-detect Tesseract on Windows if not in PATH
    import shutil
    if shutil.which('tesseract') is None:
        import os
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Text detection will be disabled.")

# YOLO is imported lazily in detect_objects() to avoid slow startup
YOLO_AVAILABLE = None  # Will be set on first use


class VideoFeatureExtractor:
    """
    The main workhorse for video analysis.
    
    Just point it at a video file and it'll extract all sorts of useful features:
    shot cuts, motion levels, on-screen text, and even what objects appear most often.
    
    Example:
        extractor = VideoFeatureExtractor("my_video.mp4")
        results = extractor.extract_all_features()
    """
    
    def __init__(self, video_path: str, 
                 shot_threshold: float = 0.5,
                 motion_sample_rate: int = 2,
                 ocr_sample_interval: int = 30):
        """
        Initialize the feature extractor.
        
        Args:
            video_path: Path to the video file
            shot_threshold: Threshold for detecting shot cuts (0-1, higher = less sensitive)
            motion_sample_rate: Process every Nth frame for motion analysis
            ocr_sample_interval: Sample every Nth frame for OCR
        """
        self.video_path = Path(video_path)
        self.shot_threshold = shot_threshold
        self.motion_sample_rate = motion_sample_rate
        self.ocr_sample_interval = ocr_sample_interval
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video and get properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def __del__(self):
        """Release video capture on destruction."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def _reset_video(self):
        """Reset video to the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized histogram for a frame."""
        # Convert to HSV for better color comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for H and S channels
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    
    def detect_shot_cuts(self) -> Dict[str, Any]:
        """
        Detect hard cuts in the video using histogram difference.
        
        Returns:
            Dictionary containing:
                - total_cuts: Number of detected shot cuts
                - cut_timestamps: List of timestamps where cuts occur
                - avg_shot_duration_seconds: Average duration between cuts
        """
        self._reset_video()
        
        cuts = []
        prev_hist = None
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Compute histogram
            curr_hist = self._compute_histogram(frame)
            
            if prev_hist is not None:
                # Compare histograms using correlation
                correlation = cv2.compareHist(
                    prev_hist.reshape(-1, 1).astype(np.float32),
                    curr_hist.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                # Low correlation indicates a potential cut
                if correlation < self.shot_threshold:
                    timestamp = frame_idx / self.fps if self.fps > 0 else 0
                    cuts.append({
                        "frame": frame_idx,
                        "timestamp_seconds": round(timestamp, 2),
                        "correlation": round(correlation, 3)
                    })
            
            prev_hist = curr_hist
            frame_idx += 1
        
        # Calculate average shot duration
        total_cuts = len(cuts)
        if total_cuts > 0:
            avg_shot_duration = self.duration / (total_cuts + 1)
        else:
            avg_shot_duration = self.duration
        
        return {
            "total_cuts": total_cuts,
            "cut_timestamps": cuts[:20],  # Limit to first 20 cuts in output
            "avg_shot_duration_seconds": round(avg_shot_duration, 2)
        }
    
    def _resize_frame(self, frame: np.ndarray, max_width: int = 320) -> np.ndarray:
        """Resize frame for faster processing while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def analyze_motion(self) -> Dict[str, Any]:
        """
        Analyze motion in the video using Farneback Optical Flow.
        
        Returns:
            Dictionary containing:
                - average_motion_magnitude: Average magnitude of motion vectors
                - max_motion_magnitude: Maximum motion detected
                - motion_intensity: Qualitative descriptor (low/medium/high)
        """
        self._reset_video()
        
        ret, prev_frame = self.cap.read()
        if not ret:
            return {"error": "Could not read first frame"}
        
        # Downscale for faster processing
        prev_frame_small = self._resize_frame(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame_small, cv2.COLOR_BGR2GRAY)
        
        motion_magnitudes = []
        frame_idx = 0
        
        # Calculate total frames to analyze for progress
        total_samples = self.frame_count // self.motion_sample_rate
        last_progress = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Sample frames for efficiency (use higher sample rate for speed)
            if frame_idx % self.motion_sample_rate != 0:
                continue
            
            # Progress indicator
            progress = len(motion_magnitudes) * 100 // max(total_samples, 1)
            if progress >= last_progress + 20:
                print(f"    Motion analysis: {progress}% complete...")
                last_progress = progress
            
            # Downscale frame for faster processing
            frame_small = self._resize_frame(frame)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Compute magnitude of flow vectors
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)
            motion_magnitudes.append(avg_magnitude)
            
            prev_gray = gray
        
        if not motion_magnitudes:
            return {"error": "No motion data collected"}
        
        avg_motion = float(np.mean(motion_magnitudes))
        max_motion = float(np.max(motion_magnitudes))
        
        # Classify motion intensity
        if avg_motion < 2:
            intensity = "low"
        elif avg_motion < 8:
            intensity = "medium"
        else:
            intensity = "high"
        
        return {
            "average_motion_magnitude": round(avg_motion, 2),
            "max_motion_magnitude": round(max_motion, 2),
            "motion_intensity": intensity,
            "frames_analyzed": len(motion_magnitudes)
        }
    
    def detect_text(self) -> Dict[str, Any]:
        """
        Detect text presence in video frames using OCR (pytesseract).
        
        Returns:
            Dictionary containing:
                - text_present_ratio: Ratio of frames containing text
                - frames_analyzed: Number of frames sampled
                - sample_keywords: List of detected text samples
        """
        if not TESSERACT_AVAILABLE:
            return {
                "error": "pytesseract not available",
                "text_present_ratio": None
            }
        
        self._reset_video()
        
        frames_with_text = 0
        frames_analyzed = 0
        detected_texts = []
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames at interval
            if frame_idx % self.ocr_sample_interval != 0:
                frame_idx += 1
                continue
            
            frame_idx += 1
            frames_analyzed += 1
            
            # Convert to grayscale and apply preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to improve OCR accuracy
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(thresh)
            
            try:
                # Extract text
                text = pytesseract.image_to_string(pil_image, timeout=5)
                text = text.strip()
                
                if text:
                    frames_with_text += 1
                    # Clean and store unique words
                    words = [w.strip() for w in text.split() if len(w.strip()) >= 3]
                    for word in words[:5]:  # Limit words per frame
                        if word.isalnum() and word not in detected_texts:
                            detected_texts.append(word)
            except Exception as e:
                # Timeout or other OCR error - skip this frame
                pass
            
            # Limit total analysis for performance
            if frames_analyzed >= 100:
                break
        
        text_ratio = frames_with_text / frames_analyzed if frames_analyzed > 0 else 0
        
        return {
            "text_present_ratio": round(text_ratio, 3),
            "frames_with_text": frames_with_text,
            "frames_analyzed": frames_analyzed,
            "sample_keywords": detected_texts[:15]  # Limit keywords in output
        }
    
    def detect_objects(self, sample_interval: int = 30) -> Dict[str, Any]:
        """
        Detect objects and people in video frames using YOLOv8.
        
        Calculates the ratio of people vs other objects detected.
        
        Args:
            sample_interval: Sample every Nth frame for detection
            
        Returns:
            Dictionary containing:
                - person_count: Total people detected across sampled frames
                - object_count: Total other objects detected
                - person_ratio: Ratio of people to total detections
                - dominant_category: 'person' or 'object'
                - top_objects: Most common object types detected
        """
        global YOLO_AVAILABLE
        
        # Lazy import YOLO to avoid slow startup
        if YOLO_AVAILABLE is None:
            try:
                print("    Importing YOLO (first time may take a moment)...")
                from ultralytics import YOLO as YOLOModel
                self._yolo_class = YOLOModel
                YOLO_AVAILABLE = True
            except ImportError:
                YOLO_AVAILABLE = False
                return {
                    "error": "ultralytics not installed. Run: pip install ultralytics torch torchvision",
                    "person_ratio": None
                }
        
        if not YOLO_AVAILABLE:
            return {
                "error": "ultralytics not available",
                "person_ratio": None
            }
        
        self._reset_video()
        
        # Load YOLOv8 model (downloads automatically on first use)
        print("    Loading YOLOv8 model...")
        model = self._yolo_class('yolov8n.pt')  # nano model for speed
        
        # COCO class names - class 0 is 'person'
        PERSON_CLASS = 0
        
        person_count = 0
        object_count = 0
        object_types = {}
        frames_analyzed = 0
        frame_idx = 0
        
        total_frames = self.frame_count
        last_progress = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames at interval
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue
            
            frame_idx += 1
            frames_analyzed += 1
            
            # Progress indicator
            progress = (frame_idx * 100) // max(total_frames, 1)
            if progress >= last_progress + 25:
                print(f"    Object detection: {progress}% complete...")
                last_progress = progress
            
            # Run YOLO inference
            results = model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        if class_id == PERSON_CLASS:
                            person_count += 1
                        else:
                            object_count += 1
                            object_types[class_name] = object_types.get(class_name, 0) + 1
            
            # Limit total analysis for performance
            if frames_analyzed >= 50:
                break
        
        total_detections = person_count + object_count
        person_ratio = person_count / total_detections if total_detections > 0 else 0
        
        # Determine dominance
        if person_ratio > 0.6:
            dominant = "person"
        elif person_ratio < 0.4:
            dominant = "object"
        else:
            dominant = "balanced"
        
        # Get top 5 object types
        top_objects = sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "person_count": person_count,
            "object_count": object_count,
            "total_detections": total_detections,
            "person_ratio": round(person_ratio, 3),
            "dominant_category": dominant,
            "top_objects": [{"type": k, "count": v} for k, v in top_objects],
            "frames_analyzed": frames_analyzed
        }
    
    def extract_all_features(self, skip_yolo: bool = False) -> Dict[str, Any]:
        """
        Extract all features from the video.
        
        Args:
            skip_yolo: If True, skip YOLO object detection (faster)
        
        Returns:
            Complete feature dictionary with video metadata and all extracted features.
        """
        print(f"Analyzing video: {self.video_path.name}")
        print(f"  Duration: {self.duration:.2f}s | FPS: {self.fps:.2f} | Resolution: {self.width}x{self.height}")
        print("-" * 50)
        
        # Video metadata
        result = {
            "video_path": str(self.video_path.absolute()),
            "video_name": self.video_path.name,
            "duration_seconds": round(self.duration, 2),
            "fps": round(self.fps, 2),
            "resolution": [self.width, self.height],
            "total_frames": self.frame_count,
            "features": {}
        }
        
        # 1. Shot Cut Detection
        print("Detecting shot cuts...")
        result["features"]["shot_cuts"] = self.detect_shot_cuts()
        print(f"  Found {result['features']['shot_cuts']['total_cuts']} cuts")
        
        # 2. Motion Analysis
        print("Analyzing motion (Optical Flow)...")
        result["features"]["motion_analysis"] = self.analyze_motion()
        if "error" not in result["features"]["motion_analysis"]:
            print(f"  Average motion: {result['features']['motion_analysis']['average_motion_magnitude']}")
            print(f"  Intensity: {result['features']['motion_analysis']['motion_intensity']}")
        
        # 3. Text Detection (OCR)
        print("Detecting text (OCR)...")
        result["features"]["text_detection"] = self.detect_text()
        if "error" not in result["features"]["text_detection"]:
            ratio = result["features"]["text_detection"]["text_present_ratio"]
            print(f"  Text present in {ratio*100:.1f}% of sampled frames")
        else:
            print(f"  Skipped: {result['features']['text_detection']['error']}")
        
        # 4. Object Detection (YOLO) - optional (I didn't test this because my PC wasn't downloading the Pytorch properly)
        if skip_yolo:
            print("Detecting objects (YOLO)... SKIPPED (--skip-yolo flag)")
            result["features"]["object_detection"] = {"skipped": True}
        else:
            print("Detecting objects (YOLO)...")
            result["features"]["object_detection"] = self.detect_objects()
            if "error" not in result["features"]["object_detection"]:
                person_ratio = result["features"]["object_detection"]["person_ratio"]
                dominant = result["features"]["object_detection"]["dominant_category"]
                print(f"  Person ratio: {person_ratio*100:.1f}% | Dominant: {dominant}")
            else:
                print(f"  Skipped: {result['features']['object_detection']['error']}")
        
        print("-" * 50)
        print("Feature extraction complete!")
        
        return result


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Video Feature Extraction Tool - Analyze videos for visual and temporal features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python video_feature_extractor.py --video sample.mp4
    python video_feature_extractor.py --video sample.mp4 --output features.json
    python video_feature_extractor.py --video sample.mp4 --shot-threshold 0.4
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to the video file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path (default: prints to stdout)"
    )
    
    parser.add_argument(
        "--shot-threshold",
        type=float,
        default=0.5,
        help="Threshold for shot cut detection (0-1, default: 0.5)"
    )
    
    parser.add_argument(
        "--motion-sample-rate",
        type=int,
        default=10,
        help="Sample every Nth frame for motion analysis (default: 10)"
    )
    
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=30,
        help="Sample every Nth frame for OCR (default: 30)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO object detection (faster, no PyTorch needed)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = VideoFeatureExtractor(
            video_path=args.video,
            shot_threshold=args.shot_threshold,
            motion_sample_rate=args.motion_sample_rate,
            ocr_sample_interval=args.ocr_interval
        )
        
        # Extract features
        features = extractor.extract_all_features(skip_yolo=args.skip_yolo)
        
        # Output results
        indent = 2 if args.pretty else None
        json_output = json.dumps(features, indent=indent)
        
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json_output)
            print(f"\nResults saved to: {output_path}")
        else:
            print("\n" + "=" * 50)
            print("EXTRACTED FEATURES (JSON):")
            print("=" * 50)
            print(json.dumps(features, indent=2))
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
