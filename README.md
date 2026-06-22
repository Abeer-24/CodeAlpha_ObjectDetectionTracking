# CodeAlpha_ObjectDetectionTracking

Real-time object detection and tracking using YOLOv8 + Deep SORT.
Supports both webcam and video file input.

## Tech Stack
- **Detection**: YOLOv8s (Ultralytics)
- **Tracking**: Deep SORT (`deep-sort-realtime`)
- **Video I/O**: OpenCV

## Setup

```bash
pip install -r requirements.txt
```

First run auto-downloads `yolov8s.pt` (~22MB) from Ultralytics.

## Usage

**Video file:**
```bash
python object_detection_tracking.py --source sample.mp4 --output out.mp4 --model yolov8s.pt --conf 0.6
```

**Webcam:**
```bash
python object_detection_tracking.py --source webcam --output out_webcam.mp4 --model yolov8s.pt --conf 0.6
```

**All flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `webcam` | `webcam` or path to video file |
| `--model` | `yolov8n.pt` | YOLOv8 weights (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `--conf` | `0.4` | Minimum detection confidence threshold |
| `--output` | `output_tracked.mp4` | Output video path |
| `--classes` | all | Filter by COCO class names e.g. `"person,car"` |
| `--cam-index` | `0` | Webcam device index |
| `--no-display` | off | Skip live preview window |

Press `q` in the preview window to stop.

## How It Works

1. **Detection**: Each frame is passed through YOLOv8, which returns bounding boxes, confidence scores, and COCO class labels for all detected objects.
2. **Tracking**: Detections are passed to Deep SORT, which assigns persistent IDs across frames using a combination of Kalman filter motion prediction and MobileNet appearance embeddings.
3. **Output**: Annotated frames are shown live and saved to an MP4 file.

## Known Limitations

During heavy occlusion (3+ people clustering together at close range), the tracker
can temporarily spawn extra track IDs when objects simultaneously enter frame in
quick succession. This is a structural limitation of Deep SORT's track-confirmation
pipeline at low frame rates (≤12fps): new detections arrive faster than existing
unconfirmed tracks can be matched, causing the cost-assignment step to treat them
as new objects rather than continuations of existing tracks.

Verified through systematic diagnostic testing across 6 parameter configurations
(appearance-based and IOU-only matching), all converging on the same root cause.
The failure is limited to the cluster entry/exit moment and self-resolves within
~12 frames as tracks age out.
