"""
CodeAlpha Task 4 — Object Detection and Tracking
Detection: YOLOv8 (Ultralytics)
Tracking:  Deep SORT (deep-sort-realtime)
Input:     webcam (--source webcam) or video file (--source path/to/file.mp4)

Usage:
    python object_detection_tracking.py --source webcam
    python object_detection_tracking.py --source sample.mp4
    python object_detection_tracking.py --source sample.mp4 --output out.mp4 --conf 0.4
"""

import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument("--source", type=str, default="webcam",
                         help="'webcam' for live camera, or path to a video file")
    parser.add_argument("--cam-index", type=int, default=0,
                         help="Webcam device index (default 0)")
    parser.add_argument("--output", type=str, default="output_tracked.mp4",
                         help="Path to save the annotated output video")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Minimum detection confidence to keep")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated COCO class names to filter, e.g. 'person,car'. "
                              "Default: all classes.")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a live preview window (useful for headless runs)")
    return parser.parse_args()


def open_capture(args):
    if args.source.lower() == "webcam":
        cap = cv2.VideoCapture(args.cam_index)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    return cap


def main():
    args = parse_args()

    model = YOLO(args.model)
    class_filter = None
    if args.classes:
        wanted = set(name.strip() for name in args.classes.split(","))
        class_filter = {idx for idx, name in model.names.items() if name in wanted}
        if not class_filter:
            print(f"WARNING: none of {wanted} matched known COCO class names; detecting all classes instead.")
            class_filter = None

    tracker = DeepSort(max_age=12, n_init=5, nms_max_overlap=0.7)

    cap = open_capture(args)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25  # webcams often report 0; fall back to a sane default
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    print(f"Source: {args.source} | {width}x{height} @ {fps:.1f}fps | model: {args.model}")
    print("Press 'q' to stop early (if preview window is open).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- Detection ---
        results = model(frame, verbose=False)[0]

        detections = []  # Deep SORT expects: ([left, top, w, h], confidence, class_name)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < args.conf:
                continue
            if class_filter is not None and cls_id not in class_filter:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            label = model.names[cls_id]
            detections.append(([x1, y1, w, h], conf, label))

        # --- Tracking ---
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class() or "object"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        writer.write(frame)

        if not args.no_display:
            cv2.imshow("Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"Done. Processed {frame_count} frames. Output saved to {args.output}")


if __name__ == "__main__":
    main()
