import cv2
import numpy as np
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = FastAPI()

# Load model once at startup
model = YOLO("yolov8n.pt")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection & Tracking</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 60px auto; padding: 20px; background: #0f0f0f; color: #eee; }
        h1 { color: #00ff88; }
        .subtitle { color: #aaa; margin-bottom: 30px; }
        input[type=file] { display: block; margin: 20px 0; padding: 10px; background: #1a1a1a; border: 1px solid #333; color: #eee; width: 100%; }
        button { background: #00ff88; color: #000; border: none; padding: 12px 30px; font-size: 16px; cursor: pointer; border-radius: 4px; font-weight: bold; }
        button:disabled { background: #555; color: #999; cursor: not-allowed; }
        #status { margin-top: 20px; padding: 15px; background: #1a1a1a; border-radius: 4px; display: none; }
        #download { display: none; margin-top: 20px; }
        #download a { background: #0088ff; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; font-weight: bold; }
        .info { background: #1a1a1a; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 3px solid #00ff88; }
    </style>
</head>
<body>
    <h1>🎯 Object Detection & Tracking</h1>
    <p class="subtitle">YOLOv8s + Deep SORT</p>
    <div class="info">
        Upload a video file (MP4, max 30MB).<br>
        Detects objects and assigns persistent tracking IDs.<br>
        Processing takes 3-8 minutes on CPU.
    </div>
    <input type="file" id="videoFile" accept="video/*">
    <button id="btn" onclick="processVideo()">🚀 Process Video</button>
    <div id="status"></div>
    <div id="download"></div>
    <script>
    async function processVideo() {
        const file = document.getElementById('videoFile').files[0];
        if (!file) { alert('Please select a video file first.'); return; }
        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const download = document.getElementById('download');
        btn.disabled = true;
        btn.textContent = 'Processing...';
        status.style.display = 'block';
        status.textContent = '⏳ Processing video... This takes 3-8 minutes on CPU. Please wait.';
        download.style.display = 'none';
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('/process', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Processing failed: ' + response.statusText);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            status.textContent = '✅ Done!';
            download.style.display = 'block';
            download.innerHTML = '<a href="' + url + '" download="tracked_output.mp4">⬇️ Download Tracked Video</a>';
        } catch (err) {
            status.textContent = '❌ Error: ' + err.message;
        } finally {
            btn.disabled = false;
            btn.textContent = '🚀 Process Video';
        }
    }
    </script>
</body>
</html>
"""

@app.post("/process")
async def process(file: UploadFile = File(...)):
    video_bytes = await file.read()
    if len(video_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    tracker = DeepSort(max_age=12, n_init=5, nms_max_overlap=0.7)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        input_path = f.name

    output_path = input_path.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, conf=0.4)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            label = model.names[cls_id]
            detections.append(([x1, y1, w, h], conf, label))
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class() or "object"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{tid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    os.unlink(input_path)

    with open(output_path, "rb") as f:
        result = f.read()
    os.unlink(output_path)

    return Response(content=result, media_type="video/mp4",
                   headers={"Content-Disposition": "attachment; filename=tracked_output.mp4"})
