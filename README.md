# Padel Game Analytics — Shot Classification System
 
A computer vision pipeline that analyzes padel match footage to detect players, track the ball, classify shots, and visualize results in real time.
 
---
 
## Project Structure
 
```
Padel Game Analytics/
├── Data/                            # (Not pushed due to large file size)
│   ├── input_sample_video.mp4       # Input/ Sample Video
    └── infernce_sample_video.mp4    # Reference Video
├── Outputs/
│   ├── annotated.mp4                # Output video with overlays (Not pushed due to large file size)
│   ├── result.json                  # Shot results (frame, timestamp, type, direction)
│   ├── result.csv                   # Same data in CSV format
│   └── analytics.json               # Shot count statistics
├── src/
│   ├── main.py                      # Main pipeline
│   ├── detector.py                  # YOLO-based object detection
│   ├── pose_estimator.py            # MediaPipe pose estimation
│   ├── tracker.py                   # Ball tracking + bounce detection
│   ├── shot_classifier.py           # Shot classification logic
│   ├── visualizer.py                # Frame overlay drawing
│   ├── dashboard.py                 # Live analytics dashboard panel
│   └── yolov8m.pt                   # YOLO model weights
├── requirements.txt
└── README.md
```
 
---
 
## Features
 
### Core
- **Player Detection** — Detects all players on court using YOLOv8m
- **Ball Detection** — Detects padel ball using YOLOv8m at high resolution
- **Racket Detection** — Detects tennis rackets using YOLOv8m
- **Ball Tracking** — Tracks ball position over time, computes velocity and direction change
- **Shot Classification** — Classifies Forehand, Backhand, and Smash using pose landmarks + ball motion fallback
- **Structured Output** — Saves results to JSON and CSV with frame, timestamp, shot type, and direction
### Extra
- **Shot Direction** — Classifies each shot as Left, Right, or Straight
- **Bounce Detection** — Detects ball bounces using trajectory analysis
- **Live Dashboard** — Real-time side panel showing shot counts, progress bars, live stats, and ball trail
- **Shot Analytics** — Percentage breakdown of each shot type saved to `analytics.json`
- **Annotated Output Video** — Full match video with bounding boxes, labels, and dashboard overlay
---
 
## Tech Stack
 
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| OpenCV | Video I/O and frame drawing |
| Ultralytics YOLOv8 | Player, ball, racket detection |
| MediaPipe | Pose estimation / body keypoints |
| NumPy | Vector math for tracking |
| Pandas | CSV export |
| PyTorch (CUDA) | GPU acceleration for YOLO |
 
---
 
## Installation
 
```bash
# Clone the repo
git clone https://github.com/yourusername/padel-game-analytics.git
cd padel-game-analytics
 
# Create virtual environment
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
```
 
---
 
## Requirements
 
```
ultralytics
opencv-python-headless
mediapipe==0.10.9
numpy
pandas
torch
torchvision
```
 
Install PyTorch with CUDA (recommended for GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
 
---
 
## Usage
 
1. Place your input video in the `Data/` folder
2. Update `VIDEO_PATH` in `main.py` if needed:
```python
VIDEO_PATH = '../Data/input_sample_video.mp4'
```
3. Run the pipeline:
```bash
cd src
python main.py
```
4. Results saved to `Outputs/`:
   - `annotated.mp4` — video with all overlays
   - `result.json` / `result.csv` — per-shot data
   - `analytics.json` — summary statistics
Press `q` to stop processing early.
 
---
 
## Output Format
 
### result.json
```json
[
    {
        "frame": 90,
        "timestamp": 3.0,
        "shot_type": "Forehand",
        "direction": "Right"
    },
    {
        "frame": 180,
        "timestamp": 6.0,
        "shot_type": "Backhand",
        "direction": "Left"
    }
]
```
 
### analytics.json
```json
{
    "total_shots": 30,
    "shot_counts": {
        "Forehand": 12,
        "Backhand": 10,
        "Smash": 5,
        "Bounce": 3
    },
    "forehand_pct": 40.0,
    "backhand_pct": 33.3,
    "smash_pct": 16.7
}
```
 
---
 
## How It Works
 
```
Input Video
    │
    ▼
ObjectDetector (YOLOv8m)
    │── Players  (green boxes)
    │── Rackets  (orange boxes)
    └── Ball     (yellow circle)
    │
    ▼
PoseEstimator (MediaPipe)
    └── Body keypoints on largest player crop
    │
    ▼
BallTracker
    │── Velocity
    │── Direction change
    └── Bounce detection
    │
    ▼
ShotClassifier
    │── Landmark-based (Smash / Forehand / Backhand)
    └── Fallback: ball motion only
    │
    ▼
Visualizer + Dashboard
    └── Annotated output video + JSON/CSV
```
 
---
 
## Performance
 
| Hardware | Processing Speed (1min 26sec video) |
|----------|--------------------------------------|
| CPU only | 6–8 minutes |
| GPU (GTX 1660 Ti) | ~1 minute |
 
Processing is optimized by running detection every 3rd frame and caching results for skipped frames.
 
---
 
## Shot Classification Logic
 
| Shot | Condition |
|------|-----------|
| Smash | Wrist above nose level + velocity > 5 |
| Forehand | Right wrist crosses right of shoulder midpoint + direction change > 10 |
| Backhand | Right wrist crosses left of shoulder midpoint + direction change > 10 |
| Fallback Smash | Velocity > 20 + direction change < 15 (no pose data) |
| Fallback Forehand | Direction change > 25 (no pose data) |
| Fallback Backhand | Direction change > 10 (no pose data) |
