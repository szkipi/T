# Tennis30 Architecture

## Philosophy: Precision Over Speed

**Not YOLO (Look Once) - We Look Multiple Times**

Tennis30 prioritizes **accuracy over real-time performance** for recorded video analysis. Perfect for 3D game reconstruction where precision matters more than speed.

---

## Multi-Pass Architecture

### **Pass 1: Ensemble Detection**
- Run 5 different models on every frame
- Vote/average detections
- Eliminate false positives through consensus

**Models:**
1. TrackNet (heatmap-based, blur-resistant)
2. YOLOv8 Fine-tuned v1
3. YOLOv8 Fine-tuned v2
4. SAMURAI (zero-shot segmentation)
5. Faster R-CNN (alternative detector)

### **Pass 2: Temporal Refinement**
- Sliding window context (±15 frames)
- Gap filling with interpolation
- Outlier detection & correction
- Bidirectional Kalman smoothing

**Methods:**
- Cubic spline interpolation
- Physics-based prediction
- ML trajectory prediction (BallRadar)

### **Pass 3: Physics Validation**
- Fit parabolic trajectories
- Apply gravity + drag model
- Bounce detection (98% accuracy)
- Validate against max velocity

**Physics Model:**
- Gravity: 9.81 m/s²
- Drag coefficient: 0.55 (tennis ball)
- Bounce damping: 0.75
- Max velocity: 180 km/h

### **Pass 4: Cross-Validation**
- Re-check detections in small ROI
- Multi-model verification
- Quality scoring (0-1)
- Flag low-confidence frames

---

## Component Architecture

```
Tennis30/
├── core/
│   ├── pipeline.py              # Main orchestrator
│   ├── ball_tracking/
│   │   └── precision_tracker.py # Multi-pass ball tracker
│   ├── pose_estimation/
│   │   └── precision_pose.py    # Multi-model pose fusion
│   ├── physics/
│   │   └── physics_engine.py    # Trajectory validation
│   ├── court_detection/
│   │   └── court_mapper.py      # 2D→3D mapping
│   └── game_state/
│       └── game_tracker.py      # Rally detection
│
├── utils/
│   ├── preprocessing/
│   │   └── video_preprocessor.py
│   ├── export/
│   │   ├── csv_exporter.py
│   │   ├── unity_exporter.py
│   │   └── unreal_exporter.py
│   └── visualization/
│
├── config/
│   └── precision_config.yaml    # Configuration
│
└── examples/
    └── basic_usage.py
```

---

## Data Flow

```
Input Video
    ↓
[Preprocessing]
    ├── Frame extraction
    └── Court detection
    ↓
[Pass 1: Ensemble]
    ├── TrackNet → heatmap
    ├── YOLO v1 → bbox
    ├── YOLO v2 → bbox
    ├── SAMURAI → mask
    └── Faster R-CNN → bbox
    ↓
[Voting/Fusion]
    └── Spatial clustering + weighted average
    ↓
[Pass 2: Temporal]
    ├── Gap filling (interpolation)
    ├── Outlier correction
    └── Kalman smoothing
    ↓
[Pass 3: Physics]
    ├── Trajectory fitting
    ├── Bounce detection
    └── Physics validation
    ↓
[Pass 4: Verification]
    ├── ROI re-check
    └── Quality scoring
    ↓
[3D Reconstruction]
    ├── 2D→3D coordinate mapping
    └── Player pose 3D
    ↓
[Export]
    ├── CSV time series
    ├── Unity JSON
    └── Unreal JSON
```

---

## Integration with Existing Repos

Tennis30 leverages 4 existing repositories:

### **1. samurai/** - SAMURAI AI
- `sam2_video_predictor.py` - Zero-shot ball segmentation
- `kalman_filter.py` - Motion prediction
- Usage: Fallback when other models fail

### **2. tennis-tracking/** - TrackNet
- `Models/tracknet.py` - Heatmap-based ball detection
- `clf.pkl` - Bounce classifier (98% accuracy)
- Usage: Primary ball detector (best for blur)

### **3. Tennis-Analysis-System/** - YOLOv8 Analysis
- `trackers/ball_tracker.py` - Fine-tuned YOLO ball
- `trackers/player_tracker.py` - Player detection
- `court_line_detector/` - Court keypoints (ResNet50)
- Usage: Player pose + court geometry

### **4. ballradar/** - Physics Prediction
- `models/player_ball.py` - Set Transformer + Bi-LSTM
- `postprocessor.py` - Physics-based correction
- Usage: Trajectory prediction & validation

---

## Accuracy Targets

| Component | Target | Method |
|-----------|--------|--------|
| Ball Detection | 99% | 5-model ensemble |
| Ball 3D Position | ±2cm | Physics fitting |
| Pose Keypoints | 95% | 4-model fusion |
| Bounce Detection | 96% | ML + physics |
| Trajectory | Parabolic | Gravity + drag |

---

## Performance

### Processing Time
- **Fast Mode** (1 pass): ~5 min/match
- **Balanced Mode** (2 passes): ~10 min/match
- **Precision Mode** (4 passes): ~15-20 min/match

### Hardware Requirements
- **Minimum**: GTX 1080, 16GB RAM
- **Recommended**: RTX 3080, 32GB RAM
- **Optimal**: RTX 4090, 64GB RAM

### Output Size
- Ball CSV: ~2MB per 1000 frames
- Player CSV: ~5MB per 1000 frames (17 keypoints)
- Unity JSON: ~10MB per match
- Total: ~25-30MB per 10-minute match

---

## Configuration

See `config/precision_config.yaml` for all tunable parameters:

- Model weights & thresholds
- Temporal window size
- Physics constants
- Export formats
- Visualization options

---

## Extension Points

Tennis30 is designed for extensibility:

1. **Add new detection models**: Modify `ball_tracking/precision_tracker.py`
2. **Custom interpolation**: Add to `_temporal_interpolation()`
3. **Physics models**: Extend `physics_engine.py`
4. **Export formats**: Create new exporter in `utils/export/`
5. **Visualization**: Add to `utils/visualization/`

---

## Roadmap

**Phase 1** (Current): Core architecture
**Phase 2**: Model integration & testing
**Phase 3**: Physics refinement
**Phase 4**: Real-world validation
**Phase 5**: Production optimization
