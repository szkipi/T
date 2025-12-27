# Legacy Repositories - Reference

## ⚠️ Archív Információ

A Tennis30 projekt átstrukturálása során a következő legacy repositories-t eltávolítottuk a git repository-ból.
Ezek továbbra is elérhetők az eredeti forrásaikból, ha szükség lenne rájuk.

---

## 📦 Eltávolított Legacy Repositories

### 1. SAMURAI (34MB)
**Eredeti forrás**: https://github.com/yangchris11/samurai

**Leírás**: Segment Anything Model for Zero-Shot Visual Tracking

**Fő komponensek**:
- SAM 2.1 alapú zero-shot tracking
- Motion-aware memory bank
- Kalman filter integráció
- Video object segmentation

**Integráció állapota tennis30/-ban**:
- ⚠️ Placeholder implementáció
- TODO: SAMURAI wrapper implementálása
- Kalman filter már portolva: `tennis30/src/core/physics/kalman.py`

**Ha szükséges**:
```bash
# Clone újra, ha kell
git clone https://github.com/yangchris11/samurai.git external/samurai
```

---

### 2. tennis-tracking (130MB)
**Eredeti forrás**: https://github.com/ArtLabss/tennis-tracking

**Leírás**: TrackNet ball tracking + bounce detection

**Fő komponensek**:
- TrackNet model (VGG-16, heatmap-based)
- Bounce classifier (TimeSeriesForest, clf.pkl)
- Court detection (Hough transform)
- Player tracking (Faster R-CNN + SORT)

**Integráció állapota tennis30/-ban**:
- ⚠️ TrackNet: Placeholder (color-based fallback)
- ✅ Bounce detection: Lehet portolni clf.pkl-t
- ✅ Court detection: Base class kész
- ✅ SORT tracking: Algoritmus referencia

**Értékes fájlok**:
- `Models/tracknet.py` - TrackNet Keras model (TODO: PyTorch konverzió)
- `clf.pkl` - Bounce classifier (98% TN, 83% TP)
- `court_detector.py` - Hough transform court detection
- `sort.py` - SORT multi-object tracking

**Ha szükséges**:
```bash
git clone https://github.com/ArtLabss/tennis-tracking.git external/tennis-tracking
```

---

### 3. Tennis-Analysis-System (6.1MB)
**Eredeti forrás**: https://github.com/abdullahtarek/tennis_analysis

**Leírás**: YOLOv8 complete tennis analysis system

**Fő komponensek**:
- YOLOv8 ball tracking (fine-tuned)
- YOLOv8 player tracking
- Court line detection (ResNet50)
- Mini court visualization
- Player statistics

**Integráció állapota tennis30/-ban**:
- ✅ YOLOv8 ball tracker: Implementálva `yolo_tracker.py`
- ⚠️ Player tracking: TODO
- ⚠️ Court detection (ResNet50): TODO
- ⚠️ Mini court viz: TODO

**Értékes fájlok**:
- `trackers/ball_tracker.py` - YOLOv8 ball tracking (referencia)
- `trackers/player_tracker.py` - YOLOv8 player tracking
- `court_line_detector/court_line_detector.py` - ResNet50 court detector
- `mini_court/mini_court.py` - 2D minimap visualization

**Ha szükséges**:
```bash
git clone https://github.com/abdullahtarek/tennis_analysis.git external/Tennis-Analysis-System
```

---

### 4. ballradar (33MB)
**Eredeti forrás**: https://github.com/airalcorn2/ballradar

**Leírás**: Physics-based ball trajectory prediction (KDD 2023)

**Fő komponensek**:
- Set Transformer (permutation invariant)
- Hierarchical Bi-LSTM
- Ball Possessor Classifier
- Ball Trajectory Regressor
- Multi-agent context awareness

**Integráció állapota tennis30/-ban**:
- ✅ Physics engine: Implementálva `physics/engine.py`
- ⚠️ ML-based prediction: Referencia
- ✅ Trajectory fitting: Parabolic fit implementálva

**Értékes fájlok**:
- `models/player_ball.py` - Set Transformer + Bi-LSTM
- `postprocessor.py` - Physics-based correction
- `datatools/trace_animator.py` - Trajectory animation

**Ha szükséges**:
```bash
git clone https://github.com/airalcorn2/ballradar.git external/ballradar
```

---

## 🔧 Miért Távolítottuk El?

### Problémák a legacy repos-okkal:
1. **Méret**: 203MB összesen a git repo-ban
2. **Duplikáció**: Sok átfedő funkció
3. **Dependency konfliktusok**: 4 különböző requirements.txt
4. **Nehéz karbantartás**: Különböző coding style-ok
5. **Nem integrált**: Külön-külön működtek

### Előnyök az eltávolítás után:
1. **Git repo méret**: 203MB → ~500KB (40x kisebb!)
2. **Egyetlen source of truth**: `tennis30/` a fő projekt
3. **Modern architektúra**: Abstract base classes, plugin system
4. **Tesztelt kód**: 33 unit + integration test
5. **Könnyű karbantartás**: Moduláris, típusbiztos kód

---

## 📋 Mit Migráltunk tennis30/-ba?

### ✅ Sikeresen migrálva:
- **Ensemble fusion** algoritmus (spatial clustering + weighted voting)
- **YOLOv8 tracker** (Ultralytics integration)
- **Kalman filter** (8D state space, bidirectional smoothing)
- **Physics engine** (gravity, drag, bounce detection, trajectory fitting)
- **4-pass pipeline** (Ensemble → Temporal → Physics → Validation)
- **Abstract base classes** (BaseBallTracker, BaseCourtDetector, BasePoseEstimator)
- **Model registry** (plugin architecture)
- **Configuration system** (YAML with variable interpolation)

### ⚠️ TODO (Phase 3):
- TrackNet PyTorch konverzió
- SAMURAI wrapper
- Court detection implementációk (ResNet50, Hough)
- Player tracking & pose estimation
- Mini court visualization
- Export utilities finalizálása

---

## 🔗 Eredeti Repos Linkek

| Repo | GitHub URL | Stars | Használat |
|------|-----------|-------|-----------|
| **SAMURAI** | [yangchris11/samurai](https://github.com/yangchris11/samurai) | - | Zero-shot tracking |
| **tennis-tracking** | [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) | 500+ | TrackNet model |
| **Tennis-Analysis-System** | [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) | 300+ | YOLOv8 analysis |
| **ballradar** | [airalcorn2/ballradar](https://github.com/airalcorn2/ballradar) | 100+ | Physics prediction |

---

## 💡 Ha Szükség Van Rájuk

### Opció 1: Git Submodules (Ajánlott fejlesztéshez)
```bash
cd /home/user/T
mkdir external

# Add as submodules
git submodule add https://github.com/yangchris11/samurai.git external/samurai
git submodule add https://github.com/ArtLabss/tennis-tracking.git external/tennis-tracking
git submodule add https://github.com/abdullahtarek/tennis_analysis.git external/Tennis-Analysis-System
git submodule add https://github.com/airalcorn2/ballradar.git external/ballradar

# Update submodules
git submodule update --init --recursive
```

### Opció 2: Pip Install (Ajánlott production-höz)
```bash
# Ha ezek pip packageként elérhetők
pip install samurai-tracking
pip install tennis-tracking
```

### Opció 3: Manual Clone (Referenciának)
```bash
mkdir legacy
cd legacy
git clone https://github.com/yangchris11/samurai.git
git clone https://github.com/ArtLabss/tennis-tracking.git
git clone https://github.com/abdullahtarek/tennis_analysis.git Tennis-Analysis-System
git clone https://github.com/airalcorn2/ballradar.git

# Add to .gitignore
echo "legacy/" >> ../.gitignore
```

---

## 📊 Tárolási Összehasonlítás

### Előtte (Legacy benne):
```
Git repository: ~203MB
├── samurai/         34MB
├── tennis-tracking/ 130MB
├── ballradar/       33MB
├── Tennis-Analysis/ 6.1MB
└── Tennis30/        142K
```

### Utána (Optimalizált):
```
Git repository: ~500KB (csak forráskód)
└── tennis30/        400KB

Model weights (külön, gitignored):
└── data/models/     ~1.1GB (letöltendő külön)
```

**Megtakarítás**: 99.75% (203MB → 500KB)

---

**Készítette**: Tennis30 refactoring, Phase 2
**Dátum**: 2025-12-27
**Verzió**: 1.0
