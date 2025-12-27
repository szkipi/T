# 🚀 Tennis30 - Gyors Teszt Feltöltött File-okkal

## 📤 Hogyan Tesztelhetsz Feltöltött File-okat

### 1. Tölts Fel Egy Tennis Képet vagy Videót

Claude Code-ban fel tudsz tölteni file-okat a beszélgetésbe. Támogatott formátumok:

**Képek:**
- `.jpg`, `.jpeg`, `.png`
- Tennis frame/frame egy labdával

**Videók:**
- `.mp4`, `.avi`, `.mov`
- Tennis mérkőzés felvétel

### 2. Futtasd a Teszt Scriptet

```bash
# Kép tesztelése
python test_upload.py /path/to/uploaded/tennis_frame.jpg

# Videó tesztelése (első 30 frame)
python test_upload.py /path/to/uploaded/tennis_video.mp4
```

## 🎯 Mit Csinál a Script?

### A) Függőségek Ellenőrzése

```
🔍 Checking dependencies...
  ✓ NumPy
  ✓ OpenCV
  ✓ PyYAML
```

### B) Ha Van NumPy/OpenCV - Ensemble Detection

**Single Frame (Kép):**
```
📸 Testing single frame: tennis_frame.jpg

✓ Image loaded: 1920x1080
🔧 Initializing ensemble tracker...
🎾 Detecting ball...

============================================================
DETECTION RESULTS:
============================================================
Position: [845, 432]
Confidence: 94.5%
Bounding box: [835, 422, 855, 442]
Models voted: 3
============================================================

💾 Visualization saved: tennis_frame_detection.jpg
```

**Video:**
```
🎬 Testing video: tennis_match.mp4

✓ Video info: 1920x1080, 30.0 FPS, 900 frames
  Processing first 30 frames...

🎾 Tracking ball...

  Frame   0: ✓ conf=92.3% pos=[845, 432]
  Frame  10: ✓ conf=88.7% pos=[892, 456]
  Frame  20: ✗ conf=12.1% pos=None
  Frame  30: ✓ conf=95.2% pos=[754, 398]

============================================================
TRACKING SUMMARY:
============================================================
Frames processed: 30
Ball detected: 27 / 30 (90.0%)
Average confidence: 86.4%
============================================================
```

### C) Ha NINCS NumPy/OpenCV - Simple Color Detection

Ha nincsenek telepítve a függőségek, fallback módban fut:

```
⚠️  Missing dependencies. Running simple color-based test only.

🎾 Simple color-based test: tennis_frame.jpg

✓ Image loaded: 1920x1080
🔍 Found 3 yellow-green objects

🎾 Potential ball detected!
   Position: (845, 432)
   Area: 156 pixels

💾 Visualization saved: tennis_frame_simple_detection.jpg
```

## 📊 Output File-ok

A script automatikusan létrehoz vizualizációkat:

- `tennis_frame_detection.jpg` - Ensemble detection eredmény
- `tennis_frame_simple_detection.jpg` - Egyszerű color-based detection

## 🎾 Példa Használat Teljes Flow

```bash
# 1. Navigálj a tennis30 könyvtárba
cd /home/user/T/tennis30

# 2. Tölts fel egy képet a Claude Code-ba (pl. "my_tennis_shot.jpg")
# A file path lesz valami ilyesmi: /tmp/upload_xxxxx/my_tennis_shot.jpg

# 3. Futtasd a tesztet
python test_upload.py /tmp/upload_xxxxx/my_tennis_shot.jpg

# 4. Nézd meg az eredményt
# Az eredmény file: /tmp/upload_xxxxx/my_tennis_shot_detection.jpg
```

## 🔧 Teljes Tennis30 Használat

Ha minden függőség telepítve van:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/user/T/tennis30/src')

from pipeline.precision import PrecisionPipeline

# Full 4-pass precision pipeline
pipeline = PrecisionPipeline.from_config(
    config_path="/home/user/T/tennis30/config/default.yaml",
    device="cpu"  # vagy "cuda" ha van GPU
)

results = pipeline.process_video(
    video_path="/path/to/uploaded/tennis_video.mp4",
    output_dir="./output",
    visualize=True
)

print(f"Accuracy: {results.get('accuracy', 0):.1%}")
print(f"Output: {results['output_path']}")
```

## 💡 Tippek

1. **Legjobb eredményhez:** Világos, éles képek, ahol a labda látható
2. **Videóknál:** A script csak az első 30 frame-et dolgozza fel alapból (gyorsaság miatt)
3. **Ha nincs labda:** A script jelezni fogja, hogy nem talált semmit
4. **Függőségek:** Telepítsd őket a legjobb eredményért:
   ```bash
   pip install numpy opencv-python pyyaml
   ```

## 🚨 Troubleshooting

**"No module named 'numpy'"**
```bash
pip install numpy opencv-python pyyaml
```

**"Cannot read image"**
- Ellenőrizd a file path-ot
- Próbálj abszolút path-ot használni

**"No tennis ball detected"**
- A kép minősége lehet gyenge
- A labda lehet túl kicsi vagy elhomályosított
- Próbáld a simple color-based módot is

---

**Készen állsz?** Tölts fel egy tennis képet vagy videót, és teszteljük! 🎾
