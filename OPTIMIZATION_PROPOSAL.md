# Tennis30 - Struktúra Optimalizálási Javaslat

## ❌ JELENLEGI PROBLÉMÁK

### 1. **Redundancia és Átfedés**

```
Probléma: 5 különböző repo összevonva, sok duplikáció

Példák:
├── tennis-tracking/court_detector.py       }
├── Tennis-Analysis-System/court_line_detector/  }  → 3x court detection!
└── Tennis30/core/court_detection/          }

├── tennis-tracking/detection.py (Faster R-CNN)  }
├── Tennis-Analysis-System/trackers/ball_tracker.py (YOLO)  }  → Ball tracking duplikáció
└── Tennis30/core/ball_tracking/precision_tracker.py       }

Hatás:
  - Kód duplikáció
  - Nehéz karbantartás
  - Verziózási problémák
```

### 2. **Dependency Management Káosz**

```
4 különböző requirements.txt:

tennis-tracking/requirements.txt:
  - tensorflow==2.3.0  (régi!)
  - opencv-python==4.1.1

Tennis-Analysis-System/requirements.txt:
  - ultralytics (YOLOv8)
  - opencv-python (verzió nincs megadva!)

ballradar/requirements.txt:
  - torch==1.9.0
  - pytorch-lightning

Tennis30/requirements.txt:
  - ultralytics
  - torch (verzió nincs megadva!)
  - samurai imports (????)

Probléma:
  ❌ Verzió konfliktusok (TensorFlow vs PyTorch)
  ❌ Nincs egységes environment
  ❌ Függőségek nincsenek pinelve
  ❌ Circular imports lehetségesek
```

### 3. **Névkonvenció Inkonzisztencia**

```
tennis-tracking/           → snake_case with dash
Tennis-Analysis-System/    → PascalCase with dash
ballradar/                 → lowercase
samurai/                   → lowercase
Tennis30/                  → PascalCase + number

Python modulok:
  - court_detector.py
  - CourtLineDetector (osztály)
  - ball_tracking (directory)
  - BallTracker (osztály)

→ Nincs egységes naming convention!
```

### 4. **Model Weights Management Hiányzik**

```
Jelenleg:
  models/
    └── ??? (nincs!)

Szükséges súlyok:
  - tracknet_best.pth         (~50MB)
  - yolov8n_tennis_v1.pt      (~6MB)
  - yolov8n_tennis_v2.pt      (~6MB)
  - sam2.1_hiera_large.pt     (~800MB!)
  - fasterrcnn_resnet50.pth   (~160MB)

Probléma:
  ❌ Súlyok nincsenek verziókezelve
  ❌ Nincs letöltési script
  ❌ Git-ben tárolni óriási
  ❌ Nincs model registry
```

### 5. **Konfiguráció Szétszórtság**

```
Jelenleg:
├── Tennis30/config/precision_config.yaml
├── Tennis-Analysis-System/input/input_video.mp4 (???)
├── ballradar/configs/ (???)
└── tennis-tracking/constants.py

Probléma:
  - Több config formátum (.yaml, .py constants)
  - Nem központosított
  - Path-ek hardcode-olva
  - Nincs environment-based config
```

### 6. **Dokumentáció Fragmentáció**

```
├── MODULES_OVERVIEW.md                    (root)
├── ARCHITECTURE_VISUALIZATION.md          (root)
├── Tennis30/README.md                     (local)
├── Tennis30/docs/ENSEMBLE_FUSION_DETAILS.md
├── tennis-tracking/README.md
├── Tennis-Analysis-System/README.md
├── ballradar/README.md
└── samurai/README.md

→ 8 README fájl!
→ Nem egyértelmű, hol kezdj
```

### 7. **Testing Hiányzik**

```
find . -name "*test*.py" -o -name "test_*"
  → 0 találat!

Probléma:
  ❌ Nincs unit test
  ❌ Nincs integration test
  ❌ Nincs CI/CD pipeline
  ❌ Manuális tesztelés
```

### 8. **Méret Probléma**

```
Jelenlegi méret:
tennis-tracking/    130M  ← Miért ekkora?
samurai/             34M
ballradar/           33M
Tennis-Analysis-System/  6.1M
Tennis30/           142K
Total:              203M

Probléma:
  - Model súlyok a repo-ban?
  - Videó fájlok a git-ben?
  - Nincs .gitignore optimalizálva
```

---

## ✅ JAVASOLT OPTIMALIZÁLT STRUKTÚRA

### 🏗️ Új Könyvtárstruktúra

```
tennis30-precision/
│
├── config/                          # ✨ Centralized configuration
│   ├── default.yaml                 # Default settings
│   ├── development.yaml             # Dev overrides
│   ├── production.yaml              # Prod settings
│   └── models.yaml                  # Model paths & weights config
│
├── data/                            # 🗃️ Data management
│   ├── raw/                         # Raw videos
│   ├── processed/                   # Processed outputs
│   ├── models/                      # Model weights (gitignored!)
│   │   ├── download_models.sh      # Auto-download script
│   │   └── .gitkeep
│   └── samples/                     # Sample test data
│
├── src/                             # 📦 Main source code
│   ├── __init__.py
│   │
│   ├── core/                        # Core functionality
│   │   ├── __init__.py
│   │   ├── court/                   # Court detection (unified!)
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract base class
│   │   │   ├── resnet_detector.py  # From Tennis-Analysis-System
│   │   │   ├── hough_detector.py   # From tennis-tracking
│   │   │   └── mapper_3d.py        # 2D→3D conversion
│   │   │
│   │   ├── tracking/                # Ball & player tracking
│   │   │   ├── __init__.py
│   │   │   ├── ball/
│   │   │   │   ├── base_tracker.py
│   │   │   │   ├── tracknet.py     # TrackNet model
│   │   │   │   ├── yolo.py         # YOLOv8 variants
│   │   │   │   ├── samurai.py      # SAMURAI wrapper
│   │   │   │   ├── rcnn.py         # Faster R-CNN
│   │   │   │   └── ensemble.py     # Ensemble fusion
│   │   │   │
│   │   │   └── player/
│   │   │       ├── detector.py     # Player detection
│   │   │       └── sort.py         # SORT tracker
│   │   │
│   │   ├── pose/                    # Pose estimation
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── yolo_pose.py
│   │   │   ├── mediapipe_pose.py
│   │   │   └── ensemble_pose.py
│   │   │
│   │   ├── physics/                 # Physics & prediction
│   │   │   ├── __init__.py
│   │   │   ├── kalman.py           # Kalman filter
│   │   │   ├── trajectory.py       # Trajectory fitting
│   │   │   ├── bounce.py           # Bounce detection
│   │   │   └── ballradar.py        # BallRadar integration
│   │   │
│   │   └── game_state/              # Game state tracking
│   │       ├── __init__.py
│   │       ├── rally.py
│   │       └── score.py
│   │
│   ├── pipeline/                    # 🔄 Processing pipelines
│   │   ├── __init__.py
│   │   ├── precision.py            # 4-pass precision pipeline
│   │   ├── realtime.py             # Fast pipeline (future)
│   │   └── batch.py                # Batch processing
│   │
│   ├── models/                      # 🧠 Model definitions
│   │   ├── __init__.py
│   │   ├── registry.py             # Model registry & loader
│   │   ├── tracknet/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   ├── samurai/
│   │   │   └── (SAMURAI integration)
│   │   └── ballradar/
│   │       └── (BallRadar integration)
│   │
│   ├── utils/                       # 🛠️ Utilities
│   │   ├── __init__.py
│   │   ├── video.py                # Video I/O
│   │   ├── visualization.py        # Drawing & rendering
│   │   ├── geometry.py             # 2D/3D math
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── export.py               # CSV, Unity, Unreal export
│   │
│   └── cli/                         # 💻 Command-line interface
│       ├── __init__.py
│       ├── main.py                 # Main CLI entry
│       ├── train.py                # Training commands
│       └── evaluate.py             # Evaluation commands
│
├── tests/                           # ✅ Testing
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_court.py
│   │   ├── test_tracking.py
│   │   ├── test_physics.py
│   │   └── test_export.py
│   ├── integration/                 # Integration tests
│   │   ├── test_pipeline.py
│   │   └── test_ensemble.py
│   ├── fixtures/                    # Test data
│   │   ├── sample_frame.jpg
│   │   └── sample_video.mp4
│   └── conftest.py                  # Pytest config
│
├── notebooks/                       # 📓 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_benchmarks.ipynb
│   └── 03_visualization_demo.ipynb
│
├── scripts/                         # 🔧 Utility scripts
│   ├── download_models.py          # Download pretrained weights
│   ├── convert_format.py           # Data format conversion
│   └── benchmark.py                # Performance benchmarking
│
├── docs/                            # 📚 Documentation
│   ├── index.md                    # Main documentation
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture/
│   │   ├── overview.md             # System architecture
│   │   ├── ensemble_voting.md     # Voting mechanism
│   │   └── models.md               # Individual models
│   ├── api/                        # API documentation
│   │   ├── core.md
│   │   ├── pipeline.md
│   │   └── utils.md
│   └── tutorials/
│       ├── basic_tracking.md
│       ├── custom_model.md
│       └── unity_integration.md
│
├── .github/                         # 🤖 CI/CD
│   └── workflows/
│       ├── tests.yml               # Run tests on PR
│       ├── lint.yml                # Code quality
│       └── release.yml             # Auto release
│
├── .gitignore                       # Git ignore rules
├── .gitattributes                   # LFS for large files
├── README.md                        # Main README
├── CONTRIBUTING.md                  # Contribution guide
├── LICENSE                          # License file
│
├── requirements.txt                 # 📦 Production dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
├── pyproject.toml                   # Modern Python config
└── Dockerfile                       # Docker containerization
```

---

## 🔧 KONKRÉT OPTIMALIZÁLÁSI LÉPÉSEK

### 1. **Dependency Management - Poetry**

**requirements.txt → pyproject.toml (Poetry)**

```toml
[tool.poetry]
name = "tennis30"
version = "1.0.0"
description = "Multi-Pass Precision Tennis Tracking"

[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.1.0"
torchvision = "^0.16.0"
opencv-python = "^4.8.0"
numpy = "^1.24.0"
ultralytics = "^8.0.0"
pyyaml = "^6.0"
scipy = "^1.11.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.0.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"

[tool.poetry.scripts]
tennis30 = "tennis30.cli.main:app"
```

**Előnyök:**
- ✅ Dependency resolution automatikus
- ✅ Lock file (reproducible builds)
- ✅ Virtual environment kezelés
- ✅ Dev dependencies elkülönítve

### 2. **Model Weights - Git LFS + Download Script**

**.gitattributes:**
```
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
```

**scripts/download_models.py:**
```python
import requests
from pathlib import Path
from tqdm import tqdm

MODEL_REGISTRY = {
    'tracknet': {
        'url': 'https://example.com/tracknet_best.pth',
        'size': '50MB',
        'checksum': 'abc123...'
    },
    'yolov8_v1': {
        'url': 'https://example.com/yolov8n_tennis_v1.pt',
        'size': '6MB',
        'checksum': 'def456...'
    },
    'sam2.1': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_large.pt',
        'size': '800MB',
        'checksum': 'ghi789...'
    }
}

def download_model(model_name, force=False):
    """Download model weights with progress bar"""
    config = MODEL_REGISTRY[model_name]
    output_path = Path('data/models') / f"{model_name}.pt"

    if output_path.exists() and not force:
        print(f"✓ {model_name} already downloaded")
        return

    print(f"⬇️  Downloading {model_name} ({config['size']})...")
    # Download with progress bar
    # Verify checksum
    # ...
```

**Használat:**
```bash
python scripts/download_models.py --all
python scripts/download_models.py tracknet yolov8_v1
```

### 3. **Centralizált Config - Hydra**

**config/default.yaml:**
```yaml
defaults:
  - models: ensemble
  - pipeline: precision
  - _self_

project:
  name: tennis30
  version: 1.0.0

paths:
  data_root: ${oc.env:DATA_ROOT,./data}
  models_dir: ${paths.data_root}/models
  output_dir: ${paths.data_root}/processed

video:
  fps: 30
  resolution: [1920, 1080]

# Hydra composable configs
```

**config/models/ensemble.yaml:**
```yaml
ball_tracking:
  ensemble:
    tracknet:
      weight: 0.30
      checkpoint: ${paths.models_dir}/tracknet_best.pth
    yolo_v1:
      weight: 0.25
      checkpoint: ${paths.models_dir}/yolov8n_tennis_v1.pt
    # ...
```

**Használat kódból:**
```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    tracker = EnsembleTracker(cfg.ball_tracking.ensemble)
    # ...
```

### 4. **Abstract Base Classes - Pluggable Architecture**

**src/core/tracking/ball/base_tracker.py:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseBallTracker(ABC):
    """Abstract base class for ball trackers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weight = config.get('weight', 1.0)

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect ball in single frame

        Returns:
            {
                'position': (x, y),
                'confidence': float,
                'bbox': (x1, y1, x2, y2),  # optional
                'mask': np.ndarray          # optional
            }
        """
        pass

    @abstractmethod
    def load_model(self, checkpoint_path: str):
        """Load pretrained weights"""
        pass

class TrackNetTracker(BaseBallTracker):
    def detect(self, frame):
        # TrackNet specific implementation
        ...

class YOLOTracker(BaseBallTracker):
    def detect(self, frame):
        # YOLO specific implementation
        ...
```

**Előnyök:**
- ✅ Egyszerű új modellek hozzáadása
- ✅ Egységes interface
- ✅ Type safety (mypy)
- ✅ Tesztelhetőség (mocking)

### 5. **Model Registry Pattern**

**src/models/registry.py:**
```python
from typing import Dict, Type
from src.core.tracking.ball.base_tracker import BaseBallTracker

class ModelRegistry:
    """Central registry for all models"""

    _ball_trackers: Dict[str, Type[BaseBallTracker]] = {}

    @classmethod
    def register_ball_tracker(cls, name: str):
        def decorator(tracker_class):
            cls._ball_trackers[name] = tracker_class
            return tracker_class
        return decorator

    @classmethod
    def get_ball_tracker(cls, name: str, config: dict) -> BaseBallTracker:
        if name not in cls._ball_trackers:
            raise ValueError(f"Unknown tracker: {name}")
        return cls._ball_trackers[name](config)

# Usage:
@ModelRegistry.register_ball_tracker('tracknet')
class TrackNetTracker(BaseBallTracker):
    ...

# Loading:
tracker = ModelRegistry.get_ball_tracker('tracknet', config)
```

### 6. **Testing Infrastructure**

**tests/unit/test_ensemble.py:**
```python
import pytest
from src.core.tracking.ball.ensemble import EnsembleBallTracker
from src.core.tracking.ball.base_tracker import BaseBallTracker

class MockTracker(BaseBallTracker):
    def __init__(self, config, return_value):
        super().__init__(config)
        self.return_value = return_value

    def detect(self, frame):
        return self.return_value

    def load_model(self, checkpoint_path):
        pass

def test_ensemble_spatial_clustering():
    """Test spatial clustering with known detections"""
    config = {
        'spatial_clustering': {'radius': 10},
        'trackers': [
            {'name': 'tracker1', 'weight': 0.5},
            {'name': 'tracker2', 'weight': 0.5}
        ]
    }

    ensemble = EnsembleBallTracker(config)

    # Mock trackers
    ensemble.trackers = [
        MockTracker({'weight': 0.5}, {'position': (100, 200), 'confidence': 0.9}),
        MockTracker({'weight': 0.5}, {'position': (102, 198), 'confidence': 0.8})
    ]

    result = ensemble.detect(None)

    # Assert weighted centroid
    assert result['position'][0] == pytest.approx(101.0, abs=0.1)
    assert result['position'][1] == pytest.approx(199.0, abs=0.1)
    assert result['confidence'] == pytest.approx(0.85, abs=0.01)
```

**CI/CD - GitHub Actions (.github/workflows/tests.yml):**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run tests
        run: |
          poetry run pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 7. **CLI Interface - Typer**

**src/cli/main.py:**
```python
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def track(
    video: Path = typer.Argument(..., help="Input video path"),
    output: Path = typer.Option("./output", help="Output directory"),
    config: Path = typer.Option(None, help="Custom config file"),
    passes: int = typer.Option(4, help="Number of refinement passes"),
    device: str = typer.Option("cuda", help="Device (cuda/cpu)")
):
    """Run precision ball tracking on video"""
    from src.pipeline.precision import PrecisionPipeline

    pipeline = PrecisionPipeline.from_config(config, device=device)
    results = pipeline.process_video(video, num_passes=passes)
    pipeline.export(results, output)

    typer.secho(f"✓ Tracking complete! Results: {output}", fg=typer.colors.GREEN)

@app.command()
def benchmark(
    video: Path,
    ground_truth: Path,
    models: str = typer.Option("all", help="Models to benchmark")
):
    """Benchmark models against ground truth"""
    ...

if __name__ == "__main__":
    app()
```

**Használat:**
```bash
tennis30 track video.mp4 --passes 4 --device cuda
tennis30 benchmark test_video.mp4 ground_truth.csv
tennis30 train tracknet --epochs 50
```

### 8. **Docker Containerization**

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application
COPY . .

# Download models
RUN python scripts/download_models.py --all

ENTRYPOINT ["tennis30"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  tennis30:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Használat:**
```bash
docker-compose run tennis30 track /app/data/video.mp4
```

---

## 📊 ELŐNYÖK ÖSSZEFOGLALÁS

### Jelenlegi vs Optimalizált

| Aspektus | Jelenlegi ❌ | Optimalizált ✅ |
|----------|-------------|----------------|
| **Struktúra** | 5 külön repo összevonva | Egységes, hierarchikus |
| **Dependencies** | 4 requirements.txt, konfliktusok | Poetry, locked dependencies |
| **Config** | Szétszórt .yaml és .py | Hydra, centralizált |
| **Model weights** | Nincs kezelve | Git LFS + download script |
| **Code reuse** | Sok duplikáció | Abstract base classes |
| **Testing** | 0 test | Unit + integration + CI/CD |
| **Documentation** | 8 README | Egyetlen docs/ struktúra |
| **CLI** | Nincs | Typer-based professzionális CLI |
| **Deployment** | Manual | Docker + docker-compose |
| **Extensibility** | Nehéz új modell | Plugin architecture |
| **Naming** | Inkonzisztens | PEP8 + egységes |

---

## 🚀 IMPLEMENTÁCIÓS TERV

### Fázis 1: Foundation (1-2 nap)
1. ✅ Új könyvtárstruktúra létrehozása
2. ✅ Poetry setup (pyproject.toml)
3. ✅ Base classes definiálása
4. ✅ Config rendszer (Hydra)
5. ✅ Git LFS setup

### Fázis 2: Migration (2-3 nap)
1. ✅ Kód áthelyezés új struktúrába
2. ✅ Import path-ek frissítése
3. ✅ Model registry implementálás
4. ✅ CLI létrehozása
5. ✅ Requirements consolidation

### Fázis 3: Testing (1-2 nap)
1. ✅ Unit tests írása
2. ✅ Integration tests
3. ✅ CI/CD pipeline setup
4. ✅ Code coverage mérés

### Fázis 4: Documentation (1 nap)
1. ✅ docs/ struktúra
2. ✅ API documentation
3. ✅ Tutorial-ok
4. ✅ README frissítés

### Fázis 5: Deployment (1 nap)
1. ✅ Dockerfile
2. ✅ Docker compose
3. ✅ Model download script
4. ✅ Release automation

**Teljes idő: ~7-10 nap**

---

## 💡 KÖVETKEZŐ LÉPÉSEK

Szeretnéd, hogy:

1. **Elkezdjem a refactor-t?** → Fázis 1-től indulunk
2. **Részletes migráció terv?** → Lépésről lépésre guide
3. **Prototípus egy részhez?** → Pl. csak a ball tracking modul
4. **Keep as-is?** → Akkor csak dokumentálom a problémákat

**Ajánlásom: Érdemes refactor-ra időt szánni most, hosszútávon sokat spórolsz vele!**
