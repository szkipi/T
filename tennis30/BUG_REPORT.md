# Tennis30 Implementation Issues - Bug Report

## 🐛 CRITICAL BUGS FOUND

### BUG #1: Configuration Path Error in EnsembleBallTracker
**File**: `src/core/tracking/ball/ensemble.py:43-44`
**Severity**: HIGH - Will crash at runtime

**Problem**:
```python
# CURRENT (BROKEN):
self.spatial_radius = config.get("spatial_clustering", {}).get("radius", 10)
self.min_models = config.get("spatial_clustering", {}).get("min_models", 2)
```

The `config` parameter is the `ball_tracking` block, but `spatial_clustering` is nested under `ensemble`:
```yaml
ball_tracking:
  ensemble:
    spatial_clustering:
      radius: 10
```

**Fix Required**:
```python
# CORRECT:
ensemble_config = config.get("ensemble", {})
self.spatial_radius = ensemble_config.get("spatial_clustering", {}).get("radius", 10)
self.min_models = ensemble_config.get("spatial_clustering", {}).get("min_models", 2)
```

**Impact**:
- `spatial_clustering` will always be empty dict
- Will use default values (10, 2) instead of config values
- Config changes to clustering will be ignored

---

### BUG #2: Import Path Issues (Potential)
**Files**: All module imports
**Severity**: MEDIUM - May cause import errors

**Problem**:
Using absolute imports like:
```python
from tennis30.src.core.tracking.ball.base_tracker import BaseBallTracker
```

But `tennis30` is not a package (no `tennis30/__init__.py`), only `tennis30/src/` is.

**Current State**:
- Directory: `tennis30/src/` (has `__init__.py`)
- But imports use `tennis30.src.*`

**Fix Options**:
1. Add `tennis30/__init__.py` (empty file)
2. Use relative imports: `from .base_tracker import BaseBallTracker`
3. Install as package: `pip install -e .`

**Impact**:
- Will work with `pip install -e .`
- Won't work with direct Python execution
- May confuse IDE autocomplete

---

### BUG #3: Missing scipy.signal dependency check
**File**: `src/core/physics/engine.py:5`
**Severity**: LOW

**Problem**:
```python
from scipy.signal import find_peaks
```

`find_peaks` used in `_detect_bounces()` but scipy is imported. No issue if scipy installed, but good to have fallback.

**Status**: Not a bug, scipy is in pyproject.toml ✓

---

### BUG #4: Edge Case - Empty clusters in ensemble fusion
**File**: `src/core/tracking/ball/ensemble.py:140-166`
**Severity**: MEDIUM

**Problem**:
```python
for i in range(len(detections)):
    if i in used:
        continue

    cluster_indices = np.where(distances[i] <= self.spatial_radius)[0]
    cluster_indices = [idx for idx in cluster_indices if idx not in used]

    if len(cluster_indices) >= self.min_models:
        clusters.append(cluster_indices)
        used.update(cluster_indices)
```

**Edge case**: If `min_models=2` but only 1 model detects, no clusters created → fallback to highest weight.

**Status**: Actually handled correctly with fallback at line 168-175 ✓

---

### BUG #5: NaN handling in Kalman smoother
**File**: `src/core/physics/kalman.py:210-225`
**Severity**: LOW - Edge case

**Problem**:
```python
if not np.isnan(measurement).any():
    kf.initiate(measurement)
    initialized = True
```

If first N frames are all NaN, filter never initializes. Then `kf.get_position()` returns `None`, and we append `np.array([np.nan, np.nan])`.

**Status**: Handled correctly, but could optimize by skipping leading NaNs.

---

### BUG #6: Type inconsistency - tuple vs list
**File**: `src/core/tracking/ball/ensemble.py:179`
**Severity**: LOW - Type hint issue

**Problem**:
```python
"position": tuple(centroid),  # Returns tuple
```

But base_tracker expects `Tuple[float, float]` and some places may expect `(int, int)`.

**Status**: centroid is `np.ndarray[float]`, converting to `tuple` is correct. No issue.

---

### BUG #7: Division by zero in weighted centroid
**File**: `src/core/tracking/ball/ensemble.py:172-178`
**Severity**: MEDIUM

**Problem**:
```python
combined_weights = cluster_weights * cluster_confidences
total_weight = combined_weights.sum()

if total_weight == 0:
    # Fallback to simple average
    centroid = cluster_positions.mean(axis=0)
    avg_confidence = cluster_confidences.mean()
else:
    # Weighted centroid
    centroid = (cluster_positions * combined_weights[:, np.newaxis]).sum(axis=0) / total_weight
```

**Edge case**: What if all confidences are 0? Then `total_weight = 0` → fallback to mean.

**Status**: Handled correctly with fallback ✓

---

### BUG #8: Missing cv2 import check
**File**: `src/pipeline/precision.py:17`
**Severity**: LOW

**Problem**:
```python
import cv2
```

No try/except. If opencv not installed, import fails.

**Status**: opencv-python in pyproject.toml, so OK. But good practice to check.

---

### BUG #9: Video loading assumes BGR format
**File**: `src/pipeline/precision.py:262-273`
**Severity**: LOW - Documentation issue

**Problem**:
```python
def _load_video(self, video_path: str) -> List[np.ndarray]:
    """Load video frames.

    Returns:
        List of frames (BGR format)
    """
```

OpenCV's `cv2.VideoCapture` returns BGR by default, but not documented in function signature.

**Status**: Documented correctly in docstring ✓

---

### BUG #10: Missing model availability check
**File**: `src/core/tracking/ball/yolo_tracker.py:63-68`
**Severity**: HIGH - Will crash if model not downloaded

**Problem**:
```python
if not model_path.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {model_path}\n"
        f"Download with: python scripts/download_models.py yolo_v1"
    )
```

Good error message, but no graceful degradation. If user doesn't download models, ensemble will fail.

**Status**: Intentional behavior - models are required. Error message is helpful ✓

---

## ✅ THINGS THAT ARE CORRECT

1. ✓ Syntax is valid (all files compile)
2. ✓ Type hints are comprehensive
3. ✓ Docstrings are detailed
4. ✓ scipy is in dependencies
5. ✓ Edge cases mostly handled (NaN, empty arrays, division by zero)
6. ✓ Error messages are helpful
7. ✓ Fallback mechanisms in place

---

## 🔧 REQUIRED FIXES

### Priority 1 (CRITICAL - Fix immediately):
1. **Fix BUG #1**: Configuration path in EnsembleBallTracker

### Priority 2 (MEDIUM - Fix soon):
2. **Fix BUG #2**: Add `tennis30/__init__.py` or document installation requirement

### Priority 3 (LOW - Nice to have):
3. Document that `pip install -e .` is required for imports
4. Add integration tests to catch these issues

---

## 📝 RECOMMENDATIONS

1. **Add integration test** that loads config and initializes EnsembleBallTracker
2. **Add `tennis30/__init__.py`** to make it a proper package
3. **Document installation** in README: `cd tennis30 && pip install -e .`
4. **Add CI/CD** to run syntax checks and import tests
5. **Type checking** with mypy to catch more issues

---

Generated: 2025-12-27
Checked files: 6 core implementation files
Total issues found: 10 (1 critical, 2 medium, 7 low/info)
