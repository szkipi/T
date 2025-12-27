# Tennis30 Ensemble Fusion - Részletes Arányok és Működés

## 📊 Jelenlegi Setup - Modell Súlyok (Weights)

### **Ball Tracking Ensemble**

| Modell | Súly (Weight) | Threshold | Státusz | Leírás |
|--------|---------------|-----------|---------|--------|
| **TrackNet** | **30%** (0.30) | 0.5 | ✅ Enabled | Heatmap-based, legjobb motion blur esetén |
| **YOLOv8 v1** | **25%** (0.25) | 0.3 | ⚠️ Disabled | Fine-tuned verzió #1 (még nincs betanítva) |
| **YOLOv8 v2** | **25%** (0.25) | 0.3 | ⚠️ Disabled | Fine-tuned verzió #2 (még nincs betanítva) |
| **SAMURAI** | **15%** (0.15) | 0.4 | ⚠️ Disabled | Zero-shot segmentation (integráció folyamatban) |
| **Faster R-CNN** | **5%** (0.05) | 0.4 | ⚠️ Disabled | Alternatív detektor |
| **ÖSSZESEN** | **100%** (1.00) | - | - | - |

---

## 🔄 Ensemble Fusion Algoritmus

### **1. Adatgyűjtés (Minden Frame-re)**

```python
# Minden modell fut MINDEN frame-en
for frame in frames:
    candidates = []

    # TrackNet detekció
    tracknet_pos = tracknet.predict(frame)
    if tracknet_pos:
        candidates.append({
            'pos': tracknet_pos,
            'model': 'tracknet',
            'weight': 0.30,
            'confidence': tracknet_pos.confidence
        })

    # YOLO v1 detekció
    yolo1_pos = yolo_v1.detect(frame)
    if yolo1_pos:
        candidates.append({
            'pos': yolo1_pos,
            'model': 'yolo_v1',
            'weight': 0.25,
            'confidence': yolo1_pos.confidence
        })

    # ... ugyanígy a többi modell
```

---

### **2. Spatial Clustering (Térbeli Csoportosítás)**

Ha több modell is talált valamit, csoportosítjuk a közeli detekciókat:

```python
def spatial_clustering(candidates, radius=10.0):
    """
    Közeli detekciók (10 pixel radius) egy klaszterbe kerülnek
    """
    clusters = []

    for candidate in candidates:
        x, y = candidate['pos']['x'], candidate['pos']['y']

        # Keresünk létező klasztert
        added = False
        for cluster in clusters:
            cx, cy = cluster['centroid']
            distance = sqrt((x - cx)² + (y - cy)²)

            if distance < 10:  # 10 pixel radius
                # Hozzáadjuk a klaszterhez
                cluster['members'].append(candidate)
                cluster['total_weight'] += candidate['weight']

                # Újraszámoljuk a centroidot (súlyozott átlag)
                cluster['centroid'] = weighted_average(cluster['members'])
                added = True
                break

        if not added:
            # Új klaszter létrehozása
            clusters.append({
                'centroid': (x, y),
                'members': [candidate],
                'total_weight': candidate['weight']
            })

    return clusters
```

---

### **3. Voting/Averaging - Konkrét Példák**

#### **Példa 1: Minden modell egyetért (ideális eset)**

```
Frame #100:
├─ TrackNet:     x=640, y=360  (confidence=0.95, weight=0.30)
├─ YOLO v1:      x=642, y=362  (confidence=0.88, weight=0.25)
├─ YOLO v2:      x=641, y=361  (confidence=0.91, weight=0.25)
├─ SAMURAI:      x=639, y=359  (confidence=0.85, weight=0.15)
└─ Faster R-CNN: x=643, y=363  (confidence=0.78, weight=0.05)

Clustering:
  - Mind az 5 detekció < 10 pixel távolságra
  - 1 klaszter keletkezik

Weighted Average:
  x = (640*0.30 + 642*0.25 + 641*0.25 + 639*0.15 + 643*0.05) / 1.0
    = (192.0 + 160.5 + 160.25 + 95.85 + 32.15) / 1.0
    = 640.75

  y = (360*0.30 + 362*0.25 + 361*0.25 + 359*0.15 + 363*0.05) / 1.0
    = 360.75

Final Position: (640.75, 360.75)
Confidence: 1.0 (100% - mind az 5 modell egyetért)
```

---

#### **Példa 2: Split vote (2 klaszter)**

```
Frame #200:
├─ TrackNet:     x=500, y=400  (weight=0.30)
├─ YOLO v1:      x=505, y=405  (weight=0.25)
├─ YOLO v2:      x=650, y=380  (weight=0.25)  ← Outlier!
├─ SAMURAI:      x=502, y=398  (weight=0.15)
└─ Faster R-CNN: x=498, y=402  (weight=0.05)

Clustering (radius=10):
  Cluster 1: TrackNet, YOLO v1, SAMURAI, Faster R-CNN
    - Position: ~(501, 401)
    - Total weight: 0.30 + 0.25 + 0.15 + 0.05 = 0.75

  Cluster 2: YOLO v2
    - Position: (650, 380)
    - Total weight: 0.25

Voting:
  ✅ Cluster 1 WINS (0.75 > 0.25)

Final Position: (501, 401)
Confidence: 0.75 (75% - többségi szavazat)
Note: YOLO v2 detekció eldobva mint outlier
```

---

#### **Példa 3: Nincs detekció (gap)**

```
Frame #300:
├─ TrackNet:     None
├─ YOLO v1:      None
├─ YOLO v2:      None
├─ SAMURAI:      None
└─ Faster R-CNN: None

Clustering:
  - 0 klaszter

Result: None (gap)
→ PASS 2 fogja interpolálni a temporal context alapján
```

---

#### **Példa 4: Csak 1 modell talál (alacsony confidence)**

```
Frame #400:
├─ TrackNet:     None
├─ YOLO v1:      x=720, y=340  (weight=0.25)
├─ YOLO v2:      None
├─ SAMURAI:      None
└─ Faster R-CNN: None

Clustering:
  Cluster 1: YOLO v1
    - Total weight: 0.25

Final Position: (720, 340)
Confidence: 0.25 (25% - csak 1 modell)
⚠️ Low confidence → PASS 4 will re-verify
```

---

## 📐 Matematikai Formula

### **Weighted Centroid Calculation**

```
         Σ (x_i * weight_i)
x_final = ──────────────────
          Σ weight_i

         Σ (y_i * weight_i)
y_final = ──────────────────
          Σ weight_i
```

**Ahol:**
- `x_i, y_i` = az i-edik modell detekciója
- `weight_i` = az i-edik modell súlya (config-ból)
- `Σ` = összegzés az adott klaszter összes tagjára

---

## 🎯 Döntési Szabályok

### **1. Több klaszter esetén (versengő detekciók)**

```python
if len(clusters) > 1:
    # Pick cluster with highest total weight
    best_cluster = max(clusters, key=lambda c: c['total_weight'])
    return best_cluster['centroid']
```

### **2. Egyetlen klaszter (konszenzus)**

```python
if len(clusters) == 1:
    # Use weighted average
    return clusters[0]['centroid']
```

### **3. Nincs detekció (gap)**

```python
if len(clusters) == 0:
    return None  # Pass 2 will interpolate
```

---

## 🔧 Finomhangolható Paraméterek

### **Config-ban állítható értékek:**

```yaml
ball_tracking:
  ensemble:
    tracknet:
      weight: 0.30        # ← Módosítható: 0.0-1.0
      threshold: 0.5      # ← Modell confidence küszöb

  spatial_clustering:
    radius: 10.0          # ← Clustering radius (pixels)

  voting:
    min_confidence: 0.5   # ← Minimum total weight elfogadáshoz
```

---

## 📊 Confidence Score Számítás

A final confidence = klaszter total weight:

```python
final_confidence = cluster['total_weight']

if final_confidence >= 0.8:
    quality = 'HIGH'      # 4-5 modell egyetért
elif final_confidence >= 0.5:
    quality = 'MEDIUM'    # 2-3 modell egyetért
else:
    quality = 'LOW'       # 1 modell vagy split vote
```

---

## 🧪 Tesztelési Forgatókönyvek

### **Scenario 1: Normál labda követés**
- 3-5 modell folyamatosan detektál
- Confidence: 0.75-1.0
- Interpolation rate: <5%

### **Scenario 2: Motion blur (gyors labda)**
- TrackNet: ✅ Detektál (heatmap-based)
- YOLO: ❌ Elveszik
- SAMURAI: ⚠️ Részben detektál
- Confidence: 0.30-0.45 (TrackNet + SAMURAI)

### **Scenario 3: Okklúzió (labda eltűnik hálónál)**
- Minden modell: ❌ None
- Gap filling: PASS 2 temporal interpolation
- Physics validation: PASS 3 trajectory fitting

### **Scenario 4: False positive (tévesen detektált objektum)**
- TrackNet: ❌ None
- YOLO v1: ✅ x=100, y=100 (bannert detektál)
- YOLO v2: ❌ None
- SAMURAI: ❌ None
- Result: confidence=0.25 → LOW → PASS 4 will reject

---

## 🎮 Gyakorlati Példa - 1 Másodperc Videó (30 frame)

```
Frame  TrackNet  YOLO_v1  YOLO_v2  SAMURAI  R-CNN  →  Ensemble  Confidence
─────────────────────────────────────────────────────────────────────────────
0      ✅ (640,360) ✅ (641,361) ✅ (640,360) ✅ (639,359) ✅ (642,362)  (640.5,360.4)  1.00
1      ✅ (645,358) ✅ (646,359) ✅ (645,358) ✅ (644,357) ✅ (647,360)  (645.3,358.3)  1.00
2      ✅ (650,356) ✅ (651,357) ❌ None     ✅ (649,355) ✅ (652,358)  (650.4,356.5)  0.75
3      ✅ (655,354) ❌ None     ❌ None     ✅ (654,353) ❌ None      (654.7,353.7)  0.45
4      ✅ (660,352) ✅ (661,353) ✅ (660,352) ✅ (659,351) ❌ None      (660.1,352.0)  0.95
...
15     ❌ None     ❌ None     ❌ None     ❌ None     ❌ None      INTERPOLATED  0.70*
16     ❌ None     ❌ None     ❌ None     ❌ None     ❌ None      INTERPOLATED  0.70*
17     ✅ (720,340) ❌ None     ✅ (721,341) ❌ None     ❌ None      (720.4,340.4)  0.55
...
29     ✅ (800,320) ✅ (801,321) ✅ (800,320) ✅ (799,319) ✅ (802,322)  (800.3,320.3)  1.00

* = PASS 2 temporal interpolation
```

**Eredmény:**
- 28/30 frame detektált (93.3%)
- 2/30 frame interpolált (6.7%)
- Átlag confidence: 0.84 (84%)

---

## 💡 Miért ez a jó?

### **Előnyök:**

1. **Redundancia**: Ha 1-2 modell fail-el, a többi kompenzál
2. **Precíziós boost**: 5 mérés átlaga pontosabb mint 1
3. **Outlier rejection**: Rossz detekciók kiszűrése vote-tal
4. **Adaptív confidence**: Tudjuk, mennyire megbízható az eredmény

### **Hátrányok:**

1. **Lassú**: 5x annyi inferencia mint YOLO alone
2. **Memóriaigényes**: Mind az 5 modell a GPU-n
3. **Komplexitás**: Több modell = több hibalehetőség

---

## 🔮 Jövőbeli Fejlesztések

1. **Dynamic weighting**: Modell súlyok változnak frame-enként (pl. TrackNet kapjon nagyobb súlyt blur esetén)
2. **Confidence-based voting**: Ne csak spatial clustering, hanem modell confidence is számítson
3. **Temporal voting**: Ne csak az aktuális frame, hanem előző frame-ek konszenzusa is
4. **ML meta-model**: Tanított modell dönt, hogy melyik ensemble eredményt fogadja el

---

**Tennis30 Ensemble Fusion - Matematikailag optimalizált, redundáns labda detektálás!** 🎾📊
