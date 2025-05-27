### **Cold Caser** V2.7.5**(*Latest*) - Developed by the *BG Gremlin Group*

# 👁️ Welcome to **Cold Caser V2.7.5**, a simple forensic suite engineered by the ***BGGremlin Group***. Designed for *Forensic Hobbiest* and *Investigative Teams*, *Cold Caser* provides a robust toolkit to swiftly analyze, compare, and annotate facial images and videos. Packed with intuitive features, precise landmark mapping, detailed symmetry analysis, and comprehensive reporting, this powerful software ensures reliable insights with every click.*Cold Caser* streamlines forensic workflows, turning critical analysis into seamless and actionable intelligence.


## 🚀 1. Prerequisites & Setup

1. **Python 3.8+**
   Make sure you have Python 3.8 or later installed.

2. **Install dependencies**

   ```bash
   pip install dlib opencv-python imutils numpy matplotlib colorama pillow
   ```

3. **Download the landmark model**

   ```bash
   cd path/to/Cold_Caser
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

   → You should now have:

   ```
   Cold_Caser/
     ├─ Cold_Caser7.5.py
     └─ shape_predictor_68_face_landmarks.dat
   ```

4. **Run the script**

   ```bash
   cd path/to/Cold_Caser
   python Cold_Caser7.5.py
   ```


## 📋 2. Main Menu Overview

When you launch, you’ll see:

```
╔════════════════════════════════════════════════════════════════╗
║  Cold Case Forensic Suite – BGGG v2.7.5                           ║
╠════╦═══════════════════════════════════════════════════════════╣
║ 1  ║ Compare Two Faces                                         ║
║ 2  ║ Analyze Single Face                                       ║
║ 3  ║ Measure Facial Distances                                  ║
║ 4  ║ Face Symmetry Analyzer                                    ║
║ 5  ║ Export Landmarks to JSON                                  ║
║ 6  ║ Generate Side-by-Side Report Image                        ║
║ 7  ║ Raw Landmark Coordinates Viewer                           ║
║ 8  ║ Gait Analysis (Video)                                     ║
║ 9  ║ Object Detection & Annotation                             ║
║ 10 ║ Generate Full Profile Report (Image [+Video])             ║
║ 11 ║ Exit                                                      ║
╚════╩═══════════════════════════════════════════════════════════╝
Select (1-11):
```

* **Enter the number** (1–11) for the tool you want.
* On any prompt, type **Ctrl+C** to quit immediately.

All **saved outputs** (images, JSON, plots) go into the `OUTPUT/` folder created next to your script:

```
Cold_Caser/
├─ Cold_Caser7.5.py
├─ shape_predictor_68_face_landmarks.dat
└─ OUTPUT/
```


## 🔍 3. Option-By-Option Deep Dive

### 1) Compare Two Faces

**Use:** Quickly overlay two faces to visually check alignment.

1. **Prompts:**

   ```
   Image 1 path:  C:\…\personA.jpg  
   Image 2 path:  C:\…\personB.jpg
   ```

2. **What it does:**

   * Detects each face.
   * Computes 68 facial landmarks.
   * Draws green dots on each landmark.
   * Resizes the second image to match the first.
   * Blends them 50/50 into an overlay.

3. **Outputs** in `OUTPUT/`:

   * `face1_landmarks.jpg` (first image with dots)
   * `face2_landmarks.jpg` (second, resized)
   * `overlay.jpg` (blended view)

4. **Display:**

   * If you have a GUI: three pop-up windows (Face 1, Face 2, Overlay); press any key to continue.
   * If headless: saves the same three files and prints their paths.

**Runtime:** \~2 s per pair on a modern PC.


### 2) Analyze Single Face

**Use:** Visualize the 68-point landmarks on one image.

1. **Prompt:**

   ```
   Image path:  C:\…\subject.jpg
   ```

2. **Process:**

   * Detects and landmark-maps the face.
   * Draws green dots.

3. **Output:**

   * `OUTPUT/Landmarks.jpg`

4. **Display:**

   * Shows in a window or saves to disk if GUI fails.

**Runtime:** \~1 s.


### 3) Measure Facial Distances

**Use:** Quantify key facial ratios in pixels.

1. **Prompt:**

   ```
   Image path:  C:\…\subject.jpg
   ```

2. **Calculates (pixels):**

   * **Inter-ocular** (eye-center to eye-center)
   * **Nose width** (nostril to nostril)
   * **Jaw width** (jaw corner to jaw corner)
   * **Nose-to-chin** (nose base to chin)
   * **Mouth width** (lip corner to corner)

3. **Output:**

   * Printed table in console.
   * (Redirect to file if desired:
     `python Cold_Caser5.5.py > distances.txt`)

**Runtime:** \~1 s.


### 4) Face Symmetry Analyzer

**Use:** Compute a symmetry score from jaw-line landmarks.

1. **Prompt:**

   ```
   Image path:  C:\…\subject.jpg
   ```

2. **Process:**

   * Calculates average left-to-right jaw deviation.
   * Lower score = more symmetric.

3. **Output:**

   * Printed symmetry score (e.g. `Symmetry score (lower better): 3.45`).

**Runtime:** \~1 s.


### 5) Export Landmarks to JSON

**Use:** Save all 68 `(x,y)` points for external analysis.

1. **Prompts:**

   ```
   Image path:  C:\…\subject.jpg  
   Output JSON filename or full path:  landmarks.json
   ```
2. **Action:**

   * Detects landmarks.
   * Writes `OUTPUT/landmarks.json` (or full path if given) with:

     ```json
     [
       {"x":345,"y":128},
       {"x":352,"y":142},
       …
     ]
     ```

### 6) Generate Side-by-Side Report Image

**Use:** Produce a single composite JPEG for reporting.

1. **Prompts:**

   ```
   Image 1 path:  C:\…\personA.jpg  
   Image 2 path:  C:\…\personB.jpg  
   Output composite filename or full path:  comparison.jpg
   ```
2. **Action:**

   * Internally runs Option 1 (auto-saving face1, face2, overlay).
   * Horizontally concatenates the three into `OUTPUT/comparison.jpg`.
3. **Output:**

   * Prints `Saved composite image to: OUTPUT/comparison.jpg`

### 7) Raw Landmark Coordinates Viewer

**Use:** Dump all 68 points to console.

1. **Prompt:**

   ```
   Image path:  C:\…\subject.jpg
   ```
2. **Output:**

   ```
   Point 00: (345,128)
   Point 01: (352,142)
   …
   Point 67: (289,403)
   ```

   (Redirect if you want to save:
   `python Cold_Caser5.5.py > raw_coords.txt`)


### 8) Gait Analysis (Video)

**Use:** Estimate walking speed from a video.

1. **Prompt:**

   ```
   Video path:  C:\…\walk.mp4
   ```

2. **Action:**

   * Runs HOG+SVM person detector on each frame.
   * Tracks the person’s center x-coordinate.
   * Computes pixel-per-second speeds.
   * Averages them.
   * Plots speeds vs. frame in `OUTPUT/gait_plot.png`.

3. **Output:**

   ```
   Avg speed px/sec: 24.37, plot saved to: OUTPUT/gait_plot.png
   ```

**Runtime:** Depends on video length (\~0.1 s per frame).



### 9) Object Detection & Annotation

**Use:** Quick face+body rectangles for situational awareness.

1. **Prompt:**

   ```
   Image path:  C:\…\scene.jpg
   ```
2. **Action:**

   * Runs Haar cascades for faces and full bodies.
   * Draws blue rectangles.
3. **Output:**

   * `OUTPUT/Detected_Objects.jpg`
   * Printed JSON of bounding boxes:

     ```json
     {
       "face":[[x,y,w,h],…],
       "body":[[x,y,w,h],…]
     }
     ```



### 10) Generate Full Profile Report

**Use:** One-stop dossier builder (image, video optional).

1. **Prompts:**

   ```
   Image path:  C:\…\subject.jpg  
   Video path (or Enter to skip):  C:\…\walk.mp4
   ```

2. **Action:**

   * EXIF/metadata extraction.
   * Landmark overlay → `Profile_Face.jpg`.
   * Object annotation → `Profile_Objects.jpg`.
   * Gait analysis → `gait_plot.png`.
   * Color palettes for eyes, skin, hair via k-means.
   * Body height & weight estimate (face-to-body ratio).
   * Saves all into `OUTPUT/`.
   * Writes `OUTPUT/Profile_Report.json` summarizing all data.

3. **Output:**

   ```
   Full profile report:
   {
     "metadata": {...},
     "objects": {...},
     "gait": {"avg_px_sec":24.37,"plot":"OUTPUT/gait_plot.png"},
     "colors": {"eyes": [...], …},
     "body_metrics": {"height_cm":172.3,"weight_kg":65.3}
   }
   Files saved:
   {
     "face_image":"OUTPUT/Profile_Face.jpg",
     "objects_image":"OUTPUT/Profile_Objects.jpg",
     "gait_plot":"OUTPUT/gait_plot.png",
     "report_json":"OUTPUT/Profile_Report.json"
   }
   ```



### 11) Exit

**Use:** Cleanly terminate the program.



## 🗂️ 4. Locating Your Outputs

All images, JSONs, and plots automatically appear in:

```
Cold_Caser/
└─ OUTPUT/
    ├─ face1_landmarks.jpg
    ├─ face2_landmarks.jpg
    ├─ overlay.jpg
    ├─ Landmarks.jpg
    ├─ landmarks.json    (if you used #5)
    ├─ comparison.jpg    (#6)
    ├─ Detected_Objects.jpg
    ├─ gait_plot.png
    ├─ Profile_Face.jpg
    ├─ Profile_Objects.jpg
    ├─ Profile_Report.json
    └─ …and any other files you generate
```

You can now drag-and-drop that `OUTPUT/` folder into case files, attach everything to an email, or feed it into your incident-management system.



## ✅ 5. Tips & Best Practices

* **Batch mode:** Use shell loops to process folders of images, redirecting console output to files.
* **Quality control:** Always eyeball `OUTPUT/overlay.jpg` after each new face pair to catch mis-detections.
* **Reference object:** Place a ruler or known-size object in the frame to calibrate height estimates more accurately.



### 👾 V3 Update Forecast

* **Logging And Directories:** Adding `logging` for timestamps (`logging.basicConfig(filename=…)`). Automated Nested Folder Creation to tidy up `OUTPUT` folder 
* **Batch Mode Baked-In** Adding In Program Options for shell loops to process folders of images, While automatically redirecting console output to files.
* **Better Detection:** In addition to Haar cascades, Adding a deep-learning detector (e.g. YOLO) for improved object detection.
