#!/usr/bin/env python3
"""
Cold Caser Forensic Suite – BGGG v2.7.5
Forensic Analysis with Auto-Save & Safe-Exit
All outputs go into the OUTPUT folder.
Developed by BGGremlin Group
MIT Licence 
"""

import os
import sys
import json
import signal
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
from colorama import init as colorama_init, Fore, Style
from PIL import Image, ExifTags

# ─── Initialization & Graceful Exit ─────────────────────────────────────────────
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
colorama_init(autoreset=True)
C1, C2, C3 = Fore.CYAN, Fore.GREEN, Fore.MAGENTA
RESET = Style.RESET_ALL

# ─── Paths & Model Load ─────────────────────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH    = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(PREDICTOR_PATH):
    print(f"{Fore.RED}[!] Landmark model not found at:\n    {PREDICTOR_PATH}\n"
          "    → Download and extract shape_predictor_68_face_landmarks.dat next to this script.")
    sys.exit(1)

# Create OUTPUT directory for all saved files
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "OUTPUT")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ─── Utility Functions ──────────────────────────────────────────────────────────
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Invalid image file: {path}")
    return img

def detect_face(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        raise RuntimeError("No face detected in image.")
    return rects[0]

def get_landmarks(img, rect):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    return face_utils.shape_to_np(shape)

def draw_landmarks(img, pts):
    out = img.copy()
    for (x, y) in pts:
        cv2.circle(out, (x, y), 2, (0, 255, 0), -1)
    return out

def safe_imshow(title, img, timeout=30000):
    try:
        cv2.imshow(title, img)
        print(f"[INFO] Displaying {title}. Press any key or wait {timeout/1000:.1f}s.")
        cv2.waitKey(int(timeout))
        cv2.destroyAllWindows()
    except Exception:
        fn = os.path.join(DEFAULT_OUTPUT_DIR, f"{title.replace(' ', '_')}.jpg")
        cv2.imwrite(fn, img)
        print(f"[HEADLESS] saved image to {fn}")

def measure_distances(pts):
    le = pts[36:42].mean(axis=0)
    re = pts[42:48].mean(axis=0)
    return {
        'interocular': float(np.linalg.norm(le - re)),
        'nose_width' : float(np.linalg.norm(pts[31] - pts[35])),
        'jaw_width'  : float(np.linalg.norm(pts[0]  - pts[16])),
        'nose_chin'  : float(np.linalg.norm(pts[30] - pts[8])),
        'mouth_width': float(np.linalg.norm(pts[48] - pts[54]))
    }

def symmetry_score(pts):
    diffs = [float(np.linalg.norm(pts[i] - pts[16 - i])) for i in range(17)]
    return float(np.mean(diffs))

def export_json(pts, path):
    data = [{'x': int(x), 'y': int(y)} for (x, y) in pts]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path

def show_raw(pts):
    for i, (x, y) in enumerate(pts):
        print(f"Point {i:02d}: ({x}, {y})")

def save_composite(path, images):
    comp = cv2.hconcat(images)
    cv2.imwrite(path, comp)
    return path

def extract_metadata(path):
    meta = {}
    try:
        img = Image.open(path)
        info = img._getexif() or {}
        for tag, val in info.items():
            name = ExifTags.TAGS.get(tag, tag)
            meta[name] = val
    except Exception:
        pass
    return meta

def detect_objects(img):
    casc_paths = {
        'face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'body': cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    }
    detections = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, fp in casc_paths.items():
        cascade = cv2.CascadeClassifier(fp)
        raw = cascade.detectMultiScale(gray, 1.1, 5)
        dets = np.array(raw) if isinstance(raw, (list, tuple)) else raw
        detections[name] = dets.tolist()
        for (x, y, w, h) in dets:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, detections

def analyze_gait(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    centers = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rects, _ = hog.detectMultiScale(frame, winStride=(8,8))
        if len(rects):
            x, y, w, h = rects[0]
            centers.append((x + w/2, y + h/2))
    cap.release()
    speeds = [abs(centers[i][0] - centers[i-1][0]) * fps for i in range(1, len(centers))]
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    plt.figure()
    plt.plot(speeds)
    plt.title('Gait Speed (px/sec)')
    plt.xlabel('Frame'); plt.ylabel('Speed')
    plot_fn = os.path.join(DEFAULT_OUTPUT_DIR, 'gait_plot.png')
    plt.savefig(plot_fn); plt.close()
    return avg_speed, plot_fn

def color_profile(img, region, clusters=3):
    x, y, w, h = region
    crop = img[y:y+h, x:x+w]
    data = crop.reshape(-1,3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(int)
    return centers.tolist()

def approximate_body_metrics(img, face_rect):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fullbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    bodies = fullbody.detectMultiScale(gray, 1.1, 5)
    if not len(bodies):
        return {'height_cm': None, 'weight_kg': None}
    x, y, w, h = bodies[0]
    face_h = face_rect.bottom() - face_rect.top()
    height_cm = (h / face_h) * 24.0  # assume face height = 24cm
    weight_kg = (height_cm/100)**2 * 22  # BMI=22
    return {'height_cm': round(height_cm,1), 'weight_kg': round(weight_kg,1)}

def generate_profile(image_path, video_path=None):
    img = load_image(image_path)
    meta = extract_metadata(image_path)
    face_rect = detect_face(img)
    pts = get_landmarks(img, face_rect)

    # Save landmark overlay
    vis = draw_landmarks(img, pts)
    face_fn = os.path.join(DEFAULT_OUTPUT_DIR, 'Profile_Face.jpg')
    cv2.imwrite(face_fn, vis)

    # Object annotation
    objs_img, objs = detect_objects(img.copy())
    objs_fn = os.path.join(DEFAULT_OUTPUT_DIR, 'Profile_Objects.jpg')
    cv2.imwrite(objs_fn, objs_img)

    gait_res = None
    if video_path:
        speed, plot_fn = analyze_gait(video_path)
        gait_res = {'avg_px_sec': speed, 'plot': plot_fn}

    # Color analysis regions
    eyes_region = [
        int(min(pts[36:42,0])), int(min(pts[36:42,1])),
        int(abs(pts[45][0]-pts[36][0])), int(abs(pts[41][1]-pts[37][1]))
    ]
    skin_region = [
        pts[1][0], pts[1][1],
        pts[15][0]-pts[1][0], pts[7][1]-pts[1][1]
    ]
    hair_region = [
        face_rect.left(),
        max(0, face_rect.top()-face_rect.height()),
        face_rect.width(), face_rect.height()
    ]
    eyes_color = color_profile(img, eyes_region)
    skin_color = color_profile(img, skin_region)
    hair_color = color_profile(img, hair_region)
    body_metrics = approximate_body_metrics(img, face_rect)

    report = {
        'metadata': meta,
        'objects': objs,
        'gait': gait_res,
        'colors': {'eyes': eyes_color, 'skin': skin_color, 'hair': hair_color},
        'body_metrics': body_metrics
    }
    rpt_fn = os.path.join(DEFAULT_OUTPUT_DIR, 'Profile_Report.json')
    with open(rpt_fn, 'w') as f:
        json.dump(report, f, indent=2)

    files = {
        'face_image'   : face_fn,
        'objects_image': objs_fn,
        'gait_plot'    : gait_res['plot'] if gait_res else None,
        'report_json'  : rpt_fn
    }
    return report, files

# ─── Compare Two Faces ──────────────────────────────────────────────────────────
def compare_two(img1_path, img2_path, show=True):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    rect1 = detect_face(img1)
    rect2 = detect_face(img2)
    lm1   = get_landmarks(img1, rect1)
    lm2   = get_landmarks(img2, rect2)
    vis1  = draw_landmarks(img1, lm1)
    vis2  = draw_landmarks(img2, lm2)
    vis2r = cv2.resize(vis2, (vis1.shape[1], vis1.shape[0]))
    overlay = cv2.addWeighted(vis1, 0.5, vis2r, 0.5, 0)

    # Auto‐save into OUTPUT
    cv2.imwrite(os.path.join(DEFAULT_OUTPUT_DIR, "face1_landmarks.jpg"), vis1)
    cv2.imwrite(os.path.join(DEFAULT_OUTPUT_DIR, "face2_landmarks.jpg"), vis2r)
    cv2.imwrite(os.path.join(DEFAULT_OUTPUT_DIR, "overlay.jpg"), overlay)

    if show:
        safe_imshow("Face 1 (landmarks)", vis1)
        safe_imshow("Face 2 (landmarks)", vis2r)
        safe_imshow("Overlay Comparison", overlay)
    else:
        print(f"Saved face1_landmarks.jpg, face2_landmarks.jpg, overlay.jpg in {DEFAULT_OUTPUT_DIR}")

    return vis1, vis2r, overlay, lm1, lm2

# ─── Main Menu ─────────────────────────────────────────────────────────────────
MENU = f"""{C1}
╔════════════════════════════════════════════════════════════════╗
║  Cold Caser Forensic Suite – BGGG v2.7.5                           ║
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
╚════╩═══════════════════════════════════════════════════════════╝{RESET}
"""

def main_menu():
    while True:
        print(MENU)
        choice = input("Select (1-11): ").strip()
        try:
            if choice == '1':
                img1 = input("Image 1 path: ").strip()
                img2 = input("Image 2 path: ").strip()
                compare_two(img1, img2)
            elif choice == '2':
                img = input("Image path: ").strip()
                image = load_image(img)
                pts   = get_landmarks(image, detect_face(image))
                out   = draw_landmarks(image, pts)
                fn    = os.path.join(DEFAULT_OUTPUT_DIR, 'Landmarks.jpg')
                cv2.imwrite(fn, out)
                print(f"Saved landmarks image to: {fn}")
                safe_imshow('Landmarks', out)
            elif choice == '3':
                img = input("Image path: ").strip()
                pts = get_landmarks(load_image(img), detect_face(load_image(img)))
                d   = measure_distances(pts)
                print(f"{C2}Measured Distances:{RESET}\n{json.dumps(d, indent=2)}")
            elif choice == '4':
                img   = input("Image path: ").strip()
                score = symmetry_score(get_landmarks(load_image(img), detect_face(load_image(img))))
                print(f"{C3}Symmetry score (lower better): {score:.2f}{RESET}")
            elif choice == '5':
                img = input("Image path: ").strip()
                raw = input("Output JSON filename or full path: ").strip()
                if not os.path.isabs(raw):
                    raw = os.path.join(DEFAULT_OUTPUT_DIR, raw)
                if not raw.lower().endswith('.json'):
                    raw += '.json'
                export_json(get_landmarks(load_image(img), detect_face(load_image(img))), raw)
                print(f"Exported landmarks JSON to: {raw}")
            elif choice == '6':
                img1    = input("Image 1 path: ").strip()
                img2    = input("Image 2 path: ").strip()
                raw_out = input("Output composite filename or full path: ").strip()
                if not os.path.isabs(raw_out):
                    raw_out = os.path.join(DEFAULT_OUTPUT_DIR, raw_out)
                name, ext = os.path.splitext(raw_out)
                if ext.lower() not in ('.jpg','.jpeg','.png','.bmp','.tiff'):
                    raw_out += '.jpg'
                vis1, vis2, ov, _, _ = compare_two(img1, img2, show=False)
                save_composite(raw_out, [vis1, vis2, ov])
                print(f"Saved composite image to: {raw_out}")
            elif choice == '7':
                img = input("Image path: ").strip()
                show_raw(get_landmarks(load_image(img), detect_face(load_image(img))))
            elif choice == '8':
                vid = input("Video path: ").strip()
                speed, plot_fn = analyze_gait(vid)
                print(f"Avg speed px/sec: {speed:.2f}, plot saved to: {plot_fn}")
            elif choice == '9':
                img = input("Image path: ").strip()
                out_img, dets = detect_objects(load_image(img))
                fn = os.path.join(DEFAULT_OUTPUT_DIR, 'Detected_Objects.jpg')
                cv2.imwrite(fn, out_img)
                print(f"Detected objects: {json.dumps(dets, indent=2)}")
                print(f"Saved annotated image to: {fn}")
            elif choice == '10':
                img = input("Image path: ").strip()
                vid = input("Video path (or Enter to skip): ").strip() or None
                rpt, files = generate_profile(img, vid)
                print(f"Full profile report:\n{json.dumps(rpt, indent=2)}")
                print(f"Files saved:\n{json.dumps(files, indent=2)}")
            elif choice == '11':
                print("Exiting. Stay sharp.")
                break
            else:
                print("Invalid choice.")
        except Exception as e:
            print(f"{Fore.RED}[Error] {e}{RESET}")

if __name__ == '__main__':
    main_menu()
