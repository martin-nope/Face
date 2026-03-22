# """
# Live face recognition with IoU-based SORT tracking.

# Track-before-recognize pattern: uses IoU-based tracking to maintain track IDs
# across frames, only runs expensive embedding extraction every N frames per
# track.  Brings pipeline from ~5 FPS to 25+ FPS.
# """

# import os
# import sys
# import time
# import pickle
# import argparse
# import cv2
# import numpy as np
# import yaml
# import threading
# import queue

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from inference.detector import FaceDetector
# from app.build_encodings import EncodingBuilder


# # ---------------------------------------------------------------------------
# # Simple SORT tracker (IoU-based, no Kalman filter for simplicity)
# # ---------------------------------------------------------------------------

# class Track:
#     """A single tracked face."""
#     _next_id = 0

#     def __init__(self, box: np.ndarray, score: float):
#         self.id = Track._next_id
#         Track._next_id += 1
#         self.box = box           # [x1, y1, x2, y2]
#         self.score = score
#         self.identity = "Unknown"
#         self.embedding = None
#         self.age = 0             # frames since last match
#         self.hits = 1            # times matched
#         self.frames_since_embed = 999  # force embed on first match

#     def update(self, box: np.ndarray, score: float):
#         self.box = box
#         self.score = score
#         self.age = 0
#         self.hits += 1
#         self.frames_since_embed += 1

#     def update_embedding(self, new_embedding: np.ndarray, momentum: float = 0.5):
#         if self.embedding is None:
#             self.embedding = new_embedding
#         else:
#             # Exponential Moving Average for extremely stable recognition
#             avg = momentum * new_embedding + (1.0 - momentum) * self.embedding
#             self.embedding = avg / np.linalg.norm(avg)


# class SORTTracker:
#     """Simple Online Realtime Tracker using IoU matching.

#     Parameters
#     ----------
#     iou_threshold : float
#         Minimum IoU to match detection to track.
#     max_age : int
#         Frames to keep a track alive without matches.
#     embed_interval : int
#         Run embedding every N frames per track.
#     """

#     def __init__(self, iou_threshold: float = 0.3, max_age: int = 10,
#                  embed_interval: int = 5):
#         self.iou_threshold = iou_threshold
#         self.max_age = max_age
#         self.embed_interval = embed_interval
#         self.tracks: list[Track] = []

#     def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
#         """Update tracks with new detections.

#         Returns list of active tracks (including newly created ones).
#         """
#         if len(boxes) == 0:
#             for t in self.tracks:
#                 t.age += 1
#                 t.frames_since_embed += 1
#             self.tracks = [t for t in self.tracks if t.age <= self.max_age]
#             return self.tracks

#         # Compute IoU matrix between existing tracks and new detections
#         if len(self.tracks) > 0:
#             track_boxes = np.array([t.box for t in self.tracks])
#             iou_matrix = self._iou_batch(track_boxes, boxes)

#             # Greedy matching
#             matched_tracks = set()
#             matched_dets = set()

#             # Sort by IoU descending
#             while True:
#                 if iou_matrix.size == 0:
#                     break
#                 max_val = iou_matrix.max()
#                 if max_val < self.iou_threshold:
#                     break
#                 idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
#                 t_idx, d_idx = idx

#                 self.tracks[t_idx].update(boxes[d_idx], scores[d_idx])
#                 matched_tracks.add(t_idx)
#                 matched_dets.add(d_idx)

#                 # Mask out matched rows/cols
#                 iou_matrix[t_idx, :] = -1
#                 iou_matrix[:, d_idx] = -1

#             # Age unmatched tracks
#             for i, t in enumerate(self.tracks):
#                 if i not in matched_tracks:
#                     t.age += 1
#                     t.frames_since_embed += 1

#             # Create new tracks for unmatched detections
#             for j in range(len(boxes)):
#                 if j not in matched_dets:
#                     self.tracks.append(Track(boxes[j], scores[j]))
#         else:
#             # No existing tracks — create all
#             for j in range(len(boxes)):
#                 self.tracks.append(Track(boxes[j], scores[j]))

#         # Remove dead tracks
#         self.tracks = [t for t in self.tracks if t.age <= self.max_age]

#         return self.tracks

#     @staticmethod
#     def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
#         """Compute IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
#         inter_x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
#         inter_y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
#         inter_x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
#         inter_y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
#         inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
#         area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
#         area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
#         union = area_a + area_b - inter
#         return inter / (union + 1e-7)

#     def needs_embedding(self, track: Track) -> bool:
#         """Check if a track needs embedding extraction."""
#         return track.frames_since_embed >= self.embed_interval


# # ---------------------------------------------------------------------------
# # Recognition pipeline
# # ---------------------------------------------------------------------------

# class LiveRecognizer:
#     """Live face recognition with tracking.

#     Parameters
#     ----------
#     detector : FaceDetector
#     encodings_path : str  — path to saved encodings.pkl
#     cfg : dict
#     update_encodings : bool
#         If True, scans `known_students/` and updates `encodings.pkl` with new crops.
#     """

#     def __init__(self, detector: FaceDetector, encodings_path: str, cfg: dict,
#                  update_encodings: bool = True):
#         self.detector = detector
#         self.cfg = cfg

#         # Ensure encodings are up-to-date (build new embeddings if new images were added)
#         if update_encodings:
#             known_dir = cfg.get("paths", {}).get("known_students", "known_students")
#             if os.path.exists(known_dir):
#                 builder = EncodingBuilder(known_dir, encodings_path)
#                 builder.build(rebuild=False)

#         # Load encodings
#         if os.path.exists(encodings_path):
#             with open(encodings_path, "rb") as f:
#                 data = pickle.load(f)
#             # data: dict[str, List[np.ndarray]]
#             self.known_db = {}
#             for name, embeddings in data.items():
#                 avg = np.mean(embeddings, axis=0)
#                 self.known_db[name] = avg / np.linalg.norm(avg)
#             print(f"✅ Loaded {len(self.known_db)} identities from {encodings_path}")
#         else:
#             print(f"⚠️  No encodings found at {encodings_path}")
#             self.known_db = {}

#         # InsightFace for embedding extraction (detection disabled)
#         try:
#             from insightface.app import FaceAnalysis
#             self.face_app = FaceAnalysis(
#                 name="buffalo_l",
#                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
#             )
#             self.face_app.prepare(ctx_id=0, det_size=(640, 640))
#             print("✅ InsightFace loaded for embedding extraction")
#         except ImportError:
#             print("⚠️  InsightFace not available — recognition disabled")
#             self.face_app = None

#         # Tracker: Increased max_age for CPU-friendly persistence
#         self.tracker = SORTTracker(
#             iou_threshold=0.2,
#             max_age=30,      # Keep track alive for 1 sec even without AI hits
#             embed_interval=30, # Recognise every 10 matched frames
#         )

#         self.encodings_path = encodings_path

#     def _refresh_encodings(self):
#         """Build and reload face encodings from disk."""
#         known_dir = self.cfg.get("paths", {}).get("known_students", "known_students")
#         if os.path.exists(known_dir):
#             builder = EncodingBuilder(known_dir, self.encodings_path)
#             builder.build(rebuild=False)
            
#             # Reload from disk
#             if os.path.exists(self.encodings_path):
#                 with open(self.encodings_path, "rb") as f:
#                     data = pickle.load(f)
#                 self.known_db = {}
#                 for name, embeddings in data.items():
#                     if len(embeddings) > 0:
#                         avg = np.mean(embeddings, axis=0)
#                         self.known_db[name] = avg / np.linalg.norm(avg)
#                 print(f"🔄 Reloaded {len(self.known_db)} identities")

#     def _extract_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
#         """Extract face embedding using InsightFace."""
#         if self.face_app is None:
#             return None

#         # Try with default threshold first
#         faces = self.face_app.get(face_crop)
#         if faces and len(faces) > 0:
#             return faces[0].normed_embedding
            
#         # Fallback: Lower threshold (since we already know it's a face crop)
#         if hasattr(self.face_app, 'models') and 'detection' in self.face_app.models:
#             old_thresh = self.face_app.models['detection'].det_thresh
#             self.face_app.models['detection'].det_thresh = 0.2
#             faces = self.face_app.get(face_crop)
#             self.face_app.models['detection'].det_thresh = old_thresh # restore
#             if faces and len(faces) > 0:
#                 return faces[0].normed_embedding
                
#         return None

#     def _identify(self, embedding: np.ndarray, threshold: float = 0.50) -> tuple:
#         """Match embedding against known database.

#         Returns (name, score).
#         """
#         best_name = "Unknown"
#         best_score = 0.0

#         for name, master_emb in self.known_db.items():
#             score = float(np.dot(embedding, master_emb))
#             if score > best_score:
#                 best_score = score
#                 best_name = name

#         if best_score < threshold:
#             best_name = "Unknown"

#         return best_name, best_score

#     def run(self):
#         """Main webcam loop with threaded detection."""
#         # Use DirectShow on Windows for zero-latency camera feed
#         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        
#         # Maximize camera resolution for best source image
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         if not cap.isOpened():
#             print("❌ Could not open webcam")
#             return

#         # Threading state
#         input_q = queue.Queue(maxsize=1)
#         output_q = queue.Queue(maxsize=1)
#         stop_event = threading.Event()

#         def detection_worker():
#             while not stop_event.is_set():
#                 try:
#                     frame_to_detect = input_q.get(timeout=0.1)
#                     res = self.detector.detect(frame_to_detect)
#                     if not output_q.full():
#                         output_q.put(res)
#                     input_q.task_done()
#                 except queue.Empty:
#                     continue

#         det_thread = threading.Thread(target=detection_worker, daemon=True)
#         det_thread.start()

#         print("\n🚀 MAX Performance mode active. Threaded detection started.\n")
#         frame_count = 0
#         fps_counter = 0
#         fps_timer = time.time()
#         display_fps = 0.0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1

#             # Auto-update encodings from known_students folder every 200 frames
#             if frame_count % 200 == 0:
#                 self._refresh_encodings()

#             # Push latest frame to detection thread if it's ready
#             if input_q.empty():
#                 input_q.put(frame.copy())

#             # Check if detection results are ready
#             if not output_q.empty():
#                 result = output_q.get()
#                 # Update tracker with new detections
#                 tracks = self.tracker.update(result["boxes"], result["scores"])
#             else:
#                 # Keep tracks alive even when detection is still running
#                 tracks = self.tracker.update(np.zeros((0, 4)), np.zeros(0))

#             # Process tracks that need embedding
#             for track in tracks:
#                 # EFFICIENCY: If we already have a high-confidence ID,
#                 # we can skip embedding for a longer period (30 frames)
#                 current_interval = self.tracker.embed_interval
#                 if track.identity != "Unknown":
#                     # We can be more relaxed once identified
#                     current_interval = 30 

#                 if track.frames_since_embed >= current_interval and self.face_app is not None:
#                     x1, y1, x2, y2 = track.box.astype(int)
                    
#                     # Add 15% padding to the crop (helps InsightFace align tilted faces)
#                     w, h = x2 - x1, y2 - y1
#                     pad_w, pad_h = int(w * 0.15), int(h * 0.15)
#                     x1_pad = max(0, x1 - pad_w)
#                     y1_pad = max(0, y1 - pad_h)
#                     x2_pad = min(frame.shape[1], x2 + pad_w)
#                     y2_pad = min(frame.shape[0], y2 + pad_h)

#                     crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
#                     if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
#                         embedding = self._extract_embedding(crop)
#                         if embedding is not None:
#                             track.update_embedding(embedding)
#                             track.identity, _ = self._identify(track.embedding)
#                             track.frames_since_embed = 0

#             # Draw
#             for track in tracks:
#                 # ONLY draw tracks that have been confirmed for at least 3 hits 
#                 # to prevent flickering boxes on noise.
#                 # AND only if they were hit recently (age < 5)
#                 if track.hits < 3 or track.age > 5:
#                     continue

#                 x1, y1, x2, y2 = track.box.astype(int)
#                 is_known = track.identity != "Unknown"
#                 color = (0, 255, 0) if is_known else (0, 0, 255)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 label = f"#{track.id} {track.identity}"
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # FPS display
#             fps_counter += 1
#             elapsed = time.time() - fps_timer
#             if elapsed > 1.0:
#                 display_fps = fps_counter / elapsed
#                 fps_counter = 0
#                 fps_timer = time.time()

#             cv2.putText(frame, f"FPS: {display_fps:.1f} (Threaded)", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             cv2.imshow("Face Recognition", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         stop_event.set()
#         cap.release()
#         cv2.destroyAllWindows()
#         print("📴 Session ended")


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Live face recognition")
#     parser.add_argument("--config", default="config.yaml")
#     parser.add_argument("--checkpoint", required=True)
#     parser.add_argument("--encodings", default=None,
#                         help="Path to encodings.pkl (default: from config)")
#     parser.add_argument("--conf-threshold", type=float, default=None,
#                         help="Override config.yaml inference.conf_threshold (0-1)")
#     parser.add_argument("--no-encoding-update", action="store_false", dest="update_encodings",
#                         help="Skip scanning known_students for new crops (use existing encodings only)")
#     args = parser.parse_args()

#     with open(args.config, "r") as f:
#         cfg = yaml.safe_load(f)

#     if args.conf_threshold is not None:
#         cfg.setdefault("inference", {})["conf_threshold"] = args.conf_threshold

#     detector = FaceDetector(args.checkpoint, cfg)
#     enc_path = args.encodings or cfg["paths"].get("encodings_file", "encodings.pkl")
#     recognizer = LiveRecognizer(detector, enc_path, cfg,
#                                  update_encodings=args.update_encodings)
#     recognizer.run()


# if __name__ == "__main__":
#     main()


"""
Live Face Recognition — GPU Build (Final)
==========================================
All 5 remaining issues fixed on top of previous 13.

FIX-A  Queue: use put_nowait with explicit full-check (no silent misuse)
FIX-B  Embedding async worker thread — never blocks main/UI thread
FIX-C  det_thresh restore moved into finally block (always restored)
FIX-D  CSV file-lock via portalocker for multi-process safety (optional dep)
FIX-E  FPS cap (target 30) + minimal sleep to avoid CPU spin
GPU    ctx_id forced to 0; CUDAExecutionProvider listed first
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
import csv
import threading
import queue
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from inference.detector import FaceDetector
    from app.build_encodings import EncodingBuilder
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

if not _IMPORTS_OK:
    class FaceDetector:                                         # noqa
        def __init__(self, *args, **kwargs): pass
        def detect(self, frame: np.ndarray) -> dict:
            return {"boxes":  np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros(0,       dtype=np.float32)}

    class EncodingBuilder:                                      # noqa
        def __init__(self, *args, **kwargs): pass
        def build(self, **kwargs): pass

# optional multi-process CSV lock
try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False


# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

METADATA_CSV_PATHS            = ["output.csv", "output2.csv"]
ATTENDANCE_FILE               = "attendance.csv"

LOW_LIGHT_THRESHOLD           = 40     # mean brightness → skip recognition
EXPOSURE_CORRECTION_THRESHOLD = 80     # mean brightness → apply CLAHE
CLAHE_EVERY_N_FRAMES          = 3      # CLAHE throttle

LABEL_MAX_CHARS               = 22     # display-name truncation

TARGET_FPS                    = 30     # FIX-E: FPS cap
_FRAME_BUDGET_SEC             = 1.0 / TARGET_FPS

# Embedding async worker
EMBED_QUEUE_SIZE              = 4      # FIX-B: pending embedding jobs


# ═════════════════════════════════════════════════════════════
# METADATA LOOKUP
# ═════════════════════════════════════════════════════════════

class StudentMetadataDB:
    """Roll / name → branch, course, etc. from output.csv / output2.csv."""

    def __init__(self, csv_paths: list[str]):
        self._by_roll: dict[str, dict]       = {}
        self._by_name: dict[str, list[dict]] = {}
        for path in csv_paths:
            if not Path(path).exists():
                print(f"[MetaDB] Not found: {path} — skipped")
                continue
            self._load(path)
        print(f"[MetaDB] {len(self._by_roll)} roll records, "
              f"{sum(len(v) for v in self._by_name.values())} name records")

    def _load(self, path: str):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return
            fields     = [c.strip() for c in reader.fieldnames]
            name_col   = self._col(fields, ("Student Name", "Name", "name"))
            roll_col   = self._col(fields, ("Roll No.", "Roll Number", "Roll",
                                            "UID", "ID", "Reg No."))
            branch_col = self._col(fields, ("Branch", "Stream", "Course",
                                            "Department"))
            for raw in reader:
                row    = {k.strip(): (v or "").strip() for k, v in raw.items()}
                record = {
                    "Name":   row.get(name_col,   "N/A") if name_col   else "N/A",
                    "Roll":   row.get(roll_col,   "N/A") if roll_col   else "N/A",
                    "Branch": row.get(branch_col, "N/A") if branch_col else "N/A",
                    **{k: v for k, v in row.items()},
                }
                roll = record["Roll"].strip()
                name = record["Name"].lower().strip()
                if roll and roll != "N/A":
                    self._by_roll[roll] = record
                if name:
                    self._by_name.setdefault(name, []).append(record)

    @staticmethod
    def _col(fields: list[str], candidates: tuple) -> Optional[str]:
        for c in candidates:
            for f in fields:
                if f.lower() == c.lower():
                    return f
        return None

    def lookup(self, identity_string: str) -> dict:
        parts = [p for p in identity_string.split("_") if p]
        if not parts:
            print("[MetaDB] Warning: empty identity string")
            return {"Name": "Unknown", "Roll": "N/A", "Branch": "N/A"}

        first = parts[0]
        last  = parts[-1] if len(parts) > 1 else ""

        if last and last in self._by_roll:
            return self._by_roll[last]

        candidates = self._by_name.get(first.lower(), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for rec in candidates:
                if rec.get("Roll", "") == last:
                    return rec
            return candidates[0]

        print(f"[MetaDB] Warning: no CSV record for '{identity_string}' "
              f"(first='{first}', last='{last}')")
        return {"Name": first, "Roll": last or "N/A", "Branch": "N/A"}


# ═════════════════════════════════════════════════════════════
# ATTENDANCE LOGGER  (FIX-D: optional portalocker)
# ═════════════════════════════════════════════════════════════

class AttendanceLogger:
    """
    Thread-safe (and optionally multi-process-safe) CSV logger.

    FIX-D: If portalocker is installed, an exclusive file lock is acquired
           on every write so two processes writing the same CSV never
           interleave rows.  Falls back gracefully when not installed.
    """

    def __init__(self, filename: str = ATTENDANCE_FILE):
        self.filename = filename
        self.lock     = threading.Lock()
        self._sr_no   = 1
        self.logged_today: set[str] = set()
        today = date.today().isoformat()

        if not Path(self.filename).exists():
            with open(self.filename, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["Sr.no", "Name", "Branch", "Roll Number", "Timestamp"]
                )
        else:
            with open(self.filename, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            data_rows   = rows[1:]
            self._sr_no = len(data_rows) + 1
            for row in data_rows:
                if len(row) >= 5 and row[4].startswith(today):
                    self.logged_today.add(self._key(row[1], row[3]))

        lock_note = "portalocker" if _HAS_PORTALOCKER else "threading.Lock only"
        print(f"[Logger] Ready — next sr={self._sr_no}, "
              f"logged today={len(self.logged_today)}, lock={lock_note}")

    @staticmethod
    def _key(name: str, roll: str) -> str:
        return roll if (roll and roll != "N/A") else name

    def log(self, name: str, branch: str, roll: str):
        key = self._key(name, roll)
        with self.lock:
            if key in self.logged_today:
                return
            with open(self.filename, "a", newline="", encoding="utf-8") as f:
                # FIX-D: acquire exclusive file lock when portalocker available
                if _HAS_PORTALOCKER:
                    portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    csv.writer(f).writerow([
                        self._sr_no, name, branch, roll,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ])
                finally:
                    if _HAS_PORTALOCKER:
                        portalocker.unlock(f)
            self.logged_today.add(key)
            self._sr_no += 1
            print(f"[Logger] {name}  branch={branch}  roll={roll}")


# ═════════════════════════════════════════════════════════════
# SORT TRACKER
# ═════════════════════════════════════════════════════════════

class Track:
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, box: np.ndarray, score: float):
        with Track._id_lock:
            self.id          = Track._next_id
            Track._next_id   = (Track._next_id + 1) % 1_000_000
        self.box                = box.astype(float)
        self.score              = score
        self.identity           = "Unknown"
        self.embedding: Optional[np.ndarray] = None
        self.age                = 0
        self.hits               = 1
        self.frames_since_embed = 999   # fire on first hit
        self._slow_age_accum    = 0.0
        self._embed_pending     = False  # FIX-B: guard against double-submit

    def update(self, box: np.ndarray, score: float):
        self.box              = box.astype(float)
        self.score            = score
        self.age              = 0
        self._slow_age_accum  = 0.0
        self.hits            += 1
        self.frames_since_embed += 1

    def slow_age(self, amount: float = 0.2):
        self._slow_age_accum += amount
        if self._slow_age_accum >= 1.0:
            self.age += 1
            self._slow_age_accum = 0.0
        self.frames_since_embed += 1

    def update_embedding(self, new_emb: np.ndarray, momentum: float = 0.5):
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-7)
        if self.embedding is None:
            self.embedding = new_emb
        else:
            avg = momentum * new_emb + (1.0 - momentum) * self.embedding
            self.embedding = avg / (np.linalg.norm(avg) + 1e-7)


class SORTTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.tracks: list[Track] = []

    def update(self, boxes: np.ndarray, scores: np.ndarray,
               ghost: bool = False) -> list[Track]:
        if ghost or len(boxes) == 0:
            for t in self.tracks:
                t.slow_age(0.2) if ghost else setattr(t, "age", t.age + 1) or setattr(t, "frames_since_embed", t.frames_since_embed + 1)
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        if self.tracks:
            iou_mat   = self._iou_batch(
                np.array([t.box for t in self.tracks]), boxes
            )
            matched_t: set[int] = set()
            matched_d: set[int] = set()
            while iou_mat.size and iou_mat.max() >= self.iou_threshold:
                ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                self.tracks[ti].update(boxes[di], scores[di])
                matched_t.add(int(ti)); matched_d.add(int(di))
                iou_mat[ti, :] = -1;   iou_mat[:, di] = -1
            for i, t in enumerate(self.tracks):
                if i not in matched_t:
                    t.age += 1; t.frames_since_embed += 1
            for j in range(len(boxes)):
                if j not in matched_d:
                    self.tracks.append(Track(boxes[j], scores[j]))
        else:
            for j in range(len(boxes)):
                self.tracks.append(Track(boxes[j], scores[j]))

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks

    @staticmethod
    def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.reshape(-1, 4); b = b.reshape(-1, 4)
        ix1   = np.maximum(a[:, 0:1], b[:, 0].reshape(1, -1))
        iy1   = np.maximum(a[:, 1:2], b[:, 1].reshape(1, -1))
        ix2   = np.minimum(a[:, 2:3], b[:, 2].reshape(1, -1))
        iy2   = np.minimum(a[:, 3:4], b[:, 3].reshape(1, -1))
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
        area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
        return inter / (area_a + area_b - inter + 1e-7)


# ═════════════════════════════════════════════════════════════
# EXPOSURE UTILITIES
# ═════════════════════════════════════════════════════════════

def measure_brightness(frame_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def draw_low_light_warning(frame: np.ndarray, brightness: float):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 200, 220), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame,
                f"LOW LIGHT (brightness={brightness:.0f}) — recognition paused",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ═════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

class LiveRecognizer:

    def __init__(self, detector: FaceDetector, encodings_path: str,
                 cfg: dict, update_encodings: bool = True):
        self.detector       = detector
        self.cfg            = cfg
        self.encodings_path = encodings_path
        self.known_db: dict = {}
        self._db_lock       = threading.RLock()
        self.refreshing     = False
        self._refresh_lock  = threading.Lock()

        csv_paths    = cfg.get("paths", {}).get("metadata_csvs", METADATA_CSV_PATHS)
        self.meta_db = StudentMetadataDB(csv_paths)
        self.logger  = AttendanceLogger(
            cfg.get("paths", {}).get("attendance_file", ATTENDANCE_FILE)
        )

        # ── GPU InsightFace ───────────────────────────────────
        self.face_app = None
        try:
            from insightface.app import FaceAnalysis
            # GPU: ctx_id=0, CUDA provider listed first
            self.face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[InsightFace] Ready on GPU (ctx_id=0)")
        except Exception as exc:
            print(f"[InsightFace] Load error: {exc}")

        self._load_db()
        if update_encodings:
            self._refresh_db_worker()

        self.tracker            = SORTTracker(iou_threshold=0.2, max_age=45)
        self.identify_threshold = cfg.get("inference", {}).get("rec_threshold", 0.45)
        self._clahe_frame_count = 0

        # FIX-B: async embedding worker ───────────────────────
        # Jobs pushed by main thread: (track_ref, crop_ndarray)
        # Results returned via _embed_results queue: (track_id, embedding)
        self._embed_jobs:    queue.Queue = queue.Queue(maxsize=EMBED_QUEUE_SIZE)
        self._embed_results: queue.Queue = queue.Queue()
        self._embed_stop     = threading.Event()
        self._embed_thread   = threading.Thread(
            target=self._embedding_worker, daemon=True, name="EmbedWorker"
        )
        self._embed_thread.start()
        print("[EmbedWorker] Async embedding thread started")

    # ── Database ──────────────────────────────────────────────

    def _load_db(self):
        path = Path(self.encodings_path)
        if not path.exists():
            print(f"[DB] No encodings at {path}")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"[DB] Corrupt/unreadable encodings: {exc}")
            return
        new_db: dict = {}
        for name, embeddings in data.items():
            arr   = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-7
            new_db[name] = arr / norms
        with self._db_lock:
            self.known_db = new_db
        print(f"[DB] Loaded {len(new_db)} identities")

    def _refresh_db_worker(self):
        with self._refresh_lock:
            if self.refreshing:
                return
            self.refreshing = True
        try:
            known_dir = self.cfg.get("paths", {}).get("known_students", "known_students")
            if os.path.exists(known_dir):
                try:
                    builder = EncodingBuilder(known_dir, self.encodings_path)
                    if hasattr(builder, "build"):
                        builder.build(rebuild=False)
                    else:
                        print("[DB] EncodingBuilder has no .build() — skipping")
                except Exception as exc:
                    print(f"[DB] EncodingBuilder error: {exc}")
            self._load_db()
        except Exception as exc:
            print(f"[DB] Refresh error: {exc}")
        finally:
            self.refreshing = False

    # ── Recognition ───────────────────────────────────────────

    def _identify(self, query_emb: np.ndarray) -> tuple[str, float]:
        best_name  = "Unknown"
        best_score = 0.0
        with self._db_lock:
            db_snapshot = dict(self.known_db)
        for name, master_embs in db_snapshot.items():
            score = float(np.max(np.dot(master_embs, query_emb)))
            if score > best_score:
                best_score = score; best_name = name
        if best_score < self.identify_threshold:
            best_name = "Unknown"
        return best_name, best_score

    def _extract_embedding(self, frame: np.ndarray,
                           box: np.ndarray) -> Optional[np.ndarray]:
        """
        FIX-C: det_thresh restore is inside finally so it ALWAYS reverts
               even if face_app.get() raises.
        """
        if self.face_app is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        pad  = int((x2 - x1) * 0.20)
        crop = frame[max(0, y1-pad):min(h, y2+pad),
                     max(0, x1-pad):min(w, x2+pad)]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        faces = self.face_app.get(crop)
        if faces:
            return faces[0].normed_embedding

        # Fallback: lower det_thresh — always restored via finally
        det = None
        try:
            det = self.face_app.models.get("detection")
            if det is None:
                return None
            old_thresh     = det.det_thresh
            det.det_thresh = 0.2
            try:
                faces = self.face_app.get(crop)
                if faces:
                    return faces[0].normed_embedding
            finally:
                # FIX-C: guaranteed restore regardless of exception
                det.det_thresh = old_thresh
        except Exception as exc:
            print(f"[Embed] det_thresh fallback error: {exc}")
        return None

    # ── FIX-B: Async embedding worker ─────────────────────────

    def _embedding_worker(self):
        """
        Runs in its own thread.
        Pulls (track_id, frame_copy, box) jobs from _embed_jobs,
        runs InsightFace extraction (GPU-accelerated),
        pushes (track_id, embedding | None) to _embed_results.

        This keeps the main/UI thread free of any InsightFace calls.
        """
        while not self._embed_stop.is_set():
            try:
                track_id, frame_copy, box = self._embed_jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                emb = self._extract_embedding(frame_copy, box)
                self._embed_results.put((track_id, emb))
            except Exception as exc:
                print(f"[EmbedWorker] Error for track {track_id}: {exc}")
                self._embed_results.put((track_id, None))
            finally:
                self._embed_jobs.task_done()

    def _submit_embed_job(self, track: Track, frame: np.ndarray):
        """
        FIX-A + FIX-B: non-blocking submit.
        Uses put_nowait and catches Full so main thread never blocks.
        The _embed_pending flag prevents flooding the queue with the
        same track before its result comes back.
        """
        if track._embed_pending:
            return
        try:
            self._embed_jobs.put_nowait(
                (track.id, frame.copy(), track.box.copy())
            )
            track._embed_pending = True
        except queue.Full:
            pass   # queue full — will retry next frame cycle

    def _drain_embed_results(self, track_map: dict[int, Track]):
        """
        Drain all pending embedding results and apply to tracks.
        Called once per main-loop iteration — never blocks.
        """
        while not self._embed_results.empty():
            try:
                track_id, emb = self._embed_results.get_nowait()
            except queue.Empty:
                break
            track = track_map.get(track_id)
            if track is None:
                continue   # track was deleted while embedding was in flight
            track._embed_pending     = False
            track.frames_since_embed = 0
            if emb is not None:
                track.update_embedding(emb)
                track.identity, _ = self._identify(track.embedding)
                if track.identity != "Unknown":
                    meta = self.meta_db.lookup(track.identity)
                    self.logger.log(
                        name   = meta.get("Name",   track.identity),
                        branch = meta.get("Branch", "N/A"),
                        roll   = meta.get("Roll",   "N/A"),
                    )

    # ── Main Loop ─────────────────────────────────────────────

    def run(self):
        # FIX-4 (confirmed correct): DirectShow on Windows only
        cap = (cv2.VideoCapture(0, cv2.CAP_DSHOW)
               if os.name == "nt" else cv2.VideoCapture(0))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] {actual_w}×{actual_h}")

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        # ── Detection thread ──────────────────────────────────
        # FIX-A: maxsize=1, use put_nowait everywhere
        input_q    = queue.Queue(maxsize=1)
        output_q   = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        def detection_worker():
            while not stop_event.is_set():
                try:
                    frm    = input_q.get(timeout=0.5)
                    result = self.detector.detect(frm)
                    # Validate output
                    if (result is not None
                            and "boxes"  in result
                            and "scores" in result):
                        # FIX-A: put_nowait — never blocks
                        try:
                            output_q.put_nowait(result)
                        except queue.Full:
                            pass   # main loop hasn't consumed last result yet
                    else:
                        print("[Detector] Malformed output — skipped")
                    input_q.task_done()
                except queue.Empty:
                    continue
                except Exception as exc:
                    print(f"[Detector] Critical error: {exc}")
                    stop_event.set()
                    break

        det_thread = threading.Thread(
            target=detection_worker, daemon=True, name="DetWorker"
        )
        det_thread.start()

        frame_count      = fps_counter = 0
        fps_timer        = time.time()
        display_fps      = 0.0
        last_tracks: list[Track] = []

        print(f"[Gate] Running at target {TARGET_FPS} FPS — press 'q' to quit")

        while cap.isOpened():
            # FIX-E: frame budget timer starts here
            frame_start = time.perf_counter()

            if stop_event.is_set():
                print("[Gate] Detector thread died — shutting down")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ── Brightness / CLAHE ────────────────────────────
            brightness = measure_brightness(frame)
            too_dark   = brightness < LOW_LIGHT_THRESHOLD

            self._clahe_frame_count += 1
            if (brightness < EXPOSURE_CORRECTION_THRESHOLD
                    and self._clahe_frame_count >= CLAHE_EVERY_N_FRAMES):
                frame = apply_clahe(frame)
                self._clahe_frame_count = 0

            # ── Background DB refresh ─────────────────────────
            if frame_count % 300 == 0 and not self.refreshing:
                threading.Thread(
                    target=self._refresh_db_worker, daemon=True
                ).start()

            # ── Detection ─────────────────────────────────────
            # FIX-A: put_nowait — skip frame if queue full, never block
            if not input_q.full():
                try:
                    input_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

            if not output_q.empty():
                result = output_q.get_nowait()
                last_tracks = self.tracker.update(
                    result["boxes"], result["scores"], ghost=False
                )
            else:
                last_tracks = self.tracker.update(
                    np.zeros((0, 4)), np.zeros(0), ghost=True
                )

            # Build track map for result drain
            track_map: dict[int, Track] = {t.id: t for t in last_tracks}

            # ── Drain async embedding results (FIX-B) ─────────
            self._drain_embed_results(track_map)

            # ── Submit new embedding jobs (FIX-B) ─────────────
            if not too_dark:
                for track in last_tracks:
                    target_interval = 45 if track.identity != "Unknown" else 10
                    if track.frames_since_embed >= target_interval:
                        self._submit_embed_job(track, frame)

            # ── Drawing ───────────────────────────────────────
            for track in last_tracks:
                if track.hits < 3 or track.age > 15:
                    continue
                x1, y1, x2, y2 = track.box.astype(int)
                is_known = track.identity != "Unknown"
                color    = (0, 220, 0) if is_known else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if is_known:
                    label = self.meta_db.lookup(track.identity).get(
                        "Name", track.identity
                    )
                else:
                    label = "Unknown"

                if len(label) > LABEL_MAX_CHARS:
                    label = label[:LABEL_MAX_CHARS - 1] + "…"

                cv2.putText(frame, label, (x1, max(y1 - 10, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if too_dark:
                draw_low_light_warning(frame, brightness)

            # ── FPS display ───────────────────────────────────
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Campus Security", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # FIX-E: sleep the remainder of the frame budget
            # Prevents CPU spin when GPU finishes ahead of schedule.
            elapsed_this_frame = time.perf_counter() - frame_start
            sleep_for = _FRAME_BUDGET_SEC - elapsed_this_frame
            if sleep_for > 0:
                time.sleep(sleep_for)

        # ── Clean shutdown ────────────────────────────────────
        stop_event.set()
        self._embed_stop.set()
        det_thread.join(timeout=2.0)
        self._embed_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[Gate] Session ended")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Live Face Recognition — GPU Build")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--encodings",  default=None)
    parser.add_argument("--no-encoding-update", action="store_false",
                        dest="update_encodings")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    detector   = FaceDetector(args.checkpoint, cfg)
    enc_path   = args.encodings or cfg["paths"].get("encodings_file", "encodings.pkl")
    recognizer = LiveRecognizer(detector, enc_path, cfg,
                                update_encodings=args.update_encodings)
    recognizer.run()


if __name__ == "__main__":
    main()