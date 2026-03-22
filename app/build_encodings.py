"""
Build and manage face encodings (embeddings) for recognition.

Scans face crops per identity, extracts embeddings via InsightFace,
stores as dict[str, List[np.ndarray]] in a .pkl file.  Supports
incremental rebuild using file modification timestamps.
"""

import os
import sys
import time
import pickle
import argparse
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Encoding builder
# ---------------------------------------------------------------------------

class EncodingBuilder:
    """Builds and manages face embeddings for the recognition pipeline.

    Parameters
    ----------
    known_students_dir : str
        Root directory with sub-folders per identity, each containing face crops.
    encodings_path : str
        Path to save/load the encodings .pkl file.
    """

    def __init__(self, known_students_dir: str, encodings_path: str):
        self.known_dir = known_students_dir
        self.enc_path = encodings_path
        self.data: dict[str, list[np.ndarray]] = {}
        self._timestamps: dict[str, float] = {}  # file → mtime

        # Load existing encodings + timestamps
        meta_path = self._meta_path()
        if os.path.exists(self.enc_path) and os.path.exists(meta_path):
            with open(self.enc_path, "rb") as f:
                self.data = pickle.load(f)
            with open(meta_path, "rb") as f:
                self._timestamps = pickle.load(f)
            print(f"📂 Loaded existing encodings: {len(self.data)} identities")
        else:
            print("📂 No existing encodings found — will build from scratch")

        # InsightFace for embedding extraction
        try:
            from insightface.app import FaceAnalysis
            self.face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError:
            print("❌ InsightFace not installed — cannot extract embeddings")
            self.face_app = None

    def _meta_path(self) -> str:
        base, ext = os.path.splitext(self.enc_path)
        return f"{base}_meta.pkl"

    def _needs_update(self, filepath: str) -> bool:
        """Check if a file needs reprocessing based on modification time."""
        current_mtime = os.path.getmtime(filepath)
        cached_mtime = self._timestamps.get(filepath, 0)
        return current_mtime > cached_mtime

    def build(self, rebuild: bool = False):
        """Scan face crops and extract embeddings.

        Parameters
        ----------
        rebuild : bool
            If True, reprocess ALL crops. If False, only process new/changed files.
        """
        if self.face_app is None:
            print("❌ Cannot build encodings without InsightFace")
            return

        if rebuild:
            print("🔄 Full rebuild requested — clearing existing encodings")
            self.data = {}
            self._timestamps = {}

        if not os.path.exists(self.known_dir):
            print(f"❌ Directory not found: {self.known_dir}")
            return

        processed = 0
        skipped = 0
        failed_files = []

        # Scan directory structure: known_students/{name}/{name}_0.jpg
        for identity_name in sorted(os.listdir(self.known_dir)):
            identity_dir = os.path.join(self.known_dir, identity_name)

            if not os.path.isdir(identity_dir):
                # Also support flat structure: known_students/Name_0.jpg
                if identity_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    name = identity_name.split("_")[0].split(".")[0]
                    filepath = os.path.join(self.known_dir, identity_name)
                    if not rebuild and not self._needs_update(filepath):
                        skipped += 1
                        continue
                    emb = self._extract(filepath)
                    if emb is not None:
                        if name not in self.data:
                            self.data[name] = []
                        self.data[name].append(emb)
                        self._timestamps[filepath] = os.path.getmtime(filepath)
                        processed += 1
                    else:
                        failed_files.append(filepath)
                continue

            for filename in sorted(os.listdir(identity_dir)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                filepath = os.path.join(identity_dir, filename)

                if not rebuild and not self._needs_update(filepath):
                    skipped += 1
                    continue

                emb = self._extract(filepath)
                if emb is not None:
                    if identity_name not in self.data:
                        self.data[identity_name] = []
                    self.data[identity_name].append(emb)
                    self._timestamps[filepath] = os.path.getmtime(filepath)
                    processed += 1
                    print(f"  ✅ {identity_name}/{filename}")
                else:
                    print(f"  ⚠️  No face in {identity_name}/{filename}")
                    failed_files.append(filepath)

        # Save
        self._save()

        # Save failed files report
        if failed_files:
            report_path = "failed_encodings.txt"
            with open(report_path, "w") as f:
                for item in failed_files:
                    f.write(f"{item}\n")
            print(f"\n⚠️  {len(failed_files)} images failed to generate encodings.")
            print(f"📄 Detailed list saved to: {report_path}")

        print(f"\n📊 Encoding Summary:")
        print(f"   Processed: {processed} crops")
        print(f"   Skipped (unchanged): {skipped} crops")
        print(f"   Identities: {len(self.data)}")
        for name, embs in self.data.items():
            print(f"     • {name}: {len(embs)} embeddings")

    def _extract(self, filepath: str) -> np.ndarray | None:
        """Extract embedding from a single image file."""
        img = cv2.imread(filepath)
        if img is None:
            return None

        # Try with default threshold first
        faces = self.face_app.get(img)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding
            
        # Fallback 1: Try with lower threshold
        if hasattr(self.face_app, 'models') and 'detection' in self.face_app.models:
            old_thresh = self.face_app.models['detection'].det_thresh
            self.face_app.models['detection'].det_thresh = 0.2
            faces = self.face_app.get(img)
            self.face_app.models['detection'].det_thresh = old_thresh # restore
            if faces and len(faces) > 0:
                return faces[0].normed_embedding

        # Fallback 2: Try with padding (sometimes helps detector context)
        h, w = img.shape[:2]
        pad_h, pad_w = int(h * 0.2), int(w * 0.2)
        padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, 
                                   cv2.BORDER_CONSTANT, value=[114, 114, 114])
        faces = self.face_app.get(padded)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding

        return None

    def _save(self):
        """Save encodings and metadata."""
        with open(self.enc_path, "wb") as f:
            pickle.dump(self.data, f)
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self._timestamps, f)
        print(f"💾 Encodings saved to {self.enc_path}")

    def get_mean_embeddings(self) -> dict[str, np.ndarray]:
        """Get mean embedding per identity for recognition."""
        mean_db = {}
        for name, embeddings in self.data.items():
            if len(embeddings) > 0:
                avg = np.mean(embeddings, axis=0)
                mean_db[name] = avg / np.linalg.norm(avg)
        return mean_db


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build face encodings")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rebuild", action="store_true",
                        help="Reprocess all crops (ignore cache)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    known_dir = cfg["paths"].get("known_students", "known_students")
    enc_path = cfg["paths"].get("encodings_file", "encodings.pkl")

    builder = EncodingBuilder(known_dir, enc_path)
    builder.build(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
