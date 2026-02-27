import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# 1. Initialize
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

DB_FOLDER = "known_students"
student_data = {} 
 
if not os.path.exists(DB_FOLDER):
    print(f"❌ Error: Folder '{DB_FOLDER}' not found!")
    os.makedirs(DB_FOLDER)
    print(f"Created '{DB_FOLDER}'. Please put your 5 images there and restart.")
    exit()

print("🔄 Processing student database...")

# 2. Enrollment
for filename in os.listdir(DB_FOLDER):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        # Get name from filename (e.g., 'Moksh_1.jpg' -> 'Moksh')
        name = filename.split('_')[0].split('.')[0]
        print(f"🔍 Found image for: {name} ({filename})")
        
        img = cv2.imread(os.path.join(DB_FOLDER, filename))
        if img is None: continue
        
        faces = app.get(img)
        if faces:
            embedding = faces[0].normed_embedding
            if name not in student_data:
                student_data[name] = []
            student_data[name].append(embedding)

# Create Final Average Embeddings
final_db = {}
for name, embeddings in student_data.items():
    avg_emb = np.mean(embeddings, axis=0)
    final_db[name] = avg_emb / np.linalg.norm(avg_emb)

if len(final_db) == 0:
    print("❌ No students loaded. Check your 'known_students' folder!")
    exit()

print(f"✅ Successfully loaded {len(final_db)} students: {list(final_db.keys())}")

# 3. Live Attendance Loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)
    for face in faces:
        live_emb = face.normed_embedding
        best_name, best_score = "Unknown", 0
        
        for name, master_emb in final_db.items():
            score = np.dot(live_emb, master_emb)
            if score > best_score:
                best_score = score
                best_name = name

        identity = best_name if best_score > 0.45 else "Unknown"
        
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"{identity} ({best_score:.2f})", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("College Gate - Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()