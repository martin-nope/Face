import cv2
from deepface import DeepFace
import os

# ===================== CONFIGURATION =====================
DB_FOLDER          = "known_students"
MODEL_NAME         = "ArcFace"
DISTANCE_THRESHOLD = 0.65
DETECTOR_BACKEND   = "ssd"   
PROCESS_EVERY_N    = 2

WINDOW_TITLE       = "Live Face Verification - Press q to quit"

print("Starting DeepFace-only live verification")
print("---------------------------------------------------\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N != 0:
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    try:
        dfs = DeepFace.find(
            img_path=frame,
            db_path=DB_FOLDER,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric="euclidean_l2",
            enforce_detection=False,
            silent=True
        )

        if dfs and len(dfs) > 0 and not dfs[0].empty:

            best = dfs[0].iloc[0]

            distance = best.get(f"{MODEL_NAME}_euclidean_l2", 999.0)
            identity = best["identity"]

            # Get face location from DeepFace
            source_x = int(best["source_x"])
            source_y = int(best["source_y"])
            source_w = int(best["source_w"])
            source_h = int(best["source_h"])

            filename = os.path.basename(identity)
            name = filename.split("_")[0]

            if distance < DISTANCE_THRESHOLD:
                label = f"{name} - Enrolled"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            print(f"Distance: {distance:.4f} | {name}")

            # Draw bounding box
            cv2.rectangle(
                frame,
                (source_x, source_y),
                (source_x + source_w, source_y + source_h),
                color,
                4
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (source_x, source_y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                1.1,
                color,
                3
            )

    except Exception as e:
        print("DeepFace error:", str(e))

    cv2.imshow(WINDOW_TITLE, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")