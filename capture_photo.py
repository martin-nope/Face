import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ===================== SETTINGS =====================
# Use the direct path to your downloaded .tflite file
MODEL_PATH = r"C:\Users\Moksh\Projects\face-recog\blaze_face_short_range.tflite"
SAVE_PATH = "known_students"
os.makedirs(SAVE_PATH, exist_ok=True)

name = input("Enter the person's name: ")
count = 0

# Initialize MediaPipe Tasks Face Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
print(f"Instructions: Press 's' to save a photo. Capture 5 photos. (Total: {count}/5)")

while count < 5:
    ret, frame = cap.read()
    if not ret: break
    
    # MediaPipe Tasks requires a mp.Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Run detection
    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # Draw box for visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Capture logic
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Crop and save
                face_crop = frame[max(0, y):y+h, max(0, x):x+w]
                if face_crop.size > 0:
                    img_name = os.path.join(SAVE_PATH, f"{name}_{count}.jpg")
                    cv2.imwrite(img_name, face_crop)
                    count += 1
                    print(f"Saved {img_name} ({count}/5)")

    cv2.putText(frame, f"Captured: {count}/5 | Press 's' to save", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Enrollment - Stay Still", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Enrollment Complete!")