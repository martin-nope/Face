from deepface import DeepFace
result = DeepFace.find(
    img_path="known_students/Moksh_001.jpg",  # Pick one of your photos
    db_path="known_students",
    model_name="ArcFace",
    distance_metric="euclidean_l2",
    enforce_detection=False
)
print(result)