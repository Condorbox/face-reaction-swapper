import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DETECTION_THRESHOLD = 0.5

# Key landmark indices for Mediapipe's face mesh
LM = {
    "nose_tip":        1,
    "chin":            152,

    # Mouth
    "mouth_left":      61,
    "mouth_right":     291,
    "mouth_top":       13,
    "mouth_bottom":    14,

    # Left eye 
    "left_eye_top":    159,
    "left_eye_bottom": 145,
    "left_eye_left":   33,
    "left_eye_right":  133,

    # Right eye
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    "right_eye_left":   362,
    "right_eye_right":  263,

    # Eyebrows
    "left_eyebrow":    105,
    "right_eyebrow":   334,
}
# --------------------------------------------------------------------------------------

base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.VIDEO,
    num_faces = 1,
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# Connect to the webcam (0 = default)
cap = cv2.VideoCapture(0)
frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read from webcam, exiting.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
    
    # VIDEO mode wants a timestamp in milliseconds
    timestamp_ms = int(frame_index * (1000 / 30))  # assuming ~30fps
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_index += 1

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Print landmarks
        for name, idx in LM.items():
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, name, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("Reaction swapper", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



