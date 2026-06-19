import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DETECTION_THRESHOLD = 0.5

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
        h, w = frame.shape[:2]
        for landmark in result.face_landmarks[0]:  # first (only) face
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Reaction swapper", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
