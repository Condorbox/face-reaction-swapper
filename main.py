import cv2

# Connect to the webcam (0 = default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read from webcam, exiting.")
        break

    cv2.imshow("Reaction swapper", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
