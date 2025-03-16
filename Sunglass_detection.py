import cv2
import sys  # Import sys for exit()

# Use the correct path (double backslashes or raw string)
face_cascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Downloads\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\ASUS\Downloads\haarcascade_eye.xml")

# Check if cascades are loaded properly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Haar cascade XML files not loaded. Check the file paths!")
    sys.exit()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Region of interest (face)
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # If no eyes detected → Assume sunglasses are present
        if len(eyes) == 0:
            text = "Sunglasses Detected!"
            color = (0, 0, 255)  # Red
        else:
            text = "No Sunglasses"
            color = (0, 255, 0)  # Green

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display output
    cv2.imshow('Sunglasses Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
