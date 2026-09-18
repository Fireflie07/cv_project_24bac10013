import cv2
import numpy as np

def main():
    # Load OpenCV's pre-trained Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load XML classifier cascade.")
        return

    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Computer Vision Emotion Classifier Running...")
    print("Press 'q' in the video window to quit.")

    # Emotion classes for simulation labeling
    emotions = ["Happy", "Neutral", "Surprised", "Focused"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab video frame.")
            break

        # Convert frame to grayscale for face detection processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )

        # Process each detected face ROI (Region of Interest)
        for (x, y, w, h) in faces:
            # Draw bounding box around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract ROI
            roi_gray = gray_frame[y:y+h, x:x+w]

            # Assign simulated prediction label based on image intensity stats
            avg_intensity = np.mean(roi_gray)
            label = emotions[int(avg_intensity) % len(emotions)]

            # Overlay classification result and confidence text on frame
            display_text = f"State: {label}"
            cv2.putText(
                frame, display_text, (x, y - 10),
                cv2.COLOR_BGR2GRAY, 0.7, (0, 255, 0), 2
            )

        # Display the output window
        cv2.imshow("Real-Time Computer Vision Classifier - Elizabeth George", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()