import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('tgmt_hand_v03_1.h5')

# Define the labels for the classes
labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}


# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the region of interest (ROI) size
roi_start = (100, 100)
roi_end = (400, 400)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle for the ROI
    cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)

    # Extract the ROI
    roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]

    # Preprocess the ROI
    resized = cv2.resize(roi, (64, 64))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 3))

    # Make predictions if there is a sign in the ROI
    if np.mean(roi) > 10:  # You can adjust this threshold based on your application
        # Make predictions
        predictions = model.predict(reshaped)
        predicted_class = np.argmax(predictions[0])
        predicted_label = labels[predicted_class]

        # Get the percentage of each class prediction
        percentages = predictions[0] * 100
        class_percentages = [(labels[i], percentages[i]) for i in range(len(labels))]

        # Sort class percentages by their prediction value
        class_percentages.sort(key=lambda x: x[1], reverse=True)

        # Display the predicted label and percentage on the frame
        text = f'Prediction: {predicted_label} ({class_percentages[0][1]:.2f}%)'
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
