import cv2
import numpy as np
from collections import deque

class ShapeRecord:
    def __init__(self, shape, size):
        self.current_shape = shape
        self.current_size = size

    def __eq__(self, other):
        if isinstance(other, ShapeRecord):
            return (self.current_shape == other.current_shape and
                    self.current_size == other.current_size)
        return False

    def __hash__(self):
        return hash((self.current_shape, self.current_size))

    def __repr__(self):
        return f"ShapeRecord(shape={self.current_shape}, size={self.current_size})"

# Function to classify the shape based on aspect ratio and contour fitting
def classify_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        # if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
        #     return "Square"
        # else:
        #     return "Rectangle"
        return "Rectangle"
    elif len(approx) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (x, y), (major_axis, minor_axis), angle = ellipse
        # Check if the major and minor axis lengths are similar (ellipse-like)
        if 0.8 <= major_axis / minor_axis <= 1.2:
            return "Ellipse"
    else:
        return "Unknown"

# Start capturing video
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path

# Create a deque to store the last few shape classifications (size 5)
history = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur and thresholding to detect objects
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    # Use morphological operations to reduce noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Detect contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    current_shape = "Unknown"
    current_size = "unknown"

    i = 0
    for contour in contours:
        if i == 0: # Skip the first contour (the whole frame)
            i = 1
            continue
        area = cv2.contourArea(contour)
        if area < 1000 or area > 30000:  # Filter small contours to avoid noise
            continue
        if area > 20000:
            current_size = "Large"
        elif area > 10000:
            current_size = "Medium"
        else:
            current_size = "Small"
        
        # Classify the detected shape
        shape = classify_shape(contour)

        # Draw the contour on the frame
        cv2.drawContours(frame, [contour], -1, (0, 200, 0), 1)

        if shape != "Unknown":
            current_shape = shape

            # Find the contour's center to display the shape
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update the history of detected shapes
    history.append(ShapeRecord(current_shape, current_size))

    # Only update display if the last 5 frames have the same shape
    if history.count(history[0]) == len(history) and history[0].current_shape != "Unknown" and history[0] != None:
        display_shape = history[0].current_shape
        displat_size = history[0].current_size
        cv2.putText(frame, f"Detected Shape: {display_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"size: {displat_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detecting shape...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Display the frame with detected shapes
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
