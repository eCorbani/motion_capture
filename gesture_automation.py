# Import necessary libraries
import cv2
import os
import numpy as np
import math


# Open a video capture object (webcam in this case)
cap = cv2.VideoCapture(0)

# Initialize a counter for detected gestures
gesture_count = 0

# Start an infinite loop for video processing
while True:
    try:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)

        # Define a 3x3 kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Define a region of interest (ROI) within the frame
        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create a binary mask for the skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to the mask
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours are found
        if contours:
            # Find the contour with the maximum area
            cnt = max(contours, key=lambda x: cv2.contourArea(x))

            # Approximate the contour to reduce the number of points
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Create a convex hull around the contour
            hull = cv2.convexHull(cnt)

            # Calculate areas and ratios for gesture recognition
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(cnt)
            ratio_area = ((hull_area - cnt_area) / cnt_area) * 100

            # Find convexity defects in the contour
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            # Reset gesture count
            gesture_count = 0

            # Iterate over defects and identify gestures
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                # Calculate distances and angles for gesture recognition
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
                d = (2 * ar) / a
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # Identify valid gestures based on angle and distance
                if angle <= 90 and d > 30:
                    gesture_count += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)
                    cv2.line(roi, start, end, [0, 255, 0], 2)

            # Display gestures on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX

            if gesture_count == 0:
                # Display waiting message or execute specific actions based on gesture
                if cnt_area < 2000:
                    cv2.putText(frame, 'Waiting data', (0, 50), font, 2,
                                (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    executed = False
                    if ratio_area < 12 and not executed:
                        cv2.putText(frame, '0 = Browser', (0, 50), font, 2,
                                    (0, 0, 255), 3, cv2.LINE_AA)
                        os.system("start Chrome.exe --window-size=600,400")
                    else:
                        cv2.putText(frame, '1 = Notepad', (0, 50), font, 2,
                                    (0, 0, 255), 3, cv2.LINE_AA)
                        os.system("start notepad.exe --window-size=600,400")

            elif gesture_count == 2:
                cv2.putText(frame, '2 = Microsoft Edge', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)
            #    os.system("start msedge.exe --window-size=600,400")

            elif gesture_count == 3 and ratio_area < 27:
                cv2.putText(frame, '3 = Firefox', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)
            #    os.system("start firefox.exe --window-size=600,400")

        # Display the processed frame
        cv2.imshow('Frame', frame)

    except Exception as e:
        # Handle exceptions and print error messages
        print(f"Error: {e}")

    # Wait for a key press and break the loop if 'Esc' key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
cap.release()
