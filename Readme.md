Gesture Recognition System

This Python code implements a real-time gesture recognition system using the OpenCV library. The system captures video from a webcam and processes it to identify hand gestures. Based on the recognized gestures, the system can trigger specific actions, such as opening different applications.
Prerequisites

Ensure you have the following libraries installed:

    OpenCV (cv2)
    NumPy (numpy)
    Math (math)
    OS (os)

Install any missing libraries using the following:

bash

pip install opencv-python numpy

Usage

    Run the script:

    bash

    python gesture_recognition.py

    The system will capture video from the default webcam (index 0).

    Make gestures within the defined region of interest (ROI) to trigger specific actions.

    Press 'Esc' to exit the application.

Gesture Recognition Logic

The code uses computer vision techniques to recognize hand gestures based on contour analysis and convexity defects. The main steps include:

    Capturing video frames from the webcam.
    Applying morphological operations and color filtering to detect the hand.
    Identifying the contours of the hand and approximating them to reduce points.
    Calculating convex hull and convexity defects for gesture analysis.
    Recognizing specific gestures based on angles and distances.

Actions Triggered by Gestures

    Gesture Count 0:
        Display "Waiting data" if hand area is below a threshold.
        Open Chrome browser if the hand area ratio is below a certain value.
        Open Notepad if the hand area ratio is above a certain value.

    Gesture Count 2:
        Open Microsoft Edge.

    Gesture Count 3:
        Open Firefox if the hand area ratio is below a certain value.

Note

    Adjust the ROI and color thresholds according to your environment for optimal performance.
    Uncomment the lines within gesture counts to activate the corresponding actions.

Troubleshooting

    If the script fails or throws errors, ensure that the required libraries are installed and accessible.

Feel free to customize the code based on your requirements and experiment with different gestures and actions. Happy coding!