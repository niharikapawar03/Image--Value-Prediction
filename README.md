This project demonstrates object tracking in images and videos using color detection with OpenCV in Python. The code runs in Google Colab, allowing users to upload images or videos and detect objects based on their color ranges.

Bounding boxes and tracking markers are drawn around the detected objects.

🚀 Features

Upload and process videos for object tracking.

Upload and process images for object detection.

Detects objects based on HSV color range.

Draws bounding boxes, center points, and labels (Tracked).

Adjustable color ranges for tracking different objects (e.g., animals, objects of specific colors).

📂 Project Structure
├── Image-Value Prediction.ipynb   # Jupyter notebook with complete code
├── README.md                      # Project documentation

🛠️ Requirements

The following libraries are required:

Python 3.x

OpenCV

NumPy

Google Colab (for easy execution & file upload interface)

Install dependencies (if not already installed):

pip install opencv-python numpy

▶️ Usage
1. Run in Google Colab

Open the notebook in Google Colab.

Upload your video or image file when prompted.

2. Video Tracking Example
from google.colab import files
uploaded = files.upload()

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

video_path = list(uploaded.keys())[0]
cap = cv2.VideoCapture(video_path)

# Define color range (HSV)
lower_color = np.array([10, 50, 50])
upper_color = np.array([30, 255, 255])

# Frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Contour detection + bounding boxes

3. Image Tracking Example
from google.colab import files
uploaded = files.upload()

img_path = next(iter(uploaded))
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, lower_color, upper_color)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes & centers

🎯 Output

Detected objects are highlighted with green bounding boxes.

Object center is marked with a red dot.

Label Tracked is displayed above detected regions.

⚙️ Customization

Modify the HSV range in the code to detect different colors:

lower_color = np.array([H_min, S_min, V_min])
upper_color = np.array([H_max, S_max, V_max])

📌 Example Applications

Wildlife/animal tracking from videos

Object detection in surveillance footage

Tracking specific colored objects in research/experiments

👩‍💻 Author

Niharika Pawar
