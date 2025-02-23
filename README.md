# YOLOv8 Real-Time Object Detection

This project utilizes the YOLOv8 model for real-time object detection using a webcam. The model identifies various objects and overlays bounding boxes along with confidence scores on the detected objects.

## Installation
To run this project, ensure you have the required dependencies installed:
```bash
pip install ultralytics opencv-python cvzone numpy
```

## Usage
1. Connect a webcam to your system.
2. Run the script to start real-time object detection:
```bash
python object_detection.py
```

### Features
- Uses `YOLOv8` for object detection.
- Detects and classifies objects in real-time.
- Draws bounding boxes and confidence scores on detected objects.

### Example Output
When an object is detected, the system will display:
```
Detected: person (Confidence: 0.85)
Detected: bicycle (Confidence: 0.78)
```

## Model & Technologies Used
- **YOLOv8** for real-time object detection
- **OpenCV** for image processing
- **cvzone** for visualization enhancements
- **Python** as the programming language

## License
This project is open-source and free to use. Feel free to modify and improve it!
