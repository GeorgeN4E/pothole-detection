import cv2 as cv
import time
import os

# Configure video source
#video_source = input("Enter 'file' for local video or 'url' for Raspberry Pi live feed: ").strip().lower()
video_source = "file"
#video_source = "url"

if video_source == 'file':
    #video_path = input("Enter the path to your local video: ").strip()
    video_path = r"D:\Sublime Text\Python\pothole-detection\Potholes.mp4"
    video_path = "D:\\Sublime Text\\Python\\pothole-detection\\Codlea_Brasov.mp4"
    cap = cv.VideoCapture(video_path)
elif video_source == 'url':
    url = input("Enter the URL of the Raspberry Pi live feed: ").strip()
    cap = cv.VideoCapture(url)
else:
    print("Invalid input! Exiting...")
    exit()

# Check if the video source is valid
if not cap.isOpened():
    print("Failed to open video source!")
    exit()

# Load YOLO model
net = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Get input video FPS and dimensions
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv.CAP_PROP_FPS)

# Desired FPS for processing
desired_fps = 10
frame_skip_interval = int(input_fps / desired_fps) if input_fps > desired_fps else 1

# Initialize video writer for the result
result = cv.VideoWriter('result.mp4', 
                        cv.VideoWriter_fourcc(*'mp4v'), 
                        desired_fps, 
                        (width, height))

# Detection parameters
Conf_threshold = 0.5
NMS_threshold = 0.4

# Start processing
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or stream.")
        break

    # Skip frames to achieve the desired FPS
    if frame_counter % frame_skip_interval != 0:
        frame_counter += 1
        continue

    # Perform object detection
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = f'Pothole: {int(score * 100)}%'
        cv.rectangle(frame, box, (0, 255, 0), 2)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame number
    frame_counter += 1
    cv.putText(frame, f'Frame: {frame_counter}', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame
    cv.imshow('Live Feed', frame)
    result.write(frame)

    # Quit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Cleanup
cap.release()
result.release()
cv.destroyAllWindows()
