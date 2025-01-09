import cv2 as cv
import time
import geocoder
import os

# Reading label names from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file
# Defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV backend for Raspberry Pi
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)       # Use CPU as target
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture("Potholes.mp4") 

# Get input video FPS and dimensions
input_fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

# Desired FPS for processing
desired_fps = 10
frame_skip_interval = int(input_fps / desired_fps)

# Initialize video writer for the result
result = cv.VideoWriter('result.mp4', 
                        cv.VideoWriter_fourcc(*'mp4v'), 
                        desired_fps, 
                        (int(width), int(height)))

# Parameters for result saving and coordinates
g = geocoder.ip('me')
result_path = "pothole_coordinates"
os.makedirs(result_path, exist_ok=True)  # Ensure directory exists
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
processed_frame_counter = 0
i = 0
b = 0

# Detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break

    # Skip frames to achieve desired FPS
    if frame_counter % frame_skip_interval != 0:
        continue

    processed_frame_counter += 1
    # Analyze the stream with detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height
        # Drawing detection boxes and saving results
        if len(scores) != 0 and scores[0] >= 0.70:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label,
                           (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                if i == 0:
                    cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                    with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                        f.write(str(g.latlng))
                    i += 1
                elif (time.time() - b) >= 2:
                    cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                    with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                        f.write(str(g.latlng))
                    b = time.time()
                    i += 1

    # Writing FPS on frame
    endingTime = time.time() - starting_time
    fps = processed_frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), 
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Showing and saving result
    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# End
cap.release()
result.release()
cv.destroyAllWindows()
