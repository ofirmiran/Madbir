from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import streamlink

# Global variables initialization
mouse_pos = (0, 0)
selected_object_id = None
track_history = defaultdict(lambda: {'points': [], 'disappeared': 0})
zoom_size = 100
zoom_center = None
selected_items = None
tk_window_open = False
zoom_visible = True


def load_models():
    model_path1 = 'yolov8x.pt'
    #5 epochs
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train5\weights\best.pt"
    #25 epochs
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train31\weights\best.pt"
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train24\weights\best.pt"
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train25\weights\best.pt"
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train26\weights\best.pt"
    #model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train31\weights\best.pt"
    model_path2 = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train32\weights\best.pt"
    #model_path2 = r"C:\Users\dvirm\PycharmProjects\pythonProject2\runs\detect\train76\weights\best.pt"
    model1 = YOLO(model_path1)
    model2 = YOLO(model_path2)
    return model1, model2


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def process_detection_results(results, model):
    sea_related_objects = ["boat", "ship", "submarine", "person", "swimmer", "float", "bird" , "car"]  # Example list, adjust according to your model's capabilities
    detections = []
    if results and results[0].boxes and len(results[0].boxes.xywh) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        labels = [model.names[int(cls_idx)] for cls_idx in class_indices]
        for box, label, confidence, track_id in zip(boxes, labels, confidences, track_ids):
            if label in sea_related_objects:
                detections.append({'bbox': box, 'label': label, 'confidence': confidence, 'track_id': track_id})
    return detections


def merge_detections(detections1, detections2, iou_threshold=0.1):
    merged_detections = []

    # Flag detections from the second model that have been merged
    merged_indices = set()

    for det1 in detections1:
        x1, y1, w1, h1 = det1['bbox']
        x1_min = x1 - w1 / 2
        y1_min = y1 - h1 / 2
        x1_max = x1 + w1 / 2
        y1_max = y1 + h1 / 2
        box1 = [x1_min, y1_min, x1_max, y1_max]

        best_iou = 0
        best_index = -1
        for index, det2 in enumerate(detections2):
            x2, y2, w2, h2 = det2['bbox']
            x2_min = x2 - w2 / 2
            y2_min = y2 - h2 / 2
            x2_max = x2 + w2 / 2
            y2_max = y2 + h2 / 2
            box2 = [x2_min, y2_min, x2_max, y2_max]

            iou = calculate_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold:
            merged_detections.append({'bbox': det1['bbox'], 'label': det1['label'], 'confidence': det1['confidence'], 'color': 'green', 'track_id': det1['track_id']})
            merged_indices.add(best_index)
        else:
            merged_detections.append({**det1, 'color': 'red'})

    # Add remaining detections from model 2 as blue
    for index, det2 in enumerate(detections2):
        if index not in merged_indices:
            merged_detections.append({**det2, 'color': 'blue'})

    return merged_detections


def draw_detections(frame, detections_with_color, font, font_scale, line_type):
    for det in detections_with_color:
        bbox = det['bbox']
        label = det['label']
        confidence = det['confidence']
        color_code = det['color']

        # Set color based on the detection source
        if color_code == 'red':
            color = (0, 0, 255)  # Red
        elif color_code == 'blue':
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        # Convert bbox from xywh to x1, y1, x2, y2 format
        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw bounding box and label with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f} (ID: {det['track_id']})"
        cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, (255, 255, 255), line_type)


def process_frame(frame, model1, model2, track_history, font, font_scale, line_type):
    global selected_items
    object_counter = defaultdict(int)

    # Process frame with both models
    results1 = model1.track(frame, persist=True)
    results2 = model2.track(frame, persist=True)

    # Pass the respective model object to process_detection_results
    detections1 = process_detection_results(results1, model1)
    detections2 = process_detection_results(results2, model2)

    # Merge detections
    merged_detections = merge_detections(detections1, detections2)


    # Now actually draw the detections on the frame
    draw_detections(frame, merged_detections, font, font_scale, line_type)



    return frame


# Ensure all other necessary functions (e.g., draw_detections, cleanup, etc.) are defined.
def cleanup(cap):
    """Release resources."""
    cap.release()
    cv2.destroyAllWindows()

import cv2

def main():
    global track_history, model1, model2

    # Load models
    model1, model2 = load_models()

    # Setup window and trackbar
    cv2.namedWindow("YOLOv8 Tracking")
    #video_path = r"C:\Users\Mizui_Meida\Desktop\קרנצ\Untitled_0003.mov" #Swim
    video_path = r"D:\הקלטות אביב נמל\19.1.25\Untitled_0029.mov"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\copy\Untitled_0000.mov"

    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0003.mov" #Swim
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0004.mov" #Swim
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0033.mov" #far ship
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0053.mov"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0038.mov" #far ship 2
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0003.mov" #far ship 2
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0052.mov" #close ship
    #video_path = r"C:\Users\Mizui_Meida\Desktop\VideoDataSaar6\49.mov" #close ship
    #video_path = r"C:\Users\Mizui_Meida\Desktop\New folder\Blackmagic HyperDeck Studio Mini_0013.mov" #submarine
    #video_path = r"C:\Users\Mizui_Meida\Desktop\New folder\Blackmagic HyperDeck Studio Mini_0045.mov" #people on boat

#aviv100

    #video_path = r"D:\vis_ir_vids\videos\07_21_21_15_32_31.mp4"
    #video_path = r"D:\vis_ir_vids\videos\07_21_21_14_42_30.mp4"
    #video_path = r"D:\vis_ir_vids\videos\07_22_21_16_32_43.mp4"

#galgal_hatsala_person
    #video_path = r"D:\mm\video\1.mp4"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\galal_sorted\20594_1710838800000_1710842400000.mp4"

#MOSP_NO_TRAIN
    #video_path = r"D:\mm\video\examp2.mov"
    #video_path = r"D:\mm\video\2_tar.mp4"
    #video_path = r"D:\mm\video\1_tar.mp4"

    #video_path = r"C:\Users\Mizui_Meida\Desktop\גילויים\Blackmagic HyperDeck Studio Mini_0042.mov"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\29.05.2024 ב\Untitled_0006.mov"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\א 29.05.2024\Untitled_0015.mov"
    #video_path = r"C:\Users\Mizui_Meida\Desktop\05.02.24\Untitled_0046.mov"

    #video_path = r"C:\Users\Mizui_Meida\Desktop\lidar-navy\recordings and img\recordings and img (2).mp4"

    #video_path = r"D:\dvir\Untitled.mp4"




    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track whether the user is interacting with the trackbar
    user_interacting = False

    def on_trackbar(val):
        nonlocal user_interacting
        user_interacting = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    cv2.createTrackbar("Frame", "YOLOv8 Tracking", 0, total_frames - 1, on_trackbar)

    # Setup for video capture, drawing, etc.
    font, font_scale, line_type = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2

    while cap.isOpened():
        # If user is not interacting, video plays normally
        if not user_interacting:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
        else:
            # When user interaction is detected, the frame is already updated by the trackbar callback
            user_interacting = False  # Reset the flag after handling
            ret, frame = cap.read()
            if not ret:
                continue  # Skip frame processing if frame is not read successfully

        # Process each frame with both models and merge their detections
        frame = process_frame(frame, model1, model2, track_history, font, font_scale, line_type)
        #cv2.imshow("YOLOv8 Tracking", frame)
        cv2.imshow("detection", frame)

        # Update trackbar position to reflect current frame
        if not user_interacting:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Frame", "YOLOv8 Tracking", current_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Cleanup resources
    cleanup(cap)

if __name__ == "__main__":
    main()






