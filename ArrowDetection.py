import cv2
import imutils
import numpy as np
from ultralytics import YOLO

# פונקציה לציור חץ מעוצב
def draw_styled_arrow(image, start_point, end_point, color, thickness=2, arrow_magnitude=10):
    """
    מצייר חץ מעוצב בתמונה.
    """
    cv2.line(image, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)
    angle = np.arctan2(start_point[1] - end_point[1], start_point[0] - end_point[0])
    p1 = (
        int(end_point[0] + arrow_magnitude * np.cos(angle + np.pi / 6)),
        int(end_point[1] + arrow_magnitude * np.sin(angle + np.pi / 6)),
    )
    p2 = (
        int(end_point[0] + arrow_magnitude * np.cos(angle - np.pi / 6)),
        int(end_point[1] + arrow_magnitude * np.sin(angle - np.pi / 6)),
    )
    points = np.array([end_point, p1, p2], dtype=np.int32)
    cv2.fillPoly(image, [points], color)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# נתיב לווידאו
video_path = r"D:\vis_ir_vids\videos\07_22_21_16_32_43.mp4"
cam = cv2.VideoCapture(video_path)

# טוען את YOLO
model = YOLO('yolov8x.pt')  # החלף למודל הרצוי

firstFrame = None
tracked_arrows = {}
MAX_LIFE = 10
next_arrow_id = 0

# משתנה לשמירת הפריים המקורי
original_frame = None

def nothing(x):
    pass

cv2.namedWindow("Settings")
cv2.createTrackbar("Area", "Settings", 11, 5000, nothing)
cv2.createTrackbar("Threshold", "Settings", 30, 255, nothing)
cv2.createTrackbar("Blur", "Settings", 8, 51, nothing)
cv2.createTrackbar("Hue Min", "Settings", 90, 180, nothing)
cv2.createTrackbar("Hue Max", "Settings", 130, 180, nothing)
cv2.createTrackbar("Merge Distance", "Settings", 9, 200, nothing)
cv2.createTrackbar("Min Arrow Distance", "Settings", 20, 200, nothing)

fps = int(cam.get(cv2.CAP_PROP_FPS))
print(f"FPS of the video: {fps}")
frame_delay = int(1000 / fps)

while True:
    ret, img = cam.read()

    if not ret:
        print("Video ended or cannot be read.")
        break

    # שמירת הפריים המקורי
    original_frame = img.copy()

    area = cv2.getTrackbarPos("Area", "Settings")
    threshold_value = cv2.getTrackbarPos("Threshold", "Settings")
    blur_value = cv2.getTrackbarPos("Blur", "Settings")
    hue_min = cv2.getTrackbarPos("Hue Min", "Settings")
    hue_max = cv2.getTrackbarPos("Hue Max", "Settings")
    merge_distance = cv2.getTrackbarPos("Merge Distance", "Settings")
    min_arrow_distance = cv2.getTrackbarPos("Min Arrow Distance", "Settings")

    if blur_value % 2 == 0:
        blur_value += 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([hue_min, 50, 50])
    upper_blue = np.array([hue_max, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (blur_value, blur_value), 0)

    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, threshold_value, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    filteredThreshImg = cv2.bitwise_and(threshImg, threshImg, mask=cv2.bitwise_not(mask))

    cnts = cv2.findContours(filteredThreshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rectangles = []
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x, y, x + w, y + h])

    rectangles = np.array(rectangles)
    merged_rectangles = []

    for i, rect1 in enumerate(rectangles):
        x1, y1, x2, y2 = rect1
        merged = False
        for j, rect2 in enumerate(merged_rectangles):
            mx1, my1, mx2, my2 = rect2
            if (x1 <= mx2 + merge_distance and x2 >= mx1 - merge_distance and
                    y1 <= my2 + merge_distance and y2 >= my1 - merge_distance):
                merged_rectangles[j] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                merged = True
                break
        if not merged:
            merged_rectangles.append([x1, y1, x2, y2])

    for x1, y1, x2, y2 in merged_rectangles:
        center_x = (x1 + x2) // 2
        top_y = y1

        matched_arrow = None
        for arrow_id, (arrow_x, arrow_y, life) in tracked_arrows.items():
            if euclidean_distance((center_x, top_y), (arrow_x, arrow_y)) < min_arrow_distance:
                matched_arrow = arrow_id
                break

        if matched_arrow is not None:
            tracked_arrows[matched_arrow] = (center_x, top_y, MAX_LIFE)
        else:
            tracked_arrows[next_arrow_id] = (center_x, top_y, MAX_LIFE)
            next_arrow_id += 1

    keys_to_remove = []
    for arrow_id in tracked_arrows:
        arrow_x, arrow_y, life = tracked_arrows[arrow_id]
        if life > 0:
            tracked_arrows[arrow_id] = (arrow_x, arrow_y, life - 1)
        else:
            keys_to_remove.append(arrow_id)

    for key in keys_to_remove:
        del tracked_arrows[key]

    for arrow_x, arrow_y, _ in tracked_arrows.values():
        arrow_start = (arrow_x, arrow_y - 20)
        arrow_end = (arrow_x, arrow_y)
        draw_styled_arrow(img, arrow_start, arrow_end, color=(0, 255, 0), thickness=3, arrow_magnitude=20)

    # זיהוי בעזרת YOLO עם TRACK_ID על הפריים המקורי
    results = model.track(original_frame, verbose=False, persist=True)
    for result in results:
        for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID {track_id}: {result.names[int(result.boxes.cls[0].item())]}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Water Mask", mask)
    cv2.imshow("cameraFeed", img)

    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
