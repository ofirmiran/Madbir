import cv2
import os
from ultralytics import YOLO
import threading
from tkinter import Tk, Button, filedialog
# Global variables
current_frame = 0  # משתנה חדש לעקוב אחרי הפריימים
total_frames = 0  # סך הפריימים בסרטון
pause_video = False  # משתנה לעצירת הסרטון
slider_drag = False  # משתנה כדי לדעת אם הסליידר נגרר

def load_model():
    # טעינת המודל YOLO
    model_path = r"C:\Users\Mizui_Meida\PycharmProjects\pythonProject3\runs\detect\train32\weights\best.pt"
    model = YOLO(model_path)
    return model

def process_detection_results(results, model):
    sea_related_objects = ["boat", "ship", "submarine", "person", "swimmer", "float"]
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

def draw_detections(frame, detections_with_color, font, font_scale, line_type):
    for det in detections_with_color:
        bbox = det['bbox']
        label = det['label']
        confidence = det['confidence']

        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text = f"{label} {confidence:.2f} (ID: {det['track_id']})"
        cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, (255, 255, 255), line_type)

def process_frame(frame, model, font, font_scale, line_type):
    # עיבוד פריימים בעזרת מודל YOLO
    results = model.track(frame, persist=True)
    detections = process_detection_results(results, model)
    draw_detections(frame, detections, font, font_scale, line_type)
    return frame

def on_trackbar(val):
    global current_frame, slider_drag
    slider_drag = True
    current_frame = val

def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()

def main(video_file=None):
    global current_frame, total_frames, pause_video, slider_drag

    model = load_model()

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # קבלת סך כל הפריימים בסרטון
    cv2.namedWindow("YOLO Video")

    # יצירת סרגל נווט (Slider) בעזרת createTrackbar
    cv2.createTrackbar('Frame', 'YOLO Video', 0, total_frames - 1, on_trackbar)

    font, font_scale, line_type = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2

    while cap.isOpened():
        if not pause_video and not slider_drag:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # קבלת הפריים הנוכחי מהווידאו

            frame = process_frame(frame, model, font, font_scale, line_type)

            cv2.imshow("YOLO Video", frame)

            # עדכון המיקום בסרגל
            cv2.setTrackbarPos('Frame', 'YOLO Video', current_frame)

            # עדכון המיקום הנוכחי בווידאו
            current_frame += 1

        # אם נגרר הסליידר, נעדכן את מיקום הווידאו
        if slider_drag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            slider_drag = False

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pause_video = not pause_video

    cleanup(cap)

def select_video():
    # בחירת קובץ וידאו מהמחשב
    video_file = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.MPG")]
    )
    if video_file:
        threading.Thread(target=main, args=(video_file,)).start()

# יצירת חלון של Tkinter לבחירת וידאו
root = Tk()
root.title("YOLO Video Detection")
root.geometry("400x200")

select_button = Button(root, text="בחר סרטון", command=select_video)
select_button.pack(pady=20)

root.mainloop()
