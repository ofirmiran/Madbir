import os
import torch
from ultralytics import YOLO
import cv2



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("GPU is available and will be used.")
        print(torch.cuda.get_device_name(0))
    else:
        print("No GPU detected. CPU will be used.")


    model = YOLO('yolov8x.pt')
    model.conf = 0.25

    model.train(data=r"C:\Users\Mizui_Meida\Desktop\database_09.06.24\data.yaml",
                epochs=100,
                imgsz=800,
                batch=8,
                device=device)

    #batch=8