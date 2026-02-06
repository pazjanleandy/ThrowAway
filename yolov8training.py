from ultralytics import YOLO

if __name__ == '__main__':
       model_path = r'D:/COLLEGE/third year/thesis/arduinosample/src/Waste-Classification-using-YOLOv8/streamlit-detection-tracking - app/weights/yolov8n.pt'
       #model_path = r'D:/COLLEGE/third year/thesis/arduinosample/src/runs/detect/train4/weights/best.pt'
       data_yaml = r'D:/COLLEGE/third year/thesis/arduinosample/src/yolo_dataset/data.yaml'
       model = YOLO(model_path)
       model.train(
           data=data_yaml,
           epochs=30,
           imgsz=640,
           augment=True  # Enable built-in augmentations
       )