from ultralytics import YOLO

model = YOLO('yolov8x') # Load model

result = model.track('input_videos/input_video.mp4', conf=0.2, save=True) # Inference on image

# print(result)
# print("Boxes:" )
# for box in result[0].boxes:
#     print(box)