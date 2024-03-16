from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='data.yaml', epochs=10)

# Tune hyperparameters on data.yaml for 30 epochs
#model.tune(data='data.yaml', epochs=10, iterations=10, optimizer='AdamW', plots=False, save=False, val=False)

