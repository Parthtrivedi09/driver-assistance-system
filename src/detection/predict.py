from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Load test image
image_path = r"data/processed/images/train/00480.jpg"  

image = cv2.imread(image_path)

# Run prediction
results = model(image, conf=0.1)
# Show results
results[0].show()

# Save result
results[0].save(filename="output.jpg")
print(results[0].boxes)

print("✅ Prediction Done! Check output.jpg")