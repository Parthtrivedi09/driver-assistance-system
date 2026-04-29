from ultralytics import YOLO
import cv2

# =========================
# LOAD TRAINED MODEL
# =========================
model = YOLO(r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\runs\detect\train\best.pt")

# =========================
# LOAD TEST IMAGE
# =========================
image_path = r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\image.png"

image = cv2.imread(image_path)

# =========================
# RUN PREDICTION
# =========================
results = model(image, conf=0.25)

# =========================
# SHOW RESULTS
# =========================
results[0].show()

# =========================
# SAVE OUTPUT IMAGE
# =========================
results[0].save(filename="output.jpg")

# =========================
# PRINT DETECTIONS
# =========================
print(results[0].boxes)

print("✅ Prediction Done! Check output.jpg")