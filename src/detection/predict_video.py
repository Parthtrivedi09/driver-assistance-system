from ultralytics import YOLO
import cv2

# =========================
# LOAD TRAINED MODEL
# =========================
model = YOLO(r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\runs\detect\train\best.pt")

# =========================
# INPUT VIDEO
# =========================
video_path = r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\videos\input\test.mp4"

cap = cv2.VideoCapture(video_path)

# =========================
# OUTPUT VIDEO SETUP
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "videos/output/output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# =========================
# PROCESS VIDEO
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO prediction
    results = model(frame)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Show live window
    cv2.imshow("Traffic Sign Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE RESOURCES
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing completed!")