from ultralytics import YOLO
import cv2
import pyttsx3
import threading

# =========================
# LOAD TRAINED MODEL
# =========================
model = YOLO(
    r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\runs\detect\train\best.pt"
)

# =========================
# VOICE FUNCTION
# =========================
def speak_warning():

    engine = pyttsx3.init()

    # Speech speed
    engine.setProperty('rate', 140)

    # Volume
    engine.setProperty('volume', 1.0)

    engine.say("STOP SIGN AHEAD")

    engine.runAndWait()

# =========================
# CONSTANTS
# =========================
CONF_THRESHOLD = 0.30

# Need 3 consecutive frames
FRAME_CONFIRMATION = 3

# Reset alert after missing frames
MAX_MISSING_FRAMES = 15

# =========================
# STATE VARIABLES
# =========================
consecutive_stop_frames = 0
missing_frames = 0
alert_active = False

# =========================
# VIDEO SOURCE
# =========================
cap = cv2.VideoCapture(
    r"C:\Users\Parth Trivedi\Desktop\CODING\Deep Learning\driver-assistance-system\videos\input\test.mp4"
)

# Webcam later:
# cap = cv2.VideoCapture(0)

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model(frame)

    stop_detected = False

    # =========================
    # PROCESS DETECTIONS
    # =========================
    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Get class name dynamically
        class_name = model.names[class_id]

        print(f"Detected Class: {class_name}")

        # =========================
        # STOP SIGN FILTER
        # =========================
        if class_name == "stop sign" and confidence >= CONF_THRESHOLD:

            stop_detected = True

            print(f"STOP SIGN DETECTED | Confidence: {confidence:.2f}")

            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                f"STOP SIGN {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =========================
    # CONSECUTIVE FRAME LOGIC
    # =========================
    if stop_detected:

        consecutive_stop_frames += 1
        missing_frames = 0

        print(f"Consecutive STOP Frames: {consecutive_stop_frames}")

        # =========================
        # TRIGGER ALERT
        # =========================
        if (
            consecutive_stop_frames >= FRAME_CONFIRMATION
            and not alert_active
        ):

            print("🚨 ALERT TRIGGERED!")

            # Voice Alert
            threading.Thread(
                target=speak_warning,
                daemon=True
            ).start()

            alert_active = True

    else:

        consecutive_stop_frames = 0
        missing_frames += 1

        # =========================
        # RESET ALERT
        # =========================
        if missing_frames > MAX_MISSING_FRAMES:

            alert_active = False

    # =========================
    # VISUAL WARNING
    # =========================
    if alert_active:

        cv2.putText(
            frame,
            "STOP SIGN AHEAD!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("STOP Sign Driver Assistant", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()