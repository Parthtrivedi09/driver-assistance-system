# src/detection/train.py

from ultralytics import YOLO
import torch
import os

def train_model():
    # =========================
    # 1. ENV CHECK (ENGINEER PRACTICE)
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================
    # 2. LOAD MODEL (TRANSFER LEARNING)
    # =========================
    model = YOLO("yolov8s.pt")   # better than nano for your case

    # =========================
    # 3. TRAIN MODEL
    # =========================
    results = model.train(
        data="data/processed/data.yaml",
        epochs=100,
        imgsz=800,              # better for small objects
        batch=8,
        device=device,

        # 🔥 robustness improvements
        patience=20,            # early stopping
        workers=4,              # faster data loading
        optimizer="AdamW",      # better optimizer
        lr0=0.001,              # learning rate

        # 🔥 augmentation (important)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # logging
        project="runs/detect",
        name="traffic_sign_model",
        exist_ok=True
    )

    return model


def evaluate_model(model):
    # =========================
    # 4. EVALUATION (VERY IMPORTANT)
    # =========================
    print("\nRunning validation...")

    metrics = model.val()

    print("\n===== EVALUATION METRICS =====")

    # Core metrics
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    print("================================\n")


if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)


