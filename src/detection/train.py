#here we are fine tuning a pretrained YOLO model on a custom traffic dataset
#basically the model already knows about car people etc we are simply using transfer learning (FINE TUNING) to use that pre trained model
from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/processed/data.yaml", # this is the configuration file where we have mentioned every location and other labels as key value pairs
        epochs=20,
        imgsz=640,
        batch=8
    )

if __name__ == "__main__":
    train_model()
    