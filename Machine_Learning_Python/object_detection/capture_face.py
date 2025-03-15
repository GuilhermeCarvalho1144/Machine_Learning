import cv2
from super_gradients.training import models
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# load models
model_path = (
    "./face_detection/checkpoints_yolo_v1/my_first_yolonas_run/average_model.pth"
)
model = models.get("yolo_nas_l", num_classes=1, checkpoint_path=model_path)

address = "https://192.168.1.102:8080/video"
cap = cv2.VideoCapture(address)


while cap.isOpened():
    _, frame = cap.read()

    preds = model.predict([frame], conf=0.5)
    plt.imshow(preds[0].draw())
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
