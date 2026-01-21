import sys
import cv2
import time
from ultralytics import YOLO

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ===================== CONFIG =====================

DATA_YAML = "my_dataset_yolo/data.yaml"

PRETRAINED_MODEL = "yolov8m.pt"
TRAINED_MODEL = "runs/detect/train/weights/best.pt"

VIDEO_PATH = "test.mp4"
OUTPUT_PATH = "output_video.mp4"

IMG_SIZE = 640
EPOCHS = 100
BATCH = 8        # CPU-safe

CONF_THRES = 0.5
IOU_THRES = 0.4

SHOW_VIDEO = True
SAVE_VIDEO = True

# ===================== UTILS =====================

def get_color(idx: int):
    return (
        int((idx * 37) % 255),
        int((idx * 17) % 255),
        int((idx * 29) % 255),
    )

def draw_boxes(frame, boxes, names):
    if boxes is None:
        return frame

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, box in enumerate(xyxy):
        class_id = cls[i]
        track_id = ids[i] if ids is not None else -1

        color = get_color(track_id if track_id != -1 else class_id)

        label = names[class_id]
        if track_id != -1:
            label += f" | ID {track_id}"

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            frame,
            label,
            (box[0], box[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame

# ===================== TRAIN =====================

def train():
    print("🚀 Training started (CPU)")

    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=4,
        project="runs/detect",
        name="train",
        exist_ok=True,
    )

    print("✅ Training finished")

# ===================== INFERENCE + TRACKING =====================

def infer():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            tracker="botsort.yaml"
        )

        # Получаем результаты
        result = results[0]

        # Работа с боксами
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # координаты
            confidences = result.boxes.conf.cpu().numpy()  # уверенность
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # классы

            # Отрисовка
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf > 0.5:  # дополнительный фильтр
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Метка с именем класса и уверенностью
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Расчет FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== ENTRY =====================

def main():
    #if len(sys.argv) < 2:
    #    print("Usage: python learn.py [train|infer]")
    #    return

    #mode = sys.argv[1]

    #if mode == "train":
    #train()
    #elif mode == "infer":
    infer()
    #else:
    #    print("Unknown mode:", mode)

if __name__ == "__main__":
    main()
