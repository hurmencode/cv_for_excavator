import cv2
from ultralytics import YOLO
import random
import time

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='my_dataset_yolo/data.yaml', epochs=100, imgsz=640, model='yolov8m.pt')

def draw_bounding_boxes_without_id(frame, results, model):
    """Рисует bounding boxes без ID (для детектора)"""
    if len(results[0].boxes) == 0:
        return frame

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, clss in zip(boxes, classes):
        if clss != 0:  # Фильтрация класса (например, игнорируем фон)
            random.seed(int(clss) + 8)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Рисуем прямоугольник
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            # Добавляем подпись класса
            label = model.names[int(clss)]
            cv2.putText(
                frame,
                label,
                (box[0], box[1] - 10),  # Смещение над коробкой
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 255, 50),
                2,
            )
    return frame

def process_video_with_tracking(model, model_detect, input_video_path, show_video=True, save_video=False, output_video_path="output_video.mp4"):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настройка записи видео
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    last_frame_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1

            # Проверка на конец видео или таймаут
            if not ret or frame is None or frame.size == 0:
                if time.time() - last_frame_time > 5:
                    print("Video ended or timeout")
                    break
                continue
            last_frame_time = time.time()

            # Обработка трекера (с ID)
            results_track = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=640)
            if results_track[0].boxes.id is not None:
                boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results_track[0].boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    random.seed(int(id))
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"Id {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Обработка детектора (без ID, только классы)
            results_detect = model_detect(frame, iou=0.4, conf=0.5, imgsz=640)
            frame = draw_bounding_boxes_without_id(frame, results_detect, model_detect)


            # Сохранение кадра
            if save_video and out is not None:
                out.write(frame)


            # Отображение в реальном времени
            if show_video:
                resized_frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
                cv2.imshow("Processed Video", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    return results_track, results_detect

# Пример использования
model = YOLO('runs/detect/train/weights/best.pt')
model_detect = YOLO('runs/detect/train/weights/best.pt')

model.fuse()
model_detect.fuse()

results_track, results_detect = process_video_with_tracking(
    model=model,
    model_detect=model_detect,
    input_video_path="test.mp4",
    show_video=True,
    save_video=True,
    output_video_path="output_video.mp4"
)