"""
Точка входа. Читает видео из папки input, прогоняет детектор + трекер,
и записывает результат в папку output.

Запуск (локально):
python main.py

В контейнере путь тот же: /app/input/crowd.mp4 -> /app/output/detected_crowd.mp4
"""
import logging
import os
from typing import List, Dict

import cv2

from detector import PeopleDetector
from tracker import PeopleTracker
from utils import draw_tracked_boxes

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
INPUT_VIDEO = "crowd.mp4"
OUTPUT_VIDEO = "detected_crowd.mp4"


def ensure_dirs():
    """Создаёт папки input/output внутри контейнера, если их нет."""
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        logging.exception("Не удалось создать input/output директории: %s", e)
        raise


def main():
    """Главный цикл обработки: загрузка модели, чтение кадров, детекция, трекинг, запись."""
    ensure_dirs()
    input_path = os.path.join(INPUT_DIR, INPUT_VIDEO)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO)

    # Инициализация детектора и трекера
    try:
        detector = PeopleDetector()  # может бросить исключение при загрузке весов
        tracker = PeopleTracker()
    except Exception:
        logging.exception("Ошибка при инициализации детектора/трекера.")
        return

    # Открываем видео
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error("Не удалось открыть входное видео: %s", input_path)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    except Exception:
        logging.exception("Ошибка при открытии видео.")
        return

    # Подготовка писателя видео — используем mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logging.error("Не удалось создать VideoWriter для: %s", output_path)
            cap.release()
            return
    except Exception:
        logging.exception("Ошибка при создании VideoWriter.")
        cap.release()
        return

    frame_idx = 0
    try:
        logging.info("Начинаем обработку: %s -> %s", input_path, output_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("Видео закончено, обработано кадров: %d", frame_idx)
                break

            frame_idx += 1
            # Детекция
            try:
                detections = detector.detect(frame)
            except Exception:
                logging.exception("Ошибка при детекции на кадре %d", frame_idx)
                detections = []

            # Трекер (предсказание + обновление)
            try:
                tracks = tracker.update(detections)
            except Exception:
                logging.exception("Ошибка в трекере на кадре %d", frame_idx)
                tracks = []

            # Отрисовка
            try:
                draw_tracked_boxes(frame, tracks)
            except Exception:
                logging.exception("Ошибка при отрисовке на кадре %d", frame_idx)

            # Запись кадра
            try:
                out.write(frame)
            except Exception:
                logging.exception("Не удалось записать кадр %d в выходное видео", frame_idx)

    except KeyboardInterrupt:
        logging.info("Прервано пользователем.")
    finally:
        cap.release()
        out.release()
        logging.info("Ресурсы освобождены. Выходной файл: %s", output_path)


if __name__ == "__main__":
    main()
