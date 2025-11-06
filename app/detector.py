"""
Модуль детектора людей на основе Ultralytics YOLOv8 (предобученные веса).
Возвращает список словарей с bbox, conf и классом.
"""
from typing import List, Dict, Tuple
import torch
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel
])
import logging

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    logging.warning("Ultralytics YOLO не доступен. Убедитесь, что пакет установлен.")

import numpy as np


class PeopleDetector:
    """
    Детектор людей. Оборачивает модель YOLO.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Загружает модель.
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO не найден в окружении.")
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            logging.exception("Не удалось загрузить модель %s: %s", model_name, e)
            raise

    def detect(self, frame) -> List[Dict]:
        """
        Выполняет детекцию на одном кадре.

        :param frame: изображение в BGR (numpy array)
        :return: список детекций вида {
            "bbox": (x1, y1, x2, y2),
            "conf": float,
            "class": "person"
        }
        """
        results = self.model(frame)  # возможное место исключения
        detections = []

        # results[0].boxes содержит список боксов
        try:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    cls_id = int(box.cls)
                    # В COCO класс 'person' обычно id == 0
                    if cls_id != 0:
                        continue
                    xyxy = box.xyxy[0] if hasattr(box, "xyxy") else box.xyxy
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf) if hasattr(box, "conf") else float(box.confidence)
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "class": "person",
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
                except Exception:
                    logging.exception("Ошибка парсинга бокса детекции.")
        except Exception:
            logging.exception("Ошибка обработки результатов модели.")
        return detections
    