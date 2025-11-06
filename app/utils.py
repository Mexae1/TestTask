"""
Утилиты визуализации: отрисовка боксов, id, confidence.
"""
from typing import List, Dict, Tuple

import cv2


def draw_tracked_boxes(frame, tracks: List[Dict]):
    """
    Рисует прямоугольники и подписи (id, класс, conf).

    :param frame: BGR изображение (numpy array)
    :param tracks: список треков {'id','bbox','class','conf'}
    """
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        tid = tr.get("id", 0)
        cls = tr.get("class", "obj")
        conf = tr.get("conf", None)
        label = f"ID:{tid} {cls}"
        if conf is not None:
            try:
                label += f" {conf:.2f}"
            except Exception:
                label += f" {conf}"

        # Выбираем цвет на основании id (постоянный)
        color = ((tid * 37) % 256, (tid * 73) % 256, (tid * 97) % 256)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)