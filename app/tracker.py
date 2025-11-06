"""
Трекер с простым сопоставлением и фильтром Калмана.

Логика:
- каждая траектория имеет KalmanFilter с состоянием [cx, cy, vx, vy]
- на каждом кадре: predict() для всех треков
- сопоставление: ближайший центр (евклид) с порогом (max_distance)
- обновление matched треков, создание новых для unmatched детекций
- удаление старых треков по счетчику missed_frames
"""
import math
import logging
from typing import List, Dict, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

_TRACK_ID = 1


class Track:
    """Одна траектория с фильтром Калмана."""
    def __init__(self, bbox: Tuple[int, int, int, int], track_id: int):
        self.id = track_id
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Состояние: [cx, cy, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([cx, cy, 0., 0.])
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self.kf.P *= 100.0
        self.kf.R *= 5.0
        self.kf.Q *= 0.01

        self.bbox = bbox
        self.missed_frames = 0
        self.hits = 1

    def predict(self):
        """Предсказание состояния."""
        try:
            self.kf.predict()
        except Exception:
            logging.exception("Ошибка Kalman predict для трека %s", self.id)

    def update(self, bbox: Tuple[int, int, int, int]):
        """Обновление состояния фильтра измерением центра bbox."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        try:
            self.kf.update(np.array([cx, cy]))
            self.bbox = bbox
            self.missed_frames = 0
            self.hits += 1
        except Exception:
            logging.exception("Ошибка Kalman update для трека %s", self.id)


class PeopleTracker:
    """Коллекция треков + сопоставление детекций с треками."""
    def __init__(self, max_distance: float = 60.0, max_missed: int = 10):
        self.tracks: List[Track] = []
        self.max_distance = max_distance
        self.max_missed = max_missed
        global _TRACK_ID
        _TRACK_ID = 1

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Обновляет трекер по списку детекций.

        :param detections: список словарей с полем 'bbox' и 'center'
        :return: список треков в формате {
            'id': int,
            'bbox': (x1,y1,x2,y2),
            'conf': float,
            'class': str
        }
        """
        global _TRACK_ID
        # 1) predict для всех треков
        for tr in self.tracks:
            tr.predict()

        # 2) подготовка центров
        det_centers = [det["center"] for det in detections]
        det_bboxes = [det["bbox"] for det in detections]

        matched_det_idx = set()
        matched_tr_idx = set()

        # 3) простое жадное сопоставление по расстоянию
        for ti, tr in enumerate(self.tracks):
            tr_center = (tr.kf.x[0], tr.kf.x[1])
            best_d = float("inf")
            best_di = None
            for di, dc in enumerate(det_centers):
                if di in matched_det_idx:
                    continue
                d = self._distance((tr_center[0], tr_center[1]), (dc[0], dc[1]))
                if d < best_d:
                    best_d = d
                    best_di = di
            if best_di is not None and best_d <= self.max_distance:
                # matched
                tr.update(det_bboxes[best_di])
                matched_det_idx.add(best_di)
                matched_tr_idx.add(ti)
            else:
                tr.missed_frames += 1

        # 4) create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_idx:
                continue
            track = Track(det["bbox"], _TRACK_ID)
            _TRACK_ID += 1
            self.tracks.append(track)

        # 5) remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]

        # 6) prepare results list for drawing
        results = []
        for tr in self.tracks:
            results.append({
                "id": tr.id,
                "bbox": tuple(map(int, tr.bbox)),
                "conf": None,
                "class": "person"
            })
        # attach confidences if detections matched (optional)
        # Простая логика: если трек был только что обновлён и есть соответствующая детекция, возьмём conf
        for di, det in enumerate(detections):
            for res in results:
                if res["bbox"] == det["bbox"]:
                    res["conf"] = det.get("conf", None)

        return results
