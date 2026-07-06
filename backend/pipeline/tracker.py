import numpy as np
from collections import OrderedDict


class Tracker:
    def __init__(self, max_missed_frames=5, iou_threshold=0.3, centroid_dist_threshold=60):
        self.next_id = 0
        self.objects = OrderedDict()
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.centroid_dist_threshold = centroid_dist_threshold

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = self._bbox_area(boxA) + self._bbox_area(boxB) - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        unmatched = list(range(len(detections)))
        for obj_id, obj in list(self.objects.items()):
            best_match, best_iou_val = None, -1
            obj_centroid = obj["centroid"]
            for i in unmatched:
                det = detections[i]
                if det["class"] != obj["class"]:
                    continue
                iou = self._iou(obj["bbox"], det["bbox"])
                det_centroid = self._centroid(det["bbox"])
                dist = np.hypot(obj_centroid[0] - det_centroid[0], obj_centroid[1] - det_centroid[1])
                qualifies = (iou > self.iou_threshold) or (dist < self.centroid_dist_threshold)
                if qualifies and iou > best_iou_val:
                    best_iou_val, best_match = iou, i

            if best_match is not None:
                det = detections[best_match]
                area_prev = self._bbox_area(obj["bbox"])
                area_now = self._bbox_area(det["bbox"])
                obj["bbox"] = det["bbox"]
                obj["centroid"] = self._centroid(det["bbox"])
                obj["missed"] = 0
                obj["area_history"].append(area_now)
                if len(obj["area_history"]) > 5:
                    obj["area_history"].pop(0)
                growth = (area_now - area_prev) / area_prev if area_prev > 0 else 0
                obj["motion"] = "approaching" if growth > 0.08 else "receding" if growth < -0.08 else "static"
                unmatched.remove(best_match)
            else:
                obj["missed"] += 1

        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]["missed"] > self.max_missed_frames:
                del self.objects[obj_id]

        for i in unmatched:
            det = detections[i]
            self.objects[self.next_id] = {
                "bbox": det["bbox"],
                "class": det["class"],
                "centroid": self._centroid(det["bbox"]),
                "area_history": [self._bbox_area(det["bbox"])],
                "missed": 0,
                "motion": "new",
            }
            self.next_id += 1
        return self.objects
