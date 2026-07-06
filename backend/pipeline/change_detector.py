class ChangeDetector:
    def __init__(self, position_bucket_size=3):
        self.last_spoken_state = {}
        self.position_bucket_size = position_bucket_size

    def _bucket_position(self, bbox, frame_width):
        cx = (bbox[0] + bbox[2]) / 2
        bucket_width = frame_width / self.position_bucket_size
        idx = int(cx // bucket_width)
        labels = ["left", "center", "right"]
        return labels[min(idx, len(labels) - 1)]

    def _bucket_distance(self, area, frame_area):
        ratio = area / frame_area
        return "near" if ratio > 0.15 else "medium" if ratio > 0.04 else "far"

    def check(self, info_objects, frame_width, frame_height):
        frame_area = frame_width * frame_height
        changes, current_state = [], {}
        for obj in info_objects:
            cls = obj["class"]
            position = self._bucket_position(obj["bbox"], frame_width)
            distance = self._bucket_distance(obj["area"], frame_area)
            current_state[cls] = {"position": position, "distance": distance}
            prev = self.last_spoken_state.get(cls)
            if prev is None:
                changes.append({**obj, "position": position, "distance": distance})
            elif prev["position"] != position or prev["distance"] != distance:
                changes.append({**obj, "position": position, "distance": distance})
        self.last_spoken_state = current_state
        return changes
