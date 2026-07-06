HAZARD_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "autorickshaw", "bicycle", "trailer", "caravan", "train", "vehicle fallback"}
HAZARD_PERSON_CLASSES = {"person", "rider"}
CLOSE_AREA_THRESHOLD = 0.15 * (512 * 512)


def classify_urgency(tracked_objects):
    results = []
    for obj_id, obj in tracked_objects.items():
        cls, motion = obj["class"], obj["motion"]
        area = obj["area_history"][-1] if obj["area_history"] else 0
        urgency = "INFO"
        if cls in HAZARD_VEHICLE_CLASSES and motion == "approaching":
            urgency = "HAZARD"
        elif cls in HAZARD_PERSON_CLASSES and motion == "approaching" and area > CLOSE_AREA_THRESHOLD:
            urgency = "HAZARD"
        elif cls in HAZARD_VEHICLE_CLASSES and area > CLOSE_AREA_THRESHOLD * 1.5:
            urgency = "HAZARD"
        results.append({
            "id": obj_id,
            "class": cls,
            "motion": motion,
            "area": area,
            "bbox": obj["bbox"],
            "urgency": urgency,
        })
    return results
