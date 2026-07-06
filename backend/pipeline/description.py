FRIENDLY_NAMES = {
    "car": "a car",
    "bus": "a bus",
    "truck": "a truck",
    "motorcycle": "a motorcycle",
    "autorickshaw": "an auto-rickshaw",
    "bicycle": "a bicycle",
    "person": "a person",
    "rider": "a rider",
    "animal": "an animal",
    "traffic light": "a traffic light",
    "traffic sign": "a traffic sign",
    "trailer": "a trailer",
    "caravan": "a caravan",
    "train": "a train",
    "vehicle fallback": "a vehicle",
}


def generate_description(obj):
    cls = obj["class"]
    urgency = obj.get("urgency", "INFO")
    motion = obj.get("motion", "static")
    position = obj.get("position", "ahead")
    distance = obj.get("distance", "medium")
    subject = FRIENDLY_NAMES.get(cls, cls)

    if urgency == "HAZARD":
        if motion == "approaching":
            return f"Careful, {subject} is approaching from your {position}."
        return f"Careful, {subject} is close on your {position}."

    distance_phrase = {"near": "close by", "medium": "nearby", "far": "further away"}.get(distance, "nearby")
    return f"There is {subject} {distance_phrase} on your {position}."
