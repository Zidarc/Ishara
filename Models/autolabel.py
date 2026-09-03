from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology

# ontology: {"text prompt sent to SAM3": "your class name"}
ontology = CaptionOntology({
    "a door": "door",
    "a staircase": "staircase",
    "an elevator door": "elevator_door",
    "a wheelchair ramp": "ramp",
    "a turnstile gate": "turnstile_gate",
    "a person": "person",
    "a chair": "chair",
    "a desk": "desk",
    "a sofa": "sofa",
    "a podium": "podium",
    "an obstacle on the floor": "floor_obstacle",
    "a wall pillar": "pillar",
    "a water cooler": "water_cooler",
    "a dustbin": "dustbin",
    "a red fire extinguisher": "fire_extinguisher",
    "a wall-mounted notice board": "notice_board",
    "a room number sign": "room_sign",
    "a head hazard warning sign": "head_hazard",
    "a vehicle": "vehicle",
    "a wall": "wall",
    "a plant": "plant",
    "a railing": "railing"
})

base_model = SegmentAnything3(ontology=ontology)

# Dataset 1: 640x640
base_model.label(
    input_folder=r"E:\GitProjects\Ishara\Models\model_code\dataset_640x640_clean",
    extension=".jpg",
    output_folder=r"E:\GitProjects\Ishara\Models\labeled_dataset_640x640"
)

# Dataset 2: 1280x1280
base_model.label(
    input_folder=r"E:\GitProjects\Ishara\Models\model_code\dataset_1280x1280_clean",
    extension=".jpg",
    output_folder=r"E:\GitProjects\Ishara\Models\labeled_dataset_1280x1280"
)

print("Done. Check labeled_dataset_640x640 and labeled_dataset_1280x1280 in the Models folder.")