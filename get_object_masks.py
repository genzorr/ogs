# Adapted from https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_local_demo.py

import os

import cv2
import numpy as np
import supervision as sv
import torch
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from tqdm import tqdm

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
grounding_model = load_model(
    model_config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
    device=device,
)

text = "object"

SCENE = "test"  # replace with actual name
PROJECT_PATH = ""  # replace with actual path to root of the project

DATA_DIR = os.path.join(PROJECT_PATH, "gs2mesh", "data", "custom", SCENE)
OUTPUT_DIR = os.path.join(PROJECT_PATH, "gs2mesh", "output")
rgbs_path = os.path.join(DATA_DIR, "images")
depths_path = os.path.join(OUTPUT_DIR, "custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p/test")
masks_path = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(masks_path, exist_ok=True)

selected_images = [1, 39, 62]  # replace with selected views

for image_id in tqdm(selected_images):
    img_path = os.path.join(rgbs_path, f"IMG_{((image_id - 1) * 10):05d}.png")

    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model, image=image, caption=text, box_threshold=0.35, text_threshold=0.25
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

    # Save masks
    print(type(masks), masks.shape)
    np.save(os.path.join(masks_path, f"{(image_id - 1):03d}.npy"), masks)

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)  # (n, 4)  # (n, h, w)

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(masks_path, f"{(image_id - 1):03d}_detection.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(masks_path, f"{(image_id - 1):03d}.jpg"), annotated_frame)
