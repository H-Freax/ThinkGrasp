import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import detectron2.data.transforms as T
from VLP.vlpart.vlpart import build_vlpart
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions
import ray
import logging
from PIL import Image
from langsam import langsamutils

# Configure logging
logging.basicConfig(level=logging.INFO)

def show_predictions_with_masks(scores, boxes, classes, masks, text_prompt):
    num_obj = len(scores)
    if num_obj == 0:
        return
    text_prompts = text_prompt.split('.')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_obj))

    for obj_ind in range(num_obj):
        box = boxes[obj_ind]
        score = scores[obj_ind]
        name = text_prompts[classes[obj_ind]]
        if score < 0.5:
            continue

        color_mask = colors[obj_ind]

        m = masks[obj_ind][0]
        img = np.ones((m.shape[0], m.shape[1], 3))
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.45)))

        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))

        label = name + ': {:.2}'.format(score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')

@ray.remote
class SegmentAnythingActor:
    def __init__(self, vlpart_checkpoint, sam_checkpoint, device="cuda"):
        self.device = device
        self.vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
        self.vlpart.to(device=self.device)
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device=self.device))

    def predict(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        # Load image
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # VLPart model inference
        preprocess = T.ResizeShortestEdge([800, 800], 1333)
        height, width = original_image.shape[:2]
        image = preprocess.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        with torch.no_grad():
            predictions = self.vlpart.inference([inputs], text_prompt=text_prompt)[0]
            # print("predictions:",predictions)

        boxes, masks = None, None
        filter_scores, filter_boxes, filter_classes = [], [], []

        if "instances" in predictions:
            instances = predictions['instances'].to('cpu')
            boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            num_obj = len(scores)
            for obj_ind in range(num_obj):
                category_score = scores[obj_ind]
                if category_score < 0.7:
                    continue
                filter_scores.append(category_score)
                filter_boxes.append(boxes[obj_ind])
                filter_classes.append(classes[obj_ind])

        if len(filter_boxes) > 0:
            # SAM model inference
            self.sam_predictor.set_image(original_image)

            boxes_filter = torch.stack(filter_boxes)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filter, original_image.shape[:2])
            masks, logits, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            # Remove small disconnected regions and holes
            fine_masks = []
            for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
                fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
            masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
            masks = torch.from_numpy(masks)
        else:
            logits=1
        phrases = [text_prompt.split('.')[cls] for cls in filter_classes]
        return masks, filter_boxes, phrases, logits

if __name__ == "__main__":
    ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus

    # Variable to control GPU usage
    use_gpu = torch.cuda.is_available()
    image_path = "color_map.png"
    text_prompt = "lemon"

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance with dynamic GPU allocation
    actor = SegmentAnythingActor.options(**actor_options).remote(
        vlpart_checkpoint="swinbase_part_0a0000.pth",
        sam_checkpoint="sam_vit_h_4b8939.pth",
        device="cuda" if use_gpu else "cpu"
    )

    masks, boxes, phrases, logits = ray.get(actor.predict.remote(image_path=image_path, text_prompt=text_prompt))

    # Display results
    print("Masks:", masks)
    print("Boxes:", boxes)
    print("Phrases:", phrases)
    print("Logits:", logits)

    # Shutdown Ray
    ray.shutdown()
