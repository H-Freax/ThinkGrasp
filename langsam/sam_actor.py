import torch
import ray
from PIL import Image
import requests

# Configure logging for Ray and other components
import logging

logging.basicConfig(level=logging.INFO)


@ray.remote
class SAMActor:
    def __init__(self, use_gpu=False):
        from transformers import SamModel, SamProcessor

        self.device = "cuda" if use_gpu else "cpu"
        # Initialize the SAM model and processor
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def predict(self, image, input_points):
        # Load and process the image
        inputs = self.processor(
            image, input_points=input_points, return_tensors="pt"
        ).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores.cpu().numpy()  # Example, adjust based on your needs

        # Return results (masks and scores)
        return masks, scores


if __name__ == "__main__":
    # Initialize Ray, replace with your cluster's address or local setup
    ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus

    # Variable to control GPU usage
    use_gpu = torch.cuda.is_available()

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance
    sam_actor = SAMActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example image URL and input point
    image_url = (
        "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    )
    image_pil = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]  # Example input point for segmentation

    # Predict masks and scores
    masks, scores = ray.get(sam_actor.predict.remote(image_pil, input_points))

    print("Scores:", scores)
    # Add your logic here to work with masks and scores

    ray.shutdown()
