from pathlib import Path
from matplotlib import pyplot as plt
from flask import Flask, request, jsonify
import logging
import numpy as np
import ray
import torch
import os
import cv2
import open3d as o3d
import base64
from PIL import Image
import io
import argparse
from engine import grasp_model
from langsam import langsamutils
from langsam.langsam_actor import LangSAM
from VLP.new_vlp_actor import SegmentAnythingActor
from openai import OpenAI
from grasp_detetor import Graspnet
import utils
from models.sac import ViLG

app = Flask(__name__)
def get_args_parser():
    parser = argparse.ArgumentParser('RefTR For Visual Grounding; FGC-GraspNet For Grasp Pose Detection',
                                     add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["img_backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_mask_branch_names', default=['bbox_attention', 'mask_head'], type=str, nargs='+')
    parser.add_argument('--lr_mask_branch_proj', default=1., type=float)
    parser.add_argument('--lr_bert_names', default=["lang_backbone"], type=str, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--warm_up_epoch', default=2, type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='StepLR', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--ckpt_cycle', default=20, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--no_decoder', default=False, action='store_true')
    parser.add_argument('--reftr_type', default='transformer_single_phrase', type=str,
                        help="using bert based reftr vs transformer based reftr")
    parser.add_argument('--pretrain_on_coco', default=False, action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to the pretrained model. If set, DETR weight will be used to initilize the network.")
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--ablation', type=str, default='none', help="Ablation")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--masks', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--freeze_reftr', action='store_true',
                        help="Train unfreeze reftr for segmentation if the flag is provided")
    parser.add_argument('--bert_model', default="bert-base-uncased", type=str,
                        help="bert model name for transformer based reftr")
    parser.add_argument('--img_bert_config', default="./configs/VinVL_VQA_base", type=str,
                        help="For bert based reftr: Path to default image bert ")
    parser.add_argument('--use_encoder_pooler', default=False, action='store_true',
                        help="For bert based reftr: Whether to enable encoder pooler ")
    parser.add_argument('--freeze_bert', action='store_true',
                        help="Whether to freeze language bert")
    parser.add_argument('--max_lang_seq', default=128, type=int,
                        help="Controls maxium number of embeddings in VLTransformer")
    parser.add_argument('--num_queries_per_phrase', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--aux_loss', action='store_true',
                        help="Enable auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_softmax_ce', action='store_true',
                        help="Whether to use cross entropy loss over all queries")
    parser.add_argument('--bbox_loss_topk', default=1, type=int,
                        help="set > 1 to enbale softmargin loss and topk picking in vg loss ")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--dataset', default='roborefit')
    parser.add_argument('--data_root', default='data/final_dataset')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--test_split', default='testA', type=str)
    parser.add_argument('--img_size', default=640, type=int)
    parser.add_argument('--img_type', default='RGB')
    parser.add_argument('--max_img_size', default=640, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/mscoco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--image_path', default='example/image')
    parser.add_argument('--depth_path', default='example/depth')
    parser.add_argument('--text_path', default='example/text')
    parser.add_argument('--checkpoint_vl_path', default='logs/checkpoint_best_r50.pth')
    parser.add_argument('--output_dir', default='outputs/roborefit',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_model_only', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run_epoch', default=500, type=int, metavar='N',
                        help='epochs for current run')
    parser.add_argument('--eval', default=False)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--checkpoint_grasp_path', default='logs/checkpoint_fgc.tar', help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=12000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    parser.add_argument('--output_dir_grasp', default='outputs/graspnet')
    return parser
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configure OpenAI API key
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Initialize Ray and load actors and models
ray.init(num_gpus=2)  # Add arguments as necessary, e.g., address, num_gpus
use_gpu = torch.cuda.is_available()
gpu_allocation = 0.8 if use_gpu else 0
actor_options = {"num_gpus": gpu_allocation}
langsam_actor = LangSAM.options(**actor_options).remote(use_gpu=use_gpu)
vlp_actor = SegmentAnythingActor.options(**actor_options).remote(
    vlpart_checkpoint="VLP/swinbase_part_0a0000.pth", sam_checkpoint="VLP/sam_vit_h_4b8939.pth",
    device="cuda" if use_gpu else "cpu")

def visualize_cropping_box(image, cropping_box):
    # Visualize the cropping box on the image
    x1, y1, x2, y2 = cropping_box
    plt.figure()
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none'))
    plt.title("Cropping Box Visualization")
    plt.show()


def select_action(bboxes, pos_bboxes, text, actions, evaluate=True):
    distances = torch.sqrt((actions[0, :, 0] - pos_bboxes[0, 0, 0]) ** 2 +
                           (actions[0, :, 1] - pos_bboxes[0, 0, 1]) ** 2)

    num_distances_to_select = min(10, distances.numel())
    top_dist_indices = torch.topk(distances, k=num_distances_to_select, largest=False).indices
    min_dist_index = top_dist_indices[0]

    return min_dist_index.item()

# Helper functions
def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def process_grasping_result(output):
    lines = output.strip().split('\n')
    result = {
        "selected_object": None,
        "cropping_box": None,
        "objects": [],
        "is_part": False
    }

    for i, line in enumerate(lines):
        if line.startswith("Selected Object/Object Part:"):
            selected_object = line.split(": ")[1].strip()
            if selected_object.startswith("[object part:"):
                result["is_part"] = True
                selected_object = selected_object[len("[object part:"):].strip(" ]")
            else:
                result["is_part"] = False
                selected_object = selected_object[len("[object:"):].strip(" ]")
            result["selected_object"] = selected_object.lower()

        elif line.startswith("Cropping Box Coordinates:"):
            coords = line.split(": ")[1].strip()[1:-1]
            result["cropping_box"] = tuple(map(int, coords.split(", ")))
        elif line.startswith("Object:"):
            obj = {
                "name": line.split(": ")[1].strip().lower(),
                "grasping_score": int(lines[i + 1].split(": ")[1].strip()),
                "material_composition": int(lines[i + 2].split(": ")[1].strip()),
                "surface_texture": int(lines[i + 3].split(": ")[1].strip()),
                "stability_assessment": int(lines[i + 4].split(": ")[1].strip()),
                "centroid_coordinates": tuple(map(int, lines[i + 5].split(": ")[1].strip()[1:-1].split(", "))),
                "preferred_grasping_location": int(lines[i + 6].split(": ")[1].strip())
            }
            result["objects"].append(obj)

    return result


def select_fallback_object(objects):
    sorted_objects = sorted(objects, key=lambda x: x['grasping_score'], reverse=True)
    return sorted_objects[0] if sorted_objects else None

def create_cropping_box_from_boxes(boxes, image_size, margin=20):
    if not boxes:
        return 0, 0, image_size[0], image_size[1]

    x1_min = float('inf')
    y1_min = float('inf')
    x2_max = float('-inf')
    y2_max = float('-inf')

    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 < x1_min:
            x1_min = x1
        if y1 < y1_min:
            y1_min = y1
        if x2 > x2_max:
            x2_max = x2
        if y2 > y2_max:
            y2_max = y2

    x1_min = max(0, x1_min - margin)
    y1_min = max(0, y1_min - margin)
    x2_max = min(image_size[0], x2_max + margin)
    y2_max = min(image_size[1], y2_max + margin)

    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)


def crop_pointcloud(pcd, cropping_box, color_image, depth_image):
    x1, y1, x2, y2 = cropping_box
    depth_crop = depth_image[y1:y2, x1:x2]
    color_crop = color_image[y1:y2, x1:x2]
    mask = (depth_crop > 0)
    points = []
    colors = []

    for i in range(depth_crop.shape[0]):
        for j in range(depth_crop.shape[1]):
            if mask[i, j]:
                x = j + x1
                y = i + y1
                z = depth_crop[i, j]
                points.append((x, y, z))
                colors.append(color_crop[i, j] / 255.0)

    points = np.array(points)
    colors = np.array(colors)

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(points)
    full_pcd.colors = o3d.utility.Vector3dVector(colors)

    return full_pcd



@app.route('/grasp_pose', methods=['POST'])
def get_grasp_pose():
    data = request.json
    rgb_image_path = data['image_path']
    depth_image_path = data['depth_path']
    text_path = data['text_path']


    img_ori = cv2.imread(rgb_image_path)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    depth_ori = np.array(Image.open(depth_image_path))
    with open(text_path, 'r') as file:
        text = file.read()

    image_pil = langsamutils.load_image(rgb_image_path)
    base64_image = load_image_as_base64(rgb_image_path)
    input_text = text


    messages = [
        {
            "role": "system",
            "content": (
                "Given a 640x480 input image and the provided instruction, perform the following steps:\n"
                "Target Object Selection:\n"
                "Identify the object in the image that best matches the instruction. If the target object is found, select it as the target object.\n"
                "If the target object is not visible, select the most cost-effective object or object part considering ease of grasping, importance, and safety.\n"
                "If the object has a handle or a part that is easier or safer to grasp, strongly prefer to select that part.\n"
                "Consider the geometric shape of the objects and the gripper's success rate when selecting the target object or object part.\n"
                "Output the name of the selected object or object part as [object:color and name] or [object part:color and name]..\n"
                "Cropping Box Calculation:\n"
                "Calculate a cropping box that includes the target object and all surrounding objects that might be relevant for grasping.\n"
                "Provide the coordinates of the cropping box in the format (top-left x, top-left y, bottom-right x, bottom-right y).\n"
                "Object Properties within Cropping Box:\n"
                "For each object within the cropping box, provide the following properties:\n"
                "Grasping Score: Evaluate the ease or difficulty of grasping the object on a scale from 0 to 100 (0 being extremely difficult, 100 being extremely easy).\n"
                "Material Composition: Evaluate the material composition of the object on a scale from 0 to 100 (0 being extremely weak, 100 being extremely strong).\n"
                "Surface Texture: Evaluate the texture of the object's surface on a scale from 0 to 100 (0 being extremely smooth, 100 being extremely rough).\n"
                "Stability Assessment: Assess the stability of the object on a scale from 0 to 100 (0 being extremely unstable, 100 being extremely stable).\n"
                "Centroid Coordinates: Provide the coordinates (x, y) of the object's center of mass across the entire image.\n"
                "Preferred Grasping Location: Divide the cropping box into a 3x3 grid and return a number from 1 to 9 indicating the preferred grasping location (1 for top-left, 9 for bottom-right).\n"
                "Additionally, consider the preferred grasping location that is most successful for the UR5 robotic arm and gripper.\n"
                "Output should be in the following format:\n"
                "Selected Object/Object Part: [object:color and name] or [object part:color and name]\n"
                "Cropping Box Coordinates: (top-left x, top-left y, bottom-right x, bottom-right y)\n"
                "Objects and Their Properties:\n"
                "Object: [color and name]\n"
                "Grasping Score: [value]\n"
                "Material Composition: [value]\n"
                "Surface Texture: [value]\n"
                "Stability Assessment: [value]\n"
                "Centroid Coordinates: (x, y)\n"
                "Preferred Grasping Location: [value]\n"
                "...\n"
                "Example Output:\n"
                "Selected Object/Object Part: [object:blue ball]\n"
                "Cropping Box Coordinates: (50, 50, 200, 200)\n"
                "Objects and Their Properties:\n"
                "Object: Blue Ball\n"
                "Grasping Score: 90\n"
                "Material Composition: 80\n"
                "Surface Texture: 20\n"
                "Stability Assessment: 95\n"
                "Centroid Coordinates: (125, 125)\n"
                "Preferred Grasping Location: 5\n"
                "Object: Yellow Bottle\n"
                "Grasping Score: 75\n"
                "Material Composition: 70\n"
                "Surface Texture: 30\n"
                "Stability Assessment: 80\n"
                "Centroid Coordinates: (100, 150)\n"
                "Preferred Grasping Location: 3\n"
                "Object: Black and Blue Scissors\n"
                "Grasping Score: 60\n"
                "Material Composition: 85\n"
                "Surface Texture: 40\n"
                "Stability Assessment: 70\n"
                "Centroid Coordinates: (175, 175)\n"
                "Preferred Grasping Location: 7"
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]

    result = {
        "selected_object": None,
        "cropping_box": None,
        "objects": []
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            temperature=0,
            max_tokens=713,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = response.choices[0].message.content
        logging.info(output)
        result = process_grasping_result(output)

        if not result['selected_object']:
            fallback_object = select_fallback_object(result['objects'])
            if fallback_object:
                goal = fallback_object['name']
            else:
                goal = input_text
        else:
            goal = result['selected_object']

        preferred_grasping_location = 5
        for obj in result['objects']:
            if obj['name'] == goal:
                preferred_grasping_location = obj.get('preferred_grasping_location', 5)

        if not result["is_part"]:
            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, goal))
        else:
            masks, boxes, phrases, logits = ray.get(
                vlp_actor.predict.remote(image_path=rgb_image_path, text_prompt=goal))

        if masks is None or masks.numel() == 0:
            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, input_text))
            if masks is None or masks.numel() == 0:
                masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, "object"))

        boxes_list = boxes.cpu().numpy().tolist()
        cropping_box = create_cropping_box_from_boxes(boxes_list, (img_ori.shape[1], img_ori.shape[0]))

        visualize_cropping_box(img_ori, cropping_box)
        ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil))
        bbox_images, bbox_positions = utils.convert_outputnew(image_pil, boxes, phrases, logits, img_ori, depth_ori, preferred_grasping_location)

        endpoint ,pcd = utils.get_and_process_data(cropping_box,img_ori, depth_ori)

        parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args([])  # Provide empty list to avoid reading command-line args
        grasp_net = grasp_model(args=args,device="cuda" if use_gpu else "cpu", image=img_ori, bbox=bbox_positions, mask=masks, text=input_text)
        gg ,gg_array= grasp_net.forward(endpoint ,pcd )
        grasp_pose_set=gg_array
        remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (32, 32))

        if bboxes is None:
            return jsonify({"error": "No bounding boxes found"}), 400

        if len(grasp_pose_set) == 1:
            action_idx = 0
        else:
            with torch.no_grad():
                action_idx = select_action(bboxes, pos_bboxes, input_text, grasps)

        action = grasp_pose_set[action_idx]
        print(action)

        chose_xyz = action[-4:-1]

        chose_rot = np.resize(np.expand_dims(action[-13:-4], axis=0), (3, 3))

        dep = action[3]

        print("chose_xyz:", chose_xyz)
        print("chose_rot:", chose_rot)
        print("dep:", dep)

        # Convert ndarrays to lists
        xyz_list = chose_xyz.tolist()
        rot_list = chose_rot.tolist()
        dep_list = dep.tolist()
        grippers = gg.to_open3d_geometry_list()
        chosen_gripper = grippers[action_idx]
        o3d.visualization.draw_geometries([pcd, chosen_gripper])

        return jsonify({
                'xyz': xyz_list,
                'rot': rot_list,
                'dep': dep_list
            })


    except Exception as e:
        logging.error(f"Error with OpenAI API request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
