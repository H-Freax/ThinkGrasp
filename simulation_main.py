import argparse
import json
import logging
import threading
import pybullet as p
import numpy as np
import random

import ray
import torch
import os
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import wandb
import utils
from constants import WORKSPACE_LIMITS
from environment_sim import Environment
from langsam import langsamutils
from langsam.langsam_actor import LangSAM
from logger import Logger
from grasp_detetor import Graspnet
from models.sac import ViLG
from VLP.new_vlp_actor import SegmentAnythingActor
from openai import OpenAI

import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


def select_action(bboxes, pos_bboxes, text, actions, evaluate=True):
    distances = torch.sqrt((actions[0, :, 0] - pos_bboxes[0, 0, 0]) ** 2 +
                           (actions[0, :, 1] - pos_bboxes[0, 0, 1]) ** 2)

    # Determine the number of distances to select (either 10 or the total number of distances if fewer)
    num_distances_to_select = min(10, distances.numel())

    # Find the indices and values of the smallest distances
    top_dist_indices = torch.topk(distances, k=num_distances_to_select, largest=False).indices

    # Select the action with the smallest distance among the top distances
    min_dist_index = top_dist_indices[0]

    return min_dist_index.item()

def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def print_grasping_result(result):
    logging.info(f"Selected Object/Object Part: {result['selected_object']}")
    logging.info(f"Cropping Box Coordinates: {result['cropping_box']}")
    logging.info("Objects and Their Properties:")
    for obj in result["objects"]:
        logging.info(f"Object: {obj['name']}")
        logging.info(f"  Grasping Score: {obj['grasping_score']}")
        logging.info(f"  Material Composition: {obj['material_composition']}")
        logging.info(f"  Surface Texture: {obj['surface_texture']}")
        logging.info(f"  Stability Assessment: {obj['stability_assessment']}")
        logging.info(f"  Centroid Coordinates: {obj['centroid_coordinates']}")
        logging.info(f"  Preferred Grasping Location: {obj['preferred_grasping_location']}")
        logging.info("")


def create_cropping_box_from_boxes(boxes, image_size, margin=20):
    if not boxes:
        # If boxes list is empty, return a default cropping box (e.g., full image)
        return 0, 0, image_size[0], image_size[1]

    # Create a cropping box that includes all detected bounding boxes
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

    # Add margin to the bounding box
    x1_min = max(0, x1_min - margin)
    y1_min = max(0, y1_min - margin)
    x2_max = min(image_size[0], x2_max + margin)
    y2_max = min(image_size[1], y2_max + margin)

    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)


def crop_pointcloud(pcd, cropping_box, color_image, depth_image, workspace_limits):
    # Convert the 2D cropping box coordinates to 3D coordinates
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
                colors.append(color_crop[i, j] / 255.0)  # Normalize the color values

    points = np.array(points)
    colors = np.array(colors)

    # Convert image coordinates to workspace coordinates
    def image_to_workspace(x, y, z, image_width, image_height, workspace_limits):
        workspace_x = workspace_limits[0][0] + (x / image_width) * (workspace_limits[0][1] - workspace_limits[0][0])
        workspace_y = workspace_limits[1][0] + (y / image_height) * (workspace_limits[1][1] - workspace_limits[1][0])
        workspace_z = z  # Assuming z is already in the correct unit (e.g., meters)
        return workspace_x, workspace_y, workspace_z

    image_height, image_width = depth_image.shape
    workspace_points = [image_to_workspace(x, y, z, image_width, image_height, workspace_limits) for x, y, z in points]

    workspace_points = np.array(workspace_points)

    # Create a grid of points for the entire workspace with z=0
    grid_x, grid_y = np.meshgrid(
        np.linspace(workspace_limits[0][0], workspace_limits[0][1], image_width),
        np.linspace(workspace_limits[1][0], workspace_limits[1][1], image_height)
    )
    grid_z = np.zeros_like(grid_x)
    full_workspace_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    full_colors = np.zeros((full_workspace_points.shape[0], 3))

    # Update the points in the cropped region with actual height
    for idx, (x, y, z) in enumerate(workspace_points):
        # Convert x, y from workspace coordinates back to image coordinates to find the corresponding index
        image_x = int((x - workspace_limits[0][0]) / (workspace_limits[0][1] - workspace_limits[0][0]) * image_width)
        image_y = int((y - workspace_limits[1][0]) / (workspace_limits[1][1] - workspace_limits[1][0]) * image_height)

        # Ensure the indices are within bounds
        if 0 <= image_x < image_width and 0 <= image_y < image_height:
            grid_index = image_x * image_height + image_y  # Correct index calculation
            full_workspace_points[grid_index, 2] = z
            full_colors[grid_index] = colors[idx]

    # Create a new point cloud from the full workspace points
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_workspace_points)
    full_pcd.colors = o3d.utility.Vector3dVector(full_colors)

    return full_pcd

def visualize_cropping_box(image, cropping_box):
    # Visualize the cropping box on the image
    x1, y1, x2, y2 = cropping_box
    plt.figure()
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none'))
    plt.title("Cropping Box Visualization")
    plt.show()



def visualize_pointcloud_with_grasps(pcd, grasp_poses):
    # Visualize the point cloud with grasp poses
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for grasp in grasp_poses:
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        grasp_frame.translate(grasp.translation)
        grasp_frame.rotate(grasp.rotation_matrix)
        vis.add_geometry(grasp_frame)

    vis.run()
    vis.destroy_window()
def process_grasping_result(output):
    lines = output.strip().split('\n')
    result = {
        "selected_object": None,
        "cropping_box": None,
        "objects": [],
        "is_part": False  # New field to indicate if the selected object is a part
    }

    for i, line in enumerate(lines):
        if line.startswith("Selected Object/Object Part:"):
            selected_object = line.split(": ")[1].strip()
            if selected_object.startswith("[object part:"):
                result["is_part"] = True  # Mark as part if "part" is in the selected object
                selected_object = selected_object[len("[object part:"):].strip(" ]")
            else:
                result["is_part"] = False  # Ensure is_part is False if "part" is not in the selected object
                selected_object = selected_object[len("[object:"):].strip(" ]")
            result["selected_object"] = selected_object


        elif line.startswith("Cropping Box Coordinates:"):
            coords = line.split(": ")[1].strip()[1:-1]
            result["cropping_box"] = tuple(map(int, coords.split(", ")))
        elif line.startswith("Object:"):
            obj = {
                "name": line.split(": ")[1].strip(),
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
    # Select the most cost-effective object to move
    sorted_objects = sorted(objects, key=lambda x: x['grasping_score'], reverse=True)
    return sorted_objects[0] if sorted_objects else None
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--evaluate', action='store', type=bool, default=True)
    parser.add_argument('--testing_case_dir', action='store', type=str, default='heavy_unseen/')
    parser.add_argument('--testing_case', action='store', type=str, default=None)

    parser.add_argument('--load_model', action='store', type=bool, default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')

    parser.add_argument('--num_episode', action='store', type=int, default=15)
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
    parser.add_argument('--max_episode_step', type=int, default=50)

    # Transformer paras
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    wandb.init(project="robotic-grasping1.0")
    ray.init(num_gpus=1) 
    use_gpu = torch.cuda.is_available()
    gpu_allocation = 1 if use_gpu else 0
    actor_options = {"num_gpus": gpu_allocation}
    langsam_actor = LangSAM.options(**actor_options).remote(use_gpu=use_gpu)

    args = parse_args()
    


    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=True)
    env.seed(args.seed)
    # load logger
    logger = Logger(case_dir=args.testing_case_dir)
    # load graspnet
    graspnet = Graspnet()

    if os.path.exists(args.testing_case_dir):
        filelist = os.listdir(args.testing_case_dir)
        filelist.sort(key=lambda x:int(x[4:6]))
    if args.testing_case != None:
        filelist = [args.testing_case]
    case = 0
    iteration = 0
    for f in filelist:
        f = os.path.join(args.testing_case_dir, f)

        logger.episode_reward_logs = []
        logger.episode_step_logs = []
        logger.episode_success_logs = []
        for episode in range(num_episode):
            episode_reward = 0
            episode_steps = 0
            done = False
            reset = False

            while not reset:
                env.reset()
                reset, lang_goal = env.add_object_push_from_file(f)
                print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

            while not done:
                # check if one of the target objects is in the workspace:
                out_of_workspace = []
                for obj_id in env.target_obj_ids:
                    pos, _, _ = env.obj_info(obj_id)
                    if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                        or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                        out_of_workspace.append(obj_id)
                if len(out_of_workspace) == len(env.target_obj_ids):
                    print("\033[031m Target objects are not in the scene!\033[0m")
                    break


                color_image, depth_image, mask_image = utils.get_true_heightmap(env)





                image = "color_map.png"
                image_pil = langsamutils.load_image(image)

                image_path = 'color_map.png'
                base64_image = load_image_as_base64(image_path)
                # 输入文本
                input_text = lang_goal

                # 配置输入消息
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Given a 224x224 input image and the provided instruction, perform the following steps:\n"
                            "Target Object Selection:\n"
                            "Identify the object in the image that best matches the instruction. If the target object is found, select it as the target object.\n"
                            "If the target object is not visible, select the most cost-effective object or object part considering ease of grasping, importance, and safety.\n"
                            "If the object has a handle or a part that is easier or safer to grasp, strongly prefer to select that part.\n"
                            "Consider the geometric shape of the objects and the gripper's success rate when selecting the target object or object part.\n"
                            "Output the name of the selected object or object part as [object:color and name] or [object part:color and name]..\n"
                            "Round object means like ball. Cup is different from mug."
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
                    # Call OpenAI API
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
                    # Output API response
                    logging.info(output)
                    result = process_grasping_result(output)
                    print_grasping_result(result)
                    with open('grasping_result_log.json', 'w') as log_file:
                        json.dump(result, log_file, indent=4)
                    wandb.log({"gpt4o_output": output})
                except Exception as e:
                    logging.error(f"Error with OpenAI API request: {e}")


                finally:

                    if not result['selected_object']:

                        fallback_object = select_fallback_object(result['objects'])

                        if fallback_object:

                            goal = fallback_object['name']

                        else:

                            goal = lang_goal

                    else:

                        goal = result['selected_object']

                    preferred_grasping_location = 5  # Default to center if not specified
                    for obj in result['objects']:
                        if obj['name'] == goal:
                            preferred_grasping_location = obj.get('preferred_grasping_location', 5)

                    masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, goal))
                    if masks is None or masks.numel() == 0:
                        masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, lang_goal))
                        if masks is None or masks.numel() == 0:
                            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, "object"))

                   boxes_list = boxes.cpu().numpy().tolist()  # Convert to list
                    cropping_box = create_cropping_box_from_boxes(boxes_list,
                                                                  (color_image.shape[1], color_image.shape[0]))

                    visualize_cropping_box(color_image, cropping_box)
                    ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil))
                    bbox_images, bbox_positions = utils.convert_output(image_pil, boxes, phrases, logits, color_image, depth_image, mask_image, preferred_grasping_location)

                    # graspnet
                    pcd = utils.get_fuse_pointcloud(env)
                    cropped_pcd = crop_pointcloud(pcd, cropping_box,color_image, depth_image, WORKSPACE_LIMITS)

                    if cropped_pcd is None or len(np.asarray(cropped_pcd.points)) == 0:
                        print("\033[031m Cropped point cloud is empty!\033[0m")
                        continue
                    # Visualize the cropped point cloud
                    # o3d.visualization.draw_geometries([pcd, cropped_pcd], window_name="Cropped Point Cloud")

                    # o3d.visualization.draw_geometries([ cropped_pcd], window_name="Cropped Point Cloud")
                    #

                    # Perform grasp detection on the cropped point cloud
                    with torch.no_grad():
                        grasp_pose_set, _, _ = graspnet.grasp_detection(cropped_pcd, env.get_true_object_poses())
                    print("Number of grasping poses", len(grasp_pose_set))
                    logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")

                    if len(grasp_pose_set) == 0:
                        with torch.no_grad():
                            grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses())
                        print("Number of grasping poses", len(grasp_pose_set))
                        logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")
                        if len(grasp_pose_set) == 0:
                            break
                    # preprocess
                    remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size))
                    logger.save_bbox_images(iteration, remain_bbox_images)
                    logger.save_heightmaps(iteration, color_image, depth_image)
                    if bboxes == None:
                        break

                    if len(grasp_pose_set) == 1:
                        action_idx = 0
                    else:
                        with torch.no_grad():
                            action_idx = select_action(bboxes, pos_bboxes, lang_goal, grasps)
                    action = grasp_pose_set[action_idx]




                    reward, done = env.step(action)
                    iteration += 1
                    episode_steps += 1
                    episode_reward += reward
                    print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))
                    wandb.log({"episode": episode, "step": episode_steps, "reward": reward})

                    if episode_steps == args.max_episode_step:
                        break


            logger.episode_reward_logs.append(episode_reward)
            logger.episode_step_logs.append(episode_steps)
            logger.episode_success_logs.append(done)
            logger.write_to_log('episode_reward', logger.episode_reward_logs)
            logger.write_to_log('episode_step', logger.episode_step_logs)
            logger.write_to_log('episode_success', logger.episode_success_logs)
            print("\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, episode_steps, round(episode_reward, 2), done))

            if episode == num_episode - 1:
                avg_success = sum(logger.episode_success_logs)/len(logger.episode_success_logs)
                avg_reward = sum(logger.episode_reward_logs)/len(logger.episode_reward_logs)
                avg_step = sum(logger.episode_step_logs)/len(logger.episode_step_logs)

                success_steps = []
                for i in range(len(logger.episode_success_logs)):
                    if logger.episode_success_logs[i]:
                        success_steps.append(logger.episode_step_logs[i])
                if len(success_steps) > 0:
                    avg_success_step = sum(success_steps) / len(success_steps)
                else:
                    avg_success_step = 1000

                result_file = os.path.join(logger.result_directory, "case" + str(case) + ".txt")
                with open(result_file, "w") as out_file:
                    out_file.write(
                        "%s %.18e %.18e %.18e %.18e\n"
                        % (
                            lang_goal,
                            avg_success,
                            avg_step,
                            avg_success_step,
                            avg_reward,
                        )
                    )
                case += 1
                print("\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(lang_goal, avg_step, avg_success_step, avg_reward, avg_success))
                logging.info(
                    f"Language goal: {lang_goal}, average steps: {avg_step}/{avg_success_step}, average reward: {avg_reward}, average success: {avg_success}")
                wandb.log({
                    "lang_goal": lang_goal,
                    "avg_success": avg_success,
                    "avg_step": avg_step,
                    "avg_success_step": avg_success_step,
                    "avg_reward": avg_reward
                })


    ray.shutdown()
