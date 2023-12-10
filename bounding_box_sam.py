import cv2
import argparse
import clip
import torch
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './segment-anything')
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image, ImageDraw
from typing import List
from tqdm import tqdm

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

@torch.no_grad()
def retriev(elements: List[Image.Image], search_text: str, preprocess, device, model) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def get_bounding_box_from_mask(mask):
    # Find all the non-zero points in the mask
    _, ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    # Get the bounding box coordinates
    tune = 40
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min = max(x_min-tune, 0)
    x_max = min(x_max+tune, 1920 - 1)
    y_min = max(y_min-tune, 0)
    y_max = min(y_max+tune, 1080 - 1)
    return [x_min, y_min, x_max, y_max]

def main(args):

    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type
    source = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    # image sorce repository
    image_files = sorted(os.listdir(source))

    start_time = time.time()

    bounding_box = np.array([430, 170, 1275, 910])
    skip_image = 0

    for i, image_file in enumerate(tqdm(image_files)):
        if i < skip_image:
            print(i * 5)
            continue
        image_file = image_files[i]
        image_path = os.path.join(source, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box[None, :],
            multimask_output=False,
        )

        mask_array = np.array(masks)
        new_bounding_box = get_bounding_box_from_mask(mask_array)
        bounding_box = np.array(new_bounding_box)

        # Transparent background
        original_image = Image.open(image_path).convert('RGBA')
        original_image_array = np.array(original_image)
        background = np.zeros_like(original_image_array)  # transparent background
        background[..., 3] = 0  # Set alpha channel to 0 (fully transparent)

        # Apply the mask to the original image
        # transparnent
        masked_image_array = np.where(mask_array[-1, :, :, None], original_image_array, background)
        # black
        # alpha_channel = np.where(mask_array[-1], 255, 0)
        # masked_image_array = np.dstack((masked_image_array[:, :, :3], alpha_channel)).astype(np.uint8) # black background
        masked_image = Image.fromarray(masked_image_array)
        overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        
        # Draw mask
        # draw = ImageDraw.Draw(overlay_image)
        # dot_color = (0, 255, 0, 255)  # Example: Green color for dots
        # for point in selected_points:
        #     draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=dot_color)

        # Save the result image
        result_image = Image.alpha_composite(masked_image.convert('RGBA'), overlay_image)
        output_path = f"{output_folder}/{image_file[:-4]}.JPG"
        result_image.convert('RGB').save(output_path)

    print(f"Total Time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation and Image Processing Script")

    parser.add_argument("-i", "--input_folder", type=str, default="Dataset/360_v2/kitchen/images_8", help="Path to the input image folder")
    parser.add_argument("-o", "--output_folder", type=str, default="Dataset/kitchen_segmented/images", help="Path to save the output images")
    parser.add_argument("-p", "--clip_prompt", type=str, default="bulldozer", help="Prompt text for CLIP model")
    parser.add_argument("--sam_checkpoint", type=str, default="segment-anything/model_checkpoint/sam_vit_h_4b8939.pth", help="Path to the SAM checkpoint file")
    parser.add_argument("--model_type", type=str, default="vit_h", help="Type of the model to use")

    args = parser.parse_args()
    main(args)