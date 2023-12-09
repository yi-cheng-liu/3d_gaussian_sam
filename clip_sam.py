import cv2
import argparse
import clip
import torch
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
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




def main(args):

    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type
    source = args.input_folder
    output_folder = args.output_folder
    clip_prompt = args.clip_prompt

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download the model weights to load them herert 
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_image = True

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    # image sorce repository
    # source = 'Dataset/kitchen/images_4'
    image_files = sorted(os.listdir(source))

    previous_selected_point = []

    start_time = time.time()

    for i, image_file in enumerate(tqdm(image_files)):
        if i < 440:
            continue
        previous_selected_point = []

        
        image_file = image_files[i]
        # get image path in order and read image
        image_path = os.path.join(source, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # if first_image:
        if True:
            masks = mask_generator.generate(image)
            # Cut out all masks
            image_masked = Image.open(image_path)
            cropped_boxes = []
            for mask in masks:
                cropped_boxes.append(segment_image(image_masked, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))
            # Load CLIP
            model, preprocess = clip.load("ViT-L/14@336px", device=device)
            # scores = retriev(cropped_boxes, "bulldozer")
            scores = retriev(cropped_boxes, f"{clip_prompt}", preprocess, device, model)
            indices = get_indices_of_values_above_threshold(scores, 0.05)
            segmentation_masks = []
            mask_pixels = []

            for seg_idx in indices:
                segmentation_mask = masks[seg_idx]["segmentation"]
                segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
                segmentation_masks.append(segmentation_mask_image)
                mask_pixels.append(segmentation_mask)

            original_image = Image.open(image_path)
            overlay_image = Image.new('RGBA', image_masked.size, (0, 0, 0, 0))
            overlay_color = (255, 0, 0, 200)
            dot_color = (0, 255, 0)  # Green color for dots

            draw = ImageDraw.Draw(overlay_image)
            for segmentation_mask_image in segmentation_masks:
                draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

            for i, mask in enumerate(mask_pixels):
                num_pixels = np.sum(mask == 1)  # Count pixels where mask is 1
                # Randomly select 10 points within the mask
                y_indices, x_indices = np.where(mask == 1)
                selected_points = random.sample(list(zip(x_indices, y_indices)), min(10, len(x_indices)))
                selected_points_nparray = np.array(selected_points)

                # Append the numpy array of selected points to previous_selected_point
                previous_selected_point.append(selected_points_nparray)

                # Draw these points on the final image
                for point in selected_points:
                    draw.point(point, fill=dot_color)

            # Concatenate all the selected points into a single numpy array
            previous_selected_point = np.vstack(previous_selected_point)

            result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)

            # Define the path where you want to save the image, including the filename and extension
            # output_path = f"Dataset/kitchen_segmented/images/{image_file[:8]}.JPG"
            output_path = f"{output_folder}/{image_file[:8]}.JPG"

            # Save the result image in the specified path
            # result_image.convert("RGB").save(output_path)
            first_image = False
        
        # else:
        predictor.set_image(image)
        input_points = previous_selected_point
        input_labels = np.array([1, 1])
        input_labels = np.ones(len(input_points))

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )

        mask_array = np.array(masks)
        
        y_indices, x_indices = np.where(mask_array[-1] == 1)

        selected_points = []
        if len(x_indices) > 0:
            selected_points = random.sample(list(zip(x_indices, y_indices)), min(20, len(x_indices)))
            previous_selected_point = np.array(selected_points)  # Update the previous_selected_point


        ######################################### black
        # # Load the original image
        # original_image = Image.open(image_path)
        # # original_image_array = np.array(original_image)
        # original_image_array = np.array(original_image.convert('RGBA'))  # Ensure the image is RGBA
        # alpha_channel = np.where(mask_array[-1], 255, 0)

        # # Create a black background image
        # black_background = np.zeros_like(original_image_array)

        # # Apply the mask to the original image
        # # masked_image_array = np.where(mask_array[-1, :, :, None], original_image_array, black_background)
        # # masked_image = Image.fromarray(masked_image_array)
        # masked_image_array = np.dstack((original_image_array[:, :, :3], alpha_channel)).astype(np.uint8)
        # masked_image = Image.fromarray(masked_image_array)

        # # Create an overlay image for the selected points
        # overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        # draw = ImageDraw.Draw(overlay_image)
        # # dot_color = (0, 255, 0, 255)  # Green color for dots
        # # for point in selected_points:
        # #     draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=dot_color)

        # # Combine the overlay with the masked image
        # # result_image = Image.alpha_composite(masked_image.convert('RGBA'), overlay_image)
        # result_image = Image.alpha_composite(masked_image, overlay_image)
        ######################################### black



        ######################################### transparent
        # Load the original image and convert it to RGBA (if it's not already)
        original_image = Image.open(image_path).convert('RGBA')
        original_image_array = np.array(original_image)

        # Create a transparent background image (RGBA)
        transparent_background = np.zeros_like(original_image_array)
        transparent_background[..., 3] = 0  # Set alpha channel to 0 (fully transparent)

        # Apply the mask to the original image
        # Where the mask is true, copy the RGB values from the original image and set alpha to 255 (opaque)
        masked_image_array = np.where(mask_array[-1, :, :, None], original_image_array, transparent_background)

        # Convert the array back to an Image
        masked_image = Image.fromarray(masked_image_array)

        # Create an overlay image for the selected points
        overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_image)
        # dot_color = (0, 255, 0, 255)  # Green color for dots
        # for point in selected_points:
        #     draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=dot_color)

        # Combine the overlay with the masked image
        result_image = Image.alpha_composite(masked_image.convert('RGBA'), overlay_image)
        ######################################### transparent



        # Save the result image
        output_path = f"{output_folder}/{image_file[:-4]}.JPG"
        result_image.convert('RGB').save(output_path)
    print(f"Total Time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation and Image Processing Script")

    parser.add_argument("-i", "--input_folder", type=str, default="Dataset/kitchen/images_4", help="Path to the input image folder")
    parser.add_argument("-o", "--output_folder", type=str, default="Dataset/kitchen_segmented/images", help="Path to save the output images")
    parser.add_argument("-p", "--clip_prompt", type=str, default="bulldozer", help="Prompt text for CLIP model")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="Path to the SAM checkpoint file")
    parser.add_argument("--model_type", type=str, default="vit_h", help="Type of the model to use")

    args = parser.parse_args()
    main(args)