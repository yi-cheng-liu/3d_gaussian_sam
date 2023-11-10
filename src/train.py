import sys
sys.path.append('~/Desktop/3d_gaussian_sam/')  # Replace with the actual path
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import time
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
print("import success. ")

def show_anns(anns):
   if len(anns) == 0:
       return
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   ax = plt.gca()
   ax.set_autoscale_on(False)
   for ann in sorted_anns:
       m = ann['segmentation']
       img = np.ones((m.shape[0], m.shape[1], 3))
       color_mask = np.random.random((1, 3)).tolist()[0]
       for i in range(3):
           img[:,:,i] = color_mask[i]
       np.dstack((img, m*0.35))
       ax.imshow(np.dstack((img, m*0.35)))

# Initilize the SAM model
model_type = "default"
checkpoint_path = "segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.cuda()
print("load model success. ")
# predictor = SamPredictor(sam)
print("initilize the sam model success. ")

# Load your image
image_path = "Dataset/kitchen/images/DSCF0656.JPG"
# predictor.set_image(image_path)
# input_prompts = ["bulldozer"]
# masks, _, _ = predictor.predict(input_prompts)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Unable to find or open the file at path: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_generator = SamAutomaticMaskGenerator(sam)
start = time.time()
masks = mask_generator.generate(image)
print(f"Total time: {time.time() - start}")
print("success. ")

plt.figure(figsize=(12, 9))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
#plt.savefig(os.path.join('outputs', "Masked_image"), bbox_inches='tight')
