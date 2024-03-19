# 3D Gaussian Splatting SAM

<p align="center">
  <img src="https://github.com/yi-cheng-liu/3d_gaussian_sam/blob/main/.assets/bulldozer.gif" alt="gif">
</p>
This is the final project for EECS542: Advanced Topics for Computer Vision. 


## 🚀 1. Motivation
The recent Neural Radiance Fields (NeRF) offers impressive results in building 3D objects given several surrounding images. However, it still has some drawbacks, such as its reliance on the Multi-Layer Perceptron (MLP) network, and the time-consuming training process. NVIDIA addresses the problem with [instant-ngp](https://github.com/NVlabs/instant-ngp), a solution that significantly accelerates NeRF's training, but still with some blurry effect on the object. Thus, to further enhance the fine details, 3D Gaussian splatting employs Gaussian-based representation. Even though, objects still have to be extracted, which Segment Anything Model (SAM) has great performance on such a task. Integrating SAM's robust segmentation capabilities with the intricate 3D Gaussian representation, we introduce a novel method aimed at delivering unparalleled quality on 3D objects given 2D images.

## 💻 2. Prerequisites

#### Gaussian Splatting
This is a new emerging 3D reconstruction tool with fast training speed and high quality. The official website can be found in [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
```bash
# Install dependencies of Gaussian Splatting
git clone --recursive https://github.com/yi-cheng-liu/3d_gaussian_sam.git

# Setting up environment of Gaussian Splatting
cd 3d_gaussian/
conda env create --file environment.yml
conda activate gaussian_splatting
cd ..
```

#### Segment Anything
This model is for segmenting the object from 2D images. Original paper could be found in the [website](https://segment-anything.com/). 
```bash
# Setting up environment of Segment-Anything
cd segment-anything
pip install -e .

# Dependencies for clip model
pip install torch opencv-python Pillow tqdm
pip install git+https://github.com/openai/CLIP.git

# The following optional dependencies are necessary for mask 
# post-processing, saving masks in COCO format, the example notebooks, 
# and exporting the model in ONNX format. 
pip install opencv-python pycocotools matplotlib onnxruntime onnx
cd ..
```

Download weights for the segmentation model from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put it in `segment-anything/model_checkpoint/`.


#### Colmap (Structure from Motion)
This model is for generating the initial point cloud from the video. Official documentation can be found in [here](https://colmap.github.io/). 
```bash
git clone https://github.com/colmap/colmap.git
```

#### NerfStudio
This folder is for viewing the training result of the Gaussian Splatting
```bash
git clone -b gaussian_splatting https://github.com/yzslab/nerfstudio.git
```
## 📊 3. Dataset
this project consists of two datasets, MipNeRF-360 and Food-360. MipNeRF-360 could be found in the official [website](https://jonbarron.info/mipnerf360/) of MipNeRF. Food-360 dataset could be found in [here](https://www.kaggle.com/datasets/liuyiche/food-360-dataset/). 

#### MipNeRF-360
```bash
cd datasets
# Dataset Pt.1
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

# Dataset Pt.2
wget https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
```
#### Food-360
```bash
# Setup the kaggle api first
kaggle datasets download liuyiche/food-360-dataset/
```

The structure of the project will look like this
```bash
├── 3d_gaussian
├── bounding_box_sam.py
├── clip_sam.py
├── colmap
├── Dataset # unzip before use
│   ├── 360_extra_scenes.zip
│   ├── 360_v2.zip
│   ├── Food-360-dataset.zip
│   └── convert_video
├── nerfstudio
├── output
├── EADME.md
├── segment-anything
│   ├── ...
│   └── model_checkpoint
└── train.py
```

## 🏃 4. Run the project
```bash
# Bounding box
python bounding_box.py
# CLIP
python clip_sam.py

# Train with Gaussian Splatting
# python train.py -s <path to COLMAP or NeRF Synthetic dataset>
python train.py -s datasets/chips/chips/images_segmented
```

> 💡 See some of our output
[🚜 *Bulldozer*](https://my.spline.design/untitled-080f52613f52436c2549075b3ca103c0/)
[🥤 *Cola-Cola*](https://my.spline.design/untitled-41db23f91cdea4b0c7f324464c729c82/)
[🍌 *Banana*](https://my.spline.design/untitled-fef6e10c43d2824caa1d48b4638b57fd/)
[🍟 *Chips*](https://my.spline.design/untitled-e82b21ae118a96f990f171db0a223322/)


## 📄 5. Related Papers
 
+ [**Gaussian-Splatting**](https://github.com/graphdeco-inria/gaussian-splatting)
+ [**Segment-Anything**](https://github.com/facebookresearch/segment-anything)
+ [**COLMAP**](https://github.com/colmap/colmap)
+ [**NerfStudio**](https://github.com/yzslab/nerfstudio)
+ [**SA3D**](https://github.com/Jumpat/SegmentAnythingin3D)


** A new paper that addressed the task with a better result
+ [**SAGA**](https://github.com/Jumpat/SegAnyGAussians)

## 📫 6. Contact

+ Tien-Li Lin, Email: tienli@umich.edu
+ Yi-Cheng Liu, Email: liuyiche@umich.edu

