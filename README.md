# 3D Gaussian Splatting SAM

<p align="center">
  <img src="https://github.com/yi-cheng-liu/3d_gaussian_sam/blob/main/.assets/bulldozer.gif" alt="gif">
</p>
This is the final project for EECS542: Advanced Topics for Computer Vision. 


## 🚀 1. Motivation
The recent Neural Radiance Fields (NeRF) offers impressive results on building 3D object given several surrounding images. However, it still has some drawbacks, such as its reliance on the Multi-Layer Perceptron (MLP) network, and the time-consuming training process. NVIDIA addresses the problem with [instant-ngp](https://github.com/NVlabs/instant-ngp), a solution that significantly accelerates NeRF's training, but still with some blurry effect on the object. Thus, to further enhance the fine details, 3D Gaussian splatting employs Gaussian-based representation. Even though, object still have to be extracted, which Segment Anything Model (SAM) has great preformance on such task. Integrating SAM's robust segmentation capabilities with the intricate 3D Gaussian representation, we introduce a novel method aimed at delivering unparalleled quality on 3D objects given 2D images.

## 💻 2. Prerequisites

#### Gaussian Splatting
This is a new emerging 3D reconstruction tool with fast training speed and high quality. Official website could be found in [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
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
This model is for generating the initial point cloud from the video. Official documentaion could be found in [here](https://colmap.github.io/). 
```bash
git clone https://github.com/colmap/colmap.git
```

#### NerfStudio
This folder is for viewing the training reuslt of the Gaussian Splatting
```bash
git clone -b gaussian_splatting https://github.com/yzslab/nerfstudio.git
```
#### Dataset
this project consists of two datasets, MipNeRF-360 and Food-360. MipNeRF-360 could be found in the official [website](https://jonbarron.info/mipnerf360/) of MipNeRF. Food-360 dataset could be found in [here](https://www.kaggle.com/datasets/liuyiche/food-360-dataset/). 



The structure of the projecct will look like this
```bash
├── 3d_gaussian
├── bounding_box_sam.py
├── clip_sam.py
├── colmap
├── Dataset
│   ├── 360_extra_scenes
│   ├── 360_v2
    ├── Food-360
│   ├── convert_video
├── nerfstudio
├── output
├── README.md
├── segment-anything
│   ├── ...
│   └── model_checkpoint
└── train.py
```



## 📄 4. Related Papers
 
+ [**Gaussian-Splatting**](https://github.com/graphdeco-inria/gaussian-splatting)
+ [**Segment-Anything**](https://github.com/facebookresearch/segment-anything)
+ [**COLMAP**](https://github.com/colmap/colmap)
+ [**NerfStudio**](https://github.com/yzslab/nerfstudio)
+ [**SA3D**](https://github.com/Jumpat/SegmentAnythingin3D)


** a new paper which addressed the task with better result
+ [**SAGA**](https://github.com/Jumpat/SegAnyGAussians)

## 📫 5. Contact

+ Tien-Li Lin, Email: tienli@umich.edu
+ Yi-Cheng Liu, Email: liuyiche@umich.edu

