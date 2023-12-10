# 3D Gaussian Splatting SAM



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```bash
# Install dependencies of Gaussian Splatting
git clone --recursive https://github.com/yi-cheng-liu/3d_gaussian_sam.git

# Setting up environment of Gaussian Splatting
cd 3d_gaussian/
conda env create --file environment.yml
conda activate gaussian_splatting
cd ..
```

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


```bash
# Colmap
git clone https://github.com/colmap/colmap.git
```

```bash
# NerfStudio
git clone -b gaussian_splatting https://github.com/yzslab/nerfstudio.git
```