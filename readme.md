# Segmentation Guided Medical Image Generation using Diffusion

This is the main repository for our work on the WS24 master practical course `Applied Deep Learning in Medicine (IN2106, IN4314)`. 

Topic: **Image Translation and Generation with Latent Diffusion**

## Structure
```
/vol/miltank/projects/practical_WS2425/diffusion/
├── code (this repository)
│   ├── amos_segmentator                : Unet based 2D segmenation model for amos data 
│   ├── eda                             : Exploratory Data Analysis
│   ├── evaluation                      : Scripts for evaluating image generations
│   ├── segmenttation-guided-diffusion  : Diff-Moddel for seg guided image generation                 
│   ├── utils                           : Utility functions
│   │   ├── dataset                     : Data related code
│   │   │   ├── amos.py                 : Custom dataset for amos data 
│   │   │   ├── custom.py               : Custom dataset for any image folder
│   │   │   ├── transforms.py           : Custom transforms for image-label pairs       
│   │   ├── filter_mask                 : Script for filtering images
│   │   ├── slicing                     : Script for slicing nifti images
│   │   └── utils.py                    : Misc Utility functions
│   └── vqvae                           : VQ-VAE model for image and mask embedding
└── data                                
    ├── amos22                          : Amos22 nifti
    ├── amos_robert                     : Amos TotalVibeSegmentor nifti
    ├── amos_robert_embeddings          : VQ-VAE embeddings
    ├── amos_robert_slices              : Slices of amos data, axial, coronal, sagittal
    ├── amos_slices                     : Slices of amos data, axial, coronal, sagittal
    └── test_data                       : Small datasubset for experimenting
```

## Description
Previous work mostly focused on segmentation of medical images. The annotation of different structures in medical images, which is a crucial step in medical image analysis. Many deep learning structures exist to successfully automate this process.

The goal of this project is to leverage modern duffusion architecture and now highly available annotations to reverse above process and generate medical images from segmentation masks. Herefor we use a generative diffusion model which reverse process is conditioned on or guided by the segmentation masks.

This project is adapted to work with the [amos22](https://amos22.grand-challenge.org/) dataset combined with enhanced segmentations of up to 72 different classes created using [TotalVibeSegmentor](https://github.com/robert-graf/TotalVibeSegmentator).

Further details can be found in the [project report](./report.pdf).

## Installation
To install the required environment, run `pip install -r requirements.txt`.

## Usage
Each folder contains names python scripts for the respective task. For more complex script calls, a bash script with default parameters is provided.

## Showcase
TODO showcase folder