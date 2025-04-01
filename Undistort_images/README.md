# Image Undistortion and Comparison Tools

This repository contains Python scripts for undistorting images using various camera models and for comparing images using Structural Similarity Index (SSIM).

## Description

The repository provides two main scripts:

1. ``: A script to undistort images based on different camera models.
2. ``: A script to compare two images using SSIM and compute their absolute difference.

The repository includes three directories with example images and corresponding camera parameters for different models:

- `AKDK_IR`
- `Fujinon185`
- `Panoramis`

## Installation

Ensure you have Python 3.8+ and the required dependencies installed. Install them using:

```bash
pip install numpy opencv-python pyyaml matplotlib scikit-image
```

## Usage

### Undistorting Images

To undistort an image, use the `undistort.py` script:

```bash
python undistort.py -i <path_to_image> -p <path_to_parameters_directory> -f <scaling_factor> -s
```

#### Arguments:

- `-i, --image`: Path to the input distorted image.
- `-p, --parameters`: Path to the directory containing camera parameters (.txt files).
- `-f, --factor`: Scaling factor for the new camera matrix (default: 1.0).
- `-s, --show`: Optional flag to display the undistorted images.

#### Example:

```bash
python3 undistort.py -i AKDK_IR/AKDK_IR.png -p AKDK_IR/ -f 1.05 -s
```

### Comparing Images

To compare two images using SSIM and compute their absolute difference, use the `SSIM.py` script:

```bash
python SSIM.py -i1 <path_to_image1> -i2 <path_to_image2> -o <output_directory>
```

#### Arguments:

- `-i1, --image1`: Path to the first input image.
- `-i2, --image2`: Path to the second input image.
- `-o, --output_dir`: Directory to save the difference image (optional, default: same directory as image1).

#### Example:

```bash
python SSIM.py -i1 ./AKDK_IR/undistort_AKDK_RPM_calibrated_factor_1.052.png -i2 ./AKDK_IR/undistort_AKDK_OFM_calibrated_factor_1.052.png -o output/
```

## Camera Models

The `undistort.py` script supports the following camera models:

- **OFM** (Omnidirectional Fisheye Model) – Uses OpenCV's `cv2.fisheye.undistortImage()`.
- **UCM** (Unified Camera Model) – Uses OpenCV's `cv2.omnidir.undistortImage()`.
- **EFM** (Equidistant Fisheye Model) – Custom implementation using inverse mapping.
- **RPM** (Radial Polynomial Model) – Uses OpenCV's `cv2.initUndistortRectifyMap()`.
- **CPM** (Cartesian Polynomial Model) – Currently unsupported (requires `py-OcamCalib`).
